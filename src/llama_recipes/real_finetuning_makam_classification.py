import json
import os
import random
from dataclasses import asdict

import fire
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import ddp_config as DDP_CONFIG
from llama_recipes.configs import (
    makam_classification_config as MAKAM_CLASSIFICATION_CONFIG,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForSequenceClassification,
)
from llama_recipes.utils.config_utils import update_config, generate_dataset_config
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset


def _strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    return cleaned


def _filter_state_dict_by_shape(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    return filtered, skipped


def _load_music_tokenizer_from_config(model_config):
    return MusicTokenizer(
        timeshift_vocab_size=model_config.onset_vocab_size,
        dur_vocab_size=model_config.dur_vocab_size,
        octave_vocab_size=model_config.octave_vocab_size,
        pitch_class_vocab_size=model_config.pitch_class_vocab_size,
        instrument_vocab_size=model_config.instrument_vocab_size,
        velocity_vocab_size=model_config.velocity_vocab_size,
        sos_token=model_config.sos_token,
        eos_token=model_config.eos_token,
        pad_token=model_config.pad_token,
        microtonal=getattr(model_config, "microtonal", False),
        pitchbend_sensitivity=getattr(model_config, "pitchbend_sensitivity", 2.0),
        microtonal_resolution=getattr(model_config, "microtonal_resolution", 1),
    )


class CompoundClassificationCollator:
    """Pad variable-length compound-token sequences for classification."""

    def __init__(self, pad_token: int):
        self.pad_token = int(pad_token)

    def __call__(self, batch):
        if len(batch) == 0:
            raise ValueError("Received an empty batch")

        max_len = max(item["input_ids"].shape[0] for item in batch)
        feature_dim = int(batch[0]["input_ids"].shape[1])
        bs = len(batch)

        input_ids = torch.full(
            (bs, max_len, feature_dim),
            fill_value=self.pad_token,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        labels = torch.zeros((bs,), dtype=torch.long)

        for i, item in enumerate(batch):
            seq = item["input_ids"]
            seq_len = seq.shape[0]
            input_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1
            labels[i] = item["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device)
    return moved


def evaluate_classification(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(batch["labels"].detach().cpu().tolist())

    if n_batches == 0:
        raise ValueError("Validation dataloader is empty")

    eval_loss = total_loss / n_batches
    correct = sum(int(p == t) for p, t in zip(all_preds, all_targets))
    accuracy = correct / max(1, len(all_targets))
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "eval_loss": eval_loss,
        "eval_accuracy": accuracy,
        "eval_macro_f1": float(macro_f1),
    }


def train_classification(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    train_cfg,
    device,
):
    model.train()
    global_step = 0
    best_macro_f1 = float("-inf")
    best_metrics = None
    running_train_losses = []
    grad_acc_steps = max(1, int(train_cfg.gradient_accumulation_steps))

    if train_cfg.save_model:
        os.makedirs(train_cfg.output_dir, exist_ok=True)

    for epoch in range(int(train_cfg.num_epochs)):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / grad_acc_steps
            loss.backward()

            if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_dataloader):
                if train_cfg.gradient_clipping and train_cfg.gradient_clipping_threshold > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clipping_threshold)
                optimizer.step()
                optimizer.zero_grad()

            raw_loss = float((loss.detach().cpu().item()) * grad_acc_steps)
            epoch_loss_sum += raw_loss
            epoch_steps += 1
            global_step += 1

            should_validate = (
                train_cfg.run_validation
                and eval_dataloader is not None
                and train_cfg.validation_interval > 0
                and global_step % train_cfg.validation_interval == 0
            )
            if should_validate:
                metrics = evaluate_classification(model, eval_dataloader, device)
                print(
                    f"step={global_step} "
                    f"eval_loss={metrics['eval_loss']:.6f} "
                    f"eval_accuracy={metrics['eval_accuracy']:.6f} "
                    f"eval_macro_f1={metrics['eval_macro_f1']:.6f}"
                )

                if metrics["eval_macro_f1"] > best_macro_f1:
                    best_macro_f1 = metrics["eval_macro_f1"]
                    best_metrics = metrics
                    if train_cfg.save_model:
                        ckpt_path = os.path.join(train_cfg.output_dir, "best_makam_classifier.pt")
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "epoch": epoch,
                                "global_step": global_step,
                                "best_eval_macro_f1": best_macro_f1,
                            },
                            ckpt_path,
                        )
                        print(f"Saved best checkpoint to {ckpt_path}")

        scheduler.step()
        epoch_train_loss = epoch_loss_sum / max(1, epoch_steps)
        running_train_losses.append(epoch_train_loss)
        print(f"epoch={epoch} train_loss={epoch_train_loss:.6f}")

        if train_cfg.run_validation and eval_dataloader is not None:
            metrics = evaluate_classification(model, eval_dataloader, device)
            print(
                f"epoch={epoch} "
                f"eval_loss={metrics['eval_loss']:.6f} "
                f"eval_accuracy={metrics['eval_accuracy']:.6f} "
                f"eval_macro_f1={metrics['eval_macro_f1']:.6f}"
            )
            if metrics["eval_macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["eval_macro_f1"]
                best_metrics = metrics
                if train_cfg.save_model:
                    ckpt_path = os.path.join(train_cfg.output_dir, "best_makam_classifier.pt")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_eval_macro_f1": best_macro_f1,
                        },
                        ckpt_path,
                    )
                    print(f"Saved best checkpoint to {ckpt_path}")

    results = {
        "avg_train_loss": sum(running_train_losses) / max(1, len(running_train_losses)),
        "best_eval_macro_f1": best_macro_f1 if best_metrics is not None else None,
        "best_eval_accuracy": best_metrics["eval_accuracy"] if best_metrics is not None else None,
        "best_eval_loss": best_metrics["eval_loss"] if best_metrics is not None else None,
        "total_steps": global_step,
    }
    return results


def main(**kwargs):
    train_cfg = TRAIN_CONFIG()
    fsdp_cfg = FSDP_CONFIG()
    ddp_cfg = DDP_CONFIG()
    makam_cfg = MAKAM_CLASSIFICATION_CONFIG()
    update_config((train_cfg, fsdp_cfg, ddp_cfg, makam_cfg), **kwargs)

    if train_cfg.enable_fsdp or train_cfg.enable_ddp:
        raise NotImplementedError(
            "real_finetuning_makam_classification.py currently supports single-device training only. "
            "DDP/FSDP support will be added in a later step."
        )

    if train_cfg.dataset != "symbtr_dataset_eval":
        raise ValueError(
            "This entrypoint is scoped to makam classification and expects --dataset symbtr_dataset_eval"
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        raise RuntimeError(
            "No CUDA/XPU device found. Current train loop uses accelerator-only execution."
        )

    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(train_cfg.seed)
    if device.type == "xpu":
        torch.xpu.manual_seed(train_cfg.seed)

    model_config_path = makam_cfg.model_config_path
    model_config = LlamaConfig.from_pretrained(model_config_path)
    tokenizer = _load_music_tokenizer_from_config(model_config)

    dataset_config = generate_dataset_config(train_cfg, kwargs)
    if not hasattr(dataset_config, "label_map"):
        raise ValueError("Dataset config must contain label_map for classification")

    label_map_path = makam_cfg.label_map_path or dataset_config.label_map
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = len(label_map)
    if num_labels < 2:
        raise ValueError(f"Invalid num_labels={num_labels}; expected at least 2")

    model_config.num_labels = num_labels
    model_config.pad_token_id = int(tokenizer.pad_token)
    model_config.problem_type = "single_label_classification"
    model = LlamaForSequenceClassification(model_config)

    # Optional initialization from a pretrained checkpoint path.
    ckpt_path = train_cfg.trained_checkpoint_path
    if ckpt_path and os.path.exists(ckpt_path):
        raw = torch.load(ckpt_path, map_location="cpu")
        state_dict = raw.get("model_state_dict", raw)
        state_dict = _strip_module_prefix(state_dict)
        state_dict, skipped = _filter_state_dict_by_shape(model, state_dict)
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=bool(makam_cfg.checkpoint_strict),
        )
        print(
            f"Loaded checkpoint from {ckpt_path}. "
            f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}, skipped_shape_mismatch={len(skipped)}"
        )

    model.to(device)
    print(f"Using device: {device}")
    print(f"Training config: {asdict(train_cfg)}")
    print(f"Makam classification config: {asdict(makam_cfg)}")
    print(f"Dataset config: {dataset_config}")
    print(f"label_map_path={label_map_path}")
    print(f"num_labels={num_labels}")

    ds_train = get_preprocessed_dataset(tokenizer, dataset_config, split="train")
    ds_val = get_preprocessed_dataset(tokenizer, dataset_config, split="test")
    print(f"--> Training Set Length = {len(ds_train)}")
    print(f"--> Validation Set Length = {len(ds_val)}")

    collator = CompoundClassificationCollator(pad_token=int(tokenizer.pad_token))
    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=train_cfg.batch_size_training,
        shuffle=True,
        collate_fn=collator,
        num_workers=train_cfg.num_workers_dataloader,
        pin_memory=True,
        drop_last=True,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        ds_val,
        batch_size=train_cfg.val_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=train_cfg.num_workers_dataloader,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_cfg.gamma)

    results = train_classification(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader if train_cfg.run_validation else None,
        optimizer=optimizer,
        scheduler=scheduler,
        train_cfg=train_cfg,
        device=device,
    )

    for key, value in results.items():
        print(f"Key: {key}, Value: {value}")


if __name__ == "__main__":
    fire.Fire(main)
