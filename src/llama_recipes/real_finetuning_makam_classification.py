import json
import os
import random
import re
from datetime import datetime
from dataclasses import asdict

import fire
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import ddp_config as DDP_CONFIG
from llama_recipes.configs import (
    makam_classification_config as MAKAM_CLASSIFICATION_CONFIG,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForCausalLM,
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


def _latest_adapter_checkpoint_dir(root_dir):
    """Return latest {epoch}-{step}.safetensors checkpoint directory under root_dir."""
    pattern = re.compile(r"^(\d+)-(\d+)\.safetensors$")
    candidates = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if not os.path.isdir(full):
            continue
        match = pattern.match(name)
        if match is None:
            continue
        epoch = int(match.group(1))
        step = int(match.group(2))
        if os.path.exists(os.path.join(full, "adapter_model.safetensors")):
            candidates.append((epoch, step, full))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _find_microtonal_llama_config(ckpt_path):
    """Find checkpoint-side llama_config.json saved during microtonal training."""
    if not ckpt_path:
        return None

    # Support passing either root dir (checkpoints/microtonal_cpt) or
    # specific adapter dir (checkpoints/microtonal_cpt/19-400.safetensors).
    search_roots = []
    if os.path.isdir(ckpt_path):
        search_roots.append(ckpt_path)
        search_roots.append(os.path.dirname(ckpt_path))

    for root in search_roots:
        if not root:
            continue
        candidate = os.path.join(root, "ddp-microtonal_cpt", "llama_config.json")
        if os.path.exists(candidate):
            return candidate

    return None


def _load_checkpoint_into_model(model, ckpt_path, strict):
    """Load either PEFT adapter dir or torch/safetensors state dict into model."""
    if os.path.isdir(ckpt_path):
        adapter_path = ckpt_path
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            latest = _latest_adapter_checkpoint_dir(ckpt_path)
            if latest is None:
                raise ValueError(
                    f"No adapter checkpoint found in directory: {ckpt_path}. "
                    "Expected adapter_model.safetensors or subdirs like 19-400.safetensors/."
                )
            adapter_path = latest

        # Adapter is CAUSAL_LM; merge it into a causal model first, then transfer
        # matching backbone tensors into the classification model.
        # Rebuild config through LlamaConfig to avoid cross-module PretrainedConfig
        # identity mismatches in mixed transformers/peft environments.
        causal_cfg = LlamaConfig.from_dict(model.config.to_dict())
        causal_model = LlamaForCausalLM(causal_cfg)
        peft_model = PeftModel.from_pretrained(
            causal_model,
            adapter_path,
            is_trainable=False,
        )
        merged_causal = peft_model.merge_and_unload()

        merged_state = merged_causal.state_dict()
        filtered_state, skipped = _filter_state_dict_by_shape(model, merged_state)
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(
            f"Loaded and merged PEFT adapter from {adapter_path}. "
            f"transferred_keys={len(filtered_state)}, missing_keys={len(missing)}, "
            f"unexpected_keys={len(unexpected)}, skipped_shape_mismatch={len(skipped)}"
        )
        return model

    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path)
    else:
        raw = torch.load(ckpt_path, map_location="cpu")
        state_dict = raw.get("model_state_dict", raw)

    state_dict = _strip_module_prefix(state_dict)
    state_dict, skipped = _filter_state_dict_by_shape(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(strict))
    print(
        f"Loaded checkpoint from {ckpt_path}. "
        f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}, skipped_shape_mismatch={len(skipped)}"
    )
    return model


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


def _create_optimizer(model, train_cfg, device):
    """Choose an optimizer that fits available VRAM on small GPUs."""
    adamw = optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    if device.type != "cuda":
        return adamw, "adamw"

    trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adam_state_bytes = trainable_numel * 2 * 4  # exp_avg + exp_avg_sq in fp32
    free_bytes, _ = torch.cuda.mem_get_info()

    # Keep headroom for allocator fragmentation and transient kernels.
    if adam_state_bytes > int(0.85 * free_bytes):
        print(
            "Low-VRAM mode: switching optimizer to SGD "
            f"(AdamW state would need ~{adam_state_bytes / (1024**3):.2f} GiB, "
            f"free ~{free_bytes / (1024**3):.2f} GiB)."
        )
        sgd = optim.SGD(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        return sgd, "sgd"

    return adamw, "adamw"


def _setup_wandb(train_cfg, makam_cfg, model_config, dataset_config, kwargs):
    if not bool(train_cfg.use_wandb):
        return None

    try:
        import wandb
    except ImportError:
        print(
            "WARNING: use_wandb=True but wandb is not installed. "
            "Continuing without wandb logging."
        )
        return None

    from llama_recipes.configs import wandb_config as WANDB_CONFIG

    wandb_cfg = WANDB_CONFIG()
    update_config(wandb_cfg, **kwargs)
    init_dict = asdict(wandb_cfg)

    try:
        run = wandb.init(**init_dict)
    except Exception as exc:
        print(
            "WARNING: Failed to initialize wandb run. "
            f"Continuing without wandb logging. Error: {exc}"
        )
        return None

    run.config.update(asdict(train_cfg), allow_val_change=True)
    run.config.update(asdict(makam_cfg), allow_val_change=True)
    run.config.update(model_config.to_dict(), allow_val_change=True)

    if hasattr(dataset_config, "__dataclass_fields__"):
        run.config.update(asdict(dataset_config), allow_val_change=True)

    return run


def _wandb_log(run, payload, step=None):
    if run is None:
        return
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=step)


def _save_metrics_and_plots(train_cfg, metrics_payload):
    if not bool(train_cfg.save_metrics):
        return None, []

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics_path = os.path.join(
        train_cfg.output_dir,
        f"metrics_data_{train_cfg.model_name}-{timestamp}.json",
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    plot_paths = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            "WARNING: Could not import matplotlib for metric plots. "
            f"Saved metrics JSON only. Error: {exc}"
        )
        return metrics_path, plot_paths

    history = metrics_payload.get("history", {})
    train_loss_by_epoch = history.get("train_loss_by_epoch", [])
    if train_loss_by_epoch:
        epochs = [item["epoch"] for item in train_loss_by_epoch]
        losses = [item["train_loss"] for item in train_loss_by_epoch]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, losses, marker="o")
        ax.set_title("Train Loss by Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        train_plot = os.path.join(
            train_cfg.output_dir,
            f"plot_train_loss_{train_cfg.model_name}-{timestamp}.png",
        )
        fig.tight_layout()
        fig.savefig(train_plot, dpi=160)
        plt.close(fig)
        plot_paths.append(train_plot)

    eval_by_epoch = history.get("eval_by_epoch", [])
    if eval_by_epoch:
        epochs = [item["epoch"] for item in eval_by_epoch]
        eval_loss = [item["eval_loss"] for item in eval_by_epoch]
        eval_acc = [item["eval_accuracy"] for item in eval_by_epoch]
        eval_f1 = [item["eval_macro_f1"] for item in eval_by_epoch]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(epochs, eval_loss, marker="o")
        axes[0].set_ylabel("Eval Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, eval_acc, marker="o")
        axes[1].set_ylabel("Eval Accuracy")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, eval_f1, marker="o")
        axes[2].set_ylabel("Eval Macro-F1")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

        eval_plot = os.path.join(
            train_cfg.output_dir,
            f"plot_eval_epoch_metrics_{train_cfg.model_name}-{timestamp}.png",
        )
        fig.tight_layout()
        fig.savefig(eval_plot, dpi=160)
        plt.close(fig)
        plot_paths.append(eval_plot)

    eval_by_step = history.get("eval_by_step", [])
    if eval_by_step:
        steps = [item["global_step"] for item in eval_by_step]
        eval_f1 = [item["eval_macro_f1"] for item in eval_by_step]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, eval_f1, marker="o")
        ax.set_title("Eval Macro-F1 by Validation Step")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Eval Macro-F1")
        ax.grid(True, alpha=0.3)
        step_plot = os.path.join(
            train_cfg.output_dir,
            f"plot_eval_f1_steps_{train_cfg.model_name}-{timestamp}.png",
        )
        fig.tight_layout()
        fig.savefig(step_plot, dpi=160)
        plt.close(fig)
        plot_paths.append(step_plot)

    return metrics_path, plot_paths


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

    labels = sorted(set(all_targets) | set(all_preds))
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=labels,
        average=None,
        zero_division=0,
    )
    conf = confusion_matrix(all_targets, all_preds, labels=labels)

    per_class_metrics = {}
    for idx, label in enumerate(labels):
        row_total = int(conf[idx].sum())
        class_acc = float(conf[idx, idx] / row_total) if row_total > 0 else 0.0
        per_class_metrics[str(label)] = {
            "accuracy": class_acc,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "eval_loss": eval_loss,
        "eval_accuracy": accuracy,
        "eval_macro_f1": float(macro_f1),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf.tolist(),
        "labels": [int(x) for x in labels],
        "predictions": [int(x) for x in all_preds],
        "targets": [int(x) for x in all_targets],
    }


def train_classification(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    train_cfg,
    device,
    wandb_run=None,
    label_map=None,
):
    model.train()
    global_step = 0
    best_macro_f1 = float("-inf")
    best_metrics = None
    running_train_losses = []
    history = {
        "train_loss_by_epoch": [],
        "eval_by_step": [],
        "eval_by_epoch": [],
    }
    grad_acc_steps = max(1, int(train_cfg.gradient_accumulation_steps))

    if train_cfg.save_model:
        os.makedirs(train_cfg.output_dir, exist_ok=True)

    for epoch in range(int(train_cfg.num_epochs)):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        optimizer.zero_grad()
        reached_max_steps = False

        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / grad_acc_steps
            loss.backward()

            if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_dataloader):
                if (
                    train_cfg.gradient_clipping
                    and train_cfg.gradient_clipping_threshold > 0.0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_cfg.gradient_clipping_threshold
                    )
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

                class_names = []
                for label_id in metrics["labels"]:
                    if isinstance(label_map, dict):
                        class_names.append(str(label_map.get(str(label_id), label_id)))
                    else:
                        class_names.append(str(label_id))

                per_class_wandb = {}
                for label_id, class_metric in metrics["per_class_metrics"].items():
                    class_name = label_id
                    if isinstance(label_map, dict):
                        class_name = str(label_map.get(str(label_id), label_id))
                    safe_name = class_name.replace("/", "_")
                    per_class_wandb[f"eval/class_accuracy_step/{safe_name}"] = float(
                        class_metric["accuracy"]
                    )
                    per_class_wandb[f"eval/class_f1_step/{safe_name}"] = float(
                        class_metric["f1"]
                    )

                history["eval_by_step"].append(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "eval_loss": float(metrics["eval_loss"]),
                        "eval_accuracy": float(metrics["eval_accuracy"]),
                        "eval_macro_f1": float(metrics["eval_macro_f1"]),
                        "per_class_metrics": metrics["per_class_metrics"],
                        "confusion_matrix": metrics["confusion_matrix"],
                        "labels": metrics["labels"],
                    }
                )
                wandb_payload = {
                    "eval/loss_step": float(metrics["eval_loss"]),
                    "eval/accuracy_step": float(metrics["eval_accuracy"]),
                    "eval/macro_f1_step": float(metrics["eval_macro_f1"]),
                    "train/epoch": int(epoch),
                }
                wandb_payload.update(per_class_wandb)
                _wandb_log(wandb_run, wandb_payload, step=global_step)

                if wandb_run is not None:
                    try:
                        import wandb

                        _wandb_log(
                            wandb_run,
                            {
                                "eval/confusion_matrix_step": wandb.plot.confusion_matrix(
                                    probs=None,
                                    y_true=metrics["targets"],
                                    preds=metrics["predictions"],
                                    class_names=class_names,
                                )
                            },
                            step=global_step,
                        )
                    except Exception as exc:
                        print(
                            "WARNING: Failed to log step confusion matrix to wandb. "
                            f"Error: {exc}"
                        )

                if metrics["eval_macro_f1"] > best_macro_f1:
                    best_macro_f1 = metrics["eval_macro_f1"]
                    best_metrics = metrics
                    if train_cfg.save_model:
                        ckpt_path = os.path.join(
                            train_cfg.output_dir, "best_makam_classifier.pt"
                        )
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

            if int(train_cfg.max_train_step) > 0 and global_step >= int(
                train_cfg.max_train_step
            ):
                reached_max_steps = True
                break

        if reached_max_steps:
            scheduler.step()
            epoch_train_loss = epoch_loss_sum / max(1, epoch_steps)
            running_train_losses.append(epoch_train_loss)
            history["train_loss_by_epoch"].append(
                {"epoch": int(epoch), "train_loss": float(epoch_train_loss)}
            )
            _wandb_log(
                wandb_run,
                {
                    "train/loss_epoch": float(epoch_train_loss),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/epoch": int(epoch),
                },
                step=global_step,
            )
            print(f"epoch={epoch} train_loss={epoch_train_loss:.6f}")
            print(f"Reached max_train_step={train_cfg.max_train_step}; stopping early.")
            break

        scheduler.step()
        epoch_train_loss = epoch_loss_sum / max(1, epoch_steps)
        running_train_losses.append(epoch_train_loss)
        history["train_loss_by_epoch"].append(
            {"epoch": int(epoch), "train_loss": float(epoch_train_loss)}
        )
        _wandb_log(
            wandb_run,
            {
                "train/loss_epoch": float(epoch_train_loss),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/epoch": int(epoch),
            },
            step=global_step,
        )
        print(f"epoch={epoch} train_loss={epoch_train_loss:.6f}")

        if train_cfg.run_validation and eval_dataloader is not None:
            metrics = evaluate_classification(model, eval_dataloader, device)
            print(
                f"epoch={epoch} "
                f"eval_loss={metrics['eval_loss']:.6f} "
                f"eval_accuracy={metrics['eval_accuracy']:.6f} "
                f"eval_macro_f1={metrics['eval_macro_f1']:.6f}"
            )

            class_names = []
            for label_id in metrics["labels"]:
                if isinstance(label_map, dict):
                    class_names.append(str(label_map.get(str(label_id), label_id)))
                else:
                    class_names.append(str(label_id))

            per_class_wandb = {}
            for label_id, class_metric in metrics["per_class_metrics"].items():
                class_name = label_id
                if isinstance(label_map, dict):
                    class_name = str(label_map.get(str(label_id), label_id))
                safe_name = class_name.replace("/", "_")
                per_class_wandb[f"eval/class_accuracy_epoch/{safe_name}"] = float(
                    class_metric["accuracy"]
                )
                per_class_wandb[f"eval/class_f1_epoch/{safe_name}"] = float(
                    class_metric["f1"]
                )

            history["eval_by_epoch"].append(
                {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "eval_loss": float(metrics["eval_loss"]),
                    "eval_accuracy": float(metrics["eval_accuracy"]),
                    "eval_macro_f1": float(metrics["eval_macro_f1"]),
                    "per_class_metrics": metrics["per_class_metrics"],
                    "confusion_matrix": metrics["confusion_matrix"],
                    "labels": metrics["labels"],
                }
            )
            wandb_payload = {
                "eval/loss_epoch": float(metrics["eval_loss"]),
                "eval/accuracy_epoch": float(metrics["eval_accuracy"]),
                "eval/macro_f1_epoch": float(metrics["eval_macro_f1"]),
                "train/epoch": int(epoch),
            }
            wandb_payload.update(per_class_wandb)
            _wandb_log(wandb_run, wandb_payload, step=global_step)

            if wandb_run is not None:
                try:
                    import wandb

                    _wandb_log(
                        wandb_run,
                        {
                            "eval/confusion_matrix_epoch": wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=metrics["targets"],
                                preds=metrics["predictions"],
                                class_names=class_names,
                            )
                        },
                        step=global_step,
                    )
                except Exception as exc:
                    print(
                        "WARNING: Failed to log epoch confusion matrix to wandb. "
                        f"Error: {exc}"
                    )
            if metrics["eval_macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["eval_macro_f1"]
                best_metrics = metrics
                if train_cfg.save_model:
                    ckpt_path = os.path.join(
                        train_cfg.output_dir, "best_makam_classifier.pt"
                    )
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
        "best_eval_accuracy": (
            best_metrics["eval_accuracy"] if best_metrics is not None else None
        ),
        "best_eval_loss": (
            best_metrics["eval_loss"] if best_metrics is not None else None
        ),
        "total_steps": global_step,
    }
    return results, history


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
    ckpt_path = train_cfg.trained_checkpoint_path
    ckpt_model_config_path = _find_microtonal_llama_config(ckpt_path)
    if ckpt_model_config_path is not None:
        model_config_path = ckpt_model_config_path
        print(
            f"Using checkpoint-side model config for compatibility: {model_config_path}"
        )

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
    if ckpt_path and os.path.exists(ckpt_path):
        model = _load_checkpoint_into_model(
            model,
            ckpt_path,
            strict=bool(makam_cfg.checkpoint_strict),
        )

    model.to(device)
    print(f"Using device: {device}")
    print(f"Training config: {asdict(train_cfg)}")
    print(f"Makam classification config: {asdict(makam_cfg)}")
    print(f"Dataset config: {dataset_config}")
    print(f"label_map_path={label_map_path}")
    print(f"num_labels={num_labels}")

    wandb_run = _setup_wandb(
        train_cfg=train_cfg,
        makam_cfg=makam_cfg,
        model_config=model_config,
        dataset_config=dataset_config,
        kwargs=kwargs,
    )

    # LoRA adapter configuration
    enable_lora = bool(train_cfg.enable_lora) or (
        bool(train_cfg.use_peft) and str(train_cfg.peft_method).lower() == "lora"
    )
    if enable_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=train_cfg.lora_r,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            target_modules=["q_proj", "v_proj"],  # attention heads
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print(
            f"LoRA enabled: r={train_cfg.lora_r}, alpha={train_cfg.lora_alpha}, dropout={train_cfg.lora_dropout}"
        )
        model.print_trainable_parameters()
    elif bool(train_cfg.use_peft):
        print(
            "WARNING: use_peft=True but only LoRA is currently implemented in this entrypoint. "
            "Set --enable_lora true or --peft_method lora."
        )

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
        drop_last=False,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        ds_val,
        batch_size=train_cfg.val_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=train_cfg.num_workers_dataloader,
        pin_memory=True,
        drop_last=False,
    )

    if train_cfg.run_validation and len(eval_dataloader) == 0:
        print(
            f"WARNING: Validation set has {len(ds_val)} samples, but batch size is {train_cfg.val_batch_size}."
        )
        print(
            "  Validation will be skipped. Consider reducing batch size or disabling run_validation."
        )
        train_cfg.run_validation = False

    optimizer, optimizer_name = _create_optimizer(model, train_cfg, device)
    print(f"Using optimizer: {optimizer_name}")
    scheduler = StepLR(optimizer, step_size=1, gamma=train_cfg.gamma)

    results, history = train_classification(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader if train_cfg.run_validation else None,
        optimizer=optimizer,
        scheduler=scheduler,
        train_cfg=train_cfg,
        device=device,
        wandb_run=wandb_run,
        label_map=label_map,
    )

    metrics_payload = {
        "results": results,
        "history": history,
        "train_config": asdict(train_cfg),
        "makam_config": asdict(makam_cfg),
        "label_map_path": label_map_path,
        "num_labels": num_labels,
    }

    metrics_path, plot_paths = _save_metrics_and_plots(train_cfg, metrics_payload)
    if metrics_path is not None:
        print(f"Saved metrics JSON to {metrics_path}")
    if plot_paths:
        print(f"Saved metric plots to: {plot_paths}")

    if wandb_run is not None:
        summary_payload = {
            "summary/avg_train_loss": float(results["avg_train_loss"]),
            "summary/total_steps": int(results["total_steps"]),
        }
        if results["best_eval_macro_f1"] is not None:
            summary_payload["summary/best_eval_macro_f1"] = float(
                results["best_eval_macro_f1"]
            )
            summary_payload["summary/best_eval_accuracy"] = float(
                results["best_eval_accuracy"]
            )
            summary_payload["summary/best_eval_loss"] = float(
                results["best_eval_loss"]
            )
        _wandb_log(wandb_run, summary_payload, step=int(results["total_steps"]))
        if metrics_path is not None:
            wandb_run.summary["metrics_json_path"] = metrics_path
        if plot_paths:
            wandb_run.summary["metrics_plot_paths"] = plot_paths
        wandb_run.finish()

    for key, value in results.items():
        print(f"Key: {key}, Value: {value}")


if __name__ == "__main__":
    fire.Fire(main)
