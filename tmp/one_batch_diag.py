#!/usr/bin/env python3
import os
import sys
import json
import traceback

# Ensure local package imports work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import makam_classification_config as MAKAM_CLASSIFICATION_CONFIG
from llama_recipes.utils.config_utils import generate_dataset_config
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForSequenceClassification,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.real_finetuning_makam_classification import (
    CompoundClassificationCollator,
    _load_checkpoint_into_model,
)


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device)
    return moved


def main():
    try:
        train_cfg = TRAIN_CONFIG()
        makam_cfg = MAKAM_CLASSIFICATION_CONFIG()

        train_cfg.dataset = "symbtr_dataset_eval"
        dataset_config = generate_dataset_config(train_cfg, {})
        print(f"dataset_config.dataset={dataset_config.dataset}")
        print(f"dataset_config.csv_file={dataset_config.csv_file} exists={os.path.exists(dataset_config.csv_file)}")
        print(f"dataset_config.label_map={dataset_config.label_map} exists={os.path.exists(dataset_config.label_map)}")

        with open(dataset_config.label_map, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        num_labels = len(label_map)
        print(f"num_labels={num_labels}")

        model_config_path = makam_cfg.model_config_path
        ckpt_path = getattr(train_cfg, 'trained_checkpoint_path', None)

        print(f"model_config_path={model_config_path}")
        model_config = LlamaConfig.from_pretrained(model_config_path)

        # build tokenizer from model config
        tokenizer = MusicTokenizer(
            timeshift_vocab_size=model_config.onset_vocab_size,
            dur_vocab_size=model_config.dur_vocab_size,
            octave_vocab_size=model_config.octave_vocab_size,
            pitch_class_vocab_size=model_config.pitch_class_vocab_size,
            instrument_vocab_size=model_config.instrument_vocab_size,
            velocity_vocab_size=model_config.velocity_vocab_size,
            sos_token=model_config.sos_token,
            eos_token=model_config.eos_token,
            pad_token=model_config.pad_token,
            microtonal=getattr(model_config, 'microtonal', False),
            pitchbend_sensitivity=getattr(model_config, 'pitchbend_sensitivity', 2.0),
            microtonal_resolution=getattr(model_config, 'microtonal_resolution', 1),
        )

        model_config.num_labels = num_labels
        model_config.pad_token_id = int(tokenizer.pad_token)
        model_config.problem_type = 'single_label_classification'

        model = LlamaForSequenceClassification(model_config)
        print('Instantiated model')

        # Optionally load checkpoint / adapter if provided
        if ckpt_path and os.path.exists(ckpt_path):
            print(f'Found ckpt_path={ckpt_path}, attempting to load')
            model = _load_checkpoint_into_model(model, ckpt_path, strict=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        model.to(device)

        ds_train = get_preprocessed_dataset(tokenizer, dataset_config, split='train')
        print('Train dataset length:', len(ds_train))

        collator = CompoundClassificationCollator(pad_token=int(tokenizer.pad_token))
        train_loader = torch.utils.data.DataLoader(
            ds_train, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0
        )

        batch = next(iter(train_loader))
        print('Batch keys:', list(batch.keys()))
        print('input_ids.shape:', getattr(batch['input_ids'], 'shape', None))
        print('attention_mask.shape:', getattr(batch['attention_mask'], 'shape', None))
        print('labels:', batch['labels'])

        model.train()
        model.zero_grad()

        batch = _move_batch_to_device(batch, device)

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        print('loss:', float(loss.detach().cpu().item()))
        print('logits.shape:', tuple(logits.shape))

        logits_cpu = logits.detach().cpu()
        print('logits stats mean/std/min/max:', float(logits_cpu.mean()), float(logits_cpu.std()), float(logits_cpu.min()), float(logits_cpu.max()))

        preds = torch.argmax(logits, dim=-1)
        print('preds:', preds.detach().cpu().tolist())

        # backward
        loss.backward()

        total_norm_sq = 0.0
        grads_report = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            gnorm = float(p.grad.detach().data.norm(2).cpu().item())
            total_norm_sq += gnorm ** 2
            if 'score' in name or 'classifier' in name or 'lm_head' in name:
                grads_report.append((name, gnorm))

        total_norm = total_norm_sq ** 0.5
        print('total_grad_norm:', total_norm)
        print('head grad norms:')
        for n, g in grads_report:
            print(f'  {n}: {g}')

        print('One-batch diagnostic completed successfully')

    except Exception as e:
        print('EXCEPTION during diagnostic:', e)
        traceback.print_exc()


if __name__ == '__main__':
    main()
