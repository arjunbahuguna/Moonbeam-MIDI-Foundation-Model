import json
import os

import pytest
import torch
from transformers import LlamaConfig, LlamaForSequenceClassification

from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.datasets.symbtr_dataset_eval import MakamClassificationDataset


def _load_tokenizer_config(model_config_path):
    with open(model_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return MusicTokenizer(
        timeshift_vocab_size=cfg["onset_vocab_size"],
        dur_vocab_size=cfg["dur_vocab_size"],
        octave_vocab_size=cfg["octave_vocab_size"],
        pitch_class_vocab_size=cfg["pitch_class_vocab_size"],
        instrument_vocab_size=cfg["instrument_vocab_size"],
        velocity_vocab_size=cfg["velocity_vocab_size"],
        sos_token=cfg.get("sos_token", -1),
        eos_token=cfg.get("eos_token", -2),
        pad_token=cfg.get("pad_token", -3),
        microtonal=cfg.get("microtonal", False),
        pitchbend_sensitivity=cfg.get("pitchbend_sensitivity", 2.0),
        microtonal_resolution=cfg.get("microtonal_resolution", 1),
    )


@pytest.mark.slow
def test_one_batch_sequence_classifier_smoke_from_label_map():
    model_cfg = "src/llama_recipes/configs/config_micro/model_config_microtonal.json"
    csv_path = "data/processed_data_symbtr/makam_classification_split.csv"
    label_map_path = "data/processed_data_symbtr/makam_label_map.json"

    required_paths = [model_cfg, csv_path, label_map_path]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        pytest.skip(f"Missing required artifact(s): {missing}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = len(label_map)
    assert num_labels > 1

    tokenizer = _load_tokenizer_config(model_cfg)

    # Keep the sequence short for an inexpensive smoke forward pass.
    dataset_cfg = type(
        "DatasetConfig",
        (),
        {
            "data_dir": "data/processed_data_symbtr",
            "csv_file": csv_path,
            "seq_len": 64,
        },
    )()

    ds = MakamClassificationDataset(dataset_cfg, tokenizer, partition="train")
    if len(ds) == 0:
        pytest.skip("No rows in train split for makam classification dataset")

    sample = ds[0]
    compound_ids = sample["input_ids"]  # [seq, 6], can include negative special values

    # LlamaForSequenceClassification expects token IDs [batch, seq] in [0, vocab_size).
    # Build a deterministic pseudo-token stream from the compound representation.
    pseudo_ids = (compound_ids + 3).clamp(min=0).reshape(1, -1)
    vocab_size = int(pseudo_ids.max().item()) + 32

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=max(512, pseudo_ids.shape[1] + 8),
        num_labels=num_labels,
        pad_token_id=0,
    )

    model = LlamaForSequenceClassification(config)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=pseudo_ids)

    assert outputs.logits.shape == (1, num_labels)
