import json
import os

import pytest
import torch
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForSequenceClassification,
)

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
    csv_path = "data_eval/symbtr4eval/train_test_split.csv"
    label_map_path = "data_eval/symbtr4eval/makam_label_map.json"

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
            "data_dir": "data_eval/symbtr4eval",
            "csv_file": csv_path,
            "label_map": label_map_path,
            "seq_len": 64,
        },
    )()

    ds = MakamClassificationDataset(dataset_cfg, tokenizer, partition="train")
    if len(ds) == 0:
        pytest.skip("No rows in train split for makam classification dataset")

    sample = ds[0]
    compound_ids = sample["input_ids"].unsqueeze(0)  # [1, seq, 6]

    # Load full config schema used by training (contains compound embedding fields),
    # then shrink dimensions for an inexpensive smoke forward pass.
    config = LlamaConfig.from_pretrained(model_cfg)
    config.hidden_size = 60
    config.intermediate_size = 120
    config.num_hidden_layers = 2
    config.num_attention_heads = 6
    config.num_key_value_heads = 6
    config.max_position_embeddings = max(512, compound_ids.shape[1] + 8)
    config.num_labels = num_labels
    config.pad_token_id = int(tokenizer.pad_token)
    config.problem_type = "single_label_classification"

    model = LlamaForSequenceClassification(config)
    model.eval()
    attention_mask = torch.ones((1, compound_ids.shape[1]), dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=compound_ids, attention_mask=attention_mask)

    assert outputs.logits.shape == (1, num_labels)
