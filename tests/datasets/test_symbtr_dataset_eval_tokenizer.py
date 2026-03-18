from types import SimpleNamespace
import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.datasets.symbtr_dataset_eval import MakamClassificationDataset


@pytest.fixture
def tokenizer():
    return MusicTokenizer(
        timeshift_vocab_size=64,
        dur_vocab_size=64,
        octave_vocab_size=16,
        pitch_class_vocab_size=12,
        instrument_vocab_size=132,
        velocity_vocab_size=132,
        microtonal=False,
    )


@pytest.fixture
def raw_tokens():
    return np.array(
        [
            [0, 10, 4, 5, 0, 80],
            [10, 8, 4, 7, 0, 70],
        ],
        dtype=np.int64,
    )


def test_player_classification_signature_is_dataset_compatible(tokenizer):

    sig = inspect.signature(tokenizer.encode_series_player_classification)
    assert "player_label" in sig.parameters
    assert sig.parameters["player_label"].default is None


def test_player_classification_equeals_encode_series_when_no_label(
    tokenizer, raw_tokens
):
    base = tokenizer.encode_series(raw_tokens, if_add_sos=True, if_add_eos=True)
    out = tokenizer.encode_series_player_classification(
        raw_tokens, if_add_sos=True, if_add_eos=True
    )

    assert out is not None
    assert isinstance(out, list)
    assert out == base
    assert out[0] == tokenizer.sos_token_compound
    assert out[-1] == tokenizer.eos_token_compound
    assert all(len(tok) == 6 for tok in out)


def test_player_label_injection_optional_path(tokenizer, raw_tokens):
    out = tokenizer.encode_series_player_classification(
        raw_tokens, if_add_sos=True, if_add_eos=True, player_label=99
    )
    assert out[0] == [99] * 6
    assert out[1] == tokenizer.sos_token_compound
    assert out[-1] == tokenizer.eos_token_compound


def test_makam_dataset_getitem_smoke(tmp_path, tokenizer):
    # tiny fake proccessed dataset
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    np.save(processed / "sample.npy", np.array([[0, 10, 4, 5, 0, 80]], dtype=np.int64))

    csv_path = tmp_path / "split.csv"

    pd.DataFrame(
        [
            {"file_base_name": "sample.npy", "split": "train", "label": 3},
        ]
    ).to_csv(csv_path, index=False)

    cfg = SimpleNamespace(data_dir=str(tmp_path), csv_file=str(csv_path), seq_len=32)

    ds = MakamClassificationDataset(cfg, tokenizer, partition="train")
    item = ds[0]

    assert set(item.keys()) == {"input_ids", "labels", "attention_mask"}
    assert item["input_ids"].dtype == torch.long
    assert item["input_ids"].ndim == 2
    assert item["input_ids"].shape[1] == 6
    assert torch.equal(item["attention_mask"], torch.ones_like(item["input_ids"]))
    assert item["labels"].dtype == torch.long
    assert item["labels"].item() == 3
    assert item["input_ids"][0].tolist() == tokenizer.sos_token_compound
    assert item["input_ids"][-1].tolist() == tokenizer.eos_token_compound
