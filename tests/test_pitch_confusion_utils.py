import importlib.util
from pathlib import Path

import numpy as np
import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "llama_recipes"
    / "utils"
    / "pitch_confusion_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("pitch_confusion_utils", _MODULE_PATH)
pcu = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(pcu)


class _DummyTokenizer:
    def __init__(self):
        # 12 western semitone cents mapped to 12 unique token IDs.
        self.pitch_dict = {c: 2000 + (c // 100) for c in range(0, 1200, 100)}
        # SOS/EOS pitch tokens.
        self.pitch_dict[1200] = 2012
        self.pitch_dict[1201] = 2013
        # A couple of microtonal bins with unique IDs.
        self.pitch_dict[1] = 3001
        self.pitch_dict[2] = 3002


def test_plot_heatmap_sparsifies_ticks_for_large_vocab(tmp_path, monkeypatch):
    token_ids = list(range(2000, 2128))
    matrix = np.zeros((len(token_ids), len(token_ids)), dtype=np.int64)
    matrix[0, 0] = 10

    captured = {}
    original_subplots = pcu.plt.subplots

    def _spy_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(pcu.plt, "subplots", _spy_subplots)

    output_path = tmp_path / "all_tokens_heatmap.png"
    pcu._plot_heatmap(
        matrix=matrix,
        token_ids=token_ids,
        output_path=str(output_path),
        title="all-token-test",
        normalize_rows=False,
    )

    ax = captured["ax"]
    assert output_path.exists()
    # Ticks should be reduced for readability when vocab is large.
    assert len(ax.get_xticks()) < len(token_ids)
    assert len(ax.get_yticks()) < len(token_ids)
    # Last token should still be shown as a labeled tick.
    assert int(ax.get_xticklabels()[-1].get_text()) == token_ids[-1]


def test_update_pitch_confusions_tracks_non_pitch_predictions():
    tokenizer = _DummyTokenizer()
    state = pcu.init_pitch_confusion_state(tokenizer)

    # labels shape: (batch, seq_len, 7). After shifting, there are 2 events.
    labels = torch.zeros((1, 3, 7), dtype=torch.long)
    # Pitch token is attribute index 4 in 7-wide labels.
    labels[0, 1, 4] = 2000
    labels[0, 2, 4] = 2001

    # generation_logits shape: (events * 6, vocab)
    vocab_size = 5000
    logits = torch.full((12, vocab_size), -1e9, dtype=torch.float32)

    # Event 0, pitch step index 3 -> predict valid pitch token 2001.
    logits[3, 2001] = 0.0
    # Event 1, pitch step index 9 -> predict non-pitch token 4999.
    logits[9, 4999] = 0.0

    pcu.update_pitch_confusions(state, labels, logits)

    assert state.total_pitch_samples == 2
    assert state.ignored_pred_non_pitch == 1

    i = state.id_to_overall_idx[2000]
    j = state.id_to_overall_idx[2001]
    assert state.matrix_overall[i, j] == 1


def test_western14_includes_sos_eos_and_is_counted():
    tokenizer = _DummyTokenizer()
    state = pcu.init_pitch_confusion_state(tokenizer)

    assert len(state.western14_token_ids) == 14
    assert tokenizer.pitch_dict[1200] in state.western14_token_ids
    assert tokenizer.pitch_dict[1201] in state.western14_token_ids

    labels = torch.zeros((1, 2, 7), dtype=torch.long)
    # One shifted event whose pitch is SOS (cents 1200)
    labels[0, 1, 4] = tokenizer.pitch_dict[1200]

    vocab_size = 5000
    logits = torch.full((6, vocab_size), -1e9, dtype=torch.float32)
    # Predict EOS (cents 1201) at pitch step index 3
    logits[3, tokenizer.pitch_dict[1201]] = 0.0

    pcu.update_pitch_confusions(state, labels, logits)

    i = state.id_to_western14_idx[tokenizer.pitch_dict[1200]]
    j = state.id_to_western14_idx[tokenizer.pitch_dict[1201]]
    assert state.matrix_western14[i, j] == 1


def test_save_artifacts_includes_western14_files(tmp_path):
    tokenizer = _DummyTokenizer()
    state = pcu.init_pitch_confusion_state(tokenizer)

    out = pcu.save_pitch_confusion_artifacts(
        state=state,
        save_dir=str(tmp_path),
        artifact_prefix="unit",
        expected_pitch_dict_size=len(state.pitch_token_ids),
    )

    assert "western14_npy" in out
    assert "western14_counts_png" in out
    assert "western14_row_norm_png" in out
    assert "overall_token_index_map_csv" in out

    for k in ("western14_npy", "western14_counts_png", "western14_row_norm_png", "overall_token_index_map_csv"):
        assert Path(out[k]).exists()