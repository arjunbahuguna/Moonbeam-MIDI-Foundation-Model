import json
import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class PitchConfusionState:
    matrix_overall: np.ndarray
    matrix_western: np.ndarray
    matrix_western14: np.ndarray
    pitch_token_ids: List[int]
    western_token_ids: List[int]
    western14_token_ids: List[int]
    id_to_overall_idx: Dict[int, int]
    id_to_western_idx: Dict[int, int]
    id_to_western14_idx: Dict[int, int]
    total_pitch_samples: int = 0
    ignored_pred_non_pitch: int = 0


def _extract_pitch_ids_from_batch(labels: torch.Tensor, generation_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover pitch-step true/pred language token IDs following the exact model flattening convention:
    labels[..., 1:, :] -> flatten -> [:, 1:] (6 attributes) -> attribute index 3 = pitch.
    """
    if labels.ndim != 3:
        raise ValueError(f"Expected labels ndim=3, got {labels.ndim}")

    shift_labels = labels[..., 1:, :].contiguous()
    labels_flat = shift_labels.view(-1, shift_labels.shape[-1])
    labels_6 = labels_flat[:, 1:].contiguous()

    pred_flat = generation_logits.argmax(dim=-1)

    n_events = min(labels_6.shape[0], pred_flat.shape[0] // 6)
    if n_events <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    labels_6 = labels_6[:n_events]
    pred_6 = pred_flat[: n_events * 6].view(n_events, 6)

    true_pitch_ids = labels_6[:, 3].long()
    pred_pitch_ids = pred_6[:, 3].long()
    return true_pitch_ids, pred_pitch_ids


def _western_token_ids_from_pitch_dict(pitch_dict: Dict[int, int]) -> List[int]:
    western = []
    for cents, token_id in pitch_dict.items():
        if cents in (1200, 1201):
            continue
        if cents % 100 == 0:
            western.append(int(token_id))
    return sorted(set(western))


def _western14_token_ids_from_pitch_dict(pitch_dict: Dict[int, int]) -> List[int]:
    # Keep semitones in musical order plus pitch SOS/EOS at the end.
    required_cents = list(range(0, 1200, 100)) + [1200, 1201]
    missing = [c for c in required_cents if c not in pitch_dict]
    if missing:
        raise ValueError(f"Missing expected cents in pitch_dict for western14: {missing}")
    return [int(pitch_dict[c]) for c in required_cents]


def init_pitch_confusion_state(tokenizer) -> PitchConfusionState:
    pitch_token_ids = sorted({int(v) for v in tokenizer.pitch_dict.values()})
    if len(pitch_token_ids) != len(tokenizer.pitch_dict):
        raise ValueError(
            "Pitch token ID uniqueness check failed: len(unique IDs) != len(pitch_dict). "
            "Cannot build unambiguous confusion matrix keyed by language token IDs."
        )

    western_token_ids = _western_token_ids_from_pitch_dict(tokenizer.pitch_dict)
    if len(western_token_ids) != 12:
        raise ValueError(
            f"Expected 12 western pitch token IDs, found {len(western_token_ids)}."
        )

    western14_token_ids = _western14_token_ids_from_pitch_dict(tokenizer.pitch_dict)
    if len(western14_token_ids) != 14:
        raise ValueError(
            f"Expected 14 western+SOS/EOS pitch token IDs, found {len(western14_token_ids)}."
        )

    id_to_overall_idx = {tid: i for i, tid in enumerate(pitch_token_ids)}
    id_to_western_idx = {tid: i for i, tid in enumerate(western_token_ids)}
    id_to_western14_idx = {tid: i for i, tid in enumerate(western14_token_ids)}

    matrix_overall = np.zeros((len(pitch_token_ids), len(pitch_token_ids)), dtype=np.int64)
    matrix_western = np.zeros((12, 12), dtype=np.int64)
    matrix_western14 = np.zeros((14, 14), dtype=np.int64)

    return PitchConfusionState(
        matrix_overall=matrix_overall,
        matrix_western=matrix_western,
        matrix_western14=matrix_western14,
        pitch_token_ids=pitch_token_ids,
        western_token_ids=western_token_ids,
        western14_token_ids=western14_token_ids,
        id_to_overall_idx=id_to_overall_idx,
        id_to_western_idx=id_to_western_idx,
        id_to_western14_idx=id_to_western14_idx,
    )


def update_pitch_confusions(state: PitchConfusionState, labels: torch.Tensor, generation_logits: torch.Tensor) -> None:
    true_ids, pred_ids = _extract_pitch_ids_from_batch(labels, generation_logits)
    if true_ids.numel() == 0:
        return

    true_np = true_ids.detach().cpu().numpy()
    pred_np = pred_ids.detach().cpu().numpy()

    for t, p in zip(true_np, pred_np):
        state.total_pitch_samples += 1
        i = state.id_to_overall_idx.get(int(t))
        j = state.id_to_overall_idx.get(int(p))
        if i is not None and j is not None:
            state.matrix_overall[i, j] += 1
        elif i is not None and j is None:
            state.ignored_pred_non_pitch += 1

        wi = state.id_to_western_idx.get(int(t))
        wj = state.id_to_western_idx.get(int(p))
        if wi is not None and wj is not None:
            state.matrix_western[wi, wj] += 1

        w14i = state.id_to_western14_idx.get(int(t))
        w14j = state.id_to_western14_idx.get(int(p))
        if w14i is not None and w14j is not None:
            state.matrix_western14[w14i, w14j] += 1


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    out = np.zeros_like(matrix, dtype=np.float64)
    nonzero = row_sums.squeeze(-1) > 0
    out[nonzero] = matrix[nonzero] / row_sums[nonzero]
    return out


def _plot_heatmap(
    matrix: np.ndarray,
    token_ids: List[int],
    output_path: str,
    title: str,
    normalize_rows: bool = False,
    max_visible_tick_labels: int = 60,
) -> None:
    data = _normalize_rows(matrix) if normalize_rows else matrix.astype(np.float64)

    n = len(token_ids)
    # Scale canvas with matrix size so dense plots remain legible.
    side = min(26, max(10, n / 55.0))
    fig, ax = plt.subplots(figsize=(side, side * 0.8), dpi=160)
    im = ax.imshow(data, interpolation="nearest", aspect="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("row-normalized" if normalize_rows else "count", rotation=-90, va="bottom")

    # Keep the full matrix, but sparsify axis labels for readability.
    if n <= max_visible_tick_labels:
        tick_positions = np.arange(n)
    else:
        stride = int(np.ceil(n / float(max_visible_tick_labels)))
        tick_positions = np.arange(0, n, stride)
        if tick_positions[-1] != n - 1:
            tick_positions = np.append(tick_positions, n - 1)
    tick_labels = [str(token_ids[i]) for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_xlabel("Predicted pitch language token ID")
    ax.set_ylabel("True pitch language token ID")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_pitch_confusion_artifacts(
    state: PitchConfusionState,
    save_dir: str,
    artifact_prefix: str,
    expected_pitch_dict_size: Optional[int] = None,
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)

    if expected_pitch_dict_size is not None and len(state.pitch_token_ids) != int(expected_pitch_dict_size):
        raise ValueError(
            f"Pitch confusion size mismatch: got {len(state.pitch_token_ids)} token IDs, "
            f"expected {expected_pitch_dict_size} from config/tokenizer pitch_dict size."
        )

    overall_npy = os.path.join(save_dir, f"{artifact_prefix}_overall_counts.npy")
    western_npy = os.path.join(save_dir, f"{artifact_prefix}_western12_counts.npy")
    western14_npy = os.path.join(save_dir, f"{artifact_prefix}_western14_counts.npy")
    overall_row_png = os.path.join(save_dir, f"{artifact_prefix}_overall_row_norm.png")
    overall_cnt_png = os.path.join(save_dir, f"{artifact_prefix}_overall_counts.png")
    western_row_png = os.path.join(save_dir, f"{artifact_prefix}_western12_row_norm.png")
    western_cnt_png = os.path.join(save_dir, f"{artifact_prefix}_western12_counts.png")
    western14_row_png = os.path.join(save_dir, f"{artifact_prefix}_western14_row_norm.png")
    western14_cnt_png = os.path.join(save_dir, f"{artifact_prefix}_western14_counts.png")
    token_map_csv = os.path.join(save_dir, f"{artifact_prefix}_overall_token_index_map.csv")
    meta_json = os.path.join(save_dir, f"{artifact_prefix}_meta.json")

    np.save(overall_npy, state.matrix_overall)
    np.save(western_npy, state.matrix_western)
    np.save(western14_npy, state.matrix_western14)

    with open(token_map_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["matrix_index", "pitch_token_id"])
        for idx, token_id in enumerate(state.pitch_token_ids):
            writer.writerow([idx, token_id])

    _plot_heatmap(
        state.matrix_overall,
        state.pitch_token_ids,
        overall_cnt_png,
        title=f"Pitch Confusion (overall, token IDs) [{artifact_prefix}]",
        normalize_rows=False,
    )
    _plot_heatmap(
        state.matrix_overall,
        state.pitch_token_ids,
        overall_row_png,
        title=f"Pitch Confusion (overall row-normalized, token IDs) [{artifact_prefix}]",
        normalize_rows=True,
    )
    _plot_heatmap(
        state.matrix_western,
        state.western_token_ids,
        western_cnt_png,
        title=f"Pitch Confusion (western 12x12, token IDs) [{artifact_prefix}]",
        normalize_rows=False,
    )
    _plot_heatmap(
        state.matrix_western,
        state.western_token_ids,
        western_row_png,
        title=f"Pitch Confusion (western 12x12 row-normalized, token IDs) [{artifact_prefix}]",
        normalize_rows=True,
    )
    _plot_heatmap(
        state.matrix_western14,
        state.western14_token_ids,
        western14_cnt_png,
        title=f"Pitch Confusion (western+SOS/EOS 14x14, token IDs) [{artifact_prefix}]",
        normalize_rows=False,
    )
    _plot_heatmap(
        state.matrix_western14,
        state.western14_token_ids,
        western14_row_png,
        title=f"Pitch Confusion (western+SOS/EOS 14x14 row-normalized, token IDs) [{artifact_prefix}]",
        normalize_rows=True,
    )

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "artifact_prefix": artifact_prefix,
                "pitch_token_ids": state.pitch_token_ids,
                "western_token_ids": state.western_token_ids,
                "western14_token_ids": state.western14_token_ids,
                "pitch_dict_size": len(state.pitch_token_ids),
                "total_pitch_samples": int(state.total_pitch_samples),
                "ignored_pred_non_pitch": int(state.ignored_pred_non_pitch),
            },
            f,
            indent=2,
        )

    return {
        "overall_npy": overall_npy,
        "western_npy": western_npy,
        "western14_npy": western14_npy,
        "overall_token_index_map_csv": token_map_csv,
        "overall_counts_png": overall_cnt_png,
        "overall_row_norm_png": overall_row_png,
        "western_counts_png": western_cnt_png,
        "western_row_norm_png": western_row_png,
        "western14_counts_png": western14_cnt_png,
        "western14_row_norm_png": western14_row_png,
        "meta_json": meta_json,
    }


def compute_western_drift_metrics(pre_western: np.ndarray, post_western: np.ndarray) -> Dict[str, float]:
    if pre_western.shape != (12, 12) or post_western.shape != (12, 12):
        raise ValueError("Western confusion matrices must both be 12x12.")

    pre_row = _normalize_rows(pre_western)
    post_row = _normalize_rows(post_western)

    fro = float(np.linalg.norm(post_row - pre_row))
    pre_diag = float(np.mean(np.diag(pre_row)))
    post_diag = float(np.mean(np.diag(post_row)))
    delta_diag = post_diag - pre_diag

    return {
        "western_row_norm_fro_diff": fro,
        "western_diag_mean_pre": pre_diag,
        "western_diag_mean_post": post_diag,
        "western_diag_mean_delta": delta_diag,
    }
