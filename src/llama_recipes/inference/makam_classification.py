import json
from typing import Dict, List

import fire
import numpy as np
import torch

from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForSequenceClassification,
)


def _load_tokenizer_from_config(model_config: LlamaConfig) -> MusicTokenizer:
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


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    return cleaned


def _filter_state_dict_by_shape(model, state_dict: Dict[str, torch.Tensor]):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    return filtered, skipped


def _window_compound_tokens(
    encoded_tokens: List[List[int]],
    max_len: int,
    stride: int,
) -> List[List[List[int]]]:
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    n = len(encoded_tokens)
    if n <= max_len:
        return [encoded_tokens]

    windows = []
    start = 0
    while start < n:
        end = min(start + max_len, n)
        windows.append(encoded_tokens[start:end])
        if end == n:
            break
        start += stride

    return windows


def _aggregate_logits(window_logits: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        return window_logits.mean(dim=0)
    if method == "max":
        return window_logits.max(dim=0).values
    raise ValueError(f"Unsupported aggregation method: {method}")


def main(
    checkpoint_path: str,
    npy_path: str,
    model_config_path: str = "src/llama_recipes/configs/config_micro/model_config_microtonal.json",
    label_map_path: str = "data_eval/symbtr4eval/makam_label_map.json",
    seq_len: int = 1200,
    window_stride: int = 600,
    aggregation: str = "mean",
):

    import os

    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be one of: mean, max")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    id_to_label = {int(v): k for k, v in label_map.items()}
    num_labels = len(label_map)

    model_config = LlamaConfig.from_pretrained(model_config_path)
    model_config.num_labels = num_labels
    tokenizer = _load_tokenizer_from_config(model_config)
    model_config.pad_token_id = int(tokenizer.pad_token)
    model_config.problem_type = "single_label_classification"

    model = LlamaForSequenceClassification(model_config)
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
    state_dict = _strip_module_prefix(state_dict)
    state_dict, skipped = _filter_state_dict_by_shape(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint {checkpoint_path}. "
        f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}, skipped_shape_mismatch={len(skipped)}"
    )

    model.to(device)
    model.eval()

    # Check if npy_path is a directory or file
    if os.path.isdir(npy_path):
        npy_files = [os.path.join(npy_path, f) for f in os.listdir(npy_path) if f.endswith('.npy')]
        results = []
        for npy_file in npy_files:
            raw_tokens = np.load(npy_file)
            encoded = tokenizer.encode_series(raw_tokens, if_add_sos=True, if_add_eos=True)
            windows = _window_compound_tokens(encoded, max_len=seq_len, stride=window_stride)

            logits_per_window = []
            with torch.no_grad():
                for w in windows:
                    input_ids = torch.tensor(w, dtype=torch.long, device=device).unsqueeze(0)
                    attention_mask = torch.ones((1, input_ids.shape[1]), dtype=torch.long, device=device)
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_per_window.append(out.logits.squeeze(0).detach().cpu())

            stacked_logits = torch.stack(logits_per_window, dim=0)
            agg_logits = _aggregate_logits(stacked_logits, method=aggregation)
            probs = torch.softmax(agg_logits, dim=-1)
            pred_id = int(torch.argmax(probs).item())
            pred_label = id_to_label.get(pred_id, str(pred_id))
            confidence = float(probs[pred_id].item())

            result = {
                "npy_path": npy_file,
                "predicted_label_id": pred_id,
                "predicted_label": pred_label,
                "confidence": confidence,
                "num_windows": len(windows),
                "window_policy": {
                    "seq_len": int(seq_len),
                    "window_stride": int(window_stride),
                    "aggregation": aggregation,
                },
            }
            results.append(result)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        raw_tokens = np.load(npy_path)
        encoded = tokenizer.encode_series(raw_tokens, if_add_sos=True, if_add_eos=True)
        windows = _window_compound_tokens(encoded, max_len=seq_len, stride=window_stride)

        logits_per_window = []
        with torch.no_grad():
            for w in windows:
                input_ids = torch.tensor(w, dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.ones((1, input_ids.shape[1]), dtype=torch.long, device=device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_per_window.append(out.logits.squeeze(0).detach().cpu())

        stacked_logits = torch.stack(logits_per_window, dim=0)
        agg_logits = _aggregate_logits(stacked_logits, method=aggregation)
        probs = torch.softmax(agg_logits, dim=-1)
        pred_id = int(torch.argmax(probs).item())
        pred_label = id_to_label.get(pred_id, str(pred_id))
        confidence = float(probs[pred_id].item())

        result = {
            "npy_path": npy_path,
            "predicted_label_id": pred_id,
            "predicted_label": pred_label,
            "confidence": confidence,
            "num_windows": len(windows),
            "window_policy": {
                "seq_len": int(seq_len),
                "window_stride": int(window_stride),
                "aggregation": aggregation,
            },
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
