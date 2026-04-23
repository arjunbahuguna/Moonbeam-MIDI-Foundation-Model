#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama_recipes.datasets.music_tokenizer import MusicTokenizer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_tokenizer(cfg: dict) -> MusicTokenizer:
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


def computed_decode_vocab_size(tok: MusicTokenizer) -> int:
    dictionaries = [
        tok.sos_out_dict,
        tok.timeshift_dict,
        tok.duration_dict,
        tok.octave_dict,
        tok.pitch_dict,
        tok.instrument_dict,
        tok.velocity_dict,
    ]
    return max(max(d.values()) for d in dictionaries) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MusicTokenizer config demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "src" / "llama_recipes" / "configs" / "model_config_small_microtonal.json",
        help="Path to a model config JSON",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "tests" / "test_tokenizer.log",
        help="Path to save output log",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tok = build_tokenizer(cfg)

    keys = [
        "onset_vocab_size",
        "dur_vocab_size",
        "octave_vocab_size",
        "pitch_class_vocab_size",
        "instrument_vocab_size",
        "velocity_vocab_size",
        "decode_vocab_size",
    ]

    # Open log file for writing
    with open(args.log, "w", encoding="utf-8") as log_file:
        def log_print(*args_to_print, **kwargs):
            """Print to both stdout and log file."""
            print(*args_to_print, **kwargs)
            print(*args_to_print, **kwargs, file=log_file)

        log_print("Config:", args.config)
        log_print("=" * 80)
        for k in keys:
            if k in cfg:
                log_print(f"{k}: {cfg[k]}")

        log_print("\n" + "=" * 80)
        log_print("computed_decode_vocab_size:", computed_decode_vocab_size(tok))
        
        log_print("\n" + "=" * 80)
        log_print("FULL DICTIONARIES:")
        log_print("=" * 80)
        
        dictionaries = {
            "sos_out_dict": tok.sos_out_dict,
            "timeshift_dict": tok.timeshift_dict,
            "duration_dict": tok.duration_dict,
            "octave_dict": tok.octave_dict,
            "pitch_dict": tok.pitch_dict,
            "instrument_dict": tok.instrument_dict,
            "velocity_dict": tok.velocity_dict,
        }
        
        for dict_name, dict_obj in dictionaries.items():
            log_print(f"\n{dict_name}:")
            for key, val in sorted(dict_obj.items()):
                log_print(f"  {key}: {val}")
        
        log_print(f"\n\nFull output saved to: {args.log}")

    # Tiny demo input: [onset, duration, octave, pitch_class, instrument, velocity]
    # demo = [[0, 12, 4, 11, 0, 64], [15, 8, 5, 0, 24, 90]]
    # encoded = tok.encode_series(demo, if_add_sos=True, if_add_eos=True)
    # labels = tok.encode_series_labels(encoded, if_added_sos=True, if_added_eos=True)

    # print("demo_tokens:", len(demo))
    # print("encoded_len:", len(encoded))
    # print("label_len:", len(labels))
    # print("first_label:", labels[0])


if __name__ == "__main__":
    main()
