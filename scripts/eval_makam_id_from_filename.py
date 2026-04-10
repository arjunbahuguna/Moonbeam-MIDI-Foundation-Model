#!/usr/bin/env python3
"""Quick evaluator for makam ID predictions.

Ground-truth makam is extracted from each sample filename (default: prefix before "--").
Predictions are read from either:
1) JSON/JSONL produced by inference scripts, or
2) CSV with columns containing path + predicted label.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _extract_gt_from_path(path: str, gt_regex: str | None) -> str:
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)

    if gt_regex:
        m = re.search(gt_regex, stem)
        if not m:
            raise ValueError(f"Could not extract GT makam from '{name}' with regex '{gt_regex}'")
        gt = m.group(1)
    else:
        # SymbTr-style default: hicaz--sazsemaisi--... -> hicaz
        if "--" in stem:
            gt = stem.split("--", 1)[0]
        else:
            # Fallback: prefix until first underscore/dash
            gt = re.split(r"[_-]", stem, maxsplit=1)[0]

    return _normalize_label(gt)


def _extract_json_payload(text: str):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object/array found in input file")


def _load_rows_from_json_file(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    payload = _extract_json_payload(text)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("JSON payload must be an object or list of objects")

    rows: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        p = item.get("npy_path") or item.get("path") or item.get("file")
        pred = (
            item.get("predicted_label")
            or item.get("prediction")
            or item.get("pred")
            or item.get("makam_pred")
        )
        if p is None or pred is None:
            continue
        rows.append({"path": str(p), "pred": _normalize_label(str(pred))})
    return rows


def _load_rows_from_csv(path: str, path_col: str, pred_col: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get(path_col)
            pred = row.get(pred_col)
            if not p or not pred:
                continue
            out.append({"path": str(p), "pred": _normalize_label(str(pred))})
    return out


def _iter_errors(rows: Iterable[Dict[str, str]], gt_regex: str | None):
    for r in rows:
        gt = _extract_gt_from_path(r["path"], gt_regex)
        pred = r["pred"]
        yield {
            "path": r["path"],
            "gt": gt,
            "pred": pred,
            "correct": int(gt == pred),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate makam ID accuracy using GT from filenames")
    parser.add_argument("--input", required=True, help="Path to JSON/JSONL/OUT/CSV file with predictions")
    parser.add_argument(
        "--format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="Input format (default: auto)",
    )
    parser.add_argument(
        "--gt-regex",
        default=None,
        help="Regex with capture group for GT makam from filename stem. Default uses prefix before '--'.",
    )
    parser.add_argument("--csv-path-col", default="npy_path", help="CSV column containing file path")
    parser.add_argument("--csv-pred-col", default="predicted_label", help="CSV column containing predicted makam")
    parser.add_argument("--show-errors", type=int, default=20, help="How many wrong examples to print")
    args = parser.parse_args()

    fmt = args.format
    if fmt == "auto":
        lower = args.input.lower()
        if lower.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "json"

    if fmt == "csv":
        rows = _load_rows_from_csv(args.input, args.csv_path_col, args.csv_pred_col)
    else:
        rows = _load_rows_from_json_file(args.input)

    if not rows:
        raise SystemExit("No valid prediction rows found")

    eval_rows = list(_iter_errors(rows, args.gt_regex))
    total = len(eval_rows)
    correct = sum(r["correct"] for r in eval_rows)
    acc = 100.0 * correct / max(1, total)

    print(f"total={total}")
    print(f"correct={correct}")
    print(f"accuracy={acc:.2f}%")

    wrong = [r for r in eval_rows if r["correct"] == 0]
    if wrong:
        pair_counts = Counter((r["gt"], r["pred"]) for r in wrong)
        print("\nTop confusions (gt -> pred):")
        for (gt, pred), c in pair_counts.most_common(10):
            print(f"{gt} -> {pred}: {c}")

        print("\nSample wrong predictions:")
        for r in wrong[: max(0, args.show_errors)]:
            print(f"gt={r['gt']}, pred={r['pred']}, path={r['path']}")


if __name__ == "__main__":
    main()
