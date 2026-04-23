#!/usr/bin/env python3
"""Auto-resolve `file_base_name` entries in makam_classification CSV by
matching them to actual `.npy` files in `data_eval/symbtr4eval/processed`.

Outputs:
 - data_eval/makam_classification/train_test_split_resolved.csv
 - data_eval/makam_classification/ambiguous_matches.csv
"""

import os
import re
import csv
from difflib import get_close_matches
import pandas as pd

CSV_IN = "data_eval/makam_classification/train_test_split.csv"
PROCESSED_DIR = "data_eval/symbtr4eval/processed"
CSV_OUT = "data_eval/makam_classification/train_test_split_resolved.csv"
AMBIG_OUT = "data_eval/makam_classification/ambiguous_matches.csv"

if not os.path.exists(CSV_IN):
    raise SystemExit(f"Input CSV not found: {CSV_IN}")
if not os.path.isdir(PROCESSED_DIR):
    raise SystemExit(f"Processed dir not found: {PROCESSED_DIR}")

proc_files = sorted([f for f in os.listdir(PROCESSED_DIR) if os.path.isfile(os.path.join(PROCESSED_DIR, f))])
proc_set = set(proc_files)

def norm(s: str) -> str:
    s2 = s.strip().lower()
    s2 = re.sub(r"[\s]+", "_", s2)
    s2 = re.sub(r"[_-]{2,}", "-", s2)
    s2 = s2.strip("-_.")
    return s2

proc_norm_map = {}
for f in proc_files:
    proc_norm_map.setdefault(norm(f), []).append(f)


df = pd.read_csv(CSV_IN)
resolved = []
ambiguous = []

for idx, row in df.iterrows():
    orig = str(row.get("file_base_name", "")).strip()
    chosen = None
    candidates = []

    if orig in proc_set:
        chosen = orig
    else:
        # Case-insensitive exact
        ci = [f for f in proc_files if f.lower() == orig.lower()]
        if len(ci) == 1:
            chosen = ci[0]
        else:
            # normalized match
            n = norm(orig)
            norm_hits = proc_norm_map.get(n, [])
            if len(norm_hits) == 1:
                chosen = norm_hits[0]
            else:
                # If CSV has trailing empty artist like '--.npy', try prefix match
                if orig.endswith("--.npy"):
                    prefix = orig.rsplit("--", 1)[0] + "--"
                    pref_hits = [f for f in proc_files if f.startswith(prefix)]
                    if len(pref_hits) == 1:
                        chosen = pref_hits[0]
                    else:
                        candidates = pref_hits
                # Fuzzy match
                if chosen is None:
                    close = get_close_matches(orig, proc_files, n=3, cutoff=0.85)
                    if len(close) == 1:
                        chosen = close[0]
                    elif len(close) > 1:
                        candidates = close
                    else:
                        # try looser cutoff
                        close2 = get_close_matches(orig, proc_files, n=3, cutoff=0.7)
                        if len(close2) == 1:
                            chosen = close2[0]
                        elif len(close2) > 1:
                            candidates = close2

    if chosen:
        resolved.append(chosen)
    else:
        # keep original in resolved but mark ambiguous
        resolved.append(orig)
        ambiguous.append({
            "index": int(idx),
            "orig": orig,
            "candidates": ";".join(candidates) if candidates else "",
        })

# attach resolved column and write outputs

df_out = df.copy()
df_out["file_base_name_resolved"] = resolved

df_out.to_csv(CSV_OUT, index=False, quoting=csv.QUOTE_MINIMAL)

# write ambiguous report

with open(AMBIG_OUT, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["index", "orig", "candidates"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in ambiguous:
        w.writerow(row)

print(f"Wrote resolved CSV: {CSV_OUT}")
print(f"Wrote ambiguous report: {AMBIG_OUT}")
print(f"Total rows: {len(df)}, ambiguous: {len(ambiguous)}")

if len(ambiguous) > 0:
    print("First 20 ambiguous rows:")
    for a in ambiguous[:20]:
        print(a)
else:
    print("All rows resolved uniquely.")
