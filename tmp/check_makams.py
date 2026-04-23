#!/usr/bin/env python3
import csv, json, os, sys

def counts(path):
    counts = {}
    n = 0
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            makam = row.get('makam') or row.get('Makam') or row.get('label') or ''
            counts[makam] = counts.get(makam, 0) + 1
    return n, counts

out = {}
paths = ['data_eval/makam_classification/train_test_split.csv', 'data_eval/symbtr4eval/train_test_split.csv']
for p in paths:
    out[p] = {}
    if not os.path.exists(p):
        out[p]['exists'] = False
        continue
    n, c = counts(p)
    out[p] = {'exists': True, 'rows': n, 'unique': len(c), 'top': sorted(c.items(), key=lambda x: -x[1])[:10]}

lm_path = 'data_eval/makam_classification/makam_label_map.json'
if os.path.exists(lm_path):
    with open(lm_path, 'r', encoding='utf-8') as f:
        lm = json.load(f)
    out['label_map'] = {'path': lm_path, 'num_labels': len(lm), 'sample_keys': list(lm.keys())[:10]}
else:
    out['label_map'] = {'exists': False}

correct_path = 'data_eval/symbtr4eval/train_test_split.csv'
if os.path.exists(correct_path):
    n, c = counts(correct_path)
    keys = sorted([k for k in c.keys() if k != ''])
    mapping = {k: i for i, k in enumerate(keys)}
    out['correct_label_map'] = {'num_labels': len(mapping), 'sample_keys': keys[:10]}
    save_path = 'data_eval/makam_classification/makam_label_map_from_symbtr4eval.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    out['saved'] = save_path

print(json.dumps(out, ensure_ascii=False, indent=2))
