#!/usr/bin/env python3
import os, sys, json
REPO_ROOT = os.getcwd()
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'llama_recipes', 'transformers_minimal', 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
from llama_recipes.configs import train_config as TRAIN_CONFIG, makam_classification_config as MAKAM_CLASSIFICATION_CONFIG
from llama_recipes.utils.config_utils import generate_dataset_config

train_cfg = TRAIN_CONFIG()
makam_cfg = MAKAM_CLASSIFICATION_CONFIG()
dataset_config = generate_dataset_config(train_cfg, {})
print('dataset_config.csv_file=', dataset_config.csv_file)
print('dataset_config.label_map=', getattr(dataset_config, 'label_map', None))
print('makam_cfg.label_map_path=', makam_cfg.label_map_path)

# show files
paths = [dataset_config.label_map, makam_cfg.label_map_path, 'data_eval/makam_classification/makam_label_map.json']
for p in paths:
    if p and os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                j = json.load(f)
            print(p, 'exists, num_labels=', len(j))
        except Exception as e:
            print(p, 'exists but cannot read:', e)
    else:
        print(p, 'not present')
