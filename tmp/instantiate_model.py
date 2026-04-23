#!/usr/bin/env python3
import os, sys, json
REPO_ROOT = os.getcwd()
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'llama_recipes', 'transformers_minimal', 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
from llama_recipes.configs import makam_classification_config as MAKAM_CFG
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import LlamaConfig, LlamaForSequenceClassification

makam_cfg = MAKAM_CFG()
model_config_path = makam_cfg.model_config_path
print('model_config_path=', model_config_path)

# load label_map
label_map_path = 'data_eval/makam_classification/makam_label_map.json'
with open(label_map_path, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
num_labels = len(label_map)
print('num_labels from label_map=', num_labels)

# load LlamaConfig from file
cfg = LlamaConfig.from_pretrained(model_config_path)
print('loaded config hidden_size=', cfg.hidden_size)
cfg.num_labels = num_labels
model = LlamaForSequenceClassification(cfg)
print('model.score weight shape=', model.score.weight.shape)
print('model.num_labels=', model.num_labels)
