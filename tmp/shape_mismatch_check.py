import torch
import json
import os
from transformers import LlamaConfig, LlamaForSequenceClassification

config_path = "src/llama_recipes/configs/model_config_small_microtonal.json"
label_map_path = "data_eval/symbtr4eval/makam_label_map.json"
checkpoint_path = "models/moonbeam_309M.pt"

with open(config_path, "r") as f:
    config_dict = json.load(f)
config = LlamaConfig(**config_dict)

with open(label_map_path, "r") as f:
    label_map = json.load(f)
config.num_labels = len(label_map)

model = LlamaForSequenceClassification(config)
model_state_dict = model.state_dict()

checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

# We mapped 'score.weight' from 'lm_head.weight' to check for mismatches
clean_state_dict = {}
for k, v in state_dict.items():
    new_k = k[7:] if k.startswith("module.") else k
    # Manually check the classification head mismatch
    if new_k == "lm_head.weight":
        target_k = "score.weight"
        if target_k in model_state_dict:
            if v.shape != model_state_dict[target_k].shape:
                 print(f"SHAPECHECK_ MISMATCH: {target_k} (mapped from {k}) | Checkpoint: {v.shape} | Model: {model_state_dict[target_k].shape}")
    clean_state_dict[new_k] = v

matched = []
mismatched = []
# (Re-running standard check but we already know score.weight vs lm_head.weight is the main one)
for k, v in clean_state_dict.items():
    if k in model_state_dict:
        if v.shape == model_state_dict[k].shape:
            matched.append(k)
        else:
            mismatched.append((k, v.shape, model_state_dict[k].shape))

for k, ckpt_shape, model_shape in mismatched:
    print(f"SHAPECHECK_ MISMATCH: {k} | Checkpoint: {ckpt_shape} | Model: {model_shape}")

msg = model.load_state_dict(clean_state_dict, strict=False)

print(f"SHAPECHECK_ SUMMARY:")
print(f"SHAPECHECK_ Matched: {len(matched)}")
print(f"SHAPECHECK_ Mismatched: {len(mismatched)}")
print(f"SHAPECHECK_ Missing (in model but not in checkpoint): {len(msg.missing_keys)}")
print(f"SHAPECHECK_ Unexpected (in checkpoint but not in model): {len(msg.unexpected_keys)}")
