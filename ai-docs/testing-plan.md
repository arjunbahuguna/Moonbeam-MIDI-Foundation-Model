# Makam Classification Pipeline Audit

## Goal
Confirm the classification evaluation stack has every dependency wired up before re-running the pipeline end to end.

---

## Confirmed Paths

| Role | Path |
|---|---|
| Source MIDI files | `data/symbtr/` |
| Output folder | `data/processed_data_symbtr/` |
| Processed `.npy` tokens | `data/processed_data_symbtr/processed/` |
| Train/test split CSV | `data/processed_data_symbtr/train_test_split.csv` |
| Classification split CSV | `data/processed_data_symbtr/makam_classification_split.csv` |
| Label map | `data/processed_data_symbtr/makam_label_map.json` |
| Model config | `src/llama_recipes/configs/config_micro/model_config_microtonal.json` |
| Dataset configs | `src/llama_recipes/configs/datasets.py` |

---

## Current State

- Source MIDI data exists under `data/symbtr/`.
- Processed token files already exist under `data/processed_data_symbtr/processed/`.
- `train_test_split.csv`, `makam_classification_split.csv`, and `makam_label_map.json` already exist under `data/processed_data_symbtr/`.
- `makam_eval.py` is already generating artifacts from `data/processed_data_symbtr/`.
- `MakamClassificationDataset` already exists in `src/llama_recipes/datasets/symbtr_dataset_eval.py` and is already registered in dataset loading utilities.
- The main Phase 2 blocker is that `symbtr_dataset_eval.py` calls `MusicTokenizer.encode_series_player_classification()`, but that tokenizer method does not exist yet.

---

## Steps

### Phase 1 — Data Preprocessing

**Step 1 — Data artifacts check** `[x]`
Fix two hardcoded args in `data_preprocess.py` before running:
- `dataset_folder`: `"data_test_Yuhang/selected_midi_SymbTr"` → `"data/symbtr"`
- `model_config`: `"src/llama_recipes/configs/model_config_microtonal.json"` → `"src/llama_recipes/configs/config_micro/model_config_microtonal.json"`

Then run `data_preprocess.py` to produce:
- `data/processed_data_symbtr/processed/*.npy`
- `data/processed_data_symbtr/train_test_split.csv`

**Step 2 — Classification split regeneration** `[x]`
Run `makam_eval.py` against `train_test_split.csv` to produce:
- `data/processed_data_symbtr/makam_classification_split.csv` (≥10 samples/makam rule applied)
- `data/processed_data_symbtr/makam_label_map.json`

**Step 3 — CSV sanity review** `[x]`
Run `makam_eval_validation.py` or `tests/test_makam_eval.py` over the regenerated CSV to cross-check per-makam counts and train/test ratios.

---

### Phase 2 — Code Wiring

**Step 4 — Tokenizer API gap** `[ ]`
Implement `MusicTokenizer.encode_series_player_classification()` in `src/llama_recipes/datasets/music_tokenizer.py` so `src/llama_recipes/datasets/symbtr_dataset_eval.py` has a callable hook.

Why this step exists:
- `MakamClassificationDataset` is already implemented and already registered.
- It currently fails only because it calls a tokenizer method that does not exist.
- This is the narrowest unblocker for getting the classification dataset to load.

Implementation sequence:

**Step 4A — Add the missing tokenizer method** `[ ]`
- Add `encode_series_player_classification(raw_token_series, if_add_sos, if_add_eos)`.
- First implementation should be a thin wrapper around the existing `encode_series()` behavior.
- Keep the contract identical to the dataset expectation: return a sequence of compound tokens suitable for `torch.tensor(..., dtype=torch.long)` in the eval dataset.

**Step 4B — Validate the tensor contract** `[ ]`
- Confirm one dataset sample returns `input_ids` with shape `[seq_len, 6]`.
- Confirm SOS and EOS are present when requested.
- Confirm truncation in `symbtr_dataset_eval.py` happens after encoding, not before.

**Step 4C — Defer true classification-token injection unless needed** `[ ]`
- There is a generic `add_new_tokens()` helper in the tokenizer, but there is no complete model-side classification-token flow wired for this task yet.
- Do not introduce a dedicated classification token in the first pass unless the classifier architecture explicitly requires it.
- The first goal is dataset loading and forward compatibility with `LlamaForSequenceClassification`.

**Step 5 — Dataset configuration** `[ ]`
Add a dataset config entry in `src/llama_recipes/configs/datasets.py` for the eval pipeline.

Recommended minimal change:
- Add `symbtr_dataset_eval` as a dataclass because that name already exists in the dataset registry.

Config fields:
- `dataset`: `symbtr_dataset_eval`
- `train_split`: `train`
- `test_split`: `test`
- `data_dir`: `data/processed_data_symbtr`
- `csv_file`: `data/processed_data_symbtr/makam_classification_split.csv`
- `label_map`: `data/processed_data_symbtr/makam_label_map.json`
- `seq_len`: start with `1200`, then tune only if truncation is excessive

Alternative:
- If we rename this config to `makam_eval_dataset`, then `dataset_utils.py` and any callers must be updated to use the new key.
- For now, prefer `symbtr_dataset_eval` to minimize churn.

**Step 6 — Classifier execution alignment** `[ ]`
Replace the old inference-oriented check with the actual missing wiring for classification.

What to verify:
- `generate_dataset_config()` can resolve the new eval dataset config.
- The classification dataset can be instantiated for both `train` and `test`.
- The model path uses `LlamaForSequenceClassification` with `num_labels` matching the size of `makam_label_map.json`.
- A single batch produces logits of shape `[batch_size, num_classes]`.

Important note:
- `recipes/inference/custom_music_generation/conditional_music_generation_batch.py` is not currently part of the makam classification path.
- The real alignment target is the classification training or evaluation entrypoint, not the music generation recipe.

**Phase 2 execution order**

1. Implement `encode_series_player_classification()`.
2. Instantiate `MakamClassificationDataset` and fetch one item.
3. Add the dataset config entry in `configs/datasets.py`.
4. Resolve that config through `generate_dataset_config()`.
5. Run one forward pass with `LlamaForSequenceClassification` using `num_labels = len(label_map)`.
6. Only after that, consider any richer classification-token or conditioning changes.

---

## Verification Checklist

1. **Artifact integrity** — For each row in `makam_classification_split.csv`, confirm a matching `.npy` exists in `data/processed_data_symbtr/processed/`.
2. **Dataset sanity** — Instantiate `MakamClassificationDataset`, fetch a single example, and verify `input_ids` tensors include SOS/EOS tokens after adding the missing tokenizer method.
3. **Config sanity** — Confirm the eval dataset config resolves through `generate_dataset_config()` without manual overrides.
4. **Classifier smoke test** — Run a one-batch forward pass and confirm logits output shape equals `num_classes` from `makam_label_map.json`.

---

## Decisions

- Use `data/processed_data_symbtr/` as the source-of-truth path because that is where the current artifacts and scripts already point.
- Keep Phase 2 narrowly scoped at first: unblock dataset loading before introducing new special-token behavior.
- Prefer a `symbtr_dataset_eval` config entry over a new config name to avoid unnecessary registry churn.
- The primary Phase 2 success criterion is not generation; it is a valid classification batch flowing into `LlamaForSequenceClassification` with the correct class count.