# Makam Classification Pipeline Audit

## Goal
Confirm the classification evaluation stack has every dependency wired up before re-running the pipeline end to end.

---

## Confirmed Paths

| Role | Path |
|---|---|
| Source MIDI files | `data/symbtr/` |
| Output folder | `processed_data_Selected_SymbTr/` |
| Processed `.npy` tokens | `processed_data_Selected_SymbTr/processed/` |
| Train/test split CSV | `processed_data_Selected_SymbTr/train_test_split.csv` |
| Classification split CSV | `processed_data_Selected_SymbTr/makam_classification_split.csv` |
| Label map | `processed_data_Selected_SymbTr/makam_label_map.json` |
| Model config | `src/llama_recipes/configs/config_micro/model_config_microtonal.json` |
| Dataset configs | `src/llama_recipes/configs/datasets.py` |

---

## Steps

### Phase 1 — Data Preprocessing

**Step 1 — Data artifacts check** `[ ]`
Fix two hardcoded args in `data_preprocess.py` before running:
- `dataset_folder`: `"data_test_Yuhang/selected_midi_SymbTr"` → `"data/symbtr"`
- `model_config`: `"src/llama_recipes/configs/model_config_microtonal.json"` → `"src/llama_recipes/configs/config_micro/model_config_microtonal.json"`

Then run `data_preprocess.py` to produce:
- `processed_data_Selected_SymbTr/processed/*.npy`
- `processed_data_Selected_SymbTr/train_test_split.csv`

**Step 2 — Classification split regeneration** `[ ]`
Run `makam_eval.py` against `train_test_split.csv` to produce:
- `processed_data_Selected_SymbTr/makam_classification_split.csv` (≥10 samples/makam rule applied)
- `processed_data_Selected_SymbTr/makam_label_map.json`

**Step 3 — CSV sanity review** `[ ]`
Run `makam_eval_validation.py` (read-only) over the regenerated CSV to cross-check per-makam counts and train/test ratios.

---

### Phase 2 — Code Wiring

**Step 4 — Tokenizer API gap** `[ ]`
Implement `MusicTokenizer.encode_series_player_classification()` in `src/llama_recipes/datasets/music_tokenizer.py`, mirroring the existing `encode_series` helpers, so `symbtr_dataset_eval.py` has a callable hook.

**Step 5 — Dataset configuration** `[ ]`
Add a `makam_eval_dataset` dataclass in `src/llama_recipes/configs/datasets.py` with:
- `data_dir`: `processed_data_Selected_SymbTr`
- `csv_file`: `processed_data_Selected_SymbTr/makam_classification_split.csv`
- `label_map`: `processed_data_Selected_SymbTr/makam_label_map.json`
- `seq_len` (to be determined)

**Step 6 — Inference alignment** `[ ]`
Walk through `recipes/inference/custom_music_generation/conditional_music_generation_batch.py` to confirm it:
- Loads `makam_classification_split.csv` and `makam_label_map.json`
- References the new `makam_eval_dataset` config
- Treats pitch values consistently (cents → fractional semitones via tokenizer)

---

## Verification Checklist

1. **Artifact integrity** — For each row in `makam_classification_split.csv`, confirm a matching `.npy` exists in `processed_data_Selected_SymbTr/processed/`.
2. **Dataset sanity** — Instantiate `MakamClassificationDataset`, fetch a single example, and verify `input_ids` tensors include SOS/EOS tokens as configured after adding the tokenizer method.
3. **Inference smoke test** — Run the small-batch inference path with one regenerated sample and confirm logits output shape equals `num_classes` from the new dataset config.

---

## Decisions

- All output paths use the `processed_data_Selected_SymbTr/` convention already present in scripts — plan references updated to match, scripts left untouched.
- Implement the new tokenizer helper (Step 4) before touching `symbtr_dataset_eval.py`, keeping changes aligned with existing encode helper style.
- Steps 1–3 must complete before Steps 4–6; within Phase 2, Steps 4 and 5 can proceed in parallel.