# Microtonal Continual Pretraining — Team TODO

**Branch:** `microtonal_cpt` (from `main`)
**Repo:** `Moonbeam-MIDI-Foundation-Model`
**Team:** Angelos, Yuhang, Julian, Arjun

---

## Implementation Status (updated 2026-03-07)

| Task | Status | Notes |
|------|--------|-------|
| **A1.** Microtonal MIDI parsing | DONE | Helper functions + midi_to_compound in music_tokenizer.py |
| **A2.** SymbTr data preprocessing | DONE | 3000/3000 files processed. Output: `~/projects/microtok/data/symbtr_processed/` (processed/*.npy + train_test_split.csv). 1.16M notes, 52 unique cent values, 83.5% microtonal, train/test=2700/300 |
| **A3.** 53-TET augmentation | NOT STARTED | Needs A2 complete |
| **A4.** Western replay data | DONE | Maestro v3: 1272/1276 files preprocessed (4 failed: large onset/duration). Output: `~/projects/microtok/data/maestro_processed/` (processed/*.npy + train_test_split.csv) |
| **B1.** Model config | DONE | model_config_microtonal.json created |
| **B2.** modeling_llama.py bugfixes | DONE | .long() cast, position_ids dtype fix |
| **B3.** Append-only pitch_dict | DONE | In music_tokenizer.py |
| **B4.** convert_to/from_language_tokens | DONE | Returns integer cents; /100.0 only in embed_tokens |
| **B5.** Weight transfer + interpolation | DONE | microtonal_utils.py: expand_vocab_with_interpolation |
| **C1.** CPT training script | DONE | real_finetuning_microtonal.py |
| **C1b.** LoRA + training strategy | DONE | Decoder unfreezing after PEFT, documented in peft.py |
| **C2.** Float pitch in training loop | DONE | pitch /100.0 in modeling_llama.py embed_tokens (line 1437), not train_utils.py |
| **C3.** L_anchor regularization | DONE | In train_utils.py train_con_gen(), logged to wandb |
| **C3b.** L_smooth regularization | DONE | Adjacent-bin pairs, logged to wandb |
| **C4.** Per-param LR groups | DONE | 3 param groups (LoRA, decoder head, GRU) + 3x gradient scaling hook for new rows in real_finetuning_microtonal.py |
| **C5.** Data mixing logic | DONE | WeightedRandomSampler in real_finetuning_microtonal.py. CLI: `--western_data_dir=... --western_csv_file=... --mixing_alpha=0.8`. Packing only. |
| **C6.** GRU accuracy logging | DONE | Per-attribute + western/micro pitch split in modeling_llama.py |
| **D1.** Inference float buffer | DONE | generation.py: float32 token buffer |
| **D2.** convert_from → cents (no /100.0) | DONE | Returns integer cents; embed_tokens handles /100.0 for both training+inference |
| **D3.** compound_to_midi pitchbend | DONE | Rewritten with pitchwheel messages |
| **D4.** Evaluation script | NOT STARTED | Can scaffold structure before dataset |
| **D4b.** Ablation table | NOT STARTED | Needs trained model |
| **D4c.** TF-AR accuracy gap | NOT STARTED | Needs trained model |
| **D5.** Western forgetting eval | NOT STARTED | Needs trained model + western test set |
| **D6.** Embedding viz | NOT STARTED | Needs trained model |
| **D7.** Makam-specific eval | NOT STARTED | Needs trained model + SymbTr metadata |
| **D8.** Listening test | NOT STARTED | Needs generated samples |
| **D9.** Seyir (melodic progression) analysis | NOT STARTED | Needs generated samples + makam seyir reference |
| **D10.** Karar (finalis) accuracy | NOT STARTED | Needs generated samples + makam karar reference |
| **D11.** Pitch stability / jitter metric | NOT STARTED | Needs generated samples |
| **D12.** Per-makam perplexity breakdown | NOT STARTED | Needs trained model + SymbTr metadata |
| **D13.** Naive quantization baseline | NOT STARTED | Needs SymbTr data (A2); no arch changes |

**Summary:** All model/tokenizer/training-loop code (A1-A2, A4, B1-B5, C1-C6, D1-D3) is DONE.
**Test suite:** 57 unit tests in `tests/test_microtonal.py` (run: `python -m pytest tests/test_microtonal.py -v --noconftest`).
**Model variants:** Both S (309M, model_config_small_microtonal.json) and M (839M, model_config_microtonal.json) are supported. All offsets derived from config — no hardcoded model-specific values.
**IMPORTANT — Model S vs M use DIFFERENT pretraining data (paper Table 6):**
- **Moonbeam-S (309M):** Pretrained on LakhMIDI only. GRU output=2341, onset/dur=1024.
- **Moonbeam-M (839M):** Pretrained on 19 datasets (paper Table 5, 81.6K hrs). GRU output=8487, onset/dur=4097.
- Western replay data must match the model variant's pretraining corpus.
**Resolution:** `microtonal_resolution` in JSON config (1=1-cent, 10=10-cent) controls tokenization grid end-to-end.
Remaining: augmentation (A3), evaluation (D4-D13).
**Ready to train:** SymbTr-only run needs no further code. Full mixed run (SymbTr + Maestro replay) is also ready.
**Pretrained checkpoints:** `models/moonbeam_309M.pt` (S) and `models/moonbeam_839M.pt` (M) — already in repo.

### Complete Training Pipeline (from preprocessed data to training)

#### Dataset path configuration

The placeholder paths in `datasets.py` (`/PATH/TO/DATA/DIR`, `/PATH/TO/CSV`) are defaults
overridden at runtime. Two approaches (both work identically):

**Option A — Edit `datasets.py` + reinstall** (Moonbeam README approach, line 46):
```python
# In src/llama_recipes/configs/datasets.py
@dataclass
class symbtr_dataset:
    data_dir: str = "/home/USER/projects/microtok/data/symbtr_processed"
    csv_file: str = "/home/USER/projects/microtok/data/symbtr_processed/train_test_split.csv"
```
Then: `pip install .`

**Option B — CLI args with dot notation** (no reinstall needed):
```bash
--symbtr_dataset.data_dir ~/projects/microtok/data/symbtr_processed \
--symbtr_dataset.csv_file ~/projects/microtok/data/symbtr_processed/train_test_split.csv
```

Path flow: CLI args → `update_config()` (config_utils.py:30-35) → dataset dataclass → dataset class constructor.

#### Recommended training hyperparameters

Based on computed data sizes and CPT literature (see `configs/training.py` for guidance comments):

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| `context_length` | **1024** | MUST match pretrained model. Moonbeam S/M both trained at 1024. Using 2048 would create unseen positional encodings. |
| `lr` | **2e-5** | CPT standard: 10-30x lower than pretraining (3e-4). Prevents catastrophic forgetting. |
| `weight_decay` | **0.01** | Mild regularization for CPT (pretraining used 0.0). |
| `gamma` | **0.85** | Same StepLR decay as pretraining (paper Section 4.1). |
| `num_epochs` | **20-30** | Small dataset (~1M tokens) needs multiple passes. Music for All (Mehta 2025) uses 20-25 epochs with early stopping on similar-sized Makam data. Use validation_interval + patience to stop early. |
| `batch_size_training` | **4** | Moonbeam pretraining used batch packing on 2x A100. 4 works on single A100 (40GB) with LoRA. |
| `mixing_alpha` | **0.8** | 80% micro / 20% western. Literature: 5-30% replay prevents forgetting. |
| `gradient_clipping` | **True** | Stabilizes early CPT steps (new vocab rows have interpolated init). |
| `gradient_clipping_threshold` | **1.0** | Standard value. Matches Music for All (Mehta 2025). |
| `use_peft` / `peft_method` | **True / lora** | LoRA + frozen backbone is standard for CPT with small target domain. Music for All uses bottleneck adapters (0.1% params) — same spirit. |
| `validation_interval` | **100** | Check val loss every 100 steps. Stop if no improvement for patience=5 intervals. |

**Data size estimates (context_length=1024):**
- SymbTr train: 2700 pieces, ~1.04M tokens → **~1004 packed chunks**
- Maestro train: 1144 pieces, ~6.32M tokens → **~1138 packed chunks**
- Combined: **~2142 chunks/epoch**
- At batch_size=4: **~536 steps/epoch**
- 20 epochs ≈ 10,720 steps, 30 epochs ≈ 16,080 steps
- With early stopping (patience 5 checks × 100 steps = stop after 500 steps of no improvement)

**Reference — Music for All (Mehta et al. 2025, arXiv:2502.07328):**
Adapts MusicGen/Mustango to Turkish Makam + Hindustani Classical with PEFT adapters.
Key findings relevant to our work:
- 20-25 epochs with early stopping (patience 5) on ~97h Makam audio
- LR 4.5-5e-5, weight_decay 0.01-0.05, grad clip 1.0, batch_size 4
- **No replay data → observed catastrophic forgetting** (Mustango creativity queries regressed). Validates our C5 replay approach.
- Adapter size 0.1% of total params (2M/2B). Similar to LoRA parameter efficiency.
- PEFT effectiveness varies by model architecture — not all models adapt equally well.

#### Quick first training run (SymbTr only, no replay)

```bash
# 1. Activate environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate microtok

# 2. Ensure packages are installed (after any source edits)
cd ~/projects/microtok/Moonbeam-MIDI-Foundation-Model
pip install .
pip install src/llama_recipes/transformers_minimal/.

# 3. Download pretrained checkpoint from HuggingFace
#    https://huggingface.co/guozixunnicolas/moonbeam-midi-foundation-model
#    Save to e.g. ~/projects/microtok/checkpoints/moonbeam_pretrained.pt

# 4. Train (single GPU, SymbTr only)
# NOTE: Do NOT use --enable_ddp with single GPU — it disables WeightedRandomSampler
#       (mixing_alpha would be ignored, falling back to uniform ~50/50 sampling)
python src/llama_recipes/real_finetuning_microtonal.py \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --val_batch_size 2 \
  --run_validation True \
  --validation_interval 100 \
  --save_metrics True \
  --dist_checkpoint_root_folder checkpoints/microtonal_cpt \
  --dist_checkpoint_folder ddp \
  --trained_checkpoint_path ~/projects/microtok/checkpoints/moonbeam_pretrained.pt \
  --pure_bf16 True \
  --use_peft True \
  --peft_method lora \
  --quantization False \
  --model_name microtonal_cpt \
  --dataset symbtr_dataset \
  --symbtr_dataset.data_dir ~/projects/microtok/data/symbtr_processed \
  --symbtr_dataset.csv_file ~/projects/microtok/data/symbtr_processed/train_test_split.csv \
  --output_dir checkpoints/microtonal_cpt \
  --batch_size_training 4 \
  --context_length 1024 \
  --num_epochs 20 \
  --gradient_clipping True \
  --gradient_clipping_threshold 1.0 \
  --use_wandb True \
  --log_interval 10 \
  --gamma 0.85
```

#### Full training run (with western replay)

Same as above, plus:
```bash
  --western_data_dir ~/projects/microtok/data/maestro_processed \
  --western_csv_file ~/projects/microtok/data/maestro_processed/train_test_split.csv \
  --mixing_alpha 0.8
```

#### Data locations summary

| Dataset | Path | Files | Notes |
|---------|------|-------|-------|
| SymbTr (microtonal) | `~/projects/microtok/data/symbtr_processed/` | 3000 .npy | DONE. CSV: `train_test_split.csv` (train=2700, test=300) |
| Maestro (western replay) | `~/projects/microtok/data/maestro_processed/` | 1272 .npy | DONE. CSV: `train_test_split.csv`. All pitches = multiples of 100 |
| Pretrained checkpoint (S) | `models/moonbeam_309M.pt` | 591 MB | In repo |
| Pretrained checkpoint (M) | `models/moonbeam_839M.pt` | 1.6 GB | In repo |

#### Ablation plan

Run these configurations to measure each component's contribution:

| Run | LoRA | Replay | L_anchor | L_smooth | Grad scaling | Notes |
|-----|------|--------|----------|----------|-------------|-------|
| 1. Full | Yes | Yes (α=0.8) | Yes | Yes | Yes (3×) | Main model |
| 2. No replay | Yes | No | Yes | Yes | Yes | Measure forgetting |
| 3. No regularization | Yes | Yes | No | No | Yes | Measure reg impact |
| 4. No LoRA (full FT) | No | Yes | Yes | Yes | Yes | LoRA vs full |
| 5. High replay | Yes | Yes (α=0.5) | Yes | Yes | Yes | Replay ratio sensitivity |

### Remaining work (for colleagues)

#### Phase 1b: Training + evaluation (needs GPU)
1. **Run training** with CLI commands above. Monitor wandb for loss curves, GRU accuracy splits.
2. **A3 (optional):** 53-TET augmentation — 52 transpositions per piece, 3.3x data increase. Run before training.
3. **D4-D8:** Evaluation scripts — scaffold after first model checkpoint.
4. **D13:** Naive quantization baseline — round microtonal pitches to nearest semitone, retrain. No arch changes.

#### Phase 2: Makam classification (adapt player_classification branch)
- Branch from `microtonal_cpt`, port `PlayerClassificationDataset` from `finetune_player_classification`
- Change: 30 player classes → 161 makam classes (labels from SymbTr filenames, already in CSV)
- Uses `LlamaForSequenceClassification` with `<classification>` token (-4)
- Provides intrinsic evaluation: can the model distinguish makams?

#### Phase 3: Conditional generation (see section below)

### Future: Conditional Generation (Phase 2+)

SymbTr filenames encode three labels: `[makam]--[form]--[usul]--[title]--[artist]`.
These are directly usable as metadata conditions via Moonbeam's existing conditional generation framework.

#### Existing conditioning infrastructure (on other branches)

The codebase already has two conditional generation patterns:

**Pattern A — Emotion conditioning (`emophia_con_gen_dataset.py`):**
Simple: prepend a single token before SOS → `[emotion_4Q] + [SOS] + music + [EOS]`.
Labels for the condition token = dummy SOS labels (not predicted by GRU).
Uses `encode_series_con_gen_emotion()` / `encode_series_labels_con_gen_emotion()`.

**Pattern B — Multi-condition (`commu_con_gen_dataset.py`, `conditional_gen_commu` branch):**
Rich: `[metadata_tokens] + [SOC] + chords + [EOC] + [SOS] + music + [EOS]`.
Uses `LlamaForCausalLM_Conditional_Generation` (modeling_llama.py:1956) — a **separate model class**
that adds supplementary embeddings, chord conditioning layers, and 10% chord dropout.
Uses `encode_series_con_gen_commu()` / `encode_series_labels_con_gen_commu()`.
Forward accepts `metadata_condition` and `bar_beat_chord_condition` tensors.

**Pattern C — Classification (`finetune_player_classification` branch):**
Uses `LlamaForSequenceClassification` with a `<classification>` token (-4).
Chunks sequences by duration. 30-class player identification.
Adaptable for **makam classification evaluation** (D7).

#### Dataset classes comparison

| Class | Path Convention | Purpose |
|-------|----------------|---------|
| `LakhDataset` | `data_dir/processed/file.npy` | Unconditional, bare minimum |
| `MergeDataset` | `data_dir/file.npy` (**no `processed/`!**) | Unified directory loader |
| `SymbTrDataset` | `data_dir/processed/file.npy` | Optional makam/form/usul conditioning |
| `Emophia_Con_Gen_Datasets` | `data_dir/processed/file.npy` | Emotion 4Q token |
| `Commu_Con_Gen_Datasets` | `data_dir/processed/file.npy` | Metadata + chords (SOC/EOC) |
| `PlayerClassificationDataset` | `data_dir/processed/file.npy` | Classification with chunking |

**Why `SymbTrDataset` has `return_conditioning`:** It pre-computes makam/form/usul integer IDs
from the CSV (which encodes them from filenames). When `return_conditioning=True`, `__getitem__`
returns `symbtr_makam_id`, `symbtr_form_id`, `symbtr_usul_id` alongside the standard
`input_ids`/`labels`/`attention_mask`. These are currently unused by `LlamaForCausalLM`
(which rejects extra kwargs) but are ready for Phase 2 conditional generation.

#### Phase 2: Makam conditional generation (simplest — Pattern A)

- Prepend makam token before SOS: `[makam_token] + [SOS] + music + [EOS]`
- Assign negative compound IDs: `makam_Hicaz = [-334, -334, -334, -334, -334, -334]` etc.
- Add `encode_series_con_gen_makam()` to music_tokenizer.py (modeled on emotion version)
- `SymbTrDataset` already has the IDs — just need to map them to token IDs
- Minimal: new encode/decode methods + new negative token registrations. Same model class.

#### Phase 3a: Makam scale degrees as temporal condition (Pattern B)

- Makam scale degrees (e.g., Rast: C D E♭↓ F G A B♭↓) encoded as chord-like condition
- Frame with SOC/EOC: `[makam_id] + [SOC] + scale_degrees + [EOC] + [SOS] + music + [EOS]`
- Requires `LlamaForCausalLM_Conditional_Generation` model class
- More powerful but more complex

#### Phase 3b: Usul (rhythmic cycle) as temporal condition

- Rhythmic pattern encoded similarly to chord progressions
- Could combine with Phase 3a: `[makam_id] + [SOC] + scale + usul + [EOC] + ...`

#### Branching strategy

- **Create `microtonal_conditional` from `microtonal_cpt`** (has all microtonal fixes)
- Port conditional generation infrastructure from `conditional_gen_commu` branch
- **Do NOT branch from `conditional_gen_commu`** (zero microtonal code there)
- Player classification from `finetune_player_classification` can be adapted for makam evaluation (D7)

---

## Overview: 4 Parallel Streams

```
STREAM A (Data)          STREAM B (Model+Tokenizer)     STREAM C (Training)        STREAM D (Inference+Eval)
─────────────────        ─────────────────────────       ──────────────────         ─────────────────────────
A1. Port MIDI parsing    B1. Model config                C1. CPT training script    D1. Inference float buffer
A2. SymbTr preprocess    B2. modeling_llama bugfixes     C1b. LoRA + training strat  D2. convert_from → frac.semi
A3. 53-TET augmentation  B3. Append-only pitch_dict      C2. Float pitch in loop    D3. compound_to_midi pitchbend
A4. Western replay data  B4. convert_to/from_lang_tokens C3. L_anchor reg.          D4. Evaluation script
                         B5. Weight transfer code        C3b. L_smooth reg.          D4b. Ablation table
                                                         C4. Per-param LR groups    D4c. TF-AR accuracy gap
                                                         C5. Data mixing logic      D5. Eval: western forgetting
                                                         C6. GRU accuracy logging   D6. Eval: embedding viz
                                                                                    D7. Eval: makam-specific
                                                                                    D8. Eval: listening test
                                                                                    D9. Eval: seyir analysis
                                                                                    D10. Eval: karar accuracy
                                                                                    D11. Eval: pitch jitter
                                                                                    D12. Eval: per-makam PPL
                                                                                    D13. Naive quant. baseline

                         ┌──────────────────────────────────────────────────────────┐
                         │  A + B must complete before C can do integration runs    │
                         │  B must complete before D1-D3                            │
                         │  C must complete before D4-D7 (need trained model)       │
                         └──────────────────────────────────────────────────────────┘
```

---

## STREAM A — Data Pipeline

### A1. Add Microtonal MIDI Parsing Utilities

**Owner:** ___
**Files:** `src/llama_recipes/datasets/music_tokenizer.py`
**Blocked by:** Nothing — start immediately

> **NOTE:** `feat/mtok_vocab` has these utilities but that branch's overall approach is WRONG
> (it destroys pretrained knowledge by changing FME base, RoPE base, and reshuffling the vocab).
> We are writing these functions fresh on our branch. They are pure math — no dependency on
> model architecture or vocab layout. Do NOT copy any config, vocab, or model changes from
> `feat/mtok_vocab`.

Add these 4 helper functions at module level (after line 20):

```python
def pitch_to_octave_pitch_class_microtonal(canonical_pitch_semitones):
    """Float semitones → (octave, pitch_class_cents 0-1199)"""
    octave = int(canonical_pitch_semitones // 12)
    pitch_class_cents = int(round((canonical_pitch_semitones % 12) * 100))
    pitch_class_cents = max(0, min(1199, pitch_class_cents))
    return octave, pitch_class_cents

def octave_pitch_class_to_pitch_microtonal(octave, pitch_class_cents):
    """Reverse: (octave, cents) → float semitones"""
    return octave * 12.0 + pitch_class_cents / 100.0

def pitchbend_to_semitones(pitchbend_value, sensitivity=2.0):
    """MIDI pitchbend int [-8192, 8191] → semitone offset float"""
    return (pitchbend_value / 8192.0) * sensitivity

def canonical_pitch_to_midi_and_pitchbend(canonical_pitch_semitones, sensitivity=2.0):
    """Float semitones → (midi_note 0-127, pitchbend int)"""
    midi_note = int(round(canonical_pitch_semitones))
    midi_note = max(0, min(127, midi_note))
    residual = canonical_pitch_semitones - midi_note
    pitchbend = int(round((residual / sensitivity) * 8192))
    pitchbend = max(-8192, min(8191, pitchbend))
    return midi_note, pitchbend
```

Modify `MusicTokenizer.__init__()` (line ~25) — add two params:

```python
def __init__(self, ..., microtonal=False, pitchbend_sensitivity=2.0):
    self.microtonal = microtonal
    self.pitchbend_sensitivity = pitchbend_sensitivity
    # ... rest unchanged
```

Modify `midi_to_compound()` (line 341) — add pitchbend tracking + note-0 filtering:

1. Add `pitchbend_state = defaultdict(int)` after `open_notes` (line 356)
2. Add `elif message.type == "pitchwheel": pitchbend_state[message.channel] = message.pitch` (after line 369)
3. Add note-0 filter: `if message.note == 0: continue` (after line 373, before pitch calc)
4. Replace pitch calculation (line 381):
   ```python
   if self.microtonal:
       bend_semitones = pitchbend_to_semitones(pitchbend_state[message.channel], self.pitchbend_sensitivity)
       canonical_pitch = message.note + bend_semitones
       octave, pitch_class = pitch_to_octave_pitch_class_microtonal(canonical_pitch)
   else:
       octave, pitch_class = pitch_to_octave_pitch_class(message.note)
   ```

**Acceptance:** `tokenizer.midi_to_compound(symb_tr_file)` returns compounds with `pitch_class` in cents (0-1199).

---

### A2. SymbTr Data Preprocessing — DONE ✓

**Owner:** ___
**Files:** `data_preprocess.py`, `model_config_microtonal.json`
**Status:** COMPLETE (2026-03-07)

**Command used:**
```bash
python data_preprocess.py \
  --dataset_name SymbTr \
  --dataset_folder ~/projects/microtok/data/symbtr_raw/midi \
  --output_folder ~/projects/microtok/data/symbtr_processed \
  --model_config src/llama_recipes/configs/model_config_microtonal.json \
  --train_ratio 0.9 \
  --microtonal \
  --pitchbend_sensitivity 2.0
```

**Results verified:**
| Metric | Value |
|--------|-------|
| Files processed | 3000/3000 (100%) |
| Total notes | 1,163,006 |
| Unique pitch cents | 52 (matches SymbTr documentation) |
| Microtonal notes | 83.5% |
| Western notes | 16.5% (multiples of 100) |
| Note length range | 18–6035 (mean 388, median 375) |
| Octave range | 4–7 (concentrated in 5-6) |
| Train/test split | 2700/300 (90/10%) |
| Unique makams | 167 (6 empty) |
| Output path | `~/projects/microtok/data/symbtr_processed/` |
| CSV file | `train_test_split.csv` (columns: file_base_name, split, length, duration, makam, form, usul, title, artist) |
| Data source | Zenodo: https://zenodo.org/records/15470412 |

---

### A3. 53-TET Transposition Augmentation

**Owner:** ___
**Files:** New script `augment_53tet.py` or integrated into `data_preprocess.py`
**Blocked by:** A2

For each SymbTr `.npy` file, generate 52 transposed copies (k=1..52 Holdrian commas):

```python
# 1 Holdrian comma ≈ 22.6 cents
for k in range(1, 53):
    shift_cents = round(k * (1200 / 53))  # k * ~22.64 cents
    transposed = original.copy()
    # Shift pitch column (index 3):
    transposed[:, 3] = transposed[:, 3] + shift_cents
    # Handle octave wrapping:
    overflow = transposed[:, 3] >= 1200
    transposed[overflow, 2] += 1  # octave up
    transposed[overflow, 3] -= 1200
    underflow = transposed[:, 3] < 0
    transposed[underflow, 2] -= 1  # octave down
    transposed[underflow, 3] += 1200
    # Filter: discard if any note has octave outside [0, 10] or MIDI note > 127
    midi_notes = transposed[:, 2] * 12 + transposed[:, 3] / 100.0
    if midi_notes.min() < 0 or midi_notes.max() > 127:
        continue
    np.save(f"{base_name}_t{k}.npy", transposed)
```

**Result:** ~3000 × 53 = ~159,000 files, ~61.5M notes (53× increase).

**Why musically valid:** Makam intervals are relative (Holdrian commas). Transposing preserves all melodic structure — only absolute pitch changes.

**Acceptance:** Augmented `.npy` files generated, CSV updated with all transpositions.

---

### A4. Western Replay Data Preparation

**Owner:** ___
**Files:** Data prep script (reuses `data_preprocess.py`)
**Blocked by:** Nothing (download western MIDI, preprocess with `--microtonal`)

Prepare a **subset of western MIDI data** for data mixing during CPT to prevent catastrophic forgetting.

#### Which dataset to use (depends on model variant):

| Model | Pretraining Data (paper Table 6) | Best Western Replay Source |
|-------|----------------------------------|---------------------------|
| **Moonbeam-S (309M)** | LakhMIDI only | LakhMIDI subset |
| **Moonbeam-M (839M)** | 19 datasets (Table 5, 81.6K hrs) | Maestro + ATEPP (or any Table 5 subset) |

**Recommended datasets for Moonbeam-M western replay** (ranked by availability + quality):

| Dataset | Hours | License | Download |
|---------|-------|---------|----------|
| **Maestro v3** | 197 | CC BY 4.0 | https://magenta.tensorflow.org/datasets/maestro |
| **ATEPP** | 997 | CC BY 4.0 | https://github.com/BetsyTang/ATEPP |
| **ASAP** | 111 | CC BY-NC-SA 4.0 | https://github.com/CPJKU/asap-dataset |
| **PiJAMA** | 217 | CC BY-NC 4.0 | https://github.com/ETH-DISCO/PiJAMA |
| **GuitarSet** | 3 | CC BY 4.0 | https://zenodo.org/records/3371780 |

**Maestro v3 is the top recommendation:** freely available, CC BY 4.0, 197 hours (~42M tokens),
high-quality piano performance MIDI, already used in Moonbeam-M pretraining. Alone it provides
enough data for 20% replay with ~3000 SymbTr files.

#### Preprocessing steps:

1. Download western MIDI to e.g. `~/projects/microtok/data/maestro_raw/`
2. **Re-preprocess with `--microtonal` flag** (same pipeline as SymbTr):
   ```bash
   python data_preprocess.py \
     --dataset_name Maestro \
     --dataset_folder ~/projects/microtok/data/maestro_raw \
     --output_folder ~/projects/microtok/data/maestro_processed \
     --model_config src/llama_recipes/configs/model_config_microtonal.json \
     --train_ratio 0.9 \
     --microtonal
   ```
   Since western MIDI has no pitchbend, `--microtonal` simply outputs `pitch * 100`
   (C=0→0, C#=1→100, ..., B=11→1100). This makes western data compatible with the
   microtonal tokenizer's pitch_dict (B3) and the `/100.0` conversion in embed_tokens (C2).
3. Output: `~/projects/microtok/data/maestro_processed/processed/*.npy` + CSV

**Why re-preprocess instead of converting existing .npy files:**
- Consistent pipeline (same tokenizer, same vocab limits, same CSV format)
- No risk of forgetting the ×100 conversion
- `data_preprocess.py` already handles the Maestro directory structure (recursive MIDI search)

**Sizing:** With α=0.8 (80% microtonal), western data ≈ 20% of total steps.
- Without augmentation: 3000 SymbTr files → need ~750 western files
- With 53-TET augmentation (A3): ~159K SymbTr files → need ~40K western files
- Maestro has ~1300 files — sufficient for non-augmented run, may need additional datasets for augmented run

**Acceptance:** Western replay `.npy` files have pitch in cents (0, 100, 200, ..., 1100).
A single microtonal tokenizer processes both western and microtonal data correctly.

---

## STREAM B — Model Architecture + Tokenizer

### B1. New Model Config

**Owner:** ___
**Files:** `src/llama_recipes/configs/model_config_microtonal.json` (NEW)
**Blocked by:** Nothing — start immediately

Copy `model_config.json` and change ONLY:

```json
{
  "microtonal": true,
  "pitchbend_sensitivity": 2.0,
  "pitch_class_vocab_size": 1202,
  "decode_vocab_size": 9675,

  // EVERYTHING ELSE UNCHANGED:
  "onset_vocab_size": 4099,
  "dur_vocab_size": 4099,
  "octave_vocab_size": 13,
  "instrument_vocab_size": 131,
  "velocity_vocab_size": 130,
  "pitch_embedding": {"method": "FME", "base": 20},
  "rope_theta_pitch": 20
  // ... all other fields identical
}
```

**Vocab math:**
```
Original:     1 + 4099 + 4099 + 13 + 14 + 131 + 130 = 8487
Microtonal:   1 + 4099 + 4099 + 13 + 1202 + 131 + 130 = 9675
New IDs:      9675 - 8487 = 1188 microtonal pitch tokens appended after position 8487
```

**Language ID layout (append-only):**
```
[sos_out:0 | timeshift:1-4099 | duration:4100-8198 | octave:8199-8211 | pitch:8212-8225 | instrument:8226-8356 | velocity:8357-8486 | pitch_micro:8487-9674]
 ↑ SAME      ↑ SAME             ↑ SAME               ↑ SAME            ↑ SAME (12+SOS+EOS)  ↑ SAME                ↑ SAME              ↑ NEW (1188 appended)
```

**CRITICAL:** `pitch_embedding.base` and `rope_theta_pitch` MUST stay at `20`. This preserves all pretrained FME + MRA knowledge. Changing the base (like `feat/mtok_vocab` did with base=1800) would destroy ALL pretrained pitch embeddings — `FME(1, base=20) ≠ FME(100, base=1800)`.

**Resolution choice (1-cent vs 10-cent):**

| Resolution | pitch_class_vocab_size | decode_vocab_size | New tokens | Empty bins | Precision |
|------------|----------------------|-------------------|------------|------------|-----------|
| **1-cent (default)** | 1202 | 9675 | 1188 | ~90% | ±0.5 cent |
| **10-cent (ablation)** | 122 | 8595 | 108 | ~50% | ±5 cent |

**Decision needed:** The review plan recommends 10-cent as the pragmatic default (fewer empty bins, small decoder expansion), but 1-cent gives maximum precision and the 53-TET augmentation (A3) fills many empty bins. **Start with 10-cent (108 new tokens), ablate 1-cent (1188 new tokens).** Update `pitch_class_vocab_size` and `decode_vocab_size` accordingly.

**Acceptance:** Config file passes JSON validation. `decode_vocab_size` = 9675.

---

### B2. `modeling_llama.py` Bugfixes (2 patches)

**Owner:** ___
**File:** `src/llama_recipes/transformers_minimal/src/transformers/models/llama/modeling_llama.py`
**Blocked by:** Nothing

**Fix 1 — Instrument embedding `.long()` cast (line 1437):**
```python
# BEFORE:
instruments = self.instrument_embedding(input_ids_tmp[..., 4])
# AFTER:
instruments = self.instrument_embedding(input_ids_tmp[..., 4].long())
```
**Why:** When `input_ids` is float (for fractional pitch), `nn.Embedding` crashes without `.long()`.

**Fix 2 — Position IDs dtype preservation (lines 940-941):**
```python
# BEFORE:
position_ids_sos = torch.where(where_sos, torch.tensor([0 for _ in range(6)]).to(hidden_states.device), position_ids)
position_ids_eos = torch.where(where_eos, torch.tensor([2**15 for _ in range(6)]).to(hidden_states.device), position_ids_sos)
# AFTER:
position_ids_sos = torch.where(where_sos, torch.tensor([0 for _ in range(6)]).to(position_ids), position_ids)
position_ids_eos = torch.where(where_eos, torch.tensor([2**15 for _ in range(6)]).to(position_ids), position_ids_sos)
```
**Why:** `.to(hidden_states.device)` only copies device, not dtype. `.to(position_ids)` preserves both. Without this, `torch.where` fails on dtype mismatch when position_ids is float.

Also check line 945+ for `where_classification` — same fix if present.

**Acceptance:** Model forward pass works with float `input_ids` tensor.

---

### B3. Append-Only `pitch_dict` in Tokenizer

**Owner:** ___
**File:** `music_tokenizer.py`, lines 57-73
**Blocked by:** A1 (needs microtonal flag), B1 (needs config values)

The current `pitch_dict` (line 62) builds a **sequential** mapping. For append-only, we need a **non-contiguous** mapping:

```python
# Replace line 62 with:
if self.microtonal:
    # Western cents (multiples of 100) → ORIGINAL pretrained IDs
    # Microtonal cents (non-multiples of 100) → APPENDED IDs after 8487
    original_decode_vocab_size = (self.sos_out_vocab_size + self.timeshift_vocab_size +
                                  self.dur_vocab_size + self.octave_vocab_size +
                                  14 +  # original 14 western pitch IDs (12 + SOS + EOS)
                                  self.instrument_vocab_size + self.velocity_vocab_size)
    # original_decode_vocab_size = 8487

    pitch_offset = (self.sos_out_vocab_size + self.timeshift_vocab_size +
                    self.dur_vocab_size + self.octave_vocab_size)
    # pitch_offset = 8212

    self.pitch_dict = {}
    micro_id = original_decode_vocab_size  # Start appending at 8487

    for c in range(self.pitch_class_vocab_size):  # 0..1201
        if c < 1200 and c % 100 == 0:
            # Western semitone: map to ORIGINAL pretrained ID
            western_pc = c // 100  # 0-11
            self.pitch_dict[c] = pitch_offset + western_pc
        elif c >= 1200:
            # SOS (1200) and EOS (1201): map to original SOS/EOS pitch IDs
            original_idx = 12 + (c - 1200)  # 12 = SOS, 13 = EOS
            self.pitch_dict[c] = pitch_offset + original_idx
        else:
            # Microtonal cent: append after original vocab
            self.pitch_dict[c] = micro_id
            micro_id += 1
    # micro_id should now be 8487 + 1188 = 9675
else:
    self.pitch_dict = {i: i + self.sos_out_vocab_size + self.timeshift_vocab_size +
                       self.dur_vocab_size + self.octave_vocab_size
                       for i in range(self.pitch_class_vocab_size)}
```

**CRITICAL invariant:** `self.pitch_dict[0] == 8212`, `self.pitch_dict[100] == 8213`, ..., `self.pitch_dict[1100] == 8223`. All 12 western pitch IDs are EXACTLY the pretrained IDs.

Update instrument_dict and velocity_dict offsets — they must NOT change:
```python
# These must KEEP their original offsets (lines 63-64), NOT shift:
if self.microtonal:
    # instrument and velocity use ORIGINAL offsets (not affected by pitch expansion)
    instr_offset = pitch_offset + 14  # 8212 + 14 = 8226 (SAME as pretrained)
    self.instrument_dict = {i: instr_offset + i for i in range(self.instrument_vocab_size)}
    vel_offset = instr_offset + self.instrument_vocab_size  # 8226 + 131 = 8357
    self.velocity_dict = {i: vel_offset + i for i in range(self.velocity_vocab_size)}
else:
    # ... original sequential code (unchanged)
```

Update all decode dicts (lines 67-73) accordingly.

**Acceptance:** `pitch_dict[0] == 8212`, `pitch_dict[50] >= 8487`, `instrument_dict[0] == 8226`, `velocity_dict[0] == 8357`.

---

### B4. `convert_to_language_tokens` and `convert_from_language_tokens`

**Owner:** ___
**File:** `music_tokenizer.py`, lines 231-265
**Blocked by:** B3

**`convert_to_language_tokens` (line 231):** The pitch value `x[4]` is now cents (0-1199) from the label path. The non-contiguous `pitch_dict` from B3 handles the mapping automatically — **no code change needed** as long as `pitch_dict` is correct.

BUT verify: `x[4]` must be castable to int for dict lookup. Since labels store integer cents, this is fine.

**`convert_from_language_tokens` (line 246):** Pitch decode must handle both original AND appended IDs:

```python
# Line 258 — currently:
pitch = self.pitch_dict_decode[x[3].item()]
# This works automatically IF pitch_dict_decode is built correctly from B3.
# The decoded value will be cents (0-1199) for microtonal, or 0-11 for western.
```

**Post-decode conversion for inference (needed in D2):** After decoding, pitch is in cents. For the next transformer input, convert to fractional semitones: `pitch_frac = pitch_cents / 100.0`.

**Acceptance:** Round-trip test: `encode → language_tokens → decode` preserves pitch value for all cent values 0-1199.

---

### B5. Weight Transfer + Interpolation Initialization Code

**Owner:** ___
**Files:** New utility function (in CPT script or `model_utils.py`)
**Blocked by:** B1

```python
def expand_vocab_with_interpolation(pretrained_state_dict, new_config, tokenizer, original_decode_vocab=8487):
    """
    Expand decoder_embedding and lm_head from 8487 → 9675 rows.
    First 8487 rows: direct copy from pretrained.
    Rows 8487+: interpolated from flanking western semitone embeddings.

    Args:
        tokenizer: MusicTokenizer instance with microtonal pitch_dict (from B3)
                   Used to build reverse_pitch_dict: language_id → cent_value
    """
    # Build reverse lookup: language_id → cent value (only for new microtonal IDs)
    reverse_pitch_dict = {v: k for k, v in tokenizer.pitch_dict.items()}

    pitch_offset = 8212  # Western pitch rows start here (pitch classes 0-11)

    def _interpolate_new_rows(old_weight, new_vocab_size):
        new_weight = torch.zeros(new_vocab_size, old_weight.shape[1])
        new_weight[:original_decode_vocab] = old_weight  # Direct copy rows 0-8486
        for new_id in range(original_decode_vocab, new_vocab_size):
            cent = reverse_pitch_dict[new_id]  # e.g., 50 → between C and C#
            lower_pc = cent // 100              # 0-11
            upper_pc = (lower_pc + 1) % 12
            frac = (cent % 100) / 100.0
            new_weight[new_id] = (1 - frac) * old_weight[pitch_offset + lower_pc] \
                                       + frac * old_weight[pitch_offset + upper_pc]
        return new_weight

    new_state = {}
    for key, value in pretrained_state_dict.items():
        if 'decoder_embedding.weight' in key:
            new_state[key] = _interpolate_new_rows(value, new_config.decode_vocab_size)
        elif 'lm_head.weight' in key:
            new_state[key] = _interpolate_new_rows(value, new_config.decode_vocab_size)
        else:
            new_state[key] = value  # All other params: direct copy
    return new_state
```

**Only 2 parameters change shape:**
| Parameter | Original | New | Transfer |
|-----------|----------|-----|----------|
| `decoder_embedding.weight` | (8487, 1536) | (9675, 1536) | Rows 0-8486: copy. 8487+: interpolate |
| `lm_head.weight` | (8487, 1536) | (9675, 1536) | Same |
| Everything else | unchanged | unchanged | Direct copy |

**Initialization strategies (implement interpolation as default, copy+noise as ablation):**

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| **Interpolation (default)** | `E[c] = (1-frac) * E[lower_pc] + frac * E[upper_pc]` | Baseline. Musically principled — cent 50 is halfway between C and C#. |
| **Copy + noise (ablation)** | `E[c] = E[nearest_pc] + ε, ε ~ N(0, σ²I)` | Alternative. σ ≈ 0.01 * ‖E‖. Simpler but less musically informed. |

**Impact: LOW** — the model will overwrite initial values quickly during CPT. Both strategies start "close enough" to the pretrained manifold. Use interpolation as default, ablate copy+noise if time permits.

**Acceptance:** After transfer, `new_emb[8212] == old_emb[8212]` (C natural), `new_emb[8487]` ≈ midpoint of `old_emb[8212]` and `old_emb[8213]` (if cent=50).

---

## STREAM C — Training Script

### C1. CPT Training Script

**Owner:** ___
**Files:** `real_finetuning_microtonal.py` (NEW, copy from `real_finetuning_uncon_gen.py`)
**Blocked by:** B1, B5

Copy `real_finetuning_uncon_gen.py` and modify the model loading section (lines 140-157):

```python
# 1. Create model with MICROTONAL config (expanded vocab)
llama_config = LlamaConfig.from_pretrained("model_config_microtonal.json")
llama_config.use_cache = use_cache
model = LlamaForCausalLM(llama_config)

# 2. Load pretrained checkpoint
model_checkpoint = torch.load(train_config.trained_checkpoint_path)
checkpoint = model_checkpoint['model_state_dict']
new_state_dict = {}
for k, v in checkpoint.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

# 3. Expand vocab with interpolation (from B5)
expanded_state_dict = expand_vocab_with_interpolation(new_state_dict, llama_config, tokenizer)

# 4. Load expanded weights
missing, unexpected = model.load_state_dict(expanded_state_dict, strict=False)
print(f"Missing: {missing}, Unexpected: {unexpected}")
# Expected: no missing keys, no unexpected keys (all shapes match now)
```

Also update tokenizer init (line 165) to pass `microtonal=True`:
```python
tokenizer = MusicTokenizer(..., microtonal=True, pitchbend_sensitivity=2.0)
```

**Acceptance:** Model loads without errors, `model.lm_head.weight.shape == (9675, 1536)`.

---

### C1b. LoRA Configuration + Training Strategy

**Owner:** ___
**File:** `src/llama_recipes/configs/peft.py`, `real_finetuning_microtonal.py`
**Blocked by:** C1

The existing codebase uses LoRA across branches with consistent settings. For microtonal CPT:

**LoRA config** (based on `conditional_gen_commu` branch which also trains decoder):
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # transformer attention only
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

**What gets LoRA vs full training:**
| Component | Training Mode | Why |
|-----------|--------------|-----|
| Transformer attention (q/k/v/o_proj) | **LoRA** (r=8, α=32) | Preserve pretrained knowledge, few trainable params |
| Transformer FFN, norms, FME | **Frozen** | These don't need adaptation for microtonal pitch |
| `decoder_embedding` (9675, 1536) | **Full training** | Must learn new microtonal pitch rows (8487+) |
| `lm_head` (1536, 9675) | **Full training** | Must output logits for new microtonal tokens |
| GRU internal layers | **Full training** | Must learn to decode new pitch vocabulary |
| `summary_projection` | **Frozen or low LR** | Maps transformer → GRU, architecture unchanged |

**Implementation:** After PEFT wrapping, ensure decoder params are trainable:
```python
model = get_peft_model(model, lora_config)
# PEFT may freeze non-LoRA params. Explicitly unfreeze decoder:
for name, param in model.named_parameters():
    if any(k in name for k in ['decoder_embedding', 'lm_head', 'decoder.', 'fc_out']):
        param.requires_grad = True
```

**Training hyperparameters:**
```python
# Optimizer
optimizer = torch.optim.AdamW(param_groups, lr=3e-5, weight_decay=0.01)

# Schedule: cosine decay with warmup
warmup_steps = int(0.05 * total_steps)  # 5% warmup
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Mixed precision
from torch.cuda.amp import autocast  # already used in train_con_gen()
```

**Warmup/freezing schedule:**
- **Steps 0 → T_freeze (10-20% of total):** Freeze western anchor embeddings `E[8212:8224]` in
  both `decoder_embedding` and `lm_head`. Only new rows (8487+) and LoRA adapters train.
  This lets microtonal rows "catch up" from interpolation init before unfreezing.
- **Steps T_freeze → end:** Unfreeze all, rely on L_anchor (C3) for stability.

```python
# Freezing logic (in training loop):
if step < T_freeze:
    with torch.no_grad():
        model.decoder_embedding.weight.grad[8212:8224] = 0
        model.lm_head.weight.grad[8212:8224] = 0
```

**Acceptance:** LoRA wrapping succeeds. Only ~2.5M LoRA params + decoder params are trainable (vs 309M total). `model.print_trainable_parameters()` confirms.

---

### C2. Float Pitch Conversion in Training Loop

**Owner:** ___
**File:** `train_utils.py`, after line 708 (batch moved to device), before line 709 (autocast)
**Blocked by:** B2

Insert between lines 708 and 709:

```python
# --- Microtonal: convert input_ids pitch from integer cents to fractional semitones ---
if getattr(train_config, 'microtonal', False):
    batch['input_ids'] = batch['input_ids'].float()
    batch['input_ids'][:, :, 3] = batch['input_ids'][:, :, 3] / 100.0  # cents → fractional semitones
    # onset/dur/octave/velocity: FME handles float ints (0.0, 1.0, ...) identically to ints
    # instrument: .long() cast is inside embed_tokens (B2 fix)
# labels stay as integers during TRAINING (teacher forcing).
# During INFERENCE, GRU output → compound tokens → next transformer input (coupled loop).
# Pitch resolution must be consistent across both paths.
```

**Why this works:**
- FME (line 1328): `inp[..., None] * self.angles` — pure float multiplication. `FME(3.0)` == `FME(3)`.
- RoPE (line 120): `position_ids[:, None, :].float()` — explicit float cast already present.
- `nn.Embedding` (instrument only): gets `.long()` cast from B2.
- Labels are NOT modified — they are integer language token IDs for `CrossEntropyLoss`.

**Acceptance:** `model(**batch)` runs without dtype errors. Loss is finite.

---

### C3. L_anchor Regularization

**Owner:** ___
**File:** `train_utils.py` (in `train_con_gen`) or in `modeling_llama.py` (in `forward`)
**Blocked by:** C1

Prevent the 12 pretrained western pitch embeddings from drifting during CPT:

```python
# Store frozen copy of pretrained western embeddings at init time:
western_pitch_ids = list(range(8212, 8224))  # IDs for pitch classes 0-11
frozen_decoder_emb = model.decoder_embedding.weight[western_pitch_ids].detach().clone()
frozen_lm_head = model.lm_head.weight[western_pitch_ids].detach().clone()

# In training loop, after loss = model(**batch).loss:
lambda_anchor = 0.5  # tune in [0.1, 1.0]
current_decoder_emb = model.decoder_embedding.weight[western_pitch_ids]
current_lm_head = model.lm_head.weight[western_pitch_ids]
L_anchor = lambda_anchor * (
    torch.mean((current_decoder_emb - frozen_decoder_emb) ** 2) +
    torch.mean((current_lm_head - frozen_lm_head) ** 2)
)
loss = loss + L_anchor
```

**Acceptance:** Western pitch embeddings move <5% from pretrained values after 1 epoch.

---

### C3b. L_smooth Regularization

**Owner:** ___
**File:** Same location as C3 (in training loop, after L_anchor)
**Blocked by:** C3

Encourage nearby cents to have nearby embeddings (addresses empty bins problem — ~90% of 1200 cent bins or ~50% of 120 bins are empty in SymbTr):

```python
# After L_anchor computation:
lambda_smooth = 0.05  # tune in [0.01, 0.1]

# Get all active microtonal pitch IDs (8487+) sorted by cent value
micro_ids = sorted(tokenizer.pitch_dict.items(), key=lambda x: x[0])  # (cent, lang_id)
micro_ids = [(c, lid) for c, lid in micro_ids if lid >= 8487]

L_smooth = 0
for i in range(len(micro_ids) - 1):
    c1, id1 = micro_ids[i]
    c2, id2 = micro_ids[i + 1]
    if c2 - c1 <= 10:  # only adjacent bins (within 10 cents)
        L_smooth += torch.sum((model.decoder_embedding.weight[id2] - model.decoder_embedding.weight[id1]) ** 2)
        L_smooth += torch.sum((model.lm_head.weight[id2] - model.lm_head.weight[id1]) ** 2)
L_smooth = lambda_smooth * L_smooth / max(len(micro_ids) - 1, 1)

loss = loss + L_anchor + L_smooth
```

**Impact: HIGH** — propagates gradient signal to untrained bins. During inference, ghost bins cause **autoregressive error cascades**: GRU samples untrained token → converts to untrained fractional semitone → fed back to transformer → bad hidden state → next GRU prediction also degraded → cascade. L_smooth ensures even untrained bins have reasonable embeddings, preventing this chain reaction. (Note: during training, teacher forcing breaks the loop, so ghost bins are harmless — but inference is what matters for generation quality.)

**Acceptance:** Embedding distances between adjacent cent bins are correlated with cent distance.

---

### C4. Per-Parameter Learning Rate Groups

**Owner:** ___
**File:** `real_finetuning_microtonal.py` (optimizer setup) or `train_utils.py`
**Blocked by:** C1

Replace single optimizer with parameter groups:

```python
# After model is loaded and PEFT is applied:
lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
decoder_existing = [model.decoder_embedding.weight, model.lm_head.weight]  # rows 0-8486 (handled via single param)
gru_params = [p for n, p in model.named_parameters() if 'decoder' in n and 'embedding' not in n and 'lm_head' not in n]

param_groups = [
    {"params": lora_params,      "lr": 3e-5},   # LoRA adapters
    {"params": decoder_existing, "lr": 3e-5},   # decoder_embedding + lm_head (full param, includes new rows)
    {"params": gru_params,       "lr": 3e-5},   # GRU internal layers
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=train_config.weight_decay)
```

**Note:** Since `decoder_embedding.weight` is a single tensor (can't split rows into param groups), we rely on L_anchor (C3) + warmup freezing (C1b) to keep pretrained rows stable. New rows learn fast because they start from interpolation (far from optimal) while pretrained rows start near-optimal.

**Optional: Gradient scaling hook** for row-level LR differentiation (review plan recommends 10× higher LR for new rows):
```python
# Register hook to scale gradients for new rows (8487+) by 3× relative to pretrained:
def scale_new_rows(grad):
    scaled = grad.clone()
    scaled[8487:] *= 3.0  # new rows get effective LR = 3e-5 * 3 = 9e-5
    return scaled
model.decoder_embedding.weight.register_hook(scale_new_rows)
model.lm_head.weight.register_hook(scale_new_rows)
```

**Acceptance:** Optimizer has distinct param groups with correct LRs.

---

### C5. Data Mixing Logic

**Owner:** ___
**File:** `real_finetuning_microtonal.py` (dataset loading section, ~line 331)
**Blocked by:** A4 (western data)

Mix microtonal + western data during CPT to prevent catastrophic forgetting.

#### C5 vs MergeDataset — NOT the same thing

| | `MergeDataset` (existing) | C5 Data Mixing (new) |
|---|---|---|
| **What** | PyTorch Dataset class | Training-time sampling strategy |
| **How** | One directory, one CSV, loads all files equally | Two separate datasets + `WeightedRandomSampler` |
| **Ratio control** | None (uniform) | Tunable `alpha` parameter |
| **Use case** | Pre-merged homogeneous data | Continual pretraining with forgetting control |

**Use C5, NOT MergeDataset.** MergeDataset provides no sampling ratio control, which is essential
for balancing new-domain learning against catastrophic forgetting.

#### How it works

**Prerequisite: Uniform cents format.** Both datasets must store pitch as integer cents.
A4 handles this by re-preprocessing western MIDI with `--microtonal` (no pitchbend → pitch×100).
Then C2's `/100.0` in embed_tokens works identically for both:
- Western: `200 / 100.0 = 2.0` (D) — correct fractional semitone
- Microtonal: `250 / 100.0 = 2.5` (D+50 cents) — correct fractional semitone

Western pitch cents (0, 100, 200, ..., 1100) map to original pretrained IDs (8212-8223) via B3's
pitch_dict. Microtonal cents map to appended IDs (8487+). A single tokenizer handles both.

#### Implementation — DONE (in `real_finetuning_microtonal.py`, lines 341-408)

**How it works (packing strategy only):**

1. Load microtonal data via `get_preprocessed_dataset` → `SymbTrDataset` (2700 train pieces)
2. Load western replay via `LakhDataset` (same `data_dir/processed/*.npy` + CSV format)
3. Pack each dataset **separately** into 1024-token chunks (homogeneous chunks)
4. Combine packed datasets with `torch.utils.data.ConcatDataset`
5. Build per-chunk weights: `alpha/n_micro` for microtonal, `(1-alpha)/n_western` for western
6. Use `WeightedRandomSampler(replacement=True)` for exact ratio control per epoch

**Why `LakhDataset` for western data (not `MergeDataset` or `SymbTrDataset`):**
- `LakhDataset` uses `data_dir/processed/filename.npy` — matches preprocessed directory layout
- `MergeDataset` uses `data_dir/filename.npy` (no `processed/` subdir) — would fail on our data
- `SymbTrDataset` would also work (handles missing conditioning columns), but `LakhDataset` is
  semantically correct for western data and avoids unnecessary conditioning overhead

**Why pack separately then combine (not combine then pack):**
- Packing combined data mixes microtonal and western notes within chunks, losing ratio control
- Separate packing produces homogeneous chunks (each is purely micro or purely western)
- `WeightedRandomSampler` then controls how many of each type are sampled per epoch

**FSDP/DDP:** Falls back to `DistributedSampler` (uniform sampling over combined dataset).
Exact ratio control requires a custom distributed weighted sampler — not needed for initial experiments.

**Single-GPU packing (no FSDP/DDP):** `get_dataloader_kwargs` returns no sampler,
so `WeightedRandomSampler` is added cleanly without conflict.

#### CLI args (added to `training.py`)

```bash
--western_data_dir ~/projects/microtok/data/maestro_processed \
--western_csv_file ~/projects/microtok/data/maestro_processed/train_test_split.csv \
--mixing_alpha 0.8
```

**Backward compatible:** Omitting `--western_data_dir` runs on microtonal data alone (original behavior).

**Acceptance:** Mixed dataloader produces batches with both western and microtonal samples. Loss converges.

---

### C6. Per-Step GRU Accuracy Logging

**Owner:** ___
**File:** `modeling_llama.py` (in forward, after line 1829) or `train_utils.py`
**Blocked by:** C1

After the GRU loss computation (line 1829), add per-step accuracy:

```python
# After: loss = self.loss_func(generation_logits, shift_labels_x_y)
# generation_logits: (N*6, decode_vocab_size), shift_labels_x_y: (N*6,)
if self.training and hasattr(self, '_log_gru_acc') and self._log_gru_acc:
    with torch.no_grad():
        preds = generation_logits.argmax(dim=-1)  # (N*6,)
        correct = (preds == shift_labels_x_y)
        # Reshape to (N, 6) to get per-step accuracy
        N = correct.shape[0] // 6
        correct_2d = correct[:N*6].view(N, 6)
        attr_names = ['timeshift', 'duration', 'octave', 'pitch', 'instrument', 'velocity']
        gru_acc = {}
        for step, name in enumerate(attr_names):
            gru_acc[f'gru_acc/{name}'] = correct_2d[:, step].float().mean().item()

        # Split pitch accuracy: western vs microtonal
        pitch_labels = shift_labels_x_y.view(-1, 6)[:, 3]  # pitch step
        is_western = (pitch_labels >= 8212) & (pitch_labels <= 8225)
        is_micro = pitch_labels >= 8487
        if is_western.any():
            gru_acc['gru_acc/pitch_western'] = correct_2d[is_western.view(N), 3].float().mean().item()
        if is_micro.any():
            gru_acc['gru_acc/pitch_micro'] = correct_2d[is_micro.view(N), 3].float().mean().item()
```

Log to wandb if available.

**Acceptance:** Training logs show per-attribute GRU accuracy, split by western/microtonal pitch.

---

## STREAM D — Inference + Evaluation

### D1. Inference: Float Token Buffer

**Owner:** ___
**File:** `recipes/inference/custom_music_generation/generation.py`, line 148
**Blocked by:** B2

```python
# BEFORE (line 148):
pad_tensor = torch.tensor(pad_id, dtype=torch.long, device="cuda").unsqueeze(0).unsqueeze(0)
# AFTER:
pad_tensor = torch.tensor(pad_id, dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0)
```

Also line 152:
```python
# BEFORE:
t_tensor = torch.tensor(t, dtype=torch.long, device="cuda")
# AFTER:
t_tensor = torch.tensor(t, dtype=torch.float32, device="cuda")
```

**Acceptance:** `tokens` buffer holds float values. Fractional pitch (e.g., 3.5) is not truncated.

---

### D2. `convert_from_language_tokens` → Integer Cents

**Owner:** ___
**File:** `music_tokenizer.py`
**Blocked by:** B3, B4

Decode language token IDs back to compound tokens. Pitch is returned as integer cents
(0-1199) — NOT divided by 100.0. The single /100.0 conversion to fractional semitones
lives in `embed_tokens` (modeling_llama.py line ~1437), which handles both training
(data loader cents) and inference (this path). This avoids the double-division bug
that would occur if both convert_from and embed_tokens each divided by 100.0.

`compound_to_midi` also converts cents/100.0 internally when `microtonal=True`.

**Acceptance:** EOS detection works (`1201 == 1201`). embed_tokens converts cents to
fractional semitones once. No double-division in the coupled inference loop.

---

### D3. `compound_to_midi` with Pitchbend

**Owner:** ___
**File:** `music_tokenizer.py`, lines 437-494
**Blocked by:** A1 (helper functions)

Write fresh (uses the A1 helper functions, NOT ported from `feat/mtok_vocab`):

```python
@staticmethod
def compound_to_midi(tokens, TIME_RESOLUTION=100, debug=False, microtonal=False, pitchbend_sensitivity=2.0):
    mid = mido.MidiFile()
    mid.ticks_per_beat = TIME_RESOLUTION // 2

    time_index = defaultdict(list)
    for _, row in enumerate(tokens):
        time_in_ticks, duration, octave, pitch_value, instrument, velocity = row

        if microtonal and isinstance(pitch_value, float) and pitch_value != int(pitch_value):
            # Fractional semitone → MIDI note + pitchbend
            canonical = octave * 12 + pitch_value
            midi_note, pitchbend = canonical_pitch_to_midi_and_pitchbend(canonical, pitchbend_sensitivity)
        else:
            midi_note = octave_pitch_class_to_pitch(octave, int(pitch_value))
            pitchbend = 0

        time_index[(time_in_ticks, 0)].append((midi_note, instrument, velocity, pitchbend))
        time_index[(time_in_ticks + duration, 1)].append((midi_note, instrument, velocity, 0))

    # ... rest of MIDI construction, adding pitchwheel message before note_on when pitchbend != 0
```

**Acceptance:** Generated MIDI files play with correct microtonal pitches in a DAW.

---

### D4. Evaluation Script — Core Metrics

**Owner:** ___
**Files:** New `evaluate_microtonal.py`
**Blocked by:** C1 (needs trained model)

Implement the following metrics on held-out SymbTr test split:

**1. Next-token perplexity** (primary metric):
```python
# Run model on test set, compute cross-entropy loss, exponentiate
test_ppl = exp(avg_cross_entropy_loss_on_test_set)
```

**2. Per-GRU-step accuracy** (from C6 logging, but also compute on test set):
```
Step 0: sos_out accuracy (should be ~100%)
Step 1: timeshift accuracy
Step 2: duration accuracy
Step 3: octave accuracy
Step 4: pitch accuracy  ← KEY METRIC
Step 5: instrument accuracy (should be ~100%, all piano)
Step 6: velocity accuracy
```

Split pitch accuracy:
- **Western pitch accuracy**: events where ground truth pitch is a western semitone (0, 100, 200, ..., 1100 cents)
- **Microtonal pitch accuracy**: events with non-western cent values

**3. Ghost bin rate** (generated music analysis):
```python
# Generate N=200 pieces autoregressively
# For each generated pitch token, check if that cent bin had ZERO training examples
ghost_rate = count(generated_pitch in empty_bins) / count(all_generated_pitches)
```

**4. Generated pitch distribution vs corpus:**
- Per pitch-class histogram of generated cent offsets vs SymbTr training distribution
- **Wasserstein distance** (earth mover's distance) per pitch class

**5. Interval distribution:**
- Compare consecutive pitch differences (in cents) in generated vs corpus music
- Turkish makam has characteristic interval patterns (Holdrian comma multiples)

**6. Tokenization fidelity** (sanity check):
```python
# Round-trip: SymbTr MIDI → tokenize → detokenize → compare
# Report RMSE in cents between original and reconstructed pitches
# For 1-cent resolution: RMSE ≤ 0.5 cent (trivial)
# Also report fraction of saturated bends (events near ±200 cents with sensitivity=2.0)
```

**Acceptance:** Evaluation script runs end-to-end, produces metrics table.

---

### D4b. Ablation Table Design

**Owner:** ___
**Blocked by:** D4, multiple training runs

Run ablations across these axes once baseline CPT works:

| Configuration | SymbTr PPL | Pitch Acc (micro) | Pitch Acc (western) | Western PPL Δ | Ghost Bin % | TF-AR Gap |
|---|---|---|---|---|---|---|
| Moonbeam pretrained (no CPT) | — | — | baseline | 0 | — | — |
| CPT, no regularization | ? | ? | ? | ? | ? | ? |
| CPT + L_anchor only | ? | ? | ? | ? | ? | ? |
| CPT + L_anchor + data mixing | ? | ? | ? | ? | ? | ? |
| CPT + L_anchor + L_smooth + data mixing | ? | ? | ? | ? | ? | ? |
| CPT + L_anchor + L_smooth + data mixing + 53-TET aug | ? | ? | ? | ? | ? | ? |
| CPT, interpolation init (default) | ? | ? | ? | ? | ? | ? |
| CPT, copy+noise init | ? | ? | ? | ? | ? | ? |
| CPT, 1-cent resolution (default) | ? | ? | ? | ? | ? | ? |
| CPT, 10-cent resolution | ? | ? | ? | ? | ? | ? |

**TF-AR Gap** = teacher-forced pitch accuracy minus autoregressive pitch accuracy (see D4c). Measures error cascade severity from the coupled inference loop.

**Key ablation axes:** (1) regularization, (2) data augmentation, (3) initialization, (4) resolution.
L_anchor + data mixing should be in all non-baseline runs.

---

### D4c. Autoregressive vs Teacher-Forced Accuracy Gap

**Owner:** ___
**Files:** In `evaluate_microtonal.py`
**Blocked by:** D4

Because Moonbeam's inference loop is coupled (GRU output feeds back to the transformer — see architecture diagram), errors compound during autoregressive generation but NOT during teacher-forced evaluation. The gap between these two directly measures cascade severity:

```python
# 1. Teacher-forced pitch accuracy (standard eval — no coupling):
#    Ground-truth input_ids go to transformer; measure GRU pitch prediction accuracy
tf_pitch_acc = eval_teacher_forced(model, test_set)

# 2. Autoregressive pitch accuracy (real generation — full coupling):
#    Generate from prompt prefix, compare generated pitches to ground-truth continuation
ar_pitch_acc = eval_autoregressive(model, test_set, prompt_len=16)

# 3. Cascade gap:
cascade_gap = tf_pitch_acc - ar_pitch_acc

# 4. Accuracy vs generation position (does it degrade over time?):
for pos in [1, 5, 10, 20, 50]:
    acc_at_pos = eval_autoregressive_at_position(model, test_set, gen_position=pos)
    # Plot: x=position, y=accuracy — slope reveals cascade rate
```

**Why this matters:**
- Small gap → GRU predictions are reliable, ghost bins don't cascade
- Large gap → need stronger L_smooth / coarser resolution / more augmentation
- Accuracy-vs-position plot shows whether errors compound or stabilize — a key insight for Moonbeam's coupled architecture
- **Paper contribution**: this metric is specific to Moonbeam's transformer-GRU loop and reveals something standard perplexity doesn't

**Acceptance:** TF-AR gap computed for all ablation runs. Accuracy-vs-position plot generated.

---

### D5. Evaluation — Western Forgetting

**Owner:** ___
**Files:** In `evaluate_microtonal.py`, uses code from `origin/finetune_player_classification` branch
**Blocked by:** D4

**1. Western PPL comparison:**
```python
# Evaluate BOTH pretrained Moonbeam AND CPT model on same western test set (Lakh)
delta_ppl = cpt_western_ppl - pretrained_western_ppl
# delta_ppl should be small (< 5% relative increase)
# Per-attribute PPL breakdown (especially pitch — should be stable for p ∈ {0,...,11})
```

**2. Downstream task benchmarks** (from Moonbeam paper Table 2):

The `origin/finetune_player_classification` branch has the exact finetuning code:
- **Script:** `real_finetuning_player_classification.py` — uses `LlamaForSequenceClassification`, loads
  pretrained with `strict=False`, applies LoRA (r=8, α=32, targets=q/k/v/o_proj)
- **Training fn:** `train_player_classification()` in `train_utils.py` — evaluates with F1 scoring
- **Config:** `player_classification_config.json` — num_labels=30, adds `<classification>` token (-4)
- **Dataset:** `player_classification_dataset.py` — loads CSV with player metadata

Run these benchmarks on the CPT model (replace pretrained checkpoint path):

| Task | Dataset | Branch/Script | Moonbeam (S) baseline | Target |
|------|---------|--------------|----------------------|--------|
| Player classification | PiJAMA30 | `finetune_player_classification` | Acc=0.649, F1=0.596 | Within 2% |
| Player classification | Pianist8 | same | Acc=0.811, F1=0.804 | Within 2% |
| Emotion classification | Emopia | same (adapt dataset) | Acc=0.636, F1=0.623 | Within 2% |
| Composer classification | GPM30 | same (adapt dataset) | Acc=0.541, F1=0.470 | Within 2% |

**Steps:**
1. Take the CPT checkpoint (after microtonal training)
2. Finetune for classification using the SAME LoRA config + script from the classification branch
3. Compare Acc/F1 to Moonbeam paper Table 2 baselines
4. Report delta — if L_anchor + data mixing work, deltas should be <2%

**3. Pitch-class confusion matrix:**
```python
# On western validation data (Lakh test set):
# Run CPT model inference, collect (predicted_pitch, true_pitch) pairs
# Build 12×12 confusion matrix for pitch classes 0-11
# Compare to pretrained Moonbeam's confusion matrix
# If L_anchor works → nearly identical patterns
# Visualize as heatmap — good paper figure
```

**Acceptance:** All forgetting metrics computed. Western PPL increase <5%. Downstream task accuracy drops <2%.

---

### D6. Evaluation — Embedding Space Visualization

**Owner:** ___
**Files:** Notebook or script
**Blocked by:** D4 (needs trained model)

**t-SNE / UMAP of `decoder_embedding` pitch rows:**

```python
# Extract pitch embedding rows
pretrained_embs = pretrained_model.decoder_embedding.weight[8212:8224]  # 12 western
cpt_western = cpt_model.decoder_embedding.weight[8212:8224]  # 12 western after CPT
cpt_micro = cpt_model.decoder_embedding.weight[8487:9675]  # 1188 microtonal

# Concatenate, run t-SNE/UMAP, plot
# Color by pitch class (C=red, C#=orange, D=yellow, ...)
# Annotate makam scale degrees (segah, hicaz, etc.)
```

**Expected:** (a) Western anchors haven't drifted far. (b) Microtonal variants cluster near parent western pitch. (c) Nearby cents have nearby embeddings.

**This makes a compelling paper figure.**

**Acceptance:** Plot generated with clear pitch-class clustering.

---

### D7. Evaluation — Makam-Specific

**Owner:** ___
**Files:** In `evaluate_microtonal.py`
**Blocked by:** D4

**Scale degree adherence:**
```python
# SymbTr metadata includes makam labels in filenames
# e.g., "hicaz--sazsemaisi--aksaksemai----tanburi_cemil_bey"
# For each generated piece, identify closest makam scale
# Measure % of generated pitches on valid scale degrees (within ±10 cents tolerance)
```

**Acceptance:** Scale adherence metric computed per makam.

---

### D8. Evaluation — Subjective Listening Test

**Owner:** ___
**Files:** Survey design + generated MIDI/audio files
**Blocked by:** D4 (needs trained model + generated pieces)

Conduct a listening test with expert evaluators to assess generated makam music quality:

**Design (following Moonbeam paper's protocol):**
1. Generate 20-30 pieces using the CPT model (unconditional, seeded with SymbTr prompts)
2. Include comparison samples: (a) real SymbTr excerpts, (b) Moonbeam pretrained (western-only) on same prompts, (c) CPT model output
3. Recruit 5-10 evaluators with Turkish makam expertise (musicologists, performers, or MIR researchers)
4. Each evaluator rates on a 1-5 Likert scale across:
   - **Pitch accuracy**: Are the microtonal intervals correct for the makam?
   - **Melodic coherence**: Does the melody follow a natural seyir (melodic progression)?
   - **Scale adherence**: Do pitches stay within the expected makam scale degrees?
   - **Overall musicality**: Does it sound like plausible makam music?
5. Blind, randomized presentation (evaluators don't know which system produced each sample)

**Analysis:**
- Mean Opinion Score (MOS) per criterion per system
- Wilcoxon signed-rank test for pairwise comparisons (following Moonbeam paper)
- Inter-rater agreement (Krippendorff's alpha or Fleiss' kappa)

**Acceptance:** Listening test completed, MOS scores and statistical tests reported. CPT model should score significantly higher than pretrained Moonbeam on pitch accuracy and scale adherence.

---

## FUTURE WORK (NOT in this sprint)

| Item | Description | Why Later |
|------|-------------|-----------|
| **L_center regularization** | Keep microtonal variants close to western anchor | LOW impact — likely skip or decay fast |
| **GMM pitch-zone tokenizer** | Data-driven pitch quantization (fewer tokens) | Alternative to uniform grid — separate experiment |
| **10-cent resolution variant** | Coarser grid (120 bins vs 1200) | Ablation — run after 1-cent baseline |
| **Copy+noise initialization** | Alternative to interpolation init | Ablation — LOW impact |
| **Microtonal-axis initialization** | Inductive bias in embedding direction | LOW impact, skip unless ablations show need |
| **Higher LoRA rank for pitch heads** | Group 3 attention heads get more capacity | Optimization — try after baseline |
| **Makam-conditional generation (Phase 2)** | Makam label as metadata condition | Depends on Phase 1 complete |
| **Makam scale as temporal condition (Phase 3)** | Scale degrees between `<soc>`/`<eoc>` | Separate paper territory |
| **Usul as temporal condition** | Rhythmic cycle encoding | Separate paper territory |

---

## Suggested Team Assignment

| Stream | Suggested Owner | Rationale |
|--------|----------------|-----------|
| **A (Data Pipeline)** | ___ | Independent, can start day 1. Heavy on preprocessing scripts. |
| **B (Model + Tokenizer)** | ___ | Requires deep understanding of Moonbeam architecture. |
| **C (Training Script)** | ___ | Integration work — benefits from understanding both A and B. |
| **D (Inference + Eval)** | ___ | Can start D1-D3 early (parallel with C), D4-D7 after training. |

**Optimal parallelism:**
- **Week 1:** A1-A3 and B1-B5 in parallel (all prereqs done)
- **Week 2:** A4 + C1-C5 (integration), D1-D3 in parallel
- **Week 3:** C6 + first training runs, D4-D7 evaluation
- **Week 4:** D8 listening test (needs generated samples from Week 3)

---

## Quick Reference: File Changes Summary

| File | Changes | Stream |
|------|---------|--------|
| `model_config_microtonal.json` | NEW file | B1 |
| `music_tokenizer.py` | +4 helper functions, modified `__init__`, `midi_to_compound`, `pitch_dict`, `convert_from_language_tokens`, `compound_to_midi` | A1, B3, B4, D2, D3 |
| `modeling_llama.py` | 2 bugfixes (lines 940-941, 1437) | B2 |
| `train_utils.py` | Float conversion (after line 708), optional L_anchor | C2, C3 |
| `real_finetuning_microtonal.py` | NEW (copy of `real_finetuning_uncon_gen.py` + weight transfer + mixing) | C1, C4, C5 |
| `generation.py` | Float buffer (line 148, 152) | D1 |
| `data_preprocess.py` | Point at SymbTr, microtonal tokenizer | A2 |
| `augment_53tet.py` | NEW script for transposition augmentation | A3 |
| `evaluate_microtonal.py` | NEW script for all evaluation metrics | D4-D8 |
| `lakh_dataset.py` | Minimal or no changes (data format is .npy) | — |
| `concatenator.py` | No changes (stores Python lists, no dtype forcing) | — |
