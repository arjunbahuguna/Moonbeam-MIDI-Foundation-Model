# Microtonal Continual Pretraining for Moonbeam — Full Analysis & TODO

> **Status (2026-03-05):** All model/tokenizer/training-loop code is IMPLEMENTED on `microtonal_cpt` branch
> (streams B, C1-C4, C6, D1-D3). **57 unit tests passing** (`tests/test_microtonal.py`).
> Remaining: data pipeline (A2-A4), data mixing (C5), evaluation (D4-D8).
> See TEAM_TODO.md for detailed per-task status. Critical path: A2 (SymbTr preprocessing) → C5 (mixing) → first training run → evaluation.
>
> **NOTE on Section 6.5.2 Change 2 below:** The original plan said `convert_from_language_tokens()` should
> divide pitch by 100.0. This was CHANGED during implementation — `convert_from` now returns **integer cents**
> (no division), and the single authoritative `/100.0` conversion happens in `embed_tokens` (modeling_llama.py
> line 1437). This handles both training and inference paths uniformly. The prose in 6.5.2 is kept for
> historical context but the ACTUAL implementation differs — see TEAM_TODO.md task D2.

## 1. Current State of the Codebase

### Branches
- **`main`**: Original Moonbeam — western 12TET only, has `real_finetuning_uncon_gen.py` (**THE script** we will base continual pretraining on)
- **`feat/mtok_vocab`**: Microtonal tokenizer (from `mtok_uniform` + formatting + config). Based on `finetune_player_classification` (downstream task), NOT on `main`.

### Training Scripts (on `main` branch)
| Script | Model Class | Training Function | Concatenator | Purpose |
|---|---|---|---|---|
| **`real_finetuning_uncon_gen.py`** | `LlamaForCausalLM` | `train_con_gen()` | `ConcatDataset_hybrid_padding_concatenating` | **Unconditional generation CPT — USE THIS** |
| `finetuning.py` | `LlamaForCausalLM` | `train()` | `ConcatDataset` (mmap-based) | Original pretraining from scratch |
| `real_finetuning_player_classification.py` | `LlamaForSequenceClassification` | `train_player_classification()` | `ConcatDataset_dummy_padding` | Downstream classification |
| `overfitting_test.py` | `LlamaForCausalLM` | `train_overfit()` | `ConcatDataset` + `ExtendedDataset` | Debug/validation |

**Key differences between `real_finetuning_uncon_gen.py` and `finetuning.py`:**
- `real_finetuning_uncon_gen.py` **loads a pretrained checkpoint** with `strict=False` (handles missing/unexpected keys) — exactly what we need for vocab expansion
- `finetuning.py` creates a **random-init model** from config (intended for training from scratch)
- `real_finetuning_uncon_gen.py` uses `train_con_gen()` which saves PEFT checkpoints and validates at intervals
- `real_finetuning_uncon_gen.py` uses `ConcatDataset_hybrid_padding_concatenating` which stores data **in-memory** (not mmap) — stores Python lists, NOT int32 numpy arrays
- Both support PEFT/LoRA with identical config

The local `model_config.json` has been partially modified from the original pretrained config:
- `onset_vocab_size=1026`, `dur_vocab_size=1026` (original pretrained: **4099**)
- `pitch_class_vocab_size=1202` (original pretrained: **14**)
- `decode_vocab_size=2341` — this assumes pitch=14, so it's inconsistent with pitch_class_vocab_size=1202
- `pitch_embedding.base=20`, `rope_theta_pitch=20` (same as original — good)

The **original pretrained config** (verified via `origin/main` git branch):
- `onset/dur=4099`, `pitch=14`, `decode_vocab_size=8487`
- Language ID layout: `[sos_out:0 | timeshift:1-4099 | duration:4100-8198 | octave:8199-8211 | pitch:8212-8225 | instrument:8226-8356 | velocity:8357-8486]`

---

## 2. Problems with Current Microtonal Tokenizer (`feat/mtok_vocab`)

### 2.1 CRITICAL: FME Input Embeddings Are Incompatible with Pretrained Weights

**Original Moonbeam (pretrained):**
- `pitch_class` input to FME: integer values `0, 1, 2, ..., 11` (semitones)
- FME base for pitch: `20`
- FME computes: `sin/cos(pitch_value * angle_rates)` where `angle_rates = 1/20^(2i/dim)`
- The model learned over 18B tokens what `FME(0)`, `FME(1)`, ..., `FME(11)` mean musically

**Current microtonal config:**
- `pitch_class` input to FME: integer values `0, 1, 2, ..., 1199` (cents)
- FME base for pitch: `1800` (changed!)
- `FME(100, base=1800)` ≠ `FME(1, base=20)` — verified numerically: `sin(1*1.0) = 0.84` vs `sin(100*1.0) = -0.51`
- **ALL pretrained pitch knowledge is destroyed** because both the input values and the base changed

### 2.2 CRITICAL: MRA Relative Attention Is Incompatible

**How MRA works:** `position_ids = input_ids` (the raw compound tokens ARE the position IDs). Attention heads are split into 6 groups, each rotated by attribute-specific RoPE:

| Head Group | Attribute | RoPE Base (original) | RoPE Base (microtonal) |
|---|---|---|---|
| 0, 4 | onset | 199999 | 199999 (same) |
| 1 | duration | 1031 | 1031 (same) |
| 2 | octave | 19 | 19 (same) |
| **3** | **pitch** | **20** | **1800** (CHANGED) |
| 5 | velocity | 131 | 131 (same) |

For pitch head group 3:
- Original: C->D relative distance = `|2-0|` = 2, with RoPE base 20
- Microtonal: C->D relative distance = `|200-0|` = 200, with RoPE base 1800
- These produce DIFFERENT rotation angles across dimensions — there is NO single base that can compensate for a 100× scale change in all RoPE dimensions simultaneously
- **The model's learned sense of "how far apart are these pitches?" is destroyed**

### 2.3 CRITICAL: GRU Decoder Language IDs Are Completely Reshuffled

The GRU decoder uses a single flat vocabulary. All attributes share one `lm_head = Linear(1536, decode_vocab_size)` and one `decoder_embedding = Embedding(decode_vocab_size, 1536)`.

**Original layout (decode_vocab_size=8487):**
```
[sos_out:0 | timeshift:1-4099 | duration:4100-8198 | octave:8199-8211 | pitch:8212-8225 | instrument:8226-8356 | velocity:8357-8486]
```

**Current microtonal layout (decode_vocab_size=3529):**
```
[sos_out:0 | timeshift:1-1026 | duration:1027-2052 | octave:2053-2065 | pitch:2066-3267 | instrument:3268-3398 | velocity:3399-3528]
```

- Every single ID is different — the pretrained decoder weights cannot be reused
- The GRU learns "at step 3, only output IDs 8199-8211 for octave" — completely wrong with new layout
- There is NO masking; the model learns valid ranges purely through training

### 2.4 Onset/Duration Vocab Changed Unnecessarily?

```
Original:    onset_vocab_size=4099, dur_vocab_size=4099
Microtonal:  onset_vocab_size=1026, dur_vocab_size=1026
```

---

## 3. The Correct Approach: Fractional Semitones + Preserve Pretrained Layout

### 3.0 Key Architectural Insight: Dual Pitch Representation

Moonbeam's architecture has **two completely different systems** for pitch — this is NOT like a standard LLM:

1. **Transformer INPUT** (`embed_tokens` → FME + MRA): **Continuous**. FME is a sinusoidal function `sin/cos(value * angle_rates)` with a learnable linear transform — there is NO lookup table. MRA/RoPE also operates on continuous values. This is fundamentally different from a standard LLM's `nn.Embedding` input.

2. **GRU DECODER OUTPUT** (`decoder_embedding` + `lm_head`): **Discrete**. A standard `nn.Embedding(decode_vocab_size, 1536)` lookup table + `nn.Linear(1536, decode_vocab_size)` classification head over a flat language vocabulary. This IS where discrete pitch token IDs live, and where initialization/regularization strategies apply.

This means we need **two pitch representations** flowing through the model:
- **Fractional semitones** (float) → transformer input via FME and MRA
- **Language token IDs** (int) → GRU decoder via decoder_embedding and lm_head

### 3.0.1 Full Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     MOONBEAM — COMPLETE DATA FLOW                      ║
║     Two REPRESENTATIONS but COUPLED paths: Transformer → GRU → loop   ║
╚══════════════════════════════════════════════════════════════════════════╝

  Dataset outputs two tensors (coupled during inference via autoregressive loop):
    input_ids  (batch, seq_len, 6)  →  Transformer input (compound tokens with floats)
    labels     (batch, seq_len, 7)  →  GRU decoder targets (language token IDs)

  ⚠ IMPORTANT: During TRAINING, these are separate (teacher forcing on both).
  During INFERENCE, GRU output → convert_from_language_tokens → tokens buffer →
  next transformer input. THE PATHS FORM A LOOP — resolution must be consistent!


───────────────── PATH 1: TRANSFORMER (continuous, accepts floats) ─────────────

  input_ids (batch, len, 6)
       │
       ├── [:,:,0] onset ────→ FME(base=199999) ──→ (batch,len,320) ─┐
       ├── [:,:,1] duration ─→ FME(base=1031)   ──→ (batch,len,320) ─┤
       ├── [:,:,2] octave ───→ FME(base=19)     ──→ (batch,len,320) ─┤
       ├── [:,:,3] pitch ────→ FME(base=20)  ✦  ──→ (batch,len,320) ─┼→ concat
       ├── [:,:,4] instrument→ nn.Embedding(131) ─→ (batch,len,320) ─┤  (1920)
       └── [:,:,5] velocity ─→ FME(base=131)    ──→ (batch,len,320) ─┘    │
                                                                           ▼
       ✦ FME = sin/cos(value × angle_rates) + bias → Linear         MLP(1920→960→1920)
         Continuous function. Accepts ANY float.                           │
         0.5 is just as valid as 0 or 1.                                   ▼
         NOT a lookup table.                                    ╔══════════════════╗
                                                                ║  TRANSFORMER     ║
       ⚠ nn.Embedding (instrument ONLY):                       ║  15 layers       ║
         Discrete lookup table. Integer indices only.           ║  MRA attention   ║
                                                                ║  (6 head groups  ║
  position_ids = input_ids (line 1697)                          ║   with per-attr  ║
  → MRA rotates each head group by its attribute:               ║   RoPE)          ║
    Heads 0,4: onset   (θ=199999)                              ╚════════╤═════════╝
    Head 1:    duration (θ=1031)                                        │
    Head 2:    octave   (θ=19)                                          ▼
    Head 3:    pitch    (θ=20)  ← accepts float!              hidden_states (batch, len, 1920)
    Head 5:    velocity (θ=131)                                         │
                                                                        ▼
                                                              summary_projection
                                                              Linear(1920 → 1536)
                                                                        │
                                                                        ▼
                                                              (batch, len, 1536)
                                                              = initial hidden state
                                                                for GRU decoder


─────────────── PATH 2: GRU DECODER (discrete, integers only) ─────────────────

  labels (batch, len, 7) — separate tensor during TRAINING (teacher forcing)
  ⚠ During INFERENCE: GRU output → language tokens → compound tokens → NEXT transformer input
       │
       ▼ shift by 1 position
       │
       ▼ flatten to (batch×(len-1), 7)
       │
       ├──────────────────────────────────────────┐
       │                                          │
       ▼                                          ▼
  labels[:, :-1]  (first 6 cols)           labels[:, 1:]  (last 6 cols)
  = TEACHER FORCING INPUT                  = PREDICTION TARGET
  [sos, tshift, dur, oct, pitch, instr]    [tshift, dur, oct, pitch, instr, vel]
       │                                          │
       ▼                                          │
  ┌─────────────────────────────┐                 │
  │ decoder_embedding           │                 │
  │ = nn.Embedding(8487, 1536)  │ ⚠ LOOKUP TABLE │
  │                             │ ⚠ integers ONLY │
  │ NOT FME!                    │ ⚠ can't do .5   │
  └──────────────┬──────────────┘                 │
                 │                                │
                 ▼                                │
  ┌─────────────────────────────┐                 │
  │ GRU (4 layers, dim=1536)   │                 │
  │ hidden_0 = summary_proj    │                 │
  │                             │                 │
  │ 6 sequential decode steps:  │                 │
  │  Step 0: sos_out → tshift   │                 │
  │  Step 1: tshift  → dur      │                 │
  │  Step 2: dur     → octave   │                 │
  │  Step 3: octave  → pitch ←─── THE pitch step  │
  │  Step 4: pitch   → instr    │                 │
  │  Step 5: instr   → velocity │                 │
  └──────────────┬──────────────┘                 │
                 │                                │
                 ▼                                │
  ┌─────────────────────────────┐                 │
  │ lm_head                     │                 │
  │ = nn.Linear(1536, 8487)     │ ⚠ CLASSIFIER   │
  │                             │ ⚠ outputs 8487  │
  │ argmax → integer class ID   │ ⚠ class logits  │
  └──────────────┬──────────────┘                 │
                 │                                │
                 ▼                                ▼
  logits (N, 8487)                     targets (N,) LongTensor
                 │                                │
                 └───────────┬────────────────────┘
                             ▼
                  ┌────────────────────┐
                  │ CrossEntropyLoss() │ ⚠ requires integer targets
                  │ loss = CE(logits,  │ ⚠ cannot accept 1032.5
                  │          targets)  │
                  └────────────────────┘


──────────── INFERENCE (autoregressive, line 1776-1785) ────────────────────────

  For each music event, GRU decodes 6 steps:

  summary_proj(transformer_output) → initial GRU hidden state
                                          │
      ┌───────────────────────────────────┘
      ▼
  decoder_embedding(sos_id) → GRU → lm_head → argmax → timeshift_id  (INT)
  decoder_embedding(tshift)  → GRU → lm_head → argmax → duration_id  (INT)
  decoder_embedding(dur_id)  → GRU → lm_head → argmax → octave_id    (INT)
  decoder_embedding(oct_id)  → GRU → lm_head → argmax → pitch_id     (INT) ←!
  decoder_embedding(pitch_id)→ GRU → lm_head → argmax → instrument_id(INT)
  decoder_embedding(instr_id)→ GRU → lm_head → argmax → velocity_id  (INT)

  Every value is an integer. No floats anywhere in the GRU path.
  pitch_id must then be converted BACK to fractional semitones
  for the next event's transformer input (see TODO 10).

  ┌──────────────────── AUTOREGRESSIVE FEEDBACK LOOP ────────────────────┐
  │                                                                      │
  │  GRU 6-step output (all integer language token IDs)                  │
  │       │                                                              │
  │       ▼                                                              │
  │  convert_from_language_tokens() → compound token (with float pitch)  │
  │       │                                                              │
  │       ▼                                                              │
  │  tokens[:, cur_pos] = next_token  ←── STORED IN TOKEN BUFFER         │
  │       │                                                              │
  │       └───→ NEXT iteration: transformer input = tokens[:, prev:cur]  │
  │             (generation.py line 218 → back to PATH 1)                │
  └──────────────────────────────────────────────────────────────────────┘
  ⚠ PATHS ARE COUPLED: GRU output becomes next transformer input!
  ⚠ Resolution MUST be consistent across both paths (10-cent everywhere).
```

### 3.1 Key Insight: FME and RoPE Accept Floats Natively

Both FME and RoPE explicitly cast inputs to float:
- FME: `inp = inp[..., None]` then `angle_rads = inp * self.angles` — pure multiplication
- RoPE: `position_ids_expanded = position_ids[:, None, :].float()` then matrix multiply

**If we represent microtonal pitches as fractional semitones (0.0-11.99) instead of cents (0-1199), using the ORIGINAL base=20:**
- `FME(0.0, base=20)` = C = EXACTLY matches pretrained `FME(0, base=20)`
- `FME(1.0, base=20)` = C# = EXACTLY matches pretrained `FME(1, base=20)`
- `FME(0.5, base=20)` = quarter-tone above C = smoothly between C and C# (novel, interpolated)
- MRA relative distance C→D = 2.0 = SAME as pretrained

### 3.2 Dual Representation: Float for Transformer, Int for GRU

```
TRANSFORMER INPUT (embed_tokens → FME + MRA):
  input_ids[..., 3] = fractional semitones (float)
  e.g., C=0.0, C+22.6cents=0.226, C#=1.0, D=2.0
  FME base = 20 (UNCHANGED from pretrained)
  MRA RoPE base = 20 (UNCHANGED from pretrained)
  → All pretrained transformer weights work perfectly
  → Microtonal pitches interpolate naturally in continuous embedding space
  → NO initialization needed — FME is a continuous function, not a lookup table

GRU DECODER OUTPUT (decoder_embedding + lm_head):
  Language token IDs = integers in flat vocab
  Pitch portion: 1200 discrete cents classes (0-1199) + SOS + EOS
  decoder_embedding = Embedding(decode_vocab_size, 1536) — LOOKUP TABLE
  lm_head = Linear(1536, decode_vocab_size) — CLASSIFICATION HEAD
  → These need to be EXPANDED for new microtonal pitch IDs
  → New rows need INITIALIZATION (copy from western anchor + noise/interpolation)
  → REGULARIZATION strategies apply here (anchor, smoothness, centering)
```

### 3.2.1 Why NOT a 7th Dimension

An alternative would be to keep pitch at 12 classes and add a 7th "zone/bend" attribute to the compound token. This can have some problems:

> **Microtonal pitches ARE pitches.** A segah (roughly E−50 cents) is not "E plus some modifier" — it is a pitch between Eb and E in a continuous frequency space.

With fractional semitones, FME naturally represents this: `FME(3.5, base=20)` falls directly between Eb and E in embedding space, exactly as it should be musically. A 7th dimension would instead represent it as `FME(4, base=20)` (E) plus a separate zone embedding, **losing the continuous pitch relationship** that FME was designed to capture.

### 3.3 New Model Config: `model_config_microtonal_v2.json`

Keep EVERYTHING the same as original except:
- `pitch_class_vocab_size`: 14 → 1214 (1200 cents + 12 reserved for SOS/EOS + compatibility padding)
  - Actually: 1202 (1200 cents values + SOS + EOS) — like the current config
- `decode_vocab_size`: 8487 → 8487 + (1202 - 14) = **9675**
  - Or more precisely: keep onset/dur/octave/instrument/velocity IDENTICAL, only expand pitch
- `microtonal`: true
- `pitchbend_sensitivity`: 2.0
- `pitch_embedding.base`: **20** (KEEP ORIGINAL — this is the fix!)
- `rope_theta_pitch`: **20** (KEEP ORIGINAL — this is the fix!)
- `onset_vocab_size`: **4099** (KEEP ORIGINAL)
- `dur_vocab_size`: **4099** (KEEP ORIGINAL)

**Language token ID layout — APPEND-ONLY (decode_vocab_size=9675):**
```
[sos_out:0 | timeshift:1-4099 | duration:4100-8198 | octave:8199-8211 | pitch_western:8212-8225 | instrument:8226-8356 | velocity:8357-8486 | pitch_microtonal:8487-9674]
 ↑ SAME                        ↑ SAME                ↑ SAME            ↑ SAME (12+SOS+EOS)       ↑ SAME               ↑ SAME              ↑ NEW (1188 appended)
```

**CRITICAL: DO NOT shift instrument/velocity IDs.** The GRU has learned that at step 5, instrument lives at IDs 8226-8356 and at step 6, velocity at 8357-8486. Shifting these would break the GRU's learned sequential pattern even if weights are copied.

Instead, append microtonal pitch IDs (non-western cent values) AFTER the original vocab:
- First 8487 IDs: **EXACTLY the pretrained layout** — no changes
- IDs 8487-9674: **1188 new microtonal pitch tokens** (cents 1-99, 101-199, ..., 1101-1199)
- `pitch_dict` becomes a non-contiguous mapping: western cents → original IDs, microtonal cents → appended IDs
- `decoder_embedding` and `lm_head`: first 8487 rows loaded directly from pretrained checkpoint, rows 8487+ initialized by interpolation

---

## 4. Detailed TODO List

### TODO 1: Create new branch from `main`
```bash
git checkout main
git checkout -b feat/microtonal_continual_pretrain
```
Do NOT base on `feat/mtok_vocab` — it's 77 commits behind main and has wrong configs.

### TODO 2: New model config — `model_config_microtonal_v2.json`
```json
{
  // ... same as model_config.json EXCEPT:
  "microtonal": true,
  "pitchbend_sensitivity": 2.0,
  "pitch_class_vocab_size": 1202,    // 1200 cents + SOS + EOS
  "pitch_embedding": {"method": "FME", "base": 20},  // KEEP ORIGINAL BASE!
  "rope_theta_pitch": 20,            // KEEP ORIGINAL BASE!
  "onset_vocab_size": 4099,          // KEEP ORIGINAL
  "dur_vocab_size": 4099,            // KEEP ORIGINAL
  "decode_vocab_size": 9675          // 1 + 4099 + 4099 + 13 + 1202 + 131 + 130
}
```

### TODO 3: Modify `MusicTokenizer` for fractional semitone input

The tokenizer needs two representations for pitch:
1. **`pitch_class_cents`** (int, 0-1199): for the GRU decoder language tokens
2. **`pitch_class_fractional`** (float, 0.0-11.99): for transformer input (FME + MRA)

Changes needed in `music_tokenizer.py`:

#### 3a. `midi_to_compound` output format
Currently outputs: `[onset_ticks, duration_ticks, octave, pitch_class, instrument, velocity]`
- For microtonal: `pitch_class` is cents (0-1199) — this is fine for GRU decoder
- **ADD**: Store `pitch_class_fractional = pitch_class_cents / 100.0` for transformer input
- The compound token needs to carry BOTH representations or we derive fractional on-the-fly

**Recommended approach:** Store cents in the .npy files (integer, compact). Convert to fractional semitones at training time in the dataset `__getitem__`:
```python
# In LakhDataset.__getitem__ or equivalent:
raw_tokens[:, 3] = raw_tokens[:, 3] / 100.0  # cents → fractional semitones for FME/MRA
```

#### 3b. `encode_single` — no change needed
Already passes through raw values.

#### 3c. `encode_series_labels` — language token path
The GRU decoder labels must use INTEGER cents (0-1199), not fractional semitones.
- `convert_to_language_tokens` maps pitch_class → language ID via `self.pitch_dict`
- `self.pitch_dict` needs to map 0-1199 (+ SOS/EOS at 1200/1201) to language IDs
- This works with current code IF `pitch_class_vocab_size=1202`

#### 3d. New: `input_ids` must contain floats for pitch

**Data flow for pitch (verified end-to-end on `main` branch):**
```
.npy files → MusicTokenizer.encode_series() → LakhDataset.__getitem__
  → ConcatDataset_hybrid_padding_concatenating._process_samples() → Python lists in memory
  → DataLoader (default collate → tensors) → train_con_gen() → model.forward(input_ids=..., labels=...)
```

**IMPORTANT update:** `real_finetuning_uncon_gen.py` (on `main`) uses `ConcatDataset_hybrid_padding_concatenating`, NOT the mmap-based `ConcatDataset` from `feat/mtok_vocab`. The hybrid concatenator stores data as **Python lists in memory** — it does NOT force int32. This simplifies our approach significantly.

**Two options for float conversion:**

**Option A (preferred): Convert in the training loop**, before `model.forward()`:
```python
# In train_con_gen(), after batch is moved to device, before model(**batch):
batch['input_ids'] = batch['input_ids'].float()
batch['input_ids'][:, :, 3] = batch['input_ids'][:, :, 3] / 100.0  # cents → fractional semitones
# Onset/dur/octave/velocity: FME handles float ints (0.0, 1.0, ...) identically to ints
# Instrument: needs .long() cast inside embed_tokens (see TODO 6)
```

**Option B: Convert in `__getitem__`** of the dataset class, so input_ids are already float when packed. Since the hybrid concatenator just stores Python lists (no dtype forcing), this also works.

Either way, **labels stay as integers** — during training they go through the GRU decoder via teacher forcing. During inference, however, GRU output is converted back to compound tokens and fed to the next transformer step (autoregressive loop), so pitch resolution must be consistent across both paths.

### TODO 4: Modify `LakhDataset` (or create `MakamDataset`)
- Load .npy with cents-based pitch (int)
- For `input_ids`: keep as int through concatenator, convert at training time
- For `labels`: keep pitch as cents (int) → for GRU decoder language tokens
- The `input_ids` tensor must support float (currently int for all western pitches)

### TODO 5: Create continual pretraining script — `real_finetuning_microtonal.py`
Based on `real_finetuning_uncon_gen.py` from `main` branch. This script already:
- Creates `LlamaForCausalLM` from config
- Loads pretrained checkpoint with `strict=False` (handles missing/unexpected keys from vocab expansion)
- Uses `train_con_gen()` (autoregressive training with periodic validation + PEFT checkpoint saving)
- Uses `ConcatDataset_hybrid_padding_concatenating` (in-memory, no int32 forcing)
- Has full PEFT/LoRA integration

**Additional steps we need to add (between checkpoint load and training):**

#### 5a. Load pretrained checkpoint into ORIGINAL architecture (8487 vocab)
```python
llama_config_orig = LlamaConfig.from_pretrained("model_config.json")  # original
model_orig = LlamaForCausalLM(llama_config_orig)
model_orig.load_state_dict(checkpoint)
```

#### 5b. Create NEW model with microtonal config (9675 vocab)
```python
llama_config_micro = LlamaConfig.from_pretrained("model_config_microtonal_v2.json")
model_micro = LlamaForCausalLM(llama_config_micro)
```

#### 5c. Transfer weights with smart initialization

**Transformer backbone (all layers):** Direct copy — architecture is IDENTICAL.
```python
# All transformer layers, FME embeddings, MRA parameters, etc.
# SAME hidden_size, num_layers, num_heads — direct parameter copy
```

**FME pitch embedding:** Direct copy — `base=20` is same, `Linear(320,320)` + bias is same.
```python
model_micro.model.pitch_embedding.load_state_dict(model_orig.model.pitch_embedding.state_dict())
```

**GRU decoder (hidden layers):** Direct copy — same architecture `(1536, 4 layers)`.
```python
model_micro.decoder.load_state_dict(model_orig.decoder.state_dict())
```

**`decoder_embedding` — Embedding(9675, 1536):**
- Rows 0-8486: **DIRECT COPY** from pretrained (all original IDs preserved exactly)
- Rows 8487-9674 (1188 new microtonal pitch tokens): Initialize from western anchor. Let `E^(0)[pc]` = pretrained row for western pitch class pc (row `8212 + pc`).

**Initialization strategies (applied to both `decoder_embedding` and `lm_head`):**

**(1) Interpolation (baseline):** Linearly interpolate between the two flanking western semitones.
```python
# For a microtonal pitch at cent value c (0-1199):
lower_pc = c // 100           # nearest lower western semitone
frac = (c % 100) / 100.0
embed_new[c] = (1 - frac) * E^(0)[lower_pc] + frac * E^(0)[(lower_pc+1) % 12]
# e.g., cents 50 → 0.5 * embed(C) + 0.5 * embed(C#)
# e.g., cents 23 → 0.77 * embed(C) + 0.23 * embed(C#)
```

**(2) Copy + noise:** Start identical to the western anchor with small perturbation.
```python
embed_new[c] = E^(0)[nearest_pc] + ε,   ε ~ N(0, σ²I)
# σ should be small relative to ||E^(0)||
```

**(3) Microtonal-axis (optional inductive bias):** Place variants along a direction in embedding space proportional to cents offset.
```python
embed_new[c] = E^(0)[nearest_pc] + β · Δc · u + ε
# u ∈ R^1536 is a unit vector (global or per-pitch-class u_pc)
# β chosen so that β · max|Δc| ≈ 1-5% of ||E^(0)||
# This makes embedding distance correlate with cents distance at init
```

**Impact assessment:** Initialization choice is LOW impact — the model will quickly overwrite initial values during CPT. All three strategies start "close enough" to the pretrained manifold. Microtonal-axis (3) adds complexity for minimal benefit since FME already encodes continuous pitch geometry on the input side.

**Recommendation:** Start with (1) interpolation as baseline (simplest, musically principled). Ablate (2) copy+noise. Skip (3) microtonal-axis unless ablations show it helps.

**`lm_head` — Linear(1536, 9675):**
Same logic as `decoder_embedding` for weight rows — first 8487 rows direct copy, new rows initialized with same strategy.

#### 5d. Training strategy
- **LoRA on transformer backbone** — preserves pretrained knowledge, few trainable params
- **Full training on GRU decoder layers** — they need to learn the new pitch vocabulary
- **Full training on `decoder_embedding` and `lm_head`** — these have the new pitch dimensions
- Consider **higher LoRA rank** for pitch-related attention heads (group 3)

#### 5e. Regularization on decoder_embedding and lm_head

These regularizers apply to the discrete pitch tokens in `decoder_embedding` and `lm_head` — NOT to FME (which is continuous and needs no regularization).

**Anchor loss** — prevent western pitch embeddings from drifting during CPT:
```python
L_anchor = λ_a * Σ_{pc=0}^{11} ||E[8212+pc] - E^(0)[pc]||²
# E^(0)[pc] = frozen copy of pretrained western embedding
```

**Smoothness loss** — encourage nearby cents to have nearby embeddings:
```python
# For each pitch class, order all its active cent tokens by cent value:
L_smooth = λ_s * Σ_{pc} Σ_{adjacent pairs (c1, c2)} ||E[c2] - E[c1]||²
```

**Centering loss** (optional) — keep microtonal variants close to their western anchor:
```python
L_center = λ_c * Σ_{pc} Σ_{microtonal cents c of pc} ||E[c] - E[8212+pc]||²
# Useful early in training, can be decayed
```

**Total objective:**
```python
L = L_pretrain + L_anchor + L_smooth + L_center
```

Note: FME embeddings do NOT need regularization — they are a continuous function with fixed base=20, unchanged from pretraining. Fractional semitone inputs naturally interpolate.

**Impact assessment for regularization:**

| Loss | Impact | Rationale |
|---|---|---|
| **L_anchor** | **HIGH** | With only 3000 files / ~1.16M notes, catastrophic forgetting of western pitch embeddings is a real risk. L_anchor is our main defense alongside data mixing. Without it, the 12 pretrained western rows (8212-8223) could drift and break western generation entirely. |
| **L_smooth** | **HIGH** | Addresses the empty bins problem (Section 5.2) AND prevents **autoregressive error cascades** (Section 5.2.1). ~90% of 1200 cent bins are empty in SymbTr — L_smooth propagates gradient signal to neighboring untrained bins. Without it, ghost tokens during inference feed untrained fractional semitones back to the transformer, compounding errors across the sequence. |
| **L_center** | **LOW** | Redundant with L_anchor for western pitches and overly constraining for microtonal ones. A segah (E−50 cents) should NOT be forced to stay close to E — it's a distinct pitch. If used at all, decay λ_c to 0 within the first 20-30% of training. |

**Recommendation:** Always use L_anchor (λ_a ≈ 0.1-1.0). Always use L_smooth (λ_s ≈ 0.01-0.1) — essential for preventing autoregressive error cascades from ghost bins (Section 5.2.1). Skip L_center or decay it rapidly.

#### 5f. CPT Schedule

**Warmup/freezing:** For first T_freeze steps (5-20% of total), freeze western anchor embeddings `E[8212:8224]` in decoder_embedding/lm_head. Train only new microtonal rows (8487+) and LoRA adapters. After warmup, unfreeze all and rely on L_anchor to maintain stability.

**Data mixing:** Train on α ≈ 0.7-0.9 microtonal (SymbTr) + (1-α) western replay data to prevent catastrophic forgetting.

**Optimization:** AdamW, conservative LR η ∈ [1e-5, 3e-5] with cosine decay, gradient clipping norm 1.0.

**Per-parameter learning rates (recommended):**
```python
param_groups = [
    {"params": lora_params,                  "lr": 3e-5},   # LoRA adapters
    {"params": decoder_existing_rows_params, "lr": 1e-5},   # Rows 0-8486 (pretrained, conservative)
    {"params": decoder_new_rows_params,      "lr": 1e-4},   # Rows 8487+ (new, needs faster learning)
    {"params": gru_internal_params,          "lr": 3e-5},   # GRU layers (adapt to new vocab)
]
```
New pitch rows (8487+) need ~3-10× higher LR than pretrained rows — they start from interpolation and must learn distinct microtonal representations. Pretrained rows should move slowly (L_anchor also helps here).

### TODO 6: Fix `input_ids` tensor dtype for float pitch

**CRITICAL:** `instrument_embedding` uses `WordEmbedding` (line 1275-1284: wraps `nn.Embedding`) which requires `LongTensor`. All other embeddings (onset, dur, octave, pitch, velocity) use `FME` which accepts floats natively. If `input_ids` is passed as float for pitch, `instrument_embedding(input_ids_tmp[..., 4])` will crash.

Verified in code:
- `instrument_embedding` config: `{"method": "WE", "vocab_size": 131}` — the ONLY `WE` embedding
- `WordEmbedding.forward()` (line 1280): calls `self.embedding(inp)` → `nn.Embedding` requires `LongTensor`
- `Fundamental_Music_Embedding.__call__()` (line 1308-1310): `inp[..., None] * self.angles` — pure float math

**Solution:** One-line fix in `embed_tokens()` (modeling_llama.py line 1402):
```python
# BEFORE:
instruments = self.instrument_embedding(input_ids_tmp[..., 4])
# AFTER:
instruments = self.instrument_embedding(input_ids_tmp[..., 4].long())
```

Keep `input_ids` as float tensor (converted at training time, see TODO 3d). All FME embeddings handle float natively (onset, dur, octave, pitch, velocity). Only instrument (the sole `WordEmbedding` / `nn.Embedding`) needs the `.long()` cast.

### TODO 6b: Fix position_ids dtype mismatch in attention — 3-LINE FIX

In `LlamaSdpaAttention.forward()` (modeling_llama.py lines 931-933), position_ids replacements use `.to(hidden_states.device)` which doesn't preserve dtype. If position_ids is float (because input_ids is float), `torch.where` will fail on dtype mismatch.

```python
# BEFORE (lines 931-933):
position_ids_sos = torch.where(where_sos, torch.tensor([0 for _ in range(6)]).to(hidden_states.device), position_ids)
position_ids_eos = torch.where(where_eos, torch.tensor([2**15 for _ in range(6)]).to(hidden_states.device), position_ids_sos)
position_ids_eos = torch.where(where_classification, torch.tensor([2**15 + 1 for _ in range(6)]).to(hidden_states.device), position_ids_eos)

# AFTER — use .to(position_ids) to preserve both device AND dtype:
position_ids_sos = torch.where(where_sos, torch.tensor([0 for _ in range(6)]).to(position_ids), position_ids)
position_ids_eos = torch.where(where_eos, torch.tensor([2**15 for _ in range(6)]).to(position_ids), position_ids_sos)
position_ids_eos = torch.where(where_classification, torch.tensor([2**15 + 1 for _ in range(6)]).to(position_ids), position_ids_eos)
```

Note: `embed_tokens` line 1395 already uses `.to(input_ids)` — correct, no change needed there.

### TODO 7: Verify `position_ids` for MRA — CONFIRMED OK

In `LlamaSdpaAttention.forward()`:
```python
position_ids = input_ids  # line 1697
cos_pitch, sin_pitch = self.rotary_emb_pitch(value_states, position_ids[:, :, 3])  # line 939
```

In `LlamaRotaryEmbedding.forward()` (line 108-122):
```python
position_ids_expanded = position_ids[:, None, :].float()  # line 112 — EXPLICIT float cast
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float())  # line 118 — float @ float
```

`position_ids[:, :, 3]` will be fractional semitones (0.0, 0.226, 1.0, 2.5, ...).
**Verified: RoPE explicitly casts position_ids to float at line 112. No code changes needed.**

### TODO 8: Handle SOS/EOS detection with float pitch — CONFIRMED OK

In `embed_tokens` (line 1388):
```python
where_sos = (input_ids[:, :, 0] == self.sos_token).unsqueeze(-1)  # onset column (index 0)
```
In MRA attention (line 926):
```python
where_sos = (position_ids[..., 0] == self.config.sos_token).unsqueeze(-1)  # onset column
```

Both check **onset column (index 0)**, NOT pitch. SOS/EOS tokens have onset=-1/-2 (int).
**As long as onset column remains integer, float pitch values don't affect detection. No change needed.**

### TODO 8b: GRU Label Structure — CONFIRMED

Verified in `encode_series_labels` (music_tokenizer.py line 259-307) and `convert_to_language_tokens` (line 321-335):

GRU labels are **7 language IDs per event**: `[sos_out, timeshift, duration, octave, pitch, instrument, velocity]`

The dicts map raw attribute values → sequential language IDs (line 173-217):
```
sos_out_dict:    {0: 0}                                        → ID 0
timeshift_dict:  {i: i + 1}                                    → IDs 1 to onset_vocab_size
duration_dict:   {i: i + 1 + onset_vocab_size}                 → IDs (onset+1) to (onset+dur)
octave_dict:     {i: i + 1 + onset + dur}                      → next 13
pitch_dict:      {i: i + 1 + onset + dur + octave}             → next pitch_class_vocab_size
instrument_dict: {i: i + 1 + onset + dur + octave + pitch}     → next 131
velocity_dict:   {i: i + 1 + onset + dur + octave + pitch + instrument} → next 130
```

GRU decodes sequentially (line 1759-1768): teacher forcing with `decoder_embedding(previous_token)` → GRU step → `lm_head` → logits over full `decode_vocab_size`. No masking — model learns valid ranges for each step through training.

### TODO 8c: Per-step GRU accuracy monitoring

During training, log **per-attribute GRU accuracy** (not just aggregate loss):
```python
# After GRU forward, for each of the 7 steps:
for step in range(7):
    step_logits = logits[:, step, :]
    step_labels = labels[:, step]
    step_acc = (step_logits.argmax(-1) == step_labels).float().mean()
    log(f"gru_acc/step_{step}_{ATTR_NAMES[step]}", step_acc)
```

This reveals if the GRU is struggling specifically with the pitch step (step 4) vs other attributes. If pitch accuracy lags behind onset/dur/octave, it signals the new pitch tokens need more training. Also monitor accuracy separately for western-range pitches vs microtonal pitches to catch forgetting.

### TODO 8d: `pitch_dict` non-sequential mapping for append-only

The current `pitch_dict` (music_tokenizer.py lines 191-198) builds a **sequential** mapping:
```python
{i: i + offset for i in range(pitch_class_vocab_size)}
```

For the append-only scheme, this must become **non-contiguous**:
```python
# Western cents (multiples of 100) → original pretrained IDs
# e.g., 0→8212, 100→8213, 200→8214, ..., 1100→8223
# Microtonal cents → appended IDs
# e.g., 1→8487, 2→8488, ..., 99→8585, 101→8586, ...
```

The exact mapping depends on the chosen resolution (1-cent, 10-cent, or GMM). Build the dict programmatically to ensure western pitches always map to their original IDs.

### TODO 9: Data preprocessing pipeline
- Port `midi_to_compound()` microtonal logic from `feat/mtok_vocab` to the new branch
- Keep `pitch_class_cents` (int 0-1199) in .npy files
- Add conversion to fractional semitones in the Dataset class, not in preprocessing
- Keep `onset_vocab_size=4099`, `dur_vocab_size=4099` (original values)
- Verify SymbTr files don't exceed these limits (they're short pieces, should be fine)

### TODO 10: Implement inference loop changes (see Section 6.5 for full analysis)

The inference pipeline is fully documented in **Section 6.5**. Summary of changes:
1. `tokens` buffer in `generate()` → `dtype=torch.float32` (currently `torch.long` truncates fractional pitches)
2. `convert_from_language_tokens()` → divide pitch values by 100.0 to get fractional semitones
3. `compound_to_midi()` → add pitchbend messages for microtonal note output
4. Instrument column `.long()` cast (same fix as training path)

### TODO 11: Investigate Gaussian Pitch-Zone Tokenizer (future improvement)

The current approach uses a uniform 1-cent grid (1200 bins), but Turkish makam pitches cluster around a small number of attractor targets (scale degrees). ~90% of the 1200 bins will be EMPTY in training data. A **Gaussian Pitch-Zone Tokenizer** (see microtok.tex) could dramatically reduce the number of new tokens:

**Idea:** For each pitch class pc, fit a Gaussian mixture model on observed cent offsets Δc from the SymbTr corpus:
```
p(Δc | pc) = Σ_{j=1}^{J_pc} π_j · N(Δc; μ_j, σ²_j)
```
- J_pc zones per pitch class (selected via BIC)
- Zone 0 = western/no-bend (μ closest to 0 cents), preserving pretrained IDs
- Post-processing: merge-by-proximity (δ≈20 cents), merge rare components (τ≈0.03)
- Set global K = max J_pc

**Token ID mapping with append-only scheme:**
```
Zone k=0 (western):  global_id(pc, 0) = 8212 + pc          → IDs 8212-8223 (PRETRAINED)
Zone k=1:            global_id(pc, 1) = 8487 + pc           → IDs 8487-8498 (NEW)
Zone k=2:            global_id(pc, 2) = 8487 + 12 + pc      → IDs 8499-8510 (NEW)
...
Zone k=K-1:          global_id(pc, K-1) = 8487 + 12*(K-2) + pc
```

This follows the `pitch_id(pc, k) = 12k + pc` formula from the tex, mapped to the global append-only layout. Total new tokens = 12*(K-1), likely **24-84** (if K=3-8) instead of 1188 — much more compact.

**FME input would use zone center:** `pitch_float = pc + μ_{pc,k} / 100.0` (still fractional semitones, still preserves all pretrained knowledge for k=0).

**Trade-off vs uniform grid:**
- **Pro:** Far fewer new tokens, musically principled, better matches the actual data distribution
- **Con:** Requires a preprocessing step (fit GMMs), adds quantization error (target RMSE < 5-8 cents), loses fine-grained resolution between zones
- **Recommendation:** Start with uniform 1-cent grid (simpler, no preprocessing). Investigate GMM zones as an optimization if the sparse decoder is a problem.

---

## 5. SymbTr/Makam-Specific Considerations

### 5.1 Pitch Distribution
Turkish makam pitches are NOT uniformly distributed across 1200 cents. They cluster around specific scale degrees defined by Holdrian commas (1 comma ≈ 22.6 cents). Most accidentals are within ±90 cents of a western note. This means:
- ~90% of the 1200 cent bins will be EMPTY in training data
- The model needs to generalize to unseen cent values between makam scale degrees
- FME's continuous nature helps here — unused cents get interpolated embeddings
- Consider: do we really need all 1200 cent values, or a smaller quantization?

### 5.2 Resolution Tradeoff & Empty Bins Problem

**The empty bins problem:** Turkish makam pitches cluster around ~30-50 attractor targets per octave (Holdrian commas). With 1-cent resolution (1200 bins), ~90% of bins will have ZERO training examples. This creates a concrete risk: during GRU inference, if the model samples an untrained bin, the `decoder_embedding` row for that bin was only ever initialized (by interpolation) and never updated by gradient — it may produce incoherent hidden states downstream.

L_smooth regularization partially addresses this (propagates gradient to neighboring bins), but the fundamental sparsity remains.

### 5.2.1 Autoregressive Error Cascades (Inference Only)

The ghost bin problem is **worse than it appears** because of the coupled autoregressive loop (Section 3.0.1). During **inference** (NOT training — teacher forcing breaks the loop):

```
GRU samples ghost token (untrained bin, e.g., cent 37)
    → convert_from_language_tokens → fractional semitone 0.37
    → stored in tokens buffer
    → NEXT iteration: transformer sees FME(0.37) as input
    → decoder_embedding row for cent 37 was only initialized (interpolation), never gradient-updated
    → transformer hidden state may be suboptimal
    → summary_projection → GRU initial state is degraded
    → NEXT GRU prediction is also less accurate
    → ERROR COMPOUNDS ACROSS THE SEQUENCE
```

During **training**, teacher forcing means the transformer always sees ground-truth `input_ids`, regardless of what the GRU predicts. So ghost bins are harmless during training — the model never "sees" its own mistakes. But during generation, errors propagate.

**Implications:**
1. **L_smooth is HIGH impact** (not MEDIUM) — it ensures ghost bin embeddings are reasonable, preventing cascade initiation
2. **10-cent resolution** further reduces risk — fewer bins = fewer ghosts = fewer cascade opportunities
3. **53-TET augmentation** helps by filling empty bins with real training signal
4. The **gap between teacher-forced and autoregressive pitch accuracy** directly measures this cascade effect (see Section 8.2)

**Resolution comparison:**

| Resolution | New tokens | Precision | Empty bins | Decoder expansion | Notes |
|---|---|---|---|---|---|
| **1-cent** (1200) | 1188 | ±0.5 cent | ~90% | +14% (8487→9675) | Maximum precision, most empty bins |
| **10-cent** (120) | 108 | ±5 cent | ~50% | +1.3% (8487→8595) | Good balance, 1 comma ≈ 2 bins |
| **Holdrian** (~53) | ~41 | ±11 cent | ~20% | +0.5% (8487→8528) | Musically motivated for makam |
| **GMM zones** (K≈5) | ~48 | varies | 0% (by design) | +0.6% (8487→8535) | Data-driven, see TODO 11 |

**Recommendation:** Start with **10-cent resolution** as a pragmatic middle ground — captures makam intervals with ±5 cent precision (well within perceptual thresholds), dramatically reduces the empty bins problem, and keeps the decoder expansion small. If higher precision is needed, move to 1-cent or GMM zones.

**If using 10-cent resolution:**
- `pitch_class_vocab_size = 122` (120 bins + SOS + EOS)
- `decode_vocab_size = 8487 + 108 = 8595`
- Fractional semitone conversion: `cents / 100.0` → same formula, just coarser
- FME input granularity: 0.0, 0.1, 0.2, ..., 11.9 — still continuous, still interpolates

### 5.3 Note-0 Rest Markers
SymbTr uses MIDI note 0 with pitchbend=-155 as rest/silence markers. The tokenizer correctly filters these out (1268 removed across 1125 files). This is already handled.

### 5.4 Instrument
SymbTr MIDIs use instrument=0 (piano). The model's pretrained instrument knowledge is fine — makam is traditionally performed on various instruments but the MIDI data is all piano-rendered.

### 5.5 Single Channel / Monophonic
Most SymbTr pieces are monophonic or have minimal polyphony. The pitchbend-per-channel approach is safe. No overlapping-pitchbend issues.

### 5.6 Dataset Size & 53-TET Transposition Augmentation

3000 files / ~1.16M notes is relatively small compared to Moonbeam's 18B token pretraining. This further motivates:
- Using LoRA (prevent overfitting with small data)
- Preserving pretrained weights as much as possible
- Potentially mixing in some western MIDI data during training to prevent catastrophic forgetting

**53-TET transposition idea:**

All SymbTr pieces are written relative to a single root (C). But in Turkish makam tradition, there is **no fixed absolute pitch for the root of a makam** — only the intervals are fixed. This is analogous to how the "Sa" in Hindustani and Carnatic ragas can be different across performances.

This means we can **augment SymbTr by transposing every piece to each of the 53 values in 53-TET** (the Holdrian comma system that underlies Turkish makam theory):

```
Original:     3,000 files  ×  ~387 notes/file  =  ~1.16M notes
Augmented:    3,000 × 53   =  159,000 files    =  ~61.5M notes  (53× increase!)
```

**Why this is musically valid:**
- Makam intervals are defined in Holdrian commas (relative), not absolute frequencies
- A piece in makam Hicaz starting on C is musically equivalent to Hicaz starting on D
- The melodic contour (seyir), interval structure, and all musical features are preserved
- Only the absolute pitch changes — which the model should be invariant to anyway

**Benefits for our CPT plan:**
1. **Solves the small dataset problem** — 61.5M notes is substantial (still small vs 18B but much more credible)
2. **Fills empty cent bins** — transposing to 53 roots spreads pitches across the full cent range, dramatically reducing sparsity
3. **Teaches transposition invariance** — the model sees the same makam structure in all keys
4. **Strengthens the case for 1-cent resolution** — with 53× more data, fewer bins are empty

**Implementation:**
- For each SymbTr file, transpose all MIDI note numbers by +k Holdrian commas (k = 0, 1, ..., 52)
- 1 Holdrian comma ≈ 22.6 cents, so transposition by k commas = shift all pitches by k × 22.6 cents
- In fractional semitones: shift by k × 0.226
- Filter out transpositions where notes fall outside MIDI range (0-127)
- Can be done at preprocessing time (multiply .npy files) or on-the-fly in the dataset class

---

## 6. Summary of Required Changes

| Component | File | Change |
|---|---|---|
| Model config | `model_config_microtonal_v2.json` | NEW FILE — keep original bases, expand pitch vocab only |
| Tokenizer | `music_tokenizer.py` | Port microtonal MIDI parsing from feat/mtok_vocab, add fractional semitone output |
| Dataset class | `lakh_dataset.py` or new `makam_dataset.py` | Convert cents→fractional semitones for input_ids, keep cents for labels |
| Training script | `real_finetuning_microtonal.py` | Copy from `real_finetuning_uncon_gen.py` (main), add vocab expansion + weight transfer + float conversion |
| Data preprocessing | `data_preprocess.py` | Port microtonal support, keep onset/dur vocab=4099 |
| Weight initialization | In training script | Interpolate new pitch decoder embeddings from pretrained western ones |
| MIDI output | `compound_to_midi()` | Port from feat/mtok_vocab, adapt for fractional semitones |

### Files that need MINIMAL changes:
- `modeling_llama.py` — **2 small fixes**: (1) `.long()` cast on instrument column in `embed_tokens` (line 1437: `input_ids_tmp[..., 4].long()`), (2) `.to(position_ids)` instead of `.to(hidden_states.device)` in attention (lines 931-933)
- `train_utils.py` → `train_con_gen()` — add `batch['input_ids'].float()` + `batch['input_ids'][:,:,3] /= 100.0` before `model(**batch)`, plus per-parameter optimizer groups
- `real_finetuning_uncon_gen.py` → copy to `real_finetuning_microtonal.py`, add weight transfer step between checkpoint load and PEFT wrapping

### Files that do NOT need changes:
- `concatenator.py` — `ConcatDataset_hybrid_padding_concatenating` stores Python lists, no dtype issues
- FME, MRA, GRU decoder architecture — all naturally handle floats

### Weight transfer summary (verified — only 2 parameters change shape):
| Parameter | Original Shape | New Shape | Transfer |
|---|---|---|---|
| `decoder_embedding.weight` | (8487, 1536) | (9675, 1536) | Rows 0-8486: direct copy. Rows 8487+: interpolate from western anchors |
| `lm_head.weight` | (8487, 1536) | (9675, 1536) | Same as above |
| `summary_projection` | (1920, 1536) | (1920, 1536) | **No change** — doesn't depend on vocab size |
| GRU internal (`gru`, `fc_out`) | (1536, 1536) | (1536, 1536) | **No change** — doesn't depend on vocab size |
| All transformer layers | unchanged | unchanged | **Direct copy** |
| All FME embeddings | unchanged | unchanged | **Direct copy** (base=20 kept) |

---

## 6.5 Inference Pipeline: GRU Language Tokens → MIDI Output

### 6.5.0 How Moonbeam Inference Works (from code)

The full autoregressive generation loop lives in `recipes/inference/custom_music_generation/generation.py` → `MusicLlama.generate()`. Here is the complete pipeline, step by step:

```
STEP 1: PROMPT ENCODING
────────────────────────
  Raw .npy data (compound tokens: [onset, dur, octave, pitch, instrument, velocity])
       │
       ▼
  tokenizer.encode_series(data, if_add_sos=True, if_add_eos=False)
       │  ← just adds SOS compound token [-1,-1,-1,-1,-1,-1] at start
       ▼
  prompt_tokens: List[List[int]]  shape = (prompt_len, 6)


STEP 2: TOKEN BUFFER INITIALIZATION
────────────────────────────────────
  tokens = tensor(bsz, total_len, 6)  ← filled with pad_token [-3,-3,-3,-3,-3,-3]
  tokens[:, :prompt_len] = prompt_tokens  ← copy prompt into buffer
  input_mask = (tokens != pad)  ← marks which positions are prompt (don't overwrite)


STEP 3: AUTOREGRESSIVE LOOP (for cur_pos in range(min_prompt_len, total_len))
─────────────────────────────────────────────────────────────────────────────

  ┌─── 3a. TRANSFORMER FORWARD PASS ───────────────────────────────────────┐
  │                                                                         │
  │  input_ids = tokens[:, prev_pos:cur_pos]     # (batch, chunk, 6)       │
  │       │                                                                 │
  │       ▼                                                                 │
  │  embed_tokens():                                                        │
  │    onset_emb   = FME(input_ids[..., 0], base=1031)     # 320-d        │
  │    dur_emb     = FME(input_ids[..., 1], base=1031)     # 320-d        │
  │    octave_emb  = FME(input_ids[..., 2], base=20)       # 320-d        │
  │    pitch_emb   = FME(input_ids[..., 3], base=20)       # 320-d  ← FLOAT OK │
  │    instr_emb   = WordEmb(input_ids[..., 4].long())     # 320-d        │
  │    velocity_emb= FME(input_ids[..., 5], base=199999)   # 320-d        │
  │    x = concat all → 1920-d                                             │
  │       │                                                                 │
  │       ▼                                                                 │
  │  24 transformer layers (with MRA: position_ids = input_ids)            │
  │       │                                                                 │
  │       ▼                                                                 │
  │  logits_shrinked = summary_projection(hidden_states)  (batch, chunk, 1536) │
  │  output.logits = logits_shrinked   ← 1536-d, NOT raw 1920-d!          │
  │  output.past_key_values = KV cache for next iteration                  │
  └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─── 3b. GRU SEQUENTIAL DECODING (6 steps per event) ───────────────────┐
  │                                                                         │
  │  hidden_state = summary_proj(transformer_output) → (num_layers, N, 1536)│
  │  next_token = sos_out_id                                                │
  │                                                                         │
  │  For each attribute in [timeshift, duration, octave, pitch, instr, vel]:│
  │    ┌──────────────────────────────────────────────────────────────────┐ │
  │    │  decoder_embedding(next_token) → GRU → lm_head → logits        │ │
  │    │       │                                                          │ │
  │    │       ▼                                                          │ │
  │    │  CONSTRAINED SAMPLING:                                           │ │
  │    │    sample_indices = attribute_dict_decode.keys()                 │ │
  │    │    ← only allow tokens valid for this attribute!                 │ │
  │    │    probs = softmax(logits / temperature)                        │ │
  │    │    next_token = top_p_sample(probs)                             │ │
  │    │    while next_token not in sample_indices: resample             │ │
  │    └──────────────────────────────────────────────────────────────────┘ │
  │                                                                         │
  │  Result: next_decoder_token_out = [sos, tshift, dur, oct, pitch, instr, vel] │
  │          (remove sos → 6 language IDs per event)                        │
  └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─── 3c. LANGUAGE TOKENS → COMPOUND TOKENS ─────────────────────────────┐
  │                                                                         │
  │  tokenizer.convert_from_language_tokens(next_decoder_token_out)        │
  │    For each token:                                                      │
  │      timeshift = timeshift_dict_decode[lang_id]   # e.g., 15 → 14     │
  │      duration  = duration_dict_decode[lang_id]    # e.g., 4200 → 100  │
  │      octave    = octave_dict_decode[lang_id]      # e.g., 8204 → 4    │
  │      pitch     = pitch_dict_decode[lang_id]       # e.g., 8215 → 3    │
  │      instrument= instrument_dict_decode[lang_id]  # e.g., 8226 → 0   │
  │      velocity  = velocity_dict_decode[lang_id]    # e.g., 8420 → 64  │
  │    → returns [tshift, dur, oct, pitch, instr, vel] as raw values       │
  └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─── 3d. ONSET ACCUMULATION ────────────────────────────────────────────┐
  │                                                                         │
  │  previous_onset = tokens[:, cur_pos-1, 0]                              │
  │  new_onset = previous_onset + delta_timeshift   ← accumulate!          │
  │  next_compound = [new_onset, dur, oct, pitch, instr, vel]              │
  │                                                                         │
  │  tokens[:, cur_pos] = next_compound   ← write back to buffer          │
  │  (only if this position isn't part of the prompt)                       │
  └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─── 3e. EOS CHECK ─────────────────────────────────────────────────────┐
  │  If ANY attribute is its EOS value → mark sequence as finished          │
  │  If all sequences finished → break                                      │
  └────────────────────────────────────────────────────────────────────────┘


STEP 4: POST-PROCESSING
────────────────────────
  tokens = tokens[:, 1:]   ← remove SOS token
  Cut to max_gen_len
  Cut at first EOS token (per attribute)
  → out_tokens: List[List[List[int]]]  (batch of compound token sequences)


STEP 5: COMPOUND TOKENS → MIDI FILE
────────────────────────────────────
  tokenizer.compound_to_midi(out_tokens)
    For each note [onset, dur, octave, pitch_class, instrument, velocity]:
      midi_note = octave * 12 + pitch_class      ← integer!
      Create note_on at time=onset, note_off at time=onset+dur
      Assign to instrument track
    → returns mido.MidiFile object
    → .save("output.mid")
```

### 6.5.1 Critical Detail: The `tokens` Buffer is `torch.long`

In the current code, the `tokens` tensor is initialized as `dtype=torch.long`:
```python
tokens = pad_tensor.expand(bsz, total_len, -1).clone()  # from torch.tensor(..., dtype=torch.long)
```

This means **every value written back into `tokens` is truncated to integer**. For the original model this is fine — all compound values are integers. But for microtonal, `pitch_class = 3.5` (fractional semitone) would be truncated to `3`.

This buffer feeds BACK into the transformer at step 3a as `input_ids`, so the pitch value reaching FME would be wrong.

### 6.5.2 What Changes for Microtonal Inference

**Change 1: `tokens` buffer dtype → `float32`**
```python
# BEFORE:
pad_tensor = torch.tensor(pad_id, dtype=torch.long, device="cuda")
# AFTER:
pad_tensor = torch.tensor(pad_id, dtype=torch.float32, device="cuda")
```
The instrument column will need an explicit `.long()` cast in `embed_tokens()` before `WordEmbedding()`. Currently there is NO `.long()` cast (line 1437: `instruments = self.instrument_embedding(input_ids_tmp[..., 4])`) — it only works because `input_ids` happens to arrive as `torch.long`. With float32 tokens, this will crash. Fix: add `.long()` at line 1437.

**Change 2: `convert_from_language_tokens()` must return fractional semitones**

Currently, `pitch_dict_decode` maps language IDs → integer pitch classes (0-11). After CPT with microtonal vocab, the decode dict will map new language IDs → cent offsets (0-1199 or 0-119 for 10-cent). We need an additional conversion step:

```python
# In convert_from_language_tokens(), after looking up raw pitch:
# Original: pitch = self.pitch_dict_decode[x[3].item()]  → e.g., 3 (integer semitone)
# Microtonal: pitch = self.pitch_dict_decode[x[3].item()]  → e.g., 350 (cents)
# Must convert: pitch_fractional = pitch / 100.0  → 3.5 (fractional semitone)
```

**BUT** `convert_from_language_tokens()` currently creates a `torch.tensor(out)` — if pitch values are floats while everything else is int, torch auto-promotes the whole tensor to float. This is actually fine since the tensor gets cast `.to(tokens)` which will now be float32.

**Change 3: `compound_to_midi()` must output pitchbend**

The current `compound_to_midi()` does `note = octave*12 + pitch_class` → integer MIDI note. For microtonal:

```python
# Fractional semitone → MIDI note + pitchbend
pitch_fractional = octave * 12 + pitch_class_fractional  # e.g., 4*12 + 3.5 = 51.5
midi_note = int(round(pitch_fractional))                  # nearest semitone: 52
bend_cents = (pitch_fractional - midi_note) * 100         # deviation: -50 cents
bend_value = int(bend_cents * 8192 / (bend_sensitivity_cents))  # MIDI pitchbend units

# Before note_on, insert pitchwheel message:
track.append(mido.Message('pitchwheel', channel=idx, pitch=bend_value, time=...))
track.append(mido.Message('note_on', note=midi_note, channel=idx, velocity=vel, time=0))
```

This is essentially the reverse of the `feat/mtok_vocab` pitchbend parsing in `midi_to_compound()`.

**Change 4: Constrained sampling must use the expanded dicts**

The GRU sampling loop iterates over `["timeshift_dict_decode", "duration_dict_decode", "octave_dict_decode", "pitch_dict_decode", ...]` and uses `sample_indices = list(getattr(self.tokenizer, attribute).keys())` to constrain sampling. After vocab expansion, `pitch_dict_decode` will have the new microtonal language IDs, so constrained sampling naturally includes them — no code change needed here.

### 6.5.3 Music Infilling (from the paper)

The Moonbeam paper (§3.3) describes **music infilling** as a sub-task of conditional generation:

> "One sub-task within this broader definition is music infilling, where the model learns to generate the missing parts or 'fills' x, given an incomplete composition c₁,...cⱼ with j events."

The infilling condition `c` uses **the same compound token format** as the music sequence. Since `c` shares absolute onsets with `x`, MRA naturally computes relative distances between generated events and context events in all dimensions. The architecture is:

```
[metadata_1 ... metadata_i] <soc> [c₁ ... cⱼ] <eoc> <sos> [x₁ ... xₜ] <eos>
```

For microtonal infilling, the context events `c` would also use fractional semitones in their pitch field. Since the entire system (FME, MRA, GRU) is adapted for microtonal in CPT, infilling would work out-of-the-box after the conditional generation finetuning in Phase 2 (Section 9).

### 6.5.4 Summary: Inference Changes Checklist

| Change | Where | Complexity |
|---|---|---|
| `tokens` buffer → `float32` | `generation.py` → `generate()` | 1 line |
| `convert_from_language_tokens()` → return fractional semitones for pitch | `music_tokenizer.py` | ~5 lines |
| `compound_to_midi()` → emit pitchbend messages | `music_tokenizer.py` | ~15 lines |
| `instrument` column `.long()` cast before `WordEmbedding` | `modeling_llama.py` (already needed for training) | 1 line |

Total: ~22 lines of changes. The inference pipeline is structurally identical to the original — we're just changing the dtype and adding a conversion step.

---

## 7. Impact Assessment — What Actually Matters

| Strategy | Impact | Why |
|---|---|---|
| **Fractional semitones (not cents)** | **CRITICAL** | Without this, ALL pretrained FME + MRA knowledge is destroyed. This is the core design decision. |
| **Append-only vocab (not reshuffled)** | **CRITICAL** | Without this, ALL pretrained GRU decoder knowledge is destroyed. Every language ID changes. |
| **L_anchor regularization** | **HIGH** | With only ~1.16M notes vs 18B pretrained, western embeddings will drift. L_anchor is essential alongside data mixing. |
| **Data mixing (α ≈ 0.7-0.9)** | **HIGH** | Prevents catastrophic forgetting. Small dataset makes this especially important. |
| **Per-parameter learning rates** | **MEDIUM** | New rows (8487+) need faster learning than pretrained rows. Practical impact on convergence speed. |
| **L_smooth regularization** | **HIGH** | Addresses empty bins — propagates gradients to untrained neighbors. During inference, ghost bins cause **error cascades** through the autoregressive loop (GRU ghost token → bad fractional semitone → bad transformer hidden state → next GRU also wrong). L_smooth is the main defense. |
| **Resolution choice (1 vs 10 cent)** | **MEDIUM** | 10-cent dramatically reduces empty bins and decoder size with minimal precision loss. Worth investigating. |
| **L_center regularization** | **LOW** | Overly constraining — microtonal pitches should be free to move away from western anchors. Skip or decay fast. |
| **Initialization choice (interp vs copy)** | **LOW** | Both start close enough to pretrained manifold. Model overwrites initial values quickly during CPT. |
| **Microtonal-axis initialization** | **LOW** | Added complexity, minimal benefit since FME already captures continuous pitch geometry on input side. |

### 7.1 Paper Contribution: Coupled Autoregressive Architecture Analysis

A key methodological contribution of this work is the analysis of Moonbeam's **coupled transformer-GRU autoregressive loop** and its implications for continual pretraining:

> We identify that Moonbeam's causal transformer and GRU decoder are not independent paths — during inference, GRU output is converted back to compound tokens and fed as the next transformer input, forming a coupled autoregressive loop. This coupling constrains microtonal pitch representation: (1) resolution must be consistent across both paths, (2) errors in the GRU decoder cascade through the transformer, and (3) the gap between teacher-forced and autoregressive accuracy directly measures cascade severity. We propose L_smooth regularization and 10-cent resolution as defenses, validated by the TF-AR accuracy gap metric.

This insight is non-trivial because Moonbeam's architecture (transformer + GRU decoder) superficially resembles an encoder-decoder model, but the causal transformer with KV cache + autoregressive feedback loop makes it fundamentally different.

---

## 8. Evaluation Protocol

### 8.1 Tokenization Fidelity (tokenizer-level, pre-training)

Only relevant for GMM zones (TODO 11) or coarse quantization. For uniform grid:
- **1-cent**: RMSE ≤ 0.5 cent (trivial)
- **10-cent**: RMSE ≤ 5 cents (still well within perceptual thresholds)

Report the **fraction of saturated bends** (events near ±200 cents with R=2) and RMSE separately on saturated vs non-saturated subsets.

### 8.2 Modeling Quality on Microtonal Data (main metric)

**Next-token perplexity on held-out SymbTr split.** This is the primary metric — does the model learn to predict Turkish makam music well?

**Per-GRU-step accuracy** (leveraging Moonbeam's architecture):
```python
# For each of the 7 GRU decode steps, report accuracy separately:
# Step 0: sos_out    (should be trivially high)
# Step 1: timeshift  (temporal structure)
# Step 2: duration   (rhythm)
# Step 3: octave     (register)
# Step 4: pitch      ← THE KEY METRIC for microtonal CPT
# Step 5: instrument (should stay high — all piano in SymbTr)
# Step 6: velocity   (dynamics)
```

Pitch accuracy (step 4) is the most informative — it directly measures whether the GRU learned the expanded microtonal vocabulary. Report this broken down:
- **Western pitch accuracy**: accuracy on events where pitch falls on a western semitone (0, 100, 200, ..., 1100 cents)
- **Microtonal pitch accuracy**: accuracy on events with non-western cent values
- This split reveals whether the model learned microtonal pitches WITHOUT forgetting western ones

### 8.2.1 Autoregressive vs Teacher-Forced Accuracy Gap (Error Cascade Metric)

Because Moonbeam's inference loop is coupled (GRU output → transformer input, see Section 3.0.1), errors compound during autoregressive generation but NOT during teacher-forced evaluation. The gap between these two metrics directly measures cascade severity:

```python
# 1. Teacher-forced accuracy (standard eval — no coupling):
#    Feed ground-truth input_ids to transformer, measure GRU pitch accuracy
tf_pitch_acc = eval_teacher_forced(model, test_set)  # e.g., 85%

# 2. Autoregressive accuracy (real generation — full coupling):
#    Generate from prompt, compare to ground-truth continuation
ar_pitch_acc = eval_autoregressive(model, test_set, prompt_len=16)  # e.g., 72%

# 3. The gap reveals coupling-induced error accumulation:
cascade_gap = tf_pitch_acc - ar_pitch_acc  # e.g., 13%

# 4. Accuracy vs sequence position (does it degrade over time?):
for pos in [1, 5, 10, 20, 50]:
    acc_at_pos = eval_autoregressive_at_position(model, test_set, gen_position=pos)
    # Expected: accuracy degrades as generation gets longer (errors compound)
```

**Why this matters:** A small gap means the model's GRU predictions are reliable enough that feeding them back doesn't hurt — ghost bins aren't causing cascades. A large gap signals that L_smooth / resolution / augmentation need strengthening.

**This is a paper-worthy metric** — it reveals something specific to Moonbeam's coupled architecture that standard perplexity doesn't capture.

### 8.3 Western Performance — Catastrophic Forgetting

**Perplexity comparison:** Evaluate pretrained Moonbeam AND CPT model on the same **western validation set** (subset of Lakh Dataset held out during CPT). Report:
- Overall perplexity change (Δ PPL)
- Per-attribute perplexity (especially pitch — should stay stable for p ∈ {0,...,11})

**Downstream task benchmarks from the Moonbeam paper:**
- Player classification on PiJAMA30 and Pianist8
- Emotion classification on Emopia
- Composer classification on GPM30

Running the CPT model on these SAME benchmarks quantifies forgetting on actual tasks, not just perplexity. This is stronger evidence and directly comparable to the Moonbeam paper's Table 2.

**Pitch-class confusion matrix:** For western validation data, compare pitch prediction confusion matrices before vs after CPT. If L_anchor works, the 12 western pitch classes should show nearly identical confusion patterns.

### 8.4 Generated Music Analysis

**Ghost bin rate (empty bins problem — inference only):**
- Generate N pieces (e.g., 100-500) autoregressively
- Count the fraction of generated pitch tokens that fall in cent bins with ZERO training examples
- High ghost bin rate → the GRU is sampling untrained tokens → causes autoregressive error cascades (Section 5.2.1) → motivates tighter resolution, stronger L_smooth, or GMM zones
- Note: ghost bins are harmless during training (teacher forcing breaks the loop), but critical during generation

**Generated pitch distribution vs corpus:**
- For each pitch class, plot histogram of generated cent offsets vs SymbTr training distribution
- Compute **Wasserstein distance** (earth mover's distance) per pitch class between generated and corpus distributions
- A well-trained model should produce cent distributions that approximate the corpus

**Interval distribution:**
- Compare intervals (consecutive pitch differences in cents) in generated vs corpus music
- Turkish makam has characteristic interval patterns (Holdrian comma multiples) that differ from western
- Good microtonal generation should reproduce these interval statistics

### 8.5 Makam-Specific Musical Evaluation

**Scale degree adherence:**
- SymbTr metadata includes makam labels (e.g., Hicaz, Rast, Hüseyni)
- For each generated piece (conditioned on a makam or evaluated post-hoc), identify the closest makam scale
- Measure what % of generated pitches fall on valid scale degrees for that makam (within ±10 cents tolerance)
- This tests whether the model learned musically meaningful pitch organization, not just statistical patterns

### 8.6 Embedding Space Visualization (qualitative)

**t-SNE/UMAP of `decoder_embedding` pitch rows:**
- Pretrained: 12 western pitch points
- After CPT: 12 western + N microtonal points
- Verify: (a) western anchors haven't moved far (L_anchor), (b) microtonal variants cluster near their western parent, (c) nearby cents have nearby embeddings (L_smooth)
- Color by pitch class, annotate makam scale degrees
- This is a compelling paper figure and directly validates the regularization strategy

### 8.7 Ablation Table Design

| Configuration | SymbTr PPL | Pitch Acc (micro) | Pitch Acc (western) | Western PPL Δ | Ghost Bin % | TF-AR Gap |
|---|---|---|---|---|---|---|
| Moonbeam pretrained (no CPT) | — | — | baseline | 0 | — | — |
| CPT, no regularization | ? | ? | ? | ? | ? | ? |
| CPT + L_anchor only | ? | ? | ? | ? | ? | ? |
| CPT + L_anchor + L_smooth | ? | ? | ? | ? | ? | ? |
| CPT + L_anchor + L_smooth + data mixing | ? | ? | ? | ? | ? | ? |
| CPT + L_anchor + L_smooth + data mixing + 53-TET aug | ? | ? | ? | ? | ? | ? |
| CPT, 10-cent resolution | ? | ? | ? | ? | ? | ? |
| CPT, 1-cent resolution | ? | ? | ? | ? | ? | ? |
| CPT, interpolation init | ? | ? | ? | ? | ? | ? |
| CPT, copy+noise init | ? | ? | ? | ? | ? | ? |

**TF-AR Gap** = teacher-forced pitch accuracy minus autoregressive pitch accuracy (Section 8.2.1). Measures coupling-induced error cascades. Small gap = ghost bins aren't causing problems. Large gap = need stronger L_smooth / coarser resolution / more augmentation.

Rows can be combined — the key ablation axes are: (1) regularization, (2) resolution, (3) initialization. Data mixing and L_anchor should be in all non-baseline runs.

### 8.8 Seyir (Melodic Progression) Analysis

Turkish makam melodies follow characteristic melodic movement patterns called **seyir**: ascending (çıkıcı), descending (inici), or ascending-descending (inici-çıkıcı). Each makam has a prescribed seyir type.

- Generate pieces (unconditionally or conditioned on makam if Phase 2 is available)
- Extract the overall pitch contour: fit a linear regression to the pitch-vs-time curve, or compute the centroid pitch in each quarter of the piece
- Classify the generated seyir as ascending/descending/mixed
- Compare to the known seyir for each makam (available in musicological references)
- Report **seyir accuracy**: % of generated pieces whose contour matches the expected seyir type

This tests whether the model learned makam-specific **melodic grammar**, not just pitch distributions.

### 8.9 Karar (Final/Resting Note) Accuracy

Each makam has a **karar** (tonic/finalis) — the note on which melodies typically resolve. For example, Rast resolves on C, Hicaz on A.

- For each generated piece, extract the final pitched note (ignoring silence/rests)
- Compute the distance in cents from the nearest expected karar for the inferred makam
- Report: (a) **karar accuracy** = % of pieces where final note is within ±25 cents of correct karar, (b) **mean karar error** in cents
- This is a simple, single-number-per-piece metric that directly measures tonal coherence

### 8.10 Pitch Stability / Jitter Metric

A well-trained model should produce **stable** pitch categories — the same scale degree should appear at a consistent cent value throughout a piece.

- For each generated piece, identify repeated pitch classes (e.g., all instances of ~E-50c)
- Cluster pitches by proximity (±15 cent threshold) to group instances of the same scale degree
- Compute the **intra-cluster standard deviation** in cents for each cluster
- Report the mean across all clusters and pieces
- Low jitter (< 5-8 cents) → model internalized stable pitch categories; high jitter → unstable representations

### 8.11 Per-Makam Perplexity Breakdown

- Report held-out perplexity **per makam** (not just aggregate over all SymbTr)
- SymbTr has ~164 makams with uneven distribution. Common makams (Hicaz, Rast, Hüseyni, ~100+ pieces each) should have lower perplexity; rare makams (~5-10 pieces) test generalization
- Plot perplexity vs. training set size per makam — expect a power-law relationship
- This reveals whether the model generalizes across the makam system or just memorizes frequent patterns

### 8.12 Naive Quantization Baseline

Compare against a **trivial baseline**: round all microtonal pitches to the nearest western semitone (no CPT, no vocab expansion).

- Train: fine-tune unmodified Moonbeam on SymbTr with all pitches snapped to nearest 12TET
- Eval: report SymbTr perplexity, pitch accuracy, interval distribution, karar accuracy
- This quantifies **what microtonal CPT adds** over naive treatment of Turkish music as western. If the naive baseline performs well on everything except pitch precision, it justifies the complexity of our approach. If it performs poorly across the board, it shows that pitch microtonality affects the entire modeling pipeline.

This baseline is cheap (no architecture changes, just data preprocessing) and essential for a convincing paper.

---

## 9. Future Work: Makam-Conditional Generation

### 9.0 How Moonbeam's Conditional Generation Works

Moonbeam has a **unified conditional generation framework** (paper §3.3, finetuned on CoMMU dataset). The input sequence is:

```
[metadata_1 ... metadata_i] <soc> [chord_1 ... chord_j] <eoc> <sos> [music_1 ... music_t] <eos>
 ↑ non-temporal conditions      ↑ temporal conditions (chords)        ↑ actual music to generate
```

**Metadata conditions** (non-temporal): prepended as special compound tokens using **negative token IDs** (e.g., -8 for key=A, -94 for velocity range). Each metadata token is a 6-dim compound token with a `metadata_tokens_pos` entry that provides position values for MRA. Conditions include: audio_key, pitch_range, num_measures, bpm, genre, track_role, instrument, rhythm, time_signature, min/max_velocity.

**Chord conditions** (temporal): tokenized as compound tokens between `<soc>` and `<eoc>`. Chords share the same absolute onset timeline as the music, so MRA naturally computes relative onset distances between music events and chord events. Chords are also fed to the GRU decoder via a separate feature extractor (`if_add_chord_in_decoder: true`).

**Key code locations (on `main` branch):**
- Config: `model_config_commu_con_gen.json` — adds `soc_token=-4`, `eoc_token=-5`, ~330 metadata tokens (-4 to -333), chord dict (62 chord types), `if_add_metadata_in_decoder`, `if_add_chord_in_decoder`
- Dataset: `commu_con_gen_dataset.py` — loads CoMMU data, prepends metadata + chord tokens
- Inference: `recipes/inference/custom_music_generation/conditional_music_generation_batch.py` — batch generation with conditions
- Tokenizer: `encode_series_con_gen_commu()` — constructs the full conditioned sequence

### 9.1 Phase 2: Makam as a Metadata Condition (simplest extension)

Add makam labels as new metadata tokens. SymbTr has ~50+ makam types (Hicaz, Rast, Hüseyni, Nihavend, Segah, etc.).

```python
# New metadata tokens (negative IDs, continuing from CoMMU's -333):
makam_tokens = {
    "makam_Hicaz": -334,
    "makam_Rast": -335,
    "makam_Huseyni": -336,
    ...  # one per makam in SymbTr (~50 makams)
}

# Each gets a metadata_tokens_pos entry (6-dim position vector for MRA):
"metadata_tokens_pos": {
    "-334": [0, 0, 0, 0, 0, 0],  # makam_Hicaz
    "-335": [0, 0, 0, 0, 0, 0],  # makam_Rast
    ...
}
```

**Training sequence:**
```
[makam_Hicaz] <sos> [note_1 ... note_t] <eos>
```

**What this enables:**
- "Generate a piece in makam Hicaz" → model generates music with correct Hicaz scale degrees
- Makam-specific evaluation: condition on makam, measure scale degree adherence
- Comparative analysis: does conditioning improve pitch accuracy vs unconditional generation?

**Implementation:**
1. Create `makam_con_gen_dataset.py` based on `commu_con_gen_dataset.py`
2. Add makam tokens to config (new negative IDs)
3. Parse SymbTr metadata for makam labels (available in filenames: e.g., `hicaz--sazsemaisi--aksaksemai----tanburi_cemil_bey`)
4. Use `encode_series_con_gen_commu()` to prepend makam tokens
5. Finetune with existing conditional generation framework

**Pros:** Minimal code changes — reuses Moonbeam's conditioning infrastructure entirely. Just new tokens and a dataset class.

**Cons:** The model only sees a categorical label. It doesn't know the scale structure of each makam (which specific pitches are valid). It must learn this implicitly from the data.

### 9.2 Phase 3: Makam Scale as Temporal Condition (more powerful)

A makam scale defines a set of allowed pitches with specific cent offsets. We could encode the scale structure as a "condition sequence" similar to how chords work:

```
[makam_Hicaz] <soc> [scale_degree_1 ... scale_degree_N] <eoc> <sos> [music ...] <eos>
```

Where each scale degree is a compound token with:
- onset = 0 (non-temporal, or could encode scale degree order)
- pitch = fractional semitone value of that scale degree
- octave = reference octave

**Why this is powerful:** MRA would compute relative pitch distances between generated notes and scale degrees. The model could "see" which scale pitches are nearby — a direct inductive bias for makam-correct generation.

**Additionally: Usul (rhythmic cycle) as temporal condition.** Turkish makam music has usul (rhythmic cycles, more complex than time signatures). SymbTr metadata includes usul labels. These could be encoded as repeating accent patterns between `<soc>`/`<eoc>`, similar to chord progressions.

### 9.3 Recommended Phasing

| Phase | Task | Complexity | Dependency |
|---|---|---|---|
| **Phase 1** (current) | Microtonal CPT — unconditional generation | Core work | None |
| **Phase 2** | Makam as metadata condition | Low — reuse existing framework | Phase 1 complete |
| **Phase 3a** | Makam scale as temporal condition | Medium — new encoding scheme | Phase 2 validated |
| **Phase 3b** | Usul as temporal condition | Medium — rhythmic encoding | Phase 2 validated |
| **Phase 4** | Joint makam + usul conditioning | High — full system | Phases 3a + 3b |

Phase 1 is our current focus. Phase 2 could be a quick follow-up experiment once CPT works. Phases 3+ would likely be a separate paper.

---

## 10. Open Questions

1. **Resolution:** Start with 10-cent (recommended) or 1-cent? See Section 5.2 for comparison. 10-cent is the pragmatic choice; 1-cent or GMM zones (TODO 11) if higher precision is needed.
2. **Mixed training ratio α:** How much western replay data? α ≈ 0.7-0.9 microtonal is the suggested range. Need to tune based on forgetting metrics.
3. **LoRA rank:** What rank for pitch attention heads (group 3) vs other heads? Higher rank on pitch heads could help.
4. **Full finetuning vs LoRA:** Given only 3000 files, LoRA is safer, but the decoder MUST be fully trained. Consider LoRA on transformer + full training on decoder.
5. **Regularization weights:** Suggested ranges: λ_a ≈ 0.1-1.0 (anchor), λ_s ≈ 0.01-0.1 (smoothness). Tune via validation loss.
6. **GMM zones (TODO 11):** If pursuing this, how to handle padded zones where some (pc,k) combos are unused? Only relevant if we move beyond uniform grid.
7. **Makam conditioning:** Should we condition generation on makam labels (available in SymbTr metadata) using Moonbeam's conditional generation framework? This could improve scale degree adherence and enable makam-specific evaluation.
