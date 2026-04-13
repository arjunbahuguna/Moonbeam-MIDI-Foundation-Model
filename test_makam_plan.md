# Test Plan: Baseline Player Recognition to Makam Classification

## Objective
Validate two things on the local RTX 3090 setup before scaling to a larger machine:
1. The original shipped classification workflow (player recognition) can run on this computer.
2. The same validation structure can be reproduced for makam classification.

This first cycle is intentionally minimal: 3 pieces per task, end-to-end sanity, then expand.

## Scope (Current Cycle)
- Baseline source for player recognition: `origin/finetune_player_classification`.
- Makam source: current branch `finetune-makam-classification`.
- Sample size: 3 pieces for player recognition + 3 pieces for makam classification.
- Goal: pipeline correctness and reproducibility, not final benchmark quality.

## Delta Findings (Apr 2026)
- Intended player-classification entrypoint is `src/llama_recipes/real_finetuning_player_classification.py` (no separate dedicated eval script in this branch).
- Intended evaluation mode is validation inside the same entrypoint (`run_validation=True`) with piece-wise test behavior (`individual_eval=True` in dataset config).
- Current branch divergence: `train_player_classification()` in `src/llama_recipes/utils/train_utils.py` is currently a compatibility wrapper to generic `train()` (specialized player metrics path from the original branch is not guaranteed here).
- Classification token is currently defined on tokenizer setup, but not actually prepended in active dataset encoding path unless explicitly requested in tokenizer call.
- Highest-priority runtime correctness blockers for sequence classification are now data-shape related (sequence-level labels and valid attention mask), not missing entry scripts.

## Infrastructure Assumptions
- GPU: NVIDIA RTX 3090 (24 GB VRAM), single-GPU execution.
- Python environment: project `.venv`.
- Runtime mode: deterministic seed where possible.
- Data movement: keep artifacts in repo-local expected paths.

## Phase 1: Baseline Snapshot (Player Recognition)
Purpose: lock the exact reference workflow from the original task.

### 1.1 Identify baseline artifacts from `origin/finetune_player_classification`
- Training entry script.
- Dataset config/class.
- Model config.
- Inference/eval script.
- Expected checkpoint path and format.

### 1.2 Freeze baseline command set
- One canonical command for data prep (if needed).
- One canonical command for mini inference/mini validation.
- One canonical command for short finetuning sanity (optional if checkpoint-only validation is used).

Acceptance criteria:
- Baseline workflow files and commands are explicitly documented and reproducible.

## Phase 2: 3090 Runtime Guardrails
Purpose: ensure both tasks are tested under the same practical constraints.

### 2.1 Fix safe defaults
- Batch size and sequence length suitable for 24 GB VRAM.
- Single-device mode only.
- Seed and deterministic options recorded.

### 2.2 Define run outcomes to record
- Pass/fail.
- First-batch forward pass success.
- Peak VRAM estimate.
- Runtime per sample.

Acceptance criteria:
- A stable local profile exists for both workflows.

## Phase 3: Player Recognition Mini End-to-End (3 Pieces)
Purpose: verify shipped task works locally.

### 3.1 Select 3 representative player-recognition pieces
- Prefer mixed lengths and non-identical metadata.

### 3.2 Run mini pipeline
- Data readiness check.
- Dataset load check.
- One forward pass check.
- Inference output check (class id + confidence/probability format).

### 3.2.a Intended path lock (current branch)
- Use only `python -m llama_recipes.real_finetuning_player_classification` as canonical player run path for this cycle.
- Treat this as train+eval unified path (no separate eval-only script currently trusted).
- Keep mini sanity command constrained (`max_train_step=1`, `max_eval_step=1`, `batch_size_training=1`, `val_batch_size=1`, `num_workers_dataloader=0`).

### 3.2.b Pre-run correctness gates (must pass before rerun)
- Dataset emits sequence-level labels for `LlamaForSequenceClassification` (one class id per sample, not per token).
- Dataset emits a real attention mask (ones for valid tokens, zeros only for padding).
- Tokenizer player-classification encode signature matches call site (already aligned in current file state).
- Player config/ckpt architecture match is maintained (`hidden_size`, `intermediate_size`, `num_hidden_layers`).

### 3.2.c Success signal for this phase
- First train step and first validation step complete without shape/signature exceptions.
- Validation metrics are logged from the active path (loss required; F1 optional for this cycle unless specialized baseline path is restored).

### 3.3 Record observations
- Any loader/config mismatch.
- Any checkpoint key/shape mismatch.
- Any memory pressure/OOM.

Acceptance criteria:
- 3/3 samples produce valid outputs without infrastructure errors.

## Phase 4: Makam Classification Mini End-to-End Mirror (3 Pieces)
Purpose: mirror the same validation structure on makam workflow.

### 4.1 Confirm required artifacts
- `makam_classification_split.csv` exists.
- `makam_label_map.json` exists.
- `processed/` token files exist and match CSV rows.

### 4.2 Select 3 makam pieces
- Prefer different makams to test label-map decoding.

### 4.3 Run mini pipeline
- Dataset load check.
- One forward pass check.
- Inference output check using makam label map.

Acceptance criteria:
- 3/3 samples produce valid makam predictions with consistent label decoding.

## Phase 5: Equivalence and Gap Review
Purpose: compare both workflows under identical test logic.

### Compare dimensions
- Artifact readiness.
- Script correctness.
- Runtime stability.
- Output sanity.
- Reproducibility under same environment.

### Classify issues
- Blocking: prevents completion.
- Non-blocking: correctness risk but pipeline still runs.
- Cosmetic/doc: does not affect execution.

Acceptance criteria:
- Clear list of what is proven working and what remains unresolved.

## Phase 6: Handoff Package for Larger Machine
Purpose: allow another team to continue without oral context.

### Deliverable contents
- Prerequisites and environment spec.
- Exact commands for both mini workflows.
- Expected outputs and acceptance checks.
- Common failure signatures and recovery steps.
- Artifact locations and naming conventions.

### Transfer rule
- Another team should be able to run the same 3+3 validation cycle unchanged.

Acceptance criteria:
- Handoff runbook is executable by a new team with no hidden assumptions.

## Verification Checklist (Current Cycle)
- [ ] Player baseline source pinned to `origin/finetune_player_classification`.
- [ ] Canonical local player run path fixed to `python -m llama_recipes.real_finetuning_player_classification`.
- [ ] Confirmed there is no separate straightforward eval-only player script in current branch.
- [ ] 3090 runtime profile recorded (batch/seq/seed/device mode).
- [ ] Player dataset returns sequence-level labels (not token-length label vectors).
- [ ] Player dataset attention mask is valid (non-zero on active tokens).
- [ ] Player mini test passed for 3 pieces.
- [ ] Add eval-only player-classification entrypoint (no training loop) and document command for future sanity checks.
- [ ] After player sanity check, restore microtonal-compatible preprocessing in `data_preprocess.py`.
- [ ] Makam mini test passed for 3 pieces.
- [ ] CSV-to-token-file integrity confirmed for tested makam rows.
- [ ] Equivalence gap summary completed.
- [ ] Larger-machine handoff section completed.

## Notes and Risks
- Temporary branch-local change: `data_preprocess.py` is currently using the player-classification-compatible flow to run the baseline sanity check.
- Required revert trigger: once player sanity passes, switch `data_preprocess.py` back to the microtonal-compatible version before continuing makam debugging/evaluation.
- If the player-classification checkpoint is unavailable, classify result as "pipeline wiring validated, model-quality validation pending checkpoint".
- Path normalization must be consistent across scripts (especially split CSV, label map, processed token directory).
- Dependency drift (especially `accelerate` utility API changes) must be documented in handoff notes.
- If baseline-comparable F1 is required, either restore the original `finetune_player_classification` training/eval logic or port its metric path explicitly; current wrapper may only provide generic validation outputs.
