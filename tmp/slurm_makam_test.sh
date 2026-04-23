#!/bin/bash

# --- SLURM Resource Directives ---
#SBATCH -J makam_classification_test
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

# --- Output Logging ---
#SBATCH -o /slurm/out/slurm.%N.%j.out
#SBATCH -e /slurm/out/slurm.%N.%j.err

# --- Matplotlib workaround ---
export MPLCONFIGDIR=/tmp/$USER/matplotlib
mkdir -p $MPLCONFIGDIR

# Ensure conda env binaries are on PATH
export PATH=/home/ywu/.conda/envs/moonbeam/bin:$PATH

# --- The Command ---
/home/ywu/.conda/envs/moonbeam/bin/torchrun --nnodes 1 --nproc_per_node 1 recipes/finetuning/real_finetuning_makam_classification.py \
  --dataset symbtr_dataset_eval \
  --trained_checkpoint_path models/moonbeam_309M.pt \
  --output_dir checkpoints/finetuned_checkpoints/makam_test_run \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --num_epochs 1 \
  --validation_interval 200 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --gamma 0.9 \
  --gradient_accumulation_steps 1 \
  --enable_lora true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --run_validation False \
  --save_model False \
  --save_metrics False \
  --use_peft True \
  --peft_method lora \
  --model_name makam_classification \
  --context_length 1203 \
  --use_wandb False \
  --use_cache False \
  --enable_ddp False \
  --pure_bf16 False
