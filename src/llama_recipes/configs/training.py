# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    # -------------------------------------------------------------
    # Hyperparameter guidance:
    #
    # MOONBEAM PRETRAINING FROM SCRATCH (paper Section 4.1, Guo & Dixon 2025):
    #   lr=3e-4, gamma=0.85 (StepLR per epoch), Adam, weight_decay=0.0,
    #   context_length=1024, mixed precision (fp16), DDP on 2x A100,
    #   <9 epochs, batch packing.
    #
    # MICROTONAL CONTINUAL PRETRAINING (CPT) — recommended:
    #   lr=1e-5 to 3e-5  (10-30x lower than pretraining to preserve knowledge)
    #   weight_decay=0.01 (mild regularization)
    #   gamma=0.85        (keep same decay schedule)
    #   gradient_clipping=True, threshold=1.0  (stabilize early training)
    #   num_epochs=20-30  (small dataset needs many passes; use early stopping patience=5)
    #   context_length=1024 (MUST match pretrained model)
    #   mixed_precision=True, use_fp16=True (same as pretraining)
    # -------------------------------------------------------------
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None
    enable_fsdp: bool=False
    enable_ddp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    validation_interval: int=200
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=1024  # Must match pretrained model (1024 for Moonbeam S/M)
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False  # Set True for CPT (stabilizes early steps)
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4  # Pretraining default. For CPT use 1e-5 to 3e-5.
    weight_decay: float=0.0  # Pretraining default. For CPT use 0.01.
    gamma: float= 0.85  # LR decay per epoch (same for pretraining and CPT)
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    trained_checkpoint_path: str = "PATH/to/saved/trained/model"
    base_checkpoint_path: str = "models/moonbeam_309M.pt"  # Used when trained_checkpoint_path points to a PEFT adapter directory
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    pure_bf16: bool = False  # Convert model to bf16 (single-GPU; FSDP/DDP use their own configs)
    use_wandb: bool = False # Enable wandb for experient tracking
    wandb_run_name: str = "ablation"  # WandB run title
    wandb_tags: str = "309M"  # Comma-separated WandB tags
    log_interval: int = 10  # Log to wandb every N training steps (reduces noise)
    enable_pitch_confusion: bool = False  # Compute/save pitch confusion artifacts during eval
    pitch_confusion_dir: str = ""  # Defaults to output_dir/pitch_confusion when empty
    pitch_confusion_max_eval_step: int = 0  # 0 = use all eval batches
    pitch_confusion_save_plots: bool = True  # Save confusion PNG/SVG plots
    pitch_confusion_log_wandb: bool = True  # Log confusion images/metrics to wandb when enabled
    validation_only: bool = False  # Run evaluation pipeline only (no training loop)
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
    model_config: str = ""  # Path to model config JSON
    microtonal: bool = False  # Master switch for microtonal mode (vocab expansion/interpolation/regularization)
    micro_lambda_anchor: float = 0.5  # L-anchor strength for western row preservation
    micro_lambda_smooth: float = 0.05  # L-smooth strength for adjacent microtonal rows
    micro_warmup_freeze_ratio: float = 0.10  # Fraction of total steps to freeze western rows early
    micro_new_row_lr_scale: float = 3.0  # Gradient scaling factor for new microtonal rows
    micro_pitch_ce_weight: float = 0.0  # Optional auxiliary CE weight on pitch tokens only
    enable_micro_pitch_ce: bool = False  # Master switch to enable/disable auxiliary pitch CE loss
    micro_train_pitch_only: bool = True  # Freeze non-pitch params in microtonal mode
    western_data_dir: str = ""  # Western replay data directory (for microtonal CPT data mixing)
    western_csv_file: str = ""  # Western data CSV file path
    mixing_alpha: float = 0.2  # Fraction of western samples when mixing (0.2 = 20% western, 80% micro)
