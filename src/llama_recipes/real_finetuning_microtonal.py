import os
import json
import dataclasses
import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import ddp_config as DDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset_hybrid_padding_concatenating
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.model_checkpointing import load_model_checkpoint_ddp
from llama_recipes.microtonal_utils import expand_vocab_with_interpolation

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    train_con_gen,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from accelerate.utils import is_xpu_available

def setup_wandb(train_config, fsdp_config, llama_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    
    # Project-specific identifiers
    init_dict["entity"] = "microtok"
    init_dict["project"] = "uncon_gen"

    # Add run name and tags for better organization and filtering in the wandb dashboard
    # Include config values for easy reference
    init_dict["name"] = "data_mixing" # Optional: set a custom run name for easier identification in the dashboard]
    init_dict["tags"] = ["309M", "symbtr_lmd"]

    # Initialize and configure wandb
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)

    # Convert the llama_config to a dictionary and then to a JSON string
    config_dict = llama_config.to_dict()
    config_json = json.dumps(config_dict, indent=4)
    
    # Get the wandb run directory
    from pathlib import Path
    # Define the file path within the wandb run directory
    folder_name = (train_config.dist_checkpoint_root_folder+ "/"+ train_config.dist_checkpoint_folder+ "-"+ train_config.model_name)
    save_dir = Path.cwd() / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config_file_path = os.path.join(save_dir, 'llama_config.json')

    # Write the JSON string to the file
    with open(config_file_path, 'w') as f:
        f.write(config_json)
        print(f"config file saved to {config_file_path}!")
    return run


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, ddp_config = TRAIN_CONFIG(), FSDP_CONFIG(), DDP_CONFIG()
    update_config((train_config, fsdp_config, ddp_config), **kwargs)
    model_config_path = train_config.model_config
    print("updated training config", train_config)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp or train_config.enable_ddp:
        setup() #enable nccl / ccl
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        # FSDP low-CPU path requires vocab expansion on rank 0, meta-device init on
        # other ranks, and sync_module_states — not yet adapted for microtonal
        raise NotImplementedError(
            "FSDP low_cpu_fsdp is not supported for microtonal CPT. "
            "Use DDP or single-GPU training."
        )

    else: #DDP and non-distributed training
        llama_config = LlamaConfig.from_pretrained(model_config_path)
        llama_config.use_cache = use_cache
        print(f"model_config:{llama_config}")

        # Create tokenizer early — needed for vocab expansion
        tokenizer = MusicTokenizer(
            timeshift_vocab_size=llama_config.onset_vocab_size,
            dur_vocab_size=llama_config.dur_vocab_size,
            octave_vocab_size=llama_config.octave_vocab_size,
            pitch_class_vocab_size=llama_config.pitch_class_vocab_size,
            instrument_vocab_size=llama_config.instrument_vocab_size,
            velocity_vocab_size=llama_config.velocity_vocab_size,
            sos_token=llama_config.sos_token,
            eos_token=llama_config.eos_token,
            pad_token=llama_config.pad_token,
            microtonal=getattr(llama_config, 'microtonal', False),
            pitchbend_sensitivity=getattr(llama_config, 'pitchbend_sensitivity', 2.0),
            microtonal_resolution=getattr(llama_config, 'microtonal_resolution', 1),
        )

        model = LlamaForCausalLM(llama_config)

        model_checkpoint = torch.load(train_config.trained_checkpoint_path)

        checkpoint = model_checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'): # Check if the keys have 'module.' prefix and remove it if necessary
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Expand decoder_embedding and lm_head for microtonal vocab
        expanded_state_dict = expand_vocab_with_interpolation(
            new_state_dict, llama_config, tokenizer
        )
        print(f"Vocab expanded: {len(new_state_dict)} keys, decoder_embedding {new_state_dict['decoder_embedding.weight'].shape} -> {expanded_state_dict['decoder_embedding.weight'].shape}")

        # Load the state_dict into the model, ignoring unmatched keys
        missing_keys, unexpected_keys = model.load_state_dict(expanded_state_dict, strict=False)
        print(f"when loading checkpoint, encounter missing keys: {missing_keys}; unexpected_keys:{unexpected_keys}")

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank==0:
            wandb_run = setup_wandb(train_config, fsdp_config, llama_config, **kwargs)


    dataset_config = generate_dataset_config(train_config, kwargs)

    print_model_size(model, train_config, rank if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare microtonal embedding regularization:
    #   L_anchor — MSE penalty preventing the 12 pretrained western pitch embeddings
    #              (rows pitch_offset..pitch_offset+11 in decoder_embedding/lm_head) from drifting
    #   L_smooth — encourages adjacent cent-resolution pitch bins to have similar embeddings
    microtonal_reg = None
    if getattr(llama_config, 'microtonal', False):
        # Direct references to the Parameter tensors — these survive PEFT/DDP wrapping
        # because wrappers delegate to the same underlying nn.Parameter objects.
        decoder_emb_weight = model.decoder_embedding.weight
        lm_head_weight = model.lm_head.weight

        # Western pitch rows in the flat GRU vocab: pitch_offset + pitch_class (0-11)
        pitch_offset = tokenizer.pitch_offset
        original_decode_vocab = tokenizer.original_decode_vocab
        western_ids = list(range(pitch_offset, pitch_offset + 12))

        # Snapshot of pretrained western embeddings (anchor targets, never updated)
        frozen_decoder_emb = decoder_emb_weight[western_ids].detach().clone()
        frozen_lm_head = lm_head_weight[western_ids].detach().clone()

        # Build index tensors for vectorized L_smooth computation.
        # Each pair (left[i], right[i]) are adjacent cent bins within 10 cents of each other
        sorted_pitches = sorted(
            [(c, lid) for c, lid in tokenizer.pitch_dict.items() if lid >= original_decode_vocab],
            key=lambda x: x[0]
        )
        left_ids, right_ids = [], []
        for i in range(len(sorted_pitches) - 1):
            c1, id1 = sorted_pitches[i]
            c2, id2 = sorted_pitches[i + 1]
            if c2 - c1 <= 10:
                left_ids.append(id1)
                right_ids.append(id2)

        # Regularization weights:
        #   lambda_anchor: MSE penalty on western pitch embeddings
        #     drifting from pretrained values. Tune in [0.1, 1.0]. Higher = more
        #     conservative (less forgetting, slower microtonal adaptation).
        #   lambda_smooth: encourages adjacent cent bins to have similar embeddings.
        #     Tune in [0.01, 0.1]. Higher = smoother embedding space, but may reduce pitch discrimination.
        #   warmup_freeze_steps: computed after dataloader creation (10% of total steps).
        #     During warmup, western pitch row gradients are zeroed so new microtonal rows can catch up from interpolation init.
        microtonal_reg = {
            'decoder_emb_weight': decoder_emb_weight,
            'lm_head_weight': lm_head_weight,
            'western_ids': western_ids,
            'frozen_decoder_emb': frozen_decoder_emb,
            'frozen_lm_head': frozen_lm_head,
            'micro_pair_ids_left': torch.tensor(left_ids, dtype=torch.long),
            'micro_pair_ids_right': torch.tensor(right_ids, dtype=torch.long),
            'lambda_anchor': 0.5,   # recommended range: [0.1, 1.0]
            'lambda_smooth': 0.05,  # recommended range: [0.01, 0.1]
            'original_decode_vocab': original_decode_vocab,
        }
        print(f"Microtonal regularization: {len(western_ids)} western anchors, "
              f"{len(left_ids)} adjacent-bin pairs for smoothness")

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    elif train_config.enable_ddp and ddp_config.pure_bf16:
        model.to(torch.bfloat16)
    elif getattr(train_config, 'pure_bf16', False):
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        # For microtonal CPT, decoder_embedding and lm_head must be fully trained
        # (not LoRA-wrapped) because they contain new vocab rows that need to learn
        # from scratch. Remove them from LoRA targets to avoid double-training
        if getattr(llama_config, 'microtonal', False):
            attention_only = ["q_proj", "v_proj", "k_proj", "o_proj"]
            peft_config.target_modules = attention_only
            print(f"Microtonal CPT: LoRA target_modules overridden to {attention_only}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if wandb_run:
            wandb_run.config.update(peft_config)

    # When using PEFT, non-LoRA params are frozen by default. For microtonal CPT we need
    # the GRU decoder head (decoder_embedding, lm_head, GRU layers) to remain trainable
    # so the model can learn the expanded pitch vocabulary.
    if getattr(llama_config, 'microtonal', False):
        for name, param in model.named_parameters():
            if any(k in name for k in ['decoder_embedding', 'lm_head', 'decoder.']):
                param.requires_grad = True

    hsdp_device_mesh = None 
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer) #TODO: dangerous

        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model) 
    elif train_config.enable_ddp: #wrap ddp code
        mixed_precision_policy, wrapping_policy = get_policies(ddp_config, rank)
        model.to(local_rank)
        model = DDP(model,
                    mixed_precision=mixed_precision_policy if not ddp_config.pure_bf16 else None, 
                    device_mesh=hsdp_device_mesh,
                    device_ids=[local_rank],
                    find_unused_parameters=False,
                    )
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")


    # Move regularization tensors to the same device as the model (once, not per step)
    if microtonal_reg is not None:
        dev = next(model.parameters()).device
        microtonal_reg['frozen_decoder_emb'] = microtonal_reg['frozen_decoder_emb'].to(dev)
        microtonal_reg['frozen_lm_head'] = microtonal_reg['frozen_lm_head'].to(dev)
        microtonal_reg['micro_pair_ids_left'] = microtonal_reg['micro_pair_ids_left'].to(dev)
        microtonal_reg['micro_pair_ids_right'] = microtonal_reg['micro_pair_ids_right'].to(dev)

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    # Load western replay data for CPT data mixing (5-30% replay prevents catastrophic forgetting)
    western_train = None
    mixing_weights = None
    if train_config.western_data_dir:
        if train_config.batching_strategy != "packing":
            raise ValueError(
                "Data mixing (western_data_dir) requires batching_strategy='packing'"
            )
        from types import SimpleNamespace
        from llama_recipes.datasets.lakh_dataset import LakhDataset
        western_config = SimpleNamespace(
            data_dir=train_config.western_data_dir,
            csv_file=train_config.western_csv_file,
        )

        # Only LakhDataset is currently supported for western replay mixing
        # Can add more as needed
        western_train = LakhDataset(western_config, tokenizer, partition="train")
        if not train_config.enable_fsdp or rank == 0:
            print(f"--> Western Replay Set Length = {len(western_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset_hybrid_padding_concatenating(dataset_train, chunk_size=train_config.context_length, split="train",data_dir = dataset_config.data_dir)

        # Pack datasets separately so each chunk is homogeneous (all-micro or
        # all-western). This is equivalent to batch-level mixing: each training
        # batch contains chunks from both domains, and the block-diagonal
        # attention mask ensures pieces within a chunk don't cross-attend 
        # regardless of domain
        if western_train is not None:
            western_train_packed = ConcatDataset_hybrid_padding_concatenating(
                western_train, chunk_size=train_config.context_length,
                split="train", data_dir=train_config.western_data_dir,
            )
            n_micro = len(dataset_train)
            n_western = len(western_train_packed)
            alpha = train_config.mixing_alpha
            # Weights sum to 1.0; each micro chunk gets alpha/n_micro,
            # each western chunk gets (1-alpha)/n_western
            mixing_weights = (
                [alpha / n_micro] * n_micro
                + [(1.0 - alpha) / n_western] * n_western
            )
            dataset_train = torch.utils.data.ConcatDataset(
                [dataset_train, western_train_packed]
            )
            if not train_config.enable_fsdp or rank == 0:
                print(
                    f"--> Mixed training: {n_micro} micro chunks + "
                    f"{n_western} western chunks, alpha={alpha}"
                )

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Override sampler for ratio-controlled mixing. Both packers start
    # sample_count=1, so attention_mask IDs overlap across datasets, this is
    # safe because _update_causal_mask compares IDs per batch element, not
    # across batch elements. For FSDP/DDP, DistributedSampler is kept
    # (uniform over combined dataset); exact ratio needs a custom sampler
    if mixing_weights is not None:
        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank == 0:
                print(
                    "WARNING: Data mixing with FSDP/DDP uses uniform sampling "
                    "over the combined dataset. mixing_alpha is approximate."
                )
        else:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                mixing_weights, num_samples=len(dataset_train), replacement=True,
            )
            train_dl_kwargs["sampler"] = sampler

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset_hybrid_padding_concatenating(dataset_val, chunk_size=train_config.context_length, split="val", data_dir = dataset_config.data_dir ) 

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Compute warmup freeze duration: for the first 10% of training steps, zero out
    # gradients on the 12 western pitch rows in decoder_embedding and lm_head.
    # This lets newly-initialized microtonal rows catch up before the pretrained
    # western rows begin adapting.
    if microtonal_reg is not None:
        total_steps = train_config.num_epochs * len(train_dataloader)
        microtonal_reg['warmup_freeze_steps'] = int(0.10 * total_steps)

    # Initialize the optimizer with per-component learning rate groups.
    # LoRA adapters and GRU decoder components have separate LR groups so they can be
    # tuned independently. All groups use the same base LR by default; differentiation
    # between pretrained and new vocab rows is handled by the gradient scaling hook below
    if getattr(llama_config, 'microtonal', False):
        lora_params, decoder_head_params, gru_params = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_' in name:
                lora_params.append(param)
            elif 'decoder_embedding' in name or 'lm_head' in name:
                decoder_head_params.append(param)
            elif 'decoder.' in name:
                gru_params.append(param)
            else:
                lora_params.append(param)  # catch-all for other trainable params

        param_groups = [
            {"params": lora_params,        "lr": train_config.lr},
            {"params": decoder_head_params, "lr": train_config.lr},
            {"params": gru_params,          "lr": train_config.lr},
        ]
        # Filter out empty groups
        param_groups = [g for g in param_groups if g["params"]]
        optimizer = optim.AdamW(param_groups, weight_decay=train_config.weight_decay)

        # Gradient scaling hook: new microtonal rows in decoder_embedding and
        # lm_head get 3x effective LR relative to pretrained rows. This compensates for
        # the interpolation initialization being far from optimal, while pretrained rows
        # start near-optimal and are additionally stabilized by L_anchor.
        original_decode_vocab = tokenizer.original_decode_vocab
        scale_factor = 3.0
        def _make_row_scaling_hook(boundary, factor):
            def hook(grad):
                scaled = grad.clone()
                scaled[boundary:] *= factor
                return scaled
            return hook
        if microtonal_reg is not None:
            microtonal_reg['decoder_emb_weight'].register_hook(
                _make_row_scaling_hook(original_decode_vocab, scale_factor))
            microtonal_reg['lm_head_weight'].register_hook(
                _make_row_scaling_hook(original_decode_vocab, scale_factor))
    elif fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    starting_epoch, starting_step = 0, 0

    # StepLR(gamma=0.85) per epoch — matches Moonbeam pretraining schedule.
    # Alternative: CosineAnnealingLR (standard for NLP CPT), requires per-step
    # lr_scheduler.step() in train_con_gen() instead of the current per-epoch call
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    print("check model trainable parameters")
    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name} | Shape: {param.shape} | Parameters: {param.numel()}")
            total_trainable += param.numel()
        else:
            print(f"Frozen: {name} | Shape: {param.shape} | Parameters: {param.numel()}")
    print(f"\nTotal Trainable Parameters: {total_trainable}")
    # Start the training process
    results = train_con_gen(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        starting_epoch,
        starting_step,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        ddp_config if train_config.enable_ddp else None,
        local_rank if (train_config.enable_fsdp or train_config.enable_ddp) else None,
        rank if (train_config.enable_fsdp or train_config.enable_ddp) else None,
        wandb_run,
        microtonal_reg=microtonal_reg,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
        if train_config.use_wandb:
            for k,v in results.items():
                wandb_run.summary[k] = v

if __name__ == "__main__":
    fire.Fire(main)
