# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime
import contextlib


import torch
import torch.nn.functional as F
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_model_checkpoint_ddp, save_peft_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.pitch_confusion_utils import (
    init_pitch_confusion_state,
    update_pitch_confusions,
    save_pitch_confusion_artifacts,
    compute_pitch_confusion_diagnostics,
    compute_western_drift_metrics,
)
# from accelerate.utils import is_xpu_available, is_ccl_available
try:
    from accelerate.utils import is_xpu_available, is_ccl_available
except ImportError:
    from accelerate.utils import is_xpu_available, is_xccl_available as is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, starting_epoch, starting_step,gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached


    # Start the training loop
    for epoch in range(starting_epoch, train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    if step < starting_step and epoch == starting_epoch:  #skip until the starting step in the first continuing epoch
                        continue
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                
                
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step%train_config.validation_interval==0 and train_config.run_validation:
                        
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model and eval_epoch_loss < best_val_loss:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                model.save_pretrained(train_config.output_dir)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        save_model_checkpoint_ddp(
                                            model, optimizer, rank, train_config, epoch=epoch, step=step
                                        )
                                        print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))     

                        """IMPORTANT"""         
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)

    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def train_overfit(model, batch, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        eval_dataloader: same as train_dataloader
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached

    # Start the training loop
    for epoch in range(train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:

                for step, batch_unused in enumerate(train_dataloader):
                    # print("batch train: ", batch['input_ids'])
                    """
                    save data as npy file for visualization
                    """

                    # Save data as npy files for the first few steps for visualization
                    if step < 5:
                        import numpy as np
                        for key in batch.keys():
                            # Convert the tensor to a NumPy array (move to CPU if needed)
                            data_array = batch[key].cpu().numpy()
                            
                            # Save the NumPy array to a file with a unique name per key and step
                            np.save(f'/data/home/acw753/musicllama/dataset_analysis/{key}_step_{step}.npy', data_array)

                    if step > 1000:
                        break

                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                
                
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step%train_config.validation_interval==0 and train_config.run_validation:
                        
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, generation_logits, generation_hidden_state, logits_shrinked = evaluation_overfit(model, train_config, batch, eval_dataloader, local_rank, tokenizer, wandb_run)

                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model and eval_epoch_loss < best_val_loss:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                model.save_pretrained(train_config.output_dir)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        save_model_checkpoint_ddp(
                                            model, optimizer, rank, train_config, epoch=epoch, step=step
                                        )
                                        torch.save(generation_logits, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_logits_epoch_{epoch}_step_{step}.pt')
                                        torch.save(generation_hidden_state, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_hidden_state_epoch_{epoch}_step_{step}.pt')
                                        torch.save(logits_shrinked, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/logits_shrinked_epoch_{epoch}_step_{step}.pt')
                                        print(f"generation logits and hidden states saved to /data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_logits_epoch_{epoch}_step_{step}.pt")
                                        print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))     

                        """IMPORTANT"""         
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)

    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def _should_run_pitch_confusion(train_config, rank=None):
    if not getattr(train_config, 'enable_pitch_confusion', False):
        return False
    if train_config.enable_fsdp or train_config.enable_ddp:
        return rank == 0
    return True

# Pitch confusion: Added utility function to find the GRU accuracy dict from the model, even if it's wrapped in multiple layers of FSDP or DDP. This is needed for the pitch confusion analysis and logging.
def _collect_gru_acc_metrics(model):
    """Find per-attribute GRU accuracy dict even under nested wrappers."""
    candidates = []
    visited = set()
    stack = [model]

    while stack:
        mod = stack.pop()
        if mod is None:
            continue
        obj_id = id(mod)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        candidates.append(mod)

        for attr in ("module", "base_model", "model"):
            child = getattr(mod, attr, None)
            if child is not None:
                stack.append(child)

    for cand in candidates:
        gru_acc = getattr(cand, "_gru_acc", None)
        if isinstance(gru_acc, dict) and len(gru_acc) > 0:
            return gru_acc
    return {}

# [Turned off by default] Auxiliary CE loss on pitch tokens to directly optimize pitch accuracy, which is critical for music generation quality. This is computed in the training loop when generation logits are available, and the tokenizer has a pitch_dict to identify which token IDs correspond to pitch targets.
def _compute_pitch_aux_ce_loss(labels, generation_logits, tokenizer, device):
    """Compute CE loss only on pitch targets to directly improve pitch accuracy."""
    if generation_logits is None or labels is None:
        return None

    if not hasattr(tokenizer, "pitch_dict"):
        return None

    pitch_ids = sorted(set(int(v) for v in tokenizer.pitch_dict.values()))
    if len(pitch_ids) == 0:
        return None

    labels_flat = labels.reshape(-1)
    logits_flat = generation_logits.reshape(-1, generation_logits.size(-1))

    min_len = min(labels_flat.numel(), logits_flat.size(0))
    if min_len <= 0:
        return None

    labels_flat = labels_flat[:min_len]
    logits_flat = logits_flat[:min_len]

    valid_mask = labels_flat >= 0
    if valid_mask.sum().item() == 0:
        return None

    pitch_id_tensor = torch.tensor(pitch_ids, device=device, dtype=labels_flat.dtype)
    pitch_mask = torch.isin(labels_flat, pitch_id_tensor)
    target_mask = valid_mask & pitch_mask

    if target_mask.sum().item() == 0:
        return None

    return F.cross_entropy(logits_flat[target_mask], labels_flat[target_mask].long())


def _maybe_log_pitch_confusion_to_wandb(train_config, wandb_run, artifacts, metrics, step):
    if wandb_run is None or not getattr(train_config, 'pitch_confusion_log_wandb', True):
        return

    log_dict = {}
    if getattr(train_config, 'pitch_confusion_save_plots', True):
        try:
            import wandb
        except ImportError:
            wandb = None

        if wandb is not None:
            image_keys = [
                ('overall_counts_png', 'pitch_confusion/overall_counts'),
                ('overall_row_norm_png', 'pitch_confusion/overall_row_norm'),
                ('western_counts_png', 'pitch_confusion/western12_counts'),
                ('western_row_norm_png', 'pitch_confusion/western12_row_norm'),
            ]
            for art_key, wb_key in image_keys:
                img_path = artifacts.get(art_key, '') if isinstance(artifacts, dict) else ''
                if img_path and os.path.exists(img_path):
                    try:
                        log_dict[wb_key] = wandb.Image(img_path)
                    except Exception:
                        # Keep metric logging alive even if an image is too large/corrupt.
                        pass

    for k, v in metrics.items():
        log_dict[f'pitch_confusion/{k}'] = v

    if log_dict:
        wandb_run.log(log_dict, step=step)

def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, starting_epoch, starting_step,gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None, microtonal_reg=None, pitch_confusion_ctx=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
        microtonal_reg: Optional dict for microtonal embedding regularization.
            When provided, adds two auxiliary losses: L_anchor (MSE penalty preventing
            pretrained western pitch embeddings from drifting) and L_smooth (encourages
            adjacent cent-resolution pitch bins to have similar embeddings).
        pitch_confusion_ctx: Optional dict controlling confusion snapshots and
            western pre/post drift evaluation.

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached

    confusion_active = _should_run_pitch_confusion(train_config, rank)
    confusion_save_dir = (
        train_config.pitch_confusion_dir
        if getattr(train_config, 'pitch_confusion_dir', '')
        else os.path.join(train_config.output_dir, 'pitch_confusion')
    )
    overall_snapshot_epochs = set()
    western_pre_matrix = None

    if confusion_active and eval_dataloader is not None:
        overall_snapshot_epochs = {0, train_config.num_epochs // 2, train_config.num_epochs - 1}

    if confusion_active and pitch_confusion_ctx is not None:
        western_eval_dataloader = pitch_confusion_ctx.get('western_eval_dataloader')
        if western_eval_dataloader is not None:
            pre_req = {
                'enabled': True,
                'artifact_prefix': 'western_pre_cpt',
                'save_dir': confusion_save_dir,
                'expected_pitch_dict_size': len(tokenizer.pitch_dict),
                'max_eval_step': getattr(train_config, 'pitch_confusion_max_eval_step', 0),
            }
            pre_out = evaluation(
                model, train_config, western_eval_dataloader, local_rank,
                tokenizer, wandb_run, confusion_request=pre_req
            )
            _, _, _, _, pre_summary = pre_out
            western_pre_matrix = pre_summary.get('matrix_western')
            if western_pre_matrix is not None:
                pitch_confusion_ctx['western_pre_matrix'] = western_pre_matrix
                pre_diag = pre_summary.get('diagnostics', {})
                pre_metrics = {
                    'western_pre_total_pitch_samples': pre_summary.get('total_pitch_samples', 0),
                    'western_pre_ignored_pred_non_pitch': pre_summary.get('ignored_pred_non_pitch', 0),
                }
                for k, v in pre_diag.items():
                    pre_metrics[f'western_pre_{k}'] = v
                _maybe_log_pitch_confusion_to_wandb(
                    train_config,
                    wandb_run,
                    pre_summary.get('artifacts', {}),
                    pre_metrics,
                    step=0,
                )

    # Start the training loop
    for epoch in range(starting_epoch, train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    if step < starting_step and epoch == starting_epoch:  #skip until the starting step in the first continuing epoch
                        continue
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        outputs = model(**batch)
                        loss = outputs.loss
                    # Microtonal embedding regularization (anchor + smoothness)
                    ce_loss = loss.detach().float().item()
                    pitch_aux_val = None
                    pitch_aux_val = None
                    pitch_aux = None
                    # Compute auxiliary pitch CE when a positive weight and generation logits exist.
                    # Adding it to the loss is gated by `enable_micro_pitch_ce`, but we still
                    # compute and record the value for `microtonal_reg` when present.
                    pitch_aux_weight = float(getattr(train_config, 'micro_pitch_ce_weight', 0.0) or 0.0)
                    if pitch_aux_weight > 0.0 and getattr(outputs, 'generation_logits', None) is not None:
                        pitch_aux = _compute_pitch_aux_ce_loss(
                            batch.get('labels', None),
                            outputs.generation_logits,
                            tokenizer,
                            loss.device,
                        )
                        if pitch_aux is not None:
                            pitch_aux_val = pitch_aux.detach().float().item()
                            if getattr(train_config, 'enable_micro_pitch_ce', True):
                                # If flag is on, then add the auxiliary pitch CE loss to the main loss with the specified weight
                                loss = loss + (pitch_aux_weight * pitch_aux)
                    if microtonal_reg is not None:
                        reg = microtonal_reg
                        w_ids = reg['western_ids']
                        # L_anchor: penalize drift of pretrained western pitch embeddings
                        L_anchor = reg['lambda_anchor'] * (
                            torch.mean((reg['decoder_emb_weight'][w_ids] - reg['frozen_decoder_emb']) ** 2) +
                            torch.mean((reg['lm_head_weight'][w_ids] - reg['frozen_lm_head']) ** 2)
                        )
                        # L_smooth: adjacent cent bins should have similar embeddings
                        left = reg['micro_pair_ids_left']
                        right = reg['micro_pair_ids_right']
                        L_smooth = (
                            torch.mean((reg['decoder_emb_weight'][right] - reg['decoder_emb_weight'][left]) ** 2) +
                            torch.mean((reg['lm_head_weight'][right] - reg['lm_head_weight'][left]) ** 2)
                        ) * reg['lambda_smooth']

                        # Add microtonal reg to the loss
                        loss = loss + L_anchor + L_smooth
                        reg['_last_L_anchor'] = L_anchor.detach().float().item()
                        reg['_last_L_smooth'] = L_smooth.detach().float().item()
                        reg['_last_ce_loss'] = ce_loss
                        if pitch_aux_val is not None:
                            reg['_last_pitch_ce_loss'] = pitch_aux_val
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        # Warmup freeze: zero gradients on western pitch rows for the first
                        # N steps, letting new microtonal rows catch up from interpolation init
                        if microtonal_reg is not None and total_train_steps <= microtonal_reg.get('warmup_freeze_steps', 0):
                            reg = microtonal_reg
                            w_ids = reg['western_ids']
                            if reg['decoder_emb_weight'].grad is not None:
                                reg['decoder_emb_weight'].grad[w_ids] = 0
                            if reg['lm_head_weight'].grad is not None:
                                reg['lm_head_weight'].grad[w_ids] = 0
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    grad_norm = model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                                if microtonal_reg is not None:
                                    microtonal_reg['_last_grad_norm'] = float(grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        # Warmup freeze: zero gradients on western pitch rows for the first
                        # N steps, letting new microtonal rows catch up from interpolation init
                        if microtonal_reg is not None and total_train_steps <= microtonal_reg.get('warmup_freeze_steps', 0):
                            reg = microtonal_reg
                            w_ids = reg['western_ids']
                            if reg['decoder_emb_weight'].grad is not None:
                                reg['decoder_emb_weight'].grad[w_ids] = 0
                            if reg['lm_head_weight'].grad is not None:
                                reg['lm_head_weight'].grad[w_ids] = 0
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    grad_norm = model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                                if microtonal_reg is not None:
                                    microtonal_reg['_last_grad_norm'] = float(grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run and step % train_config.log_interval == 0:
                        if not train_config.enable_fsdp or rank==0:
                            log_dict = {
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                                'train/lr': optimizer.param_groups[0]['lr'],
                            }
                            # Log regularization losses, diagnostics, and warmup freeze status
                            if microtonal_reg is not None:
                                if '_last_ce_loss' in microtonal_reg:
                                    log_dict['train/ce_loss'] = microtonal_reg['_last_ce_loss']
                                if '_last_L_anchor' in microtonal_reg:
                                    log_dict['train/L_anchor'] = microtonal_reg['_last_L_anchor']
                                if '_last_L_smooth' in microtonal_reg:
                                    log_dict['train/L_smooth'] = microtonal_reg['_last_L_smooth']
                                if '_last_grad_norm' in microtonal_reg:
                                    log_dict['train/grad_norm'] = microtonal_reg['_last_grad_norm']
                                log_dict['train/warmup_freeze_active'] = int(
                                    total_train_steps <= microtonal_reg.get('warmup_freeze_steps', 0))
                                # Embedding drift: L2 distance from init for new micro rows
                                reg = microtonal_reg
                                with torch.no_grad():
                                    orig_v = reg.get('original_decode_vocab')
                                    if orig_v is not None:
                                        micro_emb = reg['decoder_emb_weight'][orig_v:]
                                        micro_head = reg['lm_head_weight'][orig_v:]
                                        log_dict['train/micro_emb_norm'] = micro_emb.norm().item()
                                        log_dict['train/micro_head_norm'] = micro_head.norm().item()
                            # Log per-attribute GRU decoder accuracy
                            gru_acc = _collect_gru_acc_metrics(model)
                            if gru_acc:
                                for attr_name, acc in gru_acc.items():
                                    log_dict[f'train/gru_acc/{attr_name}'] = acc
                            if microtonal_reg is not None and '_last_pitch_ce_loss' in microtonal_reg:
                                log_dict['train/pitch_ce_loss'] = microtonal_reg['_last_pitch_ce_loss']
                            wandb_run.log(log_dict, step=epoch * len(train_dataloader) + step)

                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)


                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step%train_config.validation_interval==0:
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                # model.save_pretrained(train_config.output_dir)
                                save_peft_checkpoint(model, train_config.output_dir, epoch=epoch, step = step)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        save_model_checkpoint_ddp(
                                            model, optimizer, rank, train_config, epoch=epoch, step=step
                                        )
                                        print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))    

                        #IMPORTANT        
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        if confusion_active and eval_dataloader is not None and epoch in overall_snapshot_epochs:
            req = {
                'enabled': True,
                'artifact_prefix': f'overall_epoch_{epoch}',
                'save_dir': confusion_save_dir,
                'expected_pitch_dict_size': len(tokenizer.pitch_dict),
                'max_eval_step': getattr(train_config, 'pitch_confusion_max_eval_step', 0),
            }
            eval_out = evaluation(
                model, train_config, eval_dataloader, local_rank,
                tokenizer, wandb_run, confusion_request=req
            )
            _, _, _, _, conf_summary = eval_out
            conf_diag = conf_summary.get('diagnostics', {})
            conf_metrics = {
                f'overall_epoch_{epoch}_total_pitch_samples': conf_summary.get('total_pitch_samples', 0),
                f'overall_epoch_{epoch}_ignored_pred_non_pitch': conf_summary.get('ignored_pred_non_pitch', 0),
            }
            for k, v in conf_diag.items():
                conf_metrics[f'overall_epoch_{epoch}_{k}'] = v
            _maybe_log_pitch_confusion_to_wandb(
                train_config,
                wandb_run,
                conf_summary.get('artifacts', {}),
                conf_metrics,
                step=(epoch + 1) * len(train_dataloader),
            )

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)

    # Always capture end-of-training overall confusion metrics from the final
    # model state. This keeps evaluation aligned with CPT outcome goals even
    # when max_train_step stops training before scheduled epoch snapshots.
    if confusion_active and eval_dataloader is not None:
        final_req = {
            'enabled': True,
            'artifact_prefix': 'overall_final',
            'save_dir': confusion_save_dir,
            'expected_pitch_dict_size': len(tokenizer.pitch_dict),
            'max_eval_step': getattr(train_config, 'pitch_confusion_max_eval_step', 0),
        }
        eval_out = evaluation(
            model, train_config, eval_dataloader, local_rank,
            tokenizer, wandb_run, confusion_request=final_req
        )
        _, _, _, _, final_summary = eval_out
        final_diag = final_summary.get('diagnostics', {})
        final_metrics = {
            'overall_final_total_pitch_samples': final_summary.get('total_pitch_samples', 0),
            'overall_final_ignored_pred_non_pitch': final_summary.get('ignored_pred_non_pitch', 0),
        }
        for k, v in final_diag.items():
            final_metrics[f'overall_final_{k}'] = v
        results.update(final_metrics)
        _maybe_log_pitch_confusion_to_wandb(
            train_config,
            wandb_run,
            final_summary.get('artifacts', {}),
            final_metrics,
            step=total_train_steps,
        )

    if confusion_active and pitch_confusion_ctx is not None:
        western_eval_dataloader = pitch_confusion_ctx.get('western_eval_dataloader')
        western_pre_matrix = pitch_confusion_ctx.get('western_pre_matrix', western_pre_matrix)
        if western_eval_dataloader is not None:
            req = {
                'enabled': True,
                'artifact_prefix': 'western_post_cpt',
                'save_dir': confusion_save_dir,
                'expected_pitch_dict_size': len(tokenizer.pitch_dict),
                'max_eval_step': getattr(train_config, 'pitch_confusion_max_eval_step', 0),
            }
            eval_out = evaluation(
                model, train_config, western_eval_dataloader, local_rank,
                tokenizer, wandb_run, confusion_request=req
            )
            _, _, _, _, post_summary = eval_out
            post_diag = post_summary.get('diagnostics', {})
            drift_metrics = {
                'western_post_total_pitch_samples': post_summary.get('total_pitch_samples', 0),
                'western_post_ignored_pred_non_pitch': post_summary.get('ignored_pred_non_pitch', 0),
            }
            for k, v in post_diag.items():
                drift_metrics[f'western_post_{k}'] = v
            if western_pre_matrix is not None and post_summary.get('matrix_western') is not None:
                drift_metrics.update(
                    compute_western_drift_metrics(
                        western_pre_matrix,
                        post_summary['matrix_western'],
                    )
                )
            results.update(drift_metrics)
            _maybe_log_pitch_confusion_to_wandb(
                train_config,
                wandb_run,
                post_summary.get('artifacts', {}),
                drift_metrics,
                step=train_config.num_epochs * len(train_dataloader),
            )

    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run, confusion_request=None):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    # eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    confusion_state = None
    confusion_enabled = bool(confusion_request and confusion_request.get('enabled', False))
    confusion_max_eval_step = 0
    if confusion_enabled:
        confusion_state = init_pitch_confusion_state(tokenizer)
        confusion_max_eval_step = int(confusion_request.get('max_eval_step', 0) or 0)

    if is_xpu_available():
        eval_device = 'xpu:0'
    elif torch.cuda.is_available():
        eval_device = 'cuda:0'
    else:
        eval_device = 'cpu'

    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            if confusion_enabled and confusion_max_eval_step > 0 and total_eval_steps > confusion_max_eval_step:
                break
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to(eval_device)
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                if confusion_enabled and getattr(outputs, 'generation_logits', None) is not None:
                    update_pitch_confusions(confusion_state, batch['labels'], outputs.generation_logits)

                eval_loss += loss.detach().float()

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)

    if confusion_enabled:
        expected_pitch_dict_size = confusion_request.get('expected_pitch_dict_size', None)
        save_dir = confusion_request.get('save_dir', os.path.join(train_config.output_dir, 'pitch_confusion'))
        artifact_prefix = confusion_request.get('artifact_prefix', 'eval')
        artifacts = save_pitch_confusion_artifacts(
            confusion_state,
            save_dir=save_dir,
            artifact_prefix=artifact_prefix,
            expected_pitch_dict_size=expected_pitch_dict_size,
            save_plots=bool(getattr(train_config, 'pitch_confusion_save_plots', True)),
        )
        summary = {
            'artifacts': artifacts,
            'total_pitch_samples': int(confusion_state.total_pitch_samples),
            'ignored_pred_non_pitch': int(confusion_state.ignored_pred_non_pitch),
            'diagnostics': compute_pitch_confusion_diagnostics(confusion_state),
            'matrix_western': confusion_state.matrix_western,
            'matrix_overall': confusion_state.matrix_overall,
            'pitch_token_ids': confusion_state.pitch_token_ids,
            'western_token_ids': confusion_state.western_token_ids,
        }
        return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, summary

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

def evaluation_overfit(model,train_config, batch, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    # eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        for step, batch_unused in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            if step > 1:
                break
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                """ check generation logits and targets  """

                generation_logits = outputs.generation_logits #batch * len_x, decoder_vocab_size

                batch_size = batch['input_ids'].shape[0]
                length = batch['input_ids'].shape[1]-1 
                no_attributes = 6


                generation_logits_reshaped = torch.reshape(generation_logits, (batch_size, length, no_attributes, -1))

                # print(f"generation_logits:{generation_logits_reshaped.shape}")
                max_values, max_indices = torch.max(generation_logits_reshaped, dim=-1)
                # print(f"max_indices:{max_indices.shape}, {max_indices}")
                torch.save(generation_logits_reshaped, "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits.pth")
                torch.save(max_indices, "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits_max.pth")

                
                try:
                    decoded_tokens = tokenizer.convert_from_language_tokens(torch.max(generation_logits, dim=-1))
                    torch.save(torch.tensor(decoded_tokens), "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits_max_decoded_tokens.pth")
                    print(f"decoded_tokens:{decoded_tokens}")
                except:
                    print(f"failed to decode tokens")

                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    # eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_loss = eval_loss / 2
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, outputs.generation_logits, outputs.generation_hidden_state, outputs.logits


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params / 1e6:.2f} Million")
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")
        trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0.0
        print(f"Trainable %: {trainable_pct:.2f}%\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""


    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
