def generate_command():
    # Use python -m to avoid 'torchrun not found' errors on Windows
    base_cmd = "python recipes/finetuning/real_finetuning_microtonal.py"

    args = {
        "--lr": "3e-4",
        "--val_batch_size": "2",
        "--run_validation": "True",
        "--validation_interval": "10",
        "--save_metrics": "True",
        "--dist_checkpoint_root_folder": "checkpoints/finetuned_checkpoints/selected_symbtr_uncon_gen",
        "--dist_checkpoint_folder": "ddp",
        "--trained_checkpoint_path": "model_checkpoints/moonbeam_309M.pt",
        "--pure_bf16": "True",
        "--enable_ddp": "False",  # Disable DDP to prevent single-GPU errors on Windows
        "--use_peft": "True",
        "--peft_method": "lora",
        "--quantization": "False",
        "--model_name": "selected_symbtr",
        "--dataset": "selected_symbtr_dataset",
        "--output_dir": "checkpoints/finetuned_checkpoints/selected_symbtr_uncon_gen",
        "--batch_size_training": "1",
        "--context_length": "2048",
        "--num_epochs": "1",
        "--use_wandb": "True",
        "--gamma": "0.99"
    }

    args_str = " ".join([f"{k} {v}" for k, v in args.items()])
    full_run_cmd = f"{base_cmd} {args_str}"

    print("=" * 80)
    print("Open CMD, activate your Conda/virtual environment, then run:")
    print("\nset USE_LIBUV=0\n")
    print(full_run_cmd)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    generate_command()