add this to `recipes/finetuning/real_finetuning_uncon_gen.py`

```
sys.path.insert(0, os.path.abspath("src"))
```

to solve:

```
 (Moonbeam-MIDI-Foundation-Model) PS E:\Master\Moonbeam> python recipes/finetuning/real_finetuning_uncon_gen.py --lr 3e-4 --val_batch_size 2 --run_validation True --validation_interval 10 --save_metrics True --dist_checkpoint_root_folder checkpoints/finetuned_checkpoints/selected_symbtr_uncon_gen --dist_checkpoint_folder ddp --trained_checkpoint_path model_checkpoints/moonbeam_309M.pt --pure_bf16 True --enable_ddp False --use_peft True --peft_method lora --quantization False --model_name selected_symbtr --dataset selected_symbtr_dataset --output_dir checkpoints/finetuned_checkpoints/selected_symbtr_uncon_gen --batch_size_training 1 --context_length 2048 --num_epochs 1 -- Traceback (most recent call last):
   File "E:\Master\Moonbeam\recipes\finetuning\real_finetuning_uncon_gen.py", line 5, in <module>
     from llama_recipes.real_finetuning_uncon_gen import main
  File "E:\Master\Moonbeam\src\llama_recipes\real_finetuning_uncon_gen.py", line 22, in <module>
    from llama_recipes.datasets.music_tokenizer import MusicTokenizer
 ModuleNotFoundError: No module named 'llama_recipes.datasets.music_tokenizer'
```



------



change the path in `src/llama_recipes/real_finetuning_uncon_gen.py` from

` src\llama_recipes\real_finetuning_uncon_gen.py `

to 

`model_config_path = "src/llama_recipes/configs/config_micro"`

cause it want to take a 