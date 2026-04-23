import os
import sys
import json
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import makam_classification_config as MAKAM_CLASSIFICATION_CONFIG
from llama_recipes.utils.config_utils import generate_dataset_config
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from llama_recipes.transformers_minimal.src.transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForSequenceClassification,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.real_finetuning_makam_classification import (
    CompoundClassificationCollator,
    _load_checkpoint_into_model,
)

def main():
    train_cfg = TRAIN_CONFIG()
    train_cfg.dataset = "symbtr_dataset_eval"
    makam_cfg = MAKAM_CLASSIFICATION_CONFIG()

    dataset_config = generate_dataset_config(train_cfg, {})
    with open(dataset_config.label_map, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    num_labels = len(label_map)

    model_config = LlamaConfig.from_pretrained(makam_cfg.model_config_path)
    tokenizer = MusicTokenizer(
        timeshift_vocab_size=model_config.onset_vocab_size,
        dur_vocab_size=model_config.dur_vocab_size,
        octave_vocab_size=model_config.octave_vocab_size,
        pitch_class_vocab_size=model_config.pitch_class_vocab_size,
        instrument_vocab_size=model_config.instrument_vocab_size,
        velocity_vocab_size=model_config.velocity_vocab_size,
        sos_token=model_config.sos_token,
        eos_token=model_config.eos_token,
        pad_token=model_config.pad_token,
        microtonal=getattr(model_config, 'microtonal', False),
    )

    model_config.num_labels = num_labels
    model_config.pad_token_id = int(tokenizer.pad_token)
    model_config.problem_type = 'single_label_classification'

    model = LlamaForSequenceClassification(model_config)
    adapter_root = 'checkpoints/microtonal_cpt'
    if os.path.isdir(adapter_root):
        model = _load_checkpoint_into_model(model, adapter_root, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ds_train = get_preprocessed_dataset(tokenizer, dataset_config, split='train')
    collator = CompoundClassificationCollator(pad_token=int(tokenizer.pad_token))
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=1, collate_fn=collator)

    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    print(f'RESULT_LOSS: {float(loss.item())}')
    print(f'RESULT_LOGITS_SHAPE: {tuple(outputs.logits.shape)}')

    loss.backward()
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
             total_norm += p.grad.detach().data.norm(2).item() ** 2
    print(f'RESULT_GRAD_NORM: {total_norm ** 0.5}')

if __name__ == "__main__":
    main()
