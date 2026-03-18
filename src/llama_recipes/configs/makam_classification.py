from dataclasses import dataclass


@dataclass
class makam_classification_config:
    # Optional override. If None, falls back to dataset_config.label_map.
    label_map_path: str | None = None

    # Classification model config used to initialize LlamaForSequenceClassification.
    model_config_path: str = "src/llama_recipes/configs/config_micro/model_config_microtonal.json"

    # If True, require exact key match when loading checkpoint weights.
    checkpoint_strict: bool = False
