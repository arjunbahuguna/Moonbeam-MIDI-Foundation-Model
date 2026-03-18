import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    os.path.join(REPO_ROOT, "src", "llama_recipes", "transformers_minimal", "src"),
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import fire

from llama_recipes.real_finetuning_makam_classification import main


if __name__ == "__main__":
    fire.Fire(main)
