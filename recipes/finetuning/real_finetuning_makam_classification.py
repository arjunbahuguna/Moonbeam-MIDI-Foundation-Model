import os
import sys

sys.path.insert(0, os.path.abspath("src"))

import fire

from llama_recipes.real_finetuning_makam_classification import main


if __name__ == "__main__":
    fire.Fire(main)
