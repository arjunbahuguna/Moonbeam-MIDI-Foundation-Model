# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from llama_recipes.real_finetuning_player_classification import main

# try:
#     from llama_recipes.real_finetuning_player_classification import main
# except ModuleNotFoundError:
#     import sys, importlib, os
#     # repo root is two levels up from this file (recipes/finetuning)
#     repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#     # the package lives under src/llama_recipes
#     src_path = os.path.join(repo_root, "src")
#     # also allow direct repo root as fallback (some layouts put packages at repo root)
#     candidates = [src_path, repo_root]
#     for p in candidates:
#         if p not in sys.path and os.path.isdir(p):
#             sys.path.insert(0, p)
#     try:
#         main = importlib.import_module("llama_recipes.real_finetuning_player_classification").main
#     except Exception as e:
#         print(e)
#         raise RuntimeError(
#             "Cannot import 'llama_recipes'. Either add the repo src to PYTHONPATH or install the package:\n\n"
#             "  export PYTHONPATH=\"$PWD/src:$PYTHONPATH\"\n"
#             "  pip install -e src\n"
#         ) from e

if __name__ == "__main__":
    fire.Fire(main)