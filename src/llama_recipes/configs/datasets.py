# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class lakhmidi_dataset:
    dataset: str = "lakhmidi_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/PATH/TO/DATA/DIR"
    csv_file: str = "/PATH/TO/CSV"

@dataclass
class symbtr_dataset:
    dataset: str = "symbtr_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "processed_data_Selected_SymbTr"
    csv_file: str = "processed_data_Selected_SymbTr/train_test_split.csv"
    return_conditioning: bool = False

@dataclass
class selected_symbtr_dataset:
    dataset: str = "selected_symbtr_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "processed_data_Selected_SymbTr"
    csv_file: str = "processed_data_Selected_SymbTr/train_test_split.csv"
    return_conditioning: bool = False

@dataclass
class merge_dataset:
    dataset: str = "merge_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/PATH/TO/DATA/DIR"
    csv_file: str = "processed_data_Selected_SymbTr/train_test_split.csv"

@dataclass
class emophia_con_gen_dataset:
    dataset: str = "emophia_con_gen_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/PATH/TO/DATA/DIR"
    csv_file: str = "/PATH/TO/CSV"

@dataclass
class commu_con_gen_dataset:
    dataset: str = "commu_con_gen_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_dir: str = "/PATH/TO/DATA/DIR"
    csv_file: str = "/PATH/TO/CSV"
    additional_token_dict_path: str = "/PATH/TO/JSON"


@dataclass
class symbtr_dataset_eval:
    dataset: str = "symbtr_dataset_eval"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "data/processed_data_symbtr"
    csv_file: str = "data/processed_data_symbtr/makam_classification_split.csv"
    label_map: str = "data/processed_data_symbtr/makam_label_map.json"
    seq_len: int = 1200
    