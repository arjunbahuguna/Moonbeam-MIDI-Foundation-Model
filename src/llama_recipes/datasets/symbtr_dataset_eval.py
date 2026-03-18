import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MakamClassificationDataset(Dataset):
    """Dataset for Makam classification from preprocessed .npy files."""

    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.data_dir = dataset_config.data_dir
        self.tokenizer = tokenizer
        self.max_words = getattr(dataset_config, "seq_len", 1200)

        # Load split data from CSV
        csv_path = dataset_config.csv_file
        split_data = pd.read_csv(csv_path)

        # Filter by partition (train/test)
        self.data = split_data[split_data["split"] == partition].reset_index(drop=True)

        print(f"--> Loaded {partition} dataset: {len(self.data)} records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        file_name = item["file_base_name"]
        label = int(item["label"])

        # Load raw tokens
        file_path = os.path.join(self.data_dir, "processed", file_name)
        raw_tokens = np.load(file_path)

        # Pure classifier path: no conditioning token is injected into inputs.
        encoded_tokens = self.tokenizer.encode_series(
            raw_tokens,
            if_add_sos=True,
            if_add_eos=True,
        )

        # Truncate if exceeding max length
        if len(encoded_tokens) > self.max_words:
            encoded_tokens = encoded_tokens[: self.max_words]

        input_ids = torch.tensor(encoded_tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": torch.tensor(label, dtype=torch.long),
            "attention_mask": torch.ones_like(input_ids),
        }
