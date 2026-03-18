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
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.onset_vocab_size = int(getattr(tokenizer, "timeshift_vocab_size", 4099))
        self.dur_vocab_size = int(getattr(tokenizer, "dur_vocab_size", 4099))
        self.octave_vocab_size = int(getattr(tokenizer, "octave_vocab_size", 13))
        self.pitch_vocab_size = int(getattr(tokenizer, "pitch_class_vocab_size", 1202))
        self.instrument_vocab_size = int(getattr(tokenizer, "instrument_vocab_size", 131))
        self.velocity_vocab_size = int(getattr(tokenizer, "velocity_vocab_size", 130))

        # Load split data from CSV
        csv_path = dataset_config.csv_file
        split_data = pd.read_csv(csv_path)

        # Filter by partition (train/test)
        partition_data = split_data[split_data["split"] == partition].reset_index(
            drop=True
        )

        valid_rows = []
        missing_files = []
        for _, row in partition_data.iterrows():
            file_name = str(row["file_base_name"])
            file_path = os.path.join(self.processed_dir, file_name)
            if os.path.exists(file_path):
                valid_rows.append(row)
            else:
                missing_files.append(file_name)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

        if missing_files:
            preview = ", ".join(missing_files[:5])
            print(
                f"[symbtr_dataset_eval] Skipping {len(missing_files)} missing files for split={partition}. "
                f"Examples: {preview}"
            )

        if len(self.data) == 0:
            raise FileNotFoundError(
                f"No valid token files were found for split={partition} under {self.processed_dir}. "
                f"Check data preprocessing output and csv_file={csv_path}."
            )

        print(f"--> Loaded {partition} dataset: {len(self.data)} records")

    def __len__(self):
        return len(self.data)

    def _sanitize_raw_tokens(self, raw_tokens):
        # Classification models are trained with timeshift (delta onset), not absolute onset.
        tokens = np.asarray(raw_tokens, dtype=np.int64).copy()
        onset = tokens[:, 0]
        timeshift = np.diff(onset, prepend=0)
        tokens[:, 0] = np.clip(timeshift, 0, self.onset_vocab_size - 1)

        tokens[:, 1] = np.clip(tokens[:, 1], 0, self.dur_vocab_size - 1)
        tokens[:, 2] = np.clip(tokens[:, 2], 0, self.octave_vocab_size - 1)
        tokens[:, 3] = np.clip(tokens[:, 3], 0, self.pitch_vocab_size - 1)
        tokens[:, 4] = np.clip(tokens[:, 4], 0, self.instrument_vocab_size - 1)
        tokens[:, 5] = np.clip(tokens[:, 5], 0, self.velocity_vocab_size - 1)
        return tokens

    def __getitem__(self, index):
        item = self.data.iloc[index]
        file_name = item["file_base_name"]
        label = int(item["label"])

        # Load raw tokens
        file_path = os.path.join(self.processed_dir, file_name)
        raw_tokens = np.load(file_path)
        raw_tokens = self._sanitize_raw_tokens(raw_tokens)

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
