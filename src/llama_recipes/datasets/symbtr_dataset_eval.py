import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SymbTrDataset(Dataset):
    """
    SymbTr token dataset.

    Default behavior mirrors LakhDataset and only returns model-ready keys:
    input_ids, labels, attention_mask.

    If dataset_config.return_conditioning is True, three additional integer-ID
    fields are returned:
      - symbtr_condition_ids: [makam_id, form_id, usul_id]
      - symbtr_makam_id
      - symbtr_form_id
      - symbtr_usul_id

    Note: Returning extra fields requires a conditioning-aware collator/training
    loop. The default LlamaForCausalLM path will reject unknown kwargs.
    """

    def __init__(self, dataset_config, tokenizer, partition="train"):
        assert partition in {"train", "test", "val", "validation"}
        self.data_dir = dataset_config.data_dir
        self.tokenizer = tokenizer

        split_data = pd.read_csv(dataset_config.csv_file)
        split_data = split_data.copy()

        self.return_conditioning = bool(getattr(dataset_config, "return_conditioning", False))

        # Ensure expected columns exist; fallback to empty strings so old CSVs
        # remain compatible.
        for col in ["makam", "form", "usul"]:
            if col not in split_data.columns:
                split_data[col] = ""
            split_data[col] = split_data[col].fillna("").astype(str)

        # Build deterministic vocabularies from full CSV (not split-specific),
        # so train/test partitions share IDs.
        self.makam_to_id = self._build_label_vocab(split_data["makam"])
        self.form_to_id = self._build_label_vocab(split_data["form"])
        self.usul_to_id = self._build_label_vocab(split_data["usul"])

        part_df = split_data[split_data["split"] == partition].reset_index(drop=True)

        self.file_basenames = part_df["file_base_name"].tolist()
        self.makam_labels = part_df["makam"].tolist()
        self.form_labels = part_df["form"].tolist()
        self.usul_labels = part_df["usul"].tolist()

    @staticmethod
    def _build_label_vocab(values):
        # Reserve 0 for unknown/empty to simplify downstream conditioning logic.
        unique_values = sorted({v for v in values if v})
        vocab = {"": 0}
        for idx, value in enumerate(unique_values, start=1):
            vocab[value] = idx
        return vocab

    def _label_triplet_to_ids(self, makam, form, usul):
        return [
            self.makam_to_id.get(makam, 0),
            self.form_to_id.get(form, 0),
            self.usul_to_id.get(usul, 0),
        ]

    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, index):
        raw_tokens = np.load(os.path.join(self.data_dir, "processed", self.file_basenames[index]))

        encoded_tokens = self.tokenizer.encode_series(
            raw_tokens,
            if_add_sos=True,
            if_add_eos=True,
        )
        encoded_tokens_label = self.tokenizer.encode_series_labels(
            encoded_tokens,
            if_added_sos=True,
            if_added_eos=True,
        )

        out = {
            "input_ids": encoded_tokens,
            "labels": encoded_tokens_label,
            "attention_mask": [],
        }

        if self.return_conditioning:
            makam = self.makam_labels[index]
            form = self.form_labels[index]
            usul = self.usul_labels[index]
            cond_ids = self._label_triplet_to_ids(makam, form, usul)
            out["symbtr_condition_ids"] = cond_ids
            out["symbtr_makam_id"] = cond_ids[0]
            out["symbtr_form_id"] = cond_ids[1]
            out["symbtr_usul_id"] = cond_ids[2]

        return out
