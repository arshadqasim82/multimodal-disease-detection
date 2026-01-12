from dataclasses import dataclass
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class TextDatasetConfig:
    csv_path: str
    tokenizer_name: str
    max_length: int = 128


class TextCSVDataset(Dataset):
    """
    Expects CSV columns: text, label
    """

    def __init__(self, cfg: TextDatasetConfig):
        self.df = pd.read_csv(cfg.csv_path)
        required = {"text", "label"}
        if not required.issubset(set(self.df.columns)):
            raise ValueError(
                f"CSV must contain {required}, got {set(self.df.columns)}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.max_length = cfg.max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item
