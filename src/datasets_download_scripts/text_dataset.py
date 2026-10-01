from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class TextDatasetConfig:
    csv_path: str
    tokenizer_name: str
    max_length: int = 128
    target_col: Optional[str] = None  # "label" or "uid" or None (auto)


class TextCSVDataset(Dataset):
    """
    Supports CSVs with:
    - text + label (classification)
    - text + uid   (unlabeled paired reports)
    """

    def __init__(self, cfg: TextDatasetConfig):
        self.df = pd.read_csv(cfg.csv_path)

        if "text" not in self.df.columns:
            raise ValueError(
                f"CSV must contain 'text', got {set(self.df.columns)}")

        # Decide target column
        if cfg.target_col is not None:
            target_col = cfg.target_col
            if target_col not in self.df.columns:
                raise ValueError(
                    f"target_col='{target_col}' not in CSV columns {set(self.df.columns)}"
                )
        else:
            # auto-detect
            if "label" in self.df.columns:
                target_col = "label"
            elif "uid" in self.df.columns:
                target_col = "uid"
            else:
                target_col = None

        self.target_col = target_col
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

        if self.target_col is not None:
            if self.target_col == "label":
                item["label"] = torch.tensor(
                    int(row["label"]), dtype=torch.long)
            elif self.target_col == "uid":
                item["uid"] = torch.tensor(int(row["uid"]), dtype=torch.long)
            else:
                item[self.target_col] = torch.tensor(
                    int(row[self.target_col]), dtype=torch.long)

        return item
