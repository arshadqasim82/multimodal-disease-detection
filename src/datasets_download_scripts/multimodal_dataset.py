from dataclasses import dataclass
from typing import Dict, Optional

import random
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.image_dataset import load_image_tensor
from transformers import AutoTokenizer


@dataclass
class MultimodalDatasetConfig:
    paired_csv: str
    tokenizer_name: str
    max_length: int = 192
    image_size: int = 224
    neg_prob: float = 0.5  # probability of returning a mismatched pair


class PairedCSVMatchDataset(Dataset):
    """
    Paired CSV dataset with negative sampling to create a binary matching task.
    """

    def __init__(self, cfg: MultimodalDatasetConfig, indices=None, seed: int = 42):
        self.cfg = cfg
        self.rng = random.Random(seed)

        df = pd.read_csv(cfg.paired_csv)
        if indices is not None:
            df = df.iloc[indices]
        self.df = df.reset_index(drop=True)

        required = {"image_path", "text"}
        if not required.issubset(set(self.df.columns)):
            raise ValueError(
                f"paired_csv must contain {required}, got {set(self.df.columns)}")

        self.has_uid = "uid" in self.df.columns
        self.all_texts = self.df["text"].astype(str).tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    def __len__(self):
        return len(self.df)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])

        # image tensor
        x_img = load_image_tensor(img_path, image_size=self.cfg.image_size)

        # positive vs negative
        is_negative = (self.rng.random() < self.cfg.neg_prob)

        if not is_negative:
            text = str(row["text"])
            label = 1
        else:
            j = idx
            while j == idx:
                j = self.rng.randrange(0, len(self.all_texts))
            text = self.all_texts[j]
            label = 0

        t = self._tokenize(text)

        out = {
            "image": x_img,
            "input_ids": t["input_ids"],
            "attention_mask": t["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.has_uid:
            out["uid"] = torch.tensor(int(row["uid"]), dtype=torch.long)

        return out
