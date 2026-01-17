"""
Purpose
-------
Defines a multimodal dataset that returns aligned image + text inputs
from a single paired CSV (e.g., IU Chest X-Rays).

Pairing Strategy
----------------
Reads one CSV containing aligned rows:
- image_path: path to image file
- text: report text
- uid: group id (optional; useful for grouping / evaluation)

Labeling
--------
IU paired dataset does not provide binary disease labels by default.
This dataset can optionally return:
- uid (recommended for contrastive/retrieval training)
- a dummy label (0) if a training loop hard-requires "label"

Outputs
-------
Returns dict:
- image: tensor [3, H, W]
- input_ids: tensor [L]
- attention_mask: tensor [L]
- uid: scalar long (if available)
- label: scalar long (optional dummy, if requested)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

# we will add this helper if missing
from src.datasets.image_dataset import load_image_tensor
from src.datasets.text_dataset import TextDatasetConfig, TextCSVDataset


@dataclass
class MultimodalDatasetConfig:
    paired_csv: str                 # <- NEW: single source of truth
    tokenizer_name: str
    max_length: int = 192
    image_size: int = 224
    return_label: bool = False      # if True, returns dummy label=0


class PairedCSVMultimodalDataset(Dataset):
    """
    Reads aligned image-text pairs from one CSV.
    """

    def __init__(self, cfg: MultimodalDatasetConfig):
        self.df = pd.read_csv(cfg.paired_csv)

        required = {"image_path", "text"}
        if not required.issubset(set(self.df.columns)):
            raise ValueError(
                f"paired_csv must contain {required}, got {set(self.df.columns)}")

        # Text tokenizer dataset (reuses your existing tokenization)
        # We want uid back if it exists; otherwise it's fine.
        target_col = "uid" if "uid" in self.df.columns else None
        self.text_ds = TextCSVDataset(TextDatasetConfig(
            csv_path=cfg.paired_csv,
            tokenizer_name=cfg.tokenizer_name,
            max_length=cfg.max_length,
            target_col=target_col,
        ))

        self.image_size = cfg.image_size
        self.return_label = cfg.return_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])

        # Image tensor
        x_img = load_image_tensor(img_path, image_size=self.image_size)

        # Tokenized text
        t = self.text_ds[idx]

        out = {
            "image": x_img,
            "input_ids": t["input_ids"],
            "attention_mask": t["attention_mask"],
        }

        # include uid if available
        if "uid" in t:
            out["uid"] = t["uid"]

        # optional dummy label for compatibility
        if self.return_label:
            out["label"] = torch.tensor(0, dtype=torch.long)

        return out
