"""
Purpose
-------
Defines a multimodal dataset wrapper that returns aligned image + text inputs.

Pairing Strategy
----------------
Pairs image samples and text samples by index (engineering demonstration).
This is not patient-level fusion because the datasets are not linked.

Labeling
--------
The multimodal target label is taken from the IMAGE task
(Effusion vs No Finding), to support consistent evaluation.

Outputs
-------
Returns dict:
- image: tensor [3, H, W]
- input_ids: tensor [L]
- attention_mask: tensor [L]
- label: scalar (0/1)
"""

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset

from src.datasets.image_dataset import ImageDatasetConfig, ImageFolderBinaryDataset
from src.datasets.text_dataset import TextDatasetConfig, TextCSVDataset


@dataclass
class MultimodalDatasetConfig:
    image_root: str
    text_csv: str
    tokenizer_name: str
    max_length: int = 192
    image_size: int = 224


class PairedIndexMultimodalDataset(Dataset):
    """
    Pairs image samples and text samples by index (engineering demonstration).
    Label is taken from the IMAGE task (Effusion vs No Finding).
    """

    def __init__(self, cfg: MultimodalDatasetConfig):
        self.image_ds = ImageFolderBinaryDataset(
            ImageDatasetConfig(cfg.image_root, cfg.image_size))
        self.text_ds = TextCSVDataset(TextDatasetConfig(
            cfg.text_csv, cfg.tokenizer_name, cfg.max_length))

        self.n = min(len(self.image_ds), len(self.text_ds))

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_img, y_img, _ = self.image_ds[idx]
        t = self.text_ds[idx]
        return {
            "image": x_img,
            "input_ids": t["input_ids"],
            "attention_mask": t["attention_mask"],
            "label": y_img,
        }
