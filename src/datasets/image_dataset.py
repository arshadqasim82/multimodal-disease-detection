"""
Purpose
-------
Loads chest X-ray images for binary classification from an image folder structure,
and also provides a helper for loading a single image by path for paired datasets.

Folder Dataset Format (legacy)
------------------------------
image_root/
  0/  (class 0 images)
  1/  (class 1 images)

Outputs (legacy)
----------------
ImageFolderBinaryDataset returns: (image_tensor, label, path)

New Helper
----------
load_image_tensor(image_path, image_size) -> image_tensor
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path


@dataclass
class ImageDatasetConfig:
    image_root: str
    image_size: int = 224


def load_image_tensor(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Loads an image from a given path and returns a tensor [3, H, W].
    Minimal transforms: resize + ToTensor.

    This is used for IU paired CSV training where each row has an image_path.
    """
    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    return tfm(img)


class ImageFolderBinaryDataset(Dataset):
    """
    Loads images from:
      root/0/*.png
      root/1/*.png
    Returns (image_tensor, label, path)
    """

    def __init__(self, cfg: ImageDatasetConfig):
        self.root = Path(cfg.image_root)
        self.image_size = cfg.image_size

        self.samples = []
        for label in [0, 1]:
            class_dir = self.root / str(label)
            if class_dir.exists():
                for p in class_dir.rglob("*"):
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.samples.append((str(p), label))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found under {self.root} in class subfolders 0/ and 1/")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path, label = self.samples[idx]
        x = load_image_tensor(path, image_size=self.image_size)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, path
