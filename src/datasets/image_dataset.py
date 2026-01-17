"""
Purpose
-------
Loads NIH Chest X-ray images for binary classification.

Dataset Format
-------------
Expects a folder structure:
    data/images/
        class0/   # negative class (No Finding)
        class1/   # positive class (Effusion)

Outputs
-------
__getitem__ returns:
- x: torch.FloatTensor [3, H, W]
- y: torch.LongTensor scalar (0 or 1)
- path: str (filename for traceability)

Notes
-----
- Images are converted to grayscale then expanded to 3 channels for CNN backbones.
- Resizing is applied for fixed-size batching.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from PIL import Image
import torchvision.transforms as T


def load_image_tensor(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Loads an image from a given path and returns a normalized tensor [3,H,W].
    Assumes PNG/JPG.
    """
    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    return tfm(img)


@dataclass
class ImageDatasetConfig:
    root_dir: str
    image_size: int = 224


class ImageFolderBinaryDataset(Dataset):
    """
    Expects:
      root_dir/
        class0/*.png
        class1/*.png
    Returns: image_tensor (3,H,W), label (long), path (str)
    """

    def __init__(self, cfg: ImageDatasetConfig):
        self.root = Path(cfg.root_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ])

        self.samples: List[Tuple[str, int]] = []
        for label_name, label in [("class0", 0), ("class1", 1)]:
            class_dir = self.root / label_name
            if not class_dir.exists():
                continue
            for p in sorted(class_dir.glob("*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append((str(p), label))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found under {self.root}. Expected class0/ and class1/ folders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, path
