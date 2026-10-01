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
    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    return tfm(img)


class ImageFolderBinaryDataset(Dataset):

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
