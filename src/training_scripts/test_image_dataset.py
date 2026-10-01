from src.datasets.image_dataset import ImageDatasetConfig, ImageFolderBinaryDataset
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


def main():
    ds = ImageFolderBinaryDataset(ImageDatasetConfig(root_dir="data/images"))
    print("Image dataset size:", len(ds))
    x, y, p = ds[0]
    print("Sample shape:", tuple(x.shape), "Label:", int(y), "Path:", p)
    print("Image dataset OK ✅")


if __name__ == "__main__":
    main()
