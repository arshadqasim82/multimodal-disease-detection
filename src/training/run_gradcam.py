from src.explainability.gradcam import GradCAM
from src.models.image_encoder import DenseNetImageEncoder
from src.datasets.image_dataset import ImageDatasetConfig, ImageFolderBinaryDataset
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ImageFolderBinaryDataset(ImageDatasetConfig(root_dir="data/images"))
    loader_idx = [0, 10, 20, 30, 40, 50]  # sample indices; we’ll refine later

    model = DenseNetImageEncoder(num_classes=2).to(device)
    model.load_state_dict(torch.load(
        "reports/image_model_best.pt", map_location=device))
    model.eval()

    # target layer: DenseNet last conv block lives at model.backbone.features
    target_layer = model.backbone.features[-1]
    cam = GradCAM(model, target_layer)

    out_dir = Path("reports/figures/gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in loader_idx:
        x, y, path = ds[i]
        x = x.unsqueeze(0).to(device)
        heat, logits = cam(x)

        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = int(torch.argmax(logits, dim=1).item())

        # Plot: original (grayscale) + CAM overlay
        img = x[0].detach().cpu().permute(1, 2, 0).numpy()
        h = heat[0, 0].cpu().numpy()

        plt.figure()
        plt.imshow(img)
        plt.imshow(h, alpha=0.4)
        plt.axis("off")
        plt.title(f"y={int(y)} pred={pred} p1={prob:.3f}\n{Path(path).name}")
        plt.savefig(
            out_dir / f"gradcam_{i}_y{int(y)}_p{pred}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Saved Grad-CAM images to {out_dir}")


if __name__ == "__main__":
    main()
