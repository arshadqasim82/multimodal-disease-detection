"""
Generates a curated Grad-CAM set across TP/TN/FP/FN categories.

Outputs
-------
- reports/figures/gradcam_curated/*.png
- reports/tables/gradcam_curated_index.csv
"""

from src.explainability.gradcam import GradCAM
from src.models.image_encoder import DenseNetImageEncoder
from src.datasets.image_dataset import ImageDatasetConfig, ImageFolderBinaryDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ImageFolderBinaryDataset(ImageDatasetConfig(root_dir="data/images"))

    model = DenseNetImageEncoder(num_classes=2).to(device)
    model.load_state_dict(torch.load(
        "reports/image_model_best.pt", map_location=device))
    model.eval()

    # ✅ Ensure gradients are enabled (Grad-CAM needs them for hooks)
    torch.set_grad_enabled(True)

    target_layer = model.backbone.features[-1]
    cam = GradCAM(model, target_layer)

    out_dir = Path("reports/figures/gradcam_curated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find indices for TP/TN/FP/FN
    buckets = {"TP": [], "TN": [], "FP": [], "FN": []}

    # scan until we fill each bucket with up to k examples
    k = 3  # examples per bucket (12 total)
    for i in range(len(ds)):
        x, y, path = ds[i]
        x1 = x.unsqueeze(0).to(device)

        # ✅ IMPORTANT: no torch.no_grad() here (Grad-CAM requires grad-enabled activations)
        logits = model(x1)
        prob1 = torch.softmax(logits, dim=1)[0, 1].item()
        pred = int(torch.argmax(logits, dim=1).item())

        y_int = int(y)
        if y_int == 1 and pred == 1 and len(buckets["TP"]) < k:
            buckets["TP"].append((i, y_int, pred, prob1, path))
        elif y_int == 0 and pred == 0 and len(buckets["TN"]) < k:
            buckets["TN"].append((i, y_int, pred, prob1, path))
        elif y_int == 0 and pred == 1 and len(buckets["FP"]) < k:
            buckets["FP"].append((i, y_int, pred, prob1, path))
        elif y_int == 1 and pred == 0 and len(buckets["FN"]) < k:
            buckets["FN"].append((i, y_int, pred, prob1, path))

        if all(len(v) >= k for v in buckets.values()):
            break

    # Generate Grad-CAM figures
    rows = []
    for bucket, items in buckets.items():
        for (i, y_int, pred, prob1, path) in items:
            x, y, _ = ds[i]
            x1 = x.unsqueeze(0).to(device)

            heat, logits = cam(x1)
            img = x1[0].detach().cpu().permute(1, 2, 0).numpy()
            h = heat[0, 0].cpu().numpy()

            fname = f"{bucket}_idx{i}_y{y_int}_p{pred}.png"
            plt.figure()
            plt.imshow(img)
            plt.imshow(h, alpha=0.4)
            plt.axis("off")
            plt.title(
                f"{bucket} | y={y_int} pred={pred} p1={prob1:.3f}\n{Path(path).name}")
            plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
            plt.close()

            rows.append([bucket, i, y_int, pred, prob1,
                        Path(path).name, str(out_dir / fname)])

    # Save table
    table_path = Path("reports/tables")
    table_path.mkdir(parents=True, exist_ok=True)
    out_csv = table_path / "gradcam_curated_index.csv"
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "index", "y_true", "y_pred",
                   "p(class1)", "image_name", "figure_path"])
        w.writerows(rows)

    print("Saved curated Grad-CAM set to:", out_dir)
    print("Saved index CSV to:", out_csv)
    for b in ["TP", "TN", "FP", "FN"]:
        print(b, "count:", len(buckets[b]))


if __name__ == "__main__":
    main()
