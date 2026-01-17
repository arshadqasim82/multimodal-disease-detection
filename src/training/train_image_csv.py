import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# Dataset (reads split CSV)
# -----------------------
class CSVImageDataset(Dataset):
    def __init__(self, csv_path: str, image_roots, train: bool):
        self.df = pd.read_csv(csv_path)

        required = ["Image Index", "label"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        self.image_roots = [Path(p) for p in image_roots]
        self.train = train

        # Medically safe augmentation (NO flips)
        if train:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.90, 1.00)),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def _resolve_path(self, image_name: str) -> Path:
        # Robust: try multiple roots (NIH sample sometimes has two layouts)
        for r in self.image_roots:
            p = r / image_name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Could not find {image_name} under roots: {[str(r) for r in self.image_roots]}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["Image Index"]
        y = int(row["label"])

        img_path = self._resolve_path(image_name)
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)

        return x, y, str(img_path)


# -----------------------
# Model
# -----------------------
class DenseNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        in_feats = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_feats, 1)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(1)
        return logits


# -----------------------
# Train/Eval helpers
# -----------------------
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, ps, preds = [], [], []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        prob = torch.sigmoid(logits)

        ys.append(y.cpu().numpy())
        ps.append(prob.cpu().numpy())
        preds.append((prob > 0.5).long().cpu().numpy())

    y_true = np.concatenate(ys)
    p = np.concatenate(ps)
    y_pred = np.concatenate(preds)

    auc = roc_auc_score(y_true, p) if len(
        np.unique(y_true)) > 1 else float("nan")
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return auc, f1, cm, y_true, p


def train():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CSV splits (Phase 2 outputs)
    train_csv = "splits/train.csv"
    val_csv = "splits/val.csv"
    test_csv = "splits/test.csv"

    # NIH image roots (try both common layouts)
    image_roots = [
        "nih_sample/sample/images",
        "nih_sample/sample/sample/images",
    ]

    train_ds = CSVImageDataset(train_csv, image_roots=image_roots, train=True)
    val_ds = CSVImageDataset(val_csv, image_roots=image_roots, train=False)
    test_ds = CSVImageDataset(test_csv, image_roots=image_roots, train=False)

    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = DenseNetBinary().to(device)

    # Handle class imbalance with pos_weight
    y_train = pd.read_csv(train_csv)["label"].values
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_auc = -1.0
    best_path = Path("checkpoints/image_best_csv.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # Simple early stopping
    patience = 3
    bad_epochs = 0

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_auc, val_f1, val_cm, _, _ = eval_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | train_loss={total_loss/len(train_loader):.4f} | val_auc={val_auc:.4f} | val_f1={val_f1:.4f}")
        print("VAL CM:\n", val_cm)

        # checkpoint on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"model_state": model.state_dict()}, best_path)
            bad_epochs = 0
            print("✅ Saved new best model:", best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"⏹️ Early stopping (patience={patience}).")
                break

    # Final test with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_auc, test_f1, test_cm, y_true, p = eval_epoch(
        model, test_loader, device)
    print("\n=== FINAL TEST (patient-level split) ===")
    print(f"TEST AUC: {test_auc:.4f} | TEST F1: {test_f1:.4f}")
    print("TEST CM:\n", test_cm)

    # Save report
    Path("reports").mkdir(exist_ok=True)
    with open("reports/image_metrics_patient_split.txt", "w") as f:
        f.write(f"TEST AUC: {test_auc:.6f}\n")
        f.write(f"TEST F1: {test_f1:.6f}\n")
        f.write(f"TEST CM:\n{test_cm}\n")
        f.write(f"pos_weight (train): {float(pos_weight.item()):.6f}\n")

    print("Saved metrics to reports/image_metrics_patient_split.txt")


if __name__ == "__main__":
    train()
