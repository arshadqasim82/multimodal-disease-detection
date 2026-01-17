"""
Trains an image-only DenseNet baseline on IU paired data using a match/mismatch objective.

Task
----
Binary classification:
- label=1: the (image, report) pair is a true pair (positive)
- label=0: the report is randomly swapped (negative)

Image-only baseline uses ONLY the image but keeps the same labels.
This is a sanity baseline; it should be near chance since "matching" is not solvable from image alone.

Outputs
-------
- reports/image_model_best.pt
- reports/image_metrics.txt

Evaluation
----------
- uid-based train/val/test split (avoid leakage across views)
- ROC-AUC and F1 with confusion matrix
"""

from src.datasets.multimodal_dataset import MultimodalDatasetConfig, PairedCSVMatchDataset
from src.models.image_encoder import DenseNetImageEncoder
from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def best_f1_threshold(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy()

        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        probs.extend(p)
        labels.extend(y)

    auc = roc_auc_score(labels, probs)
    preds = (np.asarray(probs) >= threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    return float(auc), float(f1), cm, np.asarray(labels), np.asarray(probs)


def build_uid_split_indices(paired_csv: str, seed: int = 42):
    df = pd.read_csv(paired_csv)
    if "uid" not in df.columns:
        raise ValueError("paired_csv must contain 'uid' for uid-based split.")

    uids = df["uid"].unique()
    train_u, temp_u = train_test_split(uids, test_size=0.30, random_state=seed)
    val_u, test_u = train_test_split(temp_u, test_size=0.50, random_state=seed)

    train_idx = df.index[df["uid"].isin(train_u)].to_numpy()
    val_idx = df.index[df["uid"].isin(val_u)].to_numpy()
    test_idx = df.index[df["uid"].isin(test_u)].to_numpy()
    return train_idx, val_idx, test_idx


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paired_csv = "/content/multimodal-dx/IU_DIR/iu_dataset.csv"

    cfg = MultimodalDatasetConfig(
        paired_csv=paired_csv,
        # unused here, required by config
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=192,  # unused here
        image_size=224,
        neg_prob=0.5,
    )

    train_idx, val_idx, test_idx = build_uid_split_indices(
        paired_csv, seed=seed)

    # Reuse PairedCSVMatchDataset but only consume image + label
    train_ds = PairedCSVMatchDataset(cfg, indices=train_idx, seed=42)
    val_ds = PairedCSVMatchDataset(cfg, indices=val_idx, seed=123)
    test_ds = PairedCSVMatchDataset(cfg, indices=test_idx, seed=999)

    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=16,
                             shuffle=False, num_workers=2)

    model = DenseNetImageEncoder(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    best_val_auc = -1.0
    best_path = "reports/image_model_best.pt"
    Path("reports").mkdir(exist_ok=True)

    for epoch in range(5):
        model.train()
        for batch in tqdm(train_loader, desc=f"Image Epoch {epoch+1}/5"):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_auc, _, _, y_val, p_val = evaluate(
            model, val_loader, device, threshold=0.5)
        t_best, val_f1_best = best_f1_threshold(y_val, p_val)
        val_auc2, val_f1, val_cm, _, _ = evaluate(
            model, val_loader, device, threshold=t_best)

        print(
            f"Epoch {epoch+1} | VAL AUC: {val_auc2:.4f} | VAL F1@t={t_best:.2f}: {val_f1:.4f}")
        print("VAL Confusion Matrix:\n", val_cm)

        if val_auc2 > best_val_auc:
            best_val_auc = val_auc2
            torch.save(model.state_dict(), best_path)
            print("✅ Saved new best image model")

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, _, _, y_val, p_val = evaluate(model, val_loader, device, threshold=0.5)
    t_best, _ = best_f1_threshold(y_val, p_val)

    test_auc, test_f1, test_cm, _, _ = evaluate(
        model, test_loader, device, threshold=t_best)

    print("\n=== FINAL TEST RESULTS (best checkpoint, threshold tuned on VAL) ===")
    print(f"TEST AUC: {test_auc:.4f} | TEST F1@t={t_best:.2f}: {test_f1:.4f}")
    print("TEST Confusion Matrix:\n", test_cm)

    with open("reports/image_metrics.txt", "w") as f:
        f.write(f"PAIRED_CSV={paired_csv}\n")
        f.write(f"NEG_PROB={cfg.neg_prob}\n")
        f.write(f"VAL_BEST_AUC={best_val_auc:.4f}\n")
        f.write(f"BEST_THRESHOLD={t_best:.2f}\n")
        f.write(f"TEST_AUC={test_auc:.4f}\n")
        f.write(f"TEST_F1={test_f1:.4f}\n")
        f.write(f"TEST_CM=\n{test_cm}\n")

    print("Saved metrics to reports/image_metrics.txt")


if __name__ == "__main__":
    main()
