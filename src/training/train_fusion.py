"""
Trains multimodal fusion models (concat and attention) using frozen encoders.

UPDATED FOR IU PAIRED DATASET (image_path + report text)
--------------------------------------------------------
We train a binary "matching" task:
- label=1: image paired with its true report
- label=0: image paired with a random wrong report (negative sample)

Outputs
-------
- reports/fusion_concat_best.pt
- reports/fusion_attention_best.pt
- reports/fusion_*_metrics.txt

Evaluation
----------
- Split by uid to avoid leakage across views/studies
- ROC-AUC, PR-AUC
- F1 at validation-tuned threshold
- Confusion matrices
"""

from src.datasets.multimodal_dataset import MultimodalDatasetConfig, PairedCSVMatchDataset
from src.models.feature_extractors import DenseNetFeatureExtractor, BertCLSFeatureExtractor
from src.models.fusion import ConcatFusion, AttentionFusion
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
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# IMPORTANT: this expects you replaced src/datasets/multimodal_dataset.py
# with the IU paired CSV matching dataset class:


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def best_f1_threshold(labels, probs):
    """
    Choose threshold that maximizes F1 on validation set.
    """
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


def collect_probs_concat(img_enc, txt_enc, fusion, loader, device):
    img_enc.eval()
    txt_enc.eval()
    fusion.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b in loader:
            img = b["image"].to(device)
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            y = b["label"].cpu().numpy()

            fi = img_enc(img)
            ft = txt_enc(input_ids, attention_mask)
            logits = fusion(fi, ft)

            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(y)
    return np.asarray(labels), np.asarray(probs)


def collect_probs_attn(img_enc, txt_enc, fusion, loader, device):
    img_enc.eval()
    txt_enc.eval()
    fusion.eval()
    probs, labels = [], []
    all_w = []
    with torch.no_grad():
        for b in loader:
            img = b["image"].to(device)
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            y = b["label"].cpu().numpy()

            fi = img_enc(img)
            ft = txt_enc(input_ids, attention_mask)
            logits, w = fusion(fi, ft)
            all_w.append(w.cpu())

            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(y)

    w_mean = torch.cat(all_w, dim=0).mean(dim=0).tolist()  # [w_img, w_txt]
    return np.asarray(labels), np.asarray(probs), w_mean


def metrics_from_probs(labels, probs, threshold=0.5):
    auc = roc_auc_score(labels, probs)
    prauc = average_precision_score(labels, probs)
    preds = (np.asarray(probs) >= threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    return auc, prauc, f1, cm


def build_uid_split_indices(paired_csv: str, seed: int = 42):
    """
    Split by uid to avoid leakage across frontal/lateral views.
    Returns arrays of row indices for train/val/test.
    """
    df = pd.read_csv(paired_csv)
    if "uid" not in df.columns:
        raise ValueError(
            "paired_csv must contain a 'uid' column for uid-based splitting.")

    uids = df["uid"].unique()

    # 70/15/15 split
    train_u, temp_u = train_test_split(uids, test_size=0.30, random_state=seed)
    val_u, test_u = train_test_split(temp_u, test_size=0.50, random_state=seed)

    train_idx = df.index[df["uid"].isin(train_u)].to_numpy()
    val_idx = df.index[df["uid"].isin(val_u)].to_numpy()
    test_idx = df.index[df["uid"].isin(test_u)].to_numpy()

    return train_idx, val_idx, test_idx


def run_experiment(
    fusion_type: str,
    paired_csv: str,
    tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    batch_size: int = 8,
    max_length: int = 192,
    image_size: int = 224,
    neg_prob: float = 0.5,
    epochs: int = 10,
    lr: float = 2e-4,
):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MultimodalDatasetConfig(
        paired_csv=paired_csv,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        image_size=image_size,
        neg_prob=neg_prob,
    )

    train_idx, val_idx, test_idx = build_uid_split_indices(paired_csv, seed=42)

    # Build datasets with deterministic seeds for negative sampling
    train_ds = PairedCSVMatchDataset(cfg, indices=train_idx, seed=42)
    val_ds = PairedCSVMatchDataset(cfg, indices=val_idx, seed=123)
    test_ds = PairedCSVMatchDataset(cfg, indices=test_idx, seed=999)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Frozen encoders (same as your original)
    img_enc = DenseNetFeatureExtractor().to(device)
    txt_enc = BertCLSFeatureExtractor(tokenizer_name).to(device)

    for p in img_enc.parameters():
        p.requires_grad = False
    for p in txt_enc.parameters():
        p.requires_grad = False

    img_dim = img_enc.out_dim
    txt_dim = txt_enc.out_dim

    if fusion_type == "concat":
        fusion = ConcatFusion(img_dim, txt_dim, hidden=256,
                              num_classes=2).to(device)
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr)
    elif fusion_type == "attention":
        fusion = AttentionFusion(
            img_dim, txt_dim, hidden=256, num_classes=2).to(device)
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr)
    else:
        raise ValueError("fusion_type must be 'concat' or 'attention'")

    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    Path("reports").mkdir(exist_ok=True)
    best_path = f"reports/fusion_{fusion_type}_best.pt"

    for epoch in range(epochs):
        fusion.train()
        for b in tqdm(train_loader, desc=f"{fusion_type} Epoch {epoch+1}/{epochs}"):
            img = b["image"].to(device)
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            y = b["label"].to(device)

            with torch.no_grad():
                fi = img_enc(img)
                ft = txt_enc(input_ids, attention_mask)

            optimizer.zero_grad()
            if fusion_type == "concat":
                logits = fusion(fi, ft)
            else:
                logits, _ = fusion(fi, ft)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Validation
        if fusion_type == "concat":
            y_val, p_val = collect_probs_concat(
                img_enc, txt_enc, fusion, val_loader, device)
            w_mean = None
        else:
            y_val, p_val, w_mean = collect_probs_attn(
                img_enc, txt_enc, fusion, val_loader, device)

        val_auc, val_prauc, _, _ = metrics_from_probs(
            y_val, p_val, threshold=0.5)
        t_best, val_f1_best = best_f1_threshold(y_val, p_val)
        _, _, _, val_cm_best = metrics_from_probs(
            y_val, p_val, threshold=t_best)

        if w_mean is None:
            print(
                f"{fusion_type} | VAL ROC-AUC: {val_auc:.4f} | VAL PR-AUC: {val_prauc:.4f} | "
                f"best_t={t_best:.2f} | VAL F1@best_t: {val_f1_best:.4f}\nVAL CM@best_t:\n{val_cm_best}"
            )
        else:
            print(
                f"{fusion_type} | VAL ROC-AUC: {val_auc:.4f} | VAL PR-AUC: {val_prauc:.4f} | "
                f"mean[w_img,w_txt]={w_mean} | best_t={t_best:.2f} | VAL F1@best_t: {val_f1_best:.4f}\n"
                f"VAL CM@best_t:\n{val_cm_best}"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(fusion.state_dict(), best_path)
            print("✅ Saved new best fusion model")

    # Test with best checkpoint
    fusion.load_state_dict(torch.load(best_path, map_location=device))

    if fusion_type == "concat":
        y_val, p_val = collect_probs_concat(
            img_enc, txt_enc, fusion, val_loader, device)
        y_test, p_test = collect_probs_concat(
            img_enc, txt_enc, fusion, test_loader, device)
        w_mean = None
    else:
        y_val, p_val, w_mean_val = collect_probs_attn(
            img_enc, txt_enc, fusion, val_loader, device)
        y_test, p_test, w_mean_test = collect_probs_attn(
            img_enc, txt_enc, fusion, test_loader, device)
        w_mean = w_mean_test

    t_best, _ = best_f1_threshold(y_val, p_val)
    test_auc, test_prauc, test_f1, test_cm = metrics_from_probs(
        y_test, p_test, threshold=t_best)

    print("\n=== FINAL TEST RESULTS (best checkpoint, threshold tuned on VAL) ===")
    if w_mean is None:
        print(
            f"TEST ROC-AUC: {test_auc:.4f} | TEST PR-AUC: {test_prauc:.4f} | TEST F1@t={t_best:.2f}: {test_f1:.4f}")
    else:
        print(
            f"TEST ROC-AUC: {test_auc:.4f} | TEST PR-AUC: {test_prauc:.4f} | "
            f"TEST F1@t={t_best:.2f}: {test_f1:.4f} | mean[w_img,w_txt]={w_mean}"
        )
    print("TEST CM:\n", test_cm)

    # Save metrics
    metrics_path = f"reports/fusion_{fusion_type}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"PAIRED_CSV={paired_csv}\n")
        f.write(f"NEG_PROB={neg_prob}\n")
        f.write(f"VAL_BEST_ROC_AUC={best_val_auc:.4f}\n")
        f.write(f"BEST_THRESHOLD={t_best:.2f}\n")
        f.write(f"TEST_ROC_AUC={test_auc:.4f}\n")
        f.write(f"TEST_PR_AUC={test_prauc:.4f}\n")
        f.write(f"TEST_F1={test_f1:.4f}\n")
        f.write(f"TEST_CM=\n{test_cm}\n")
        if w_mean is not None:
            f.write(f"MEAN_WEIGHTS={w_mean}\n")

    print(f"Saved metrics to {metrics_path}")


def main():
    # UPDATE THIS PATH to your IU paired CSV
    paired_csv = "/content/multimodal-dx/IU_DIR/iu_dataset.csv"

    print("Running CONCAT experiment...")
    run_experiment("concat", paired_csv=paired_csv)

    print("\nRunning ATTENTION experiment...")
    run_experiment("attention", paired_csv=paired_csv)


if __name__ == "__main__":
    main()
