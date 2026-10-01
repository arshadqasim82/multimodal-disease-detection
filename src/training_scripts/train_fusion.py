

from src.datasets.multimodal_dataset import (
    MultimodalDatasetConfig,
    PairedCSVMatchDataset,
)
from src.models.feature_extractors import (
    DenseNetFeatureExtractor,
    BertCLSFeatureExtractor,
)
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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, average_precision_score


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


def metrics_from_probs(labels, probs, threshold):
    auc = roc_auc_score(labels, probs)
    prauc = average_precision_score(labels, probs)
    preds = (np.asarray(probs) >= threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    return auc, prauc, f1, cm


@torch.no_grad()
def collect_probs_concat(img_enc, txt_enc, fusion, loader, device):
    img_enc.eval()
    txt_enc.eval()
    fusion.eval()
    probs, labels = [], []

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


@torch.no_grad()
def collect_probs_attn(img_enc, txt_enc, fusion, loader, device):
    img_enc.eval()
    txt_enc.eval()
    fusion.eval()
    probs, labels = [], []
    weights = []

    for b in loader:
        img = b["image"].to(device)
        input_ids = b["input_ids"].to(device)
        attention_mask = b["attention_mask"].to(device)
        y = b["label"].cpu().numpy()

        fi = img_enc(img)
        ft = txt_enc(input_ids, attention_mask)
        logits, w = fusion(fi, ft)
        weights.append(w.cpu())

        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p)
        labels.extend(y)

    w_mean = torch.cat(weights, dim=0).mean(dim=0).tolist()
    return np.asarray(labels), np.asarray(probs), w_mean


def build_uid_split_indices(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    if "uid" not in df.columns:
        raise ValueError("iu_dataset.csv must contain a 'uid' column")

    uids = df["uid"].unique()
    train_u, temp_u = train_test_split(uids, test_size=0.30, random_state=seed)
    val_u, test_u = train_test_split(temp_u, test_size=0.50, random_state=seed)

    train_idx = df.index[df["uid"].isin(train_u)].to_numpy()
    val_idx = df.index[df["uid"].isin(val_u)].to_numpy()
    test_idx = df.index[df["uid"].isin(test_u)].to_numpy()

    return train_idx, val_idx, test_idx


def run_experiment(fusion_type: str, paired_csv: str):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MultimodalDatasetConfig(
        paired_csv=paired_csv,
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=192,
        image_size=224,
        neg_prob=0.5,
    )

    train_idx, val_idx, test_idx = build_uid_split_indices(paired_csv)

    train_ds = PairedCSVMatchDataset(cfg, indices=train_idx, seed=42)
    val_ds = PairedCSVMatchDataset(cfg, indices=val_idx, seed=123)
    test_ds = PairedCSVMatchDataset(cfg, indices=test_idx, seed=999)

    train_loader = DataLoader(train_ds, batch_size=8,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8,
                             shuffle=False, num_workers=2)

    img_enc = DenseNetFeatureExtractor().to(device)
    txt_enc = BertCLSFeatureExtractor(
        "emilyalsentzer/Bio_ClinicalBERT").to(device)

    for p in img_enc.parameters():
        p.requires_grad = False
    for p in txt_enc.parameters():
        p.requires_grad = False

    img_dim = img_enc.out_dim
    txt_dim = txt_enc.out_dim

    if fusion_type == "concat":
        fusion = ConcatFusion(img_dim, txt_dim, hidden=256,
                              num_classes=2).to(device)
    elif fusion_type == "attention":
        fusion = AttentionFusion(
            img_dim, txt_dim, hidden=256, num_classes=2).to(device)
    else:
        raise ValueError("fusion_type must be 'concat' or 'attention'")

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    Path("reports").mkdir(exist_ok=True)
    best_auc = -1.0
    best_path = f"reports/fusion_{fusion_type}_best.pt"

    for epoch in range(10):
        fusion.train()
        for b in tqdm(train_loader, desc=f"{fusion_type} Epoch {epoch+1}/10"):
            img = b["image"].to(device)
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            y = b["label"].to(device)

            with torch.no_grad():
                fi = img_enc(img)
                ft = txt_enc(input_ids, attention_mask)

            optimizer.zero_grad()
            logits = fusion(
                fi, ft) if fusion_type == "concat" else fusion(fi, ft)[0]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        if fusion_type == "concat":
            yv, pv = collect_probs_concat(
                img_enc, txt_enc, fusion, val_loader, device)
            w_mean = None
        else:
            yv, pv, w_mean = collect_probs_attn(
                img_enc, txt_enc, fusion, val_loader, device)

        t_best, _ = best_f1_threshold(yv, pv)
        val_auc, val_prauc, val_f1, val_cm = metrics_from_probs(yv, pv, t_best)

        print(
            f"{fusion_type.upper()} | VAL AUC={val_auc:.4f} | PR-AUC={val_prauc:.4f} | F1={val_f1:.4f}")
        print("VAL CM:\n", val_cm)
        if w_mean is not None:
            print("Mean attention weights [image, text]:", w_mean)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(fusion.state_dict(), best_path)
            print("✅ Saved new best model")

    fusion.load_state_dict(torch.load(best_path, map_location=device))

    if fusion_type == "concat":
        yt, pt = collect_probs_concat(
            img_enc, txt_enc, fusion, test_loader, device)
    else:
        yt, pt, w_test = collect_probs_attn(
            img_enc, txt_enc, fusion, test_loader, device)

    t_best, _ = best_f1_threshold(yv, pv)
    test_auc, test_prauc, test_f1, test_cm = metrics_from_probs(yt, pt, t_best)

    print("\n=== FINAL TEST RESULTS ===")
    print(
        f"TEST AUC={test_auc:.4f} | PR-AUC={test_prauc:.4f} | F1={test_f1:.4f}")
    print("TEST CM:\n", test_cm)

    with open(f"reports/fusion_{fusion_type}_metrics.txt", "w") as f:
        f.write(f"CSV={paired_csv}\n")
        f.write(f"TEST_AUC={test_auc:.4f}\n")
        f.write(f"TEST_PR_AUC={test_prauc:.4f}\n")
        f.write(f"TEST_F1={test_f1:.4f}\n")
        f.write(f"TEST_CM=\n{test_cm}\n")


def main():
    paired_csv = "/content/multimodal-dx/{IU_DIR}/iu_dataset.csv"
    print("Using dataset:", paired_csv)

    print("\nRunning CONCAT fusion...")
    run_experiment("concat", paired_csv)

    print("\nRunning ATTENTION fusion...")
    run_experiment("attention", paired_csv)


if __name__ == "__main__":
    main()
