# src/training/train_fusion.py
from src.models.fusion import ConcatFusion, AttentionFusion
from src.models.feature_extractors import DenseNetFeatureExtractor, BertCLSFeatureExtractor
from src.datasets.multimodal_dataset import MultimodalDatasetConfig, PairedIndexMultimodalDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch
import numpy as np
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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


def stratified_split_indices(ds, seed=42):
    """
    Stratify by image label (the target label of the multimodal dataset).
    """
    labels = []
    for i in range(len(ds)):
        labels.append(int(ds[i]["label"]))
    labels = np.asarray(labels)

    idx = np.arange(len(ds))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return train_idx, val_idx, test_idx


def run_experiment(fusion_type: str):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MultimodalDatasetConfig(
        image_root="data/images",
        text_csv="data/clinical_text.csv",
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=192,
        image_size=224
    )
    ds = PairedIndexMultimodalDataset(cfg)

    train_idx, val_idx, test_idx = stratified_split_indices(ds, seed=42)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=8,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8,
                             shuffle=False, num_workers=2)

    img_enc = DenseNetFeatureExtractor().to(device)
    txt_enc = BertCLSFeatureExtractor(
        "emilyalsentzer/Bio_ClinicalBERT").to(device)

    # Freeze encoders
    for p in img_enc.parameters():
        p.requires_grad = False
    for p in txt_enc.parameters():
        p.requires_grad = False

    img_dim = img_enc.out_dim
    txt_dim = txt_enc.out_dim

    if fusion_type == "concat":
        fusion = ConcatFusion(img_dim, txt_dim, hidden=256,
                              num_classes=2).to(device)
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=2e-4)
    elif fusion_type == "attention":
        fusion = AttentionFusion(
            img_dim, txt_dim, hidden=256, num_classes=2).to(device)
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=2e-4)
    else:
        raise ValueError("fusion_type must be concat or attention")

    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_path = f"reports/fusion_{fusion_type}_best.pt"
    Path("reports").mkdir(exist_ok=True)

    for epoch in range(10):
        fusion.train()
        for b in tqdm(train_loader, desc=f"{fusion_type} Epoch {epoch+1}"):
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

        # Validation metrics + threshold selection
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
            print(f"{fusion_type} | VAL ROC-AUC: {val_auc:.4f} | VAL PR-AUC: {val_prauc:.4f} | "
                  f"best_t={t_best:.2f} | VAL F1@best_t: {val_f1_best:.4f}\nVAL CM@best_t:\n{val_cm_best}")
        else:
            print(f"{fusion_type} | VAL ROC-AUC: {val_auc:.4f} | VAL PR-AUC: {val_prauc:.4f} | "
                  f"mean[w_img,w_txt]={w_mean} | best_t={t_best:.2f} | VAL F1@best_t: {val_f1_best:.4f}\n"
                  f"VAL CM@best_t:\n{val_cm_best}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(fusion.state_dict(), best_path)
            print("✅ Saved new best fusion model")

    # Final test evaluation using best checkpoint and best threshold from VAL of that checkpoint
    fusion.load_state_dict(torch.load(best_path, map_location=device))

    if fusion_type == "concat":
        y_val, p_val = collect_probs_concat(
            img_enc, txt_enc, fusion, val_loader, device)
        y_test, p_test = collect_probs_concat(
            img_enc, txt_enc, fusion, test_loader, device)
        w_mean = None
    else:
        y_val, p_val, w_mean = collect_probs_attn(
            img_enc, txt_enc, fusion, val_loader, device)
        y_test, p_test, w_mean_test = collect_probs_attn(
            img_enc, txt_enc, fusion, test_loader, device)
        # report test mean weights too
        w_mean = w_mean_test

    t_best, _ = best_f1_threshold(y_val, p_val)

    test_auc, test_prauc, test_f1, test_cm = metrics_from_probs(
        y_test, p_test, threshold=t_best)

    print("\n=== FINAL TEST RESULTS (best checkpoint, threshold tuned on VAL) ===")
    if w_mean is None:
        print(
            f"TEST ROC-AUC: {test_auc:.4f} | TEST PR-AUC: {test_prauc:.4f} | TEST F1@t={t_best:.2f}: {test_f1:.4f}")
    else:
        print(f"TEST ROC-AUC: {test_auc:.4f} | TEST PR-AUC: {test_prauc:.4f} | TEST F1@t={t_best:.2f}: {test_f1:.4f} "
              f"| mean[w_img,w_txt]={w_mean}")
    print("TEST CM:\n", test_cm)

    # Save metrics
    metrics_path = f"reports/fusion_{fusion_type}_metrics.txt"
    with open(metrics_path, "w") as f:
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
    print("Running CONCAT experiment...")
    run_experiment("concat")
    print("\nRunning ATTENTION experiment...")
    run_experiment("attention")


if __name__ == "__main__":
    main()
