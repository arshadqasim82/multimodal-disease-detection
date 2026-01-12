from src.models.fusion import ConcatFusion, AttentionFusion
from src.models.feature_extractors import DenseNetFeatureExtractor, BertCLSFeatureExtractor
from src.datasets.multimodal_dataset import MultimodalDatasetConfig, PairedIndexMultimodalDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
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


def evaluate_concat(img_enc, txt_enc, fusion, loader, device):
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

    auc = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return auc, f1, cm


def evaluate_attn(img_enc, txt_enc, fusion, loader, device):
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

    auc = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    w_mean = torch.cat(all_w, dim=0).mean(dim=0).tolist()  # [w_img, w_txt]
    return auc, f1, cm, w_mean


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

    n_total = len(ds)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=g)

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
    Path("reports").mkdir(exist_ok=True)
    best_path = f"reports/fusion_{fusion_type}_best.pt"

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

        # Validation
        if fusion_type == "concat":
            val_auc, val_f1, val_cm = evaluate_concat(
                img_enc, txt_enc, fusion, val_loader, device)
            print(
                f"{fusion_type} | VAL AUC: {val_auc:.4f} | VAL F1: {val_f1:.4f}\nVAL CM:\n{val_cm}")
        else:
            val_auc, val_f1, val_cm, w_mean = evaluate_attn(
                img_enc, txt_enc, fusion, val_loader, device)
            print(
                f"{fusion_type} | VAL AUC: {val_auc:.4f} | VAL F1: {val_f1:.4f} | mean[w_img,w_txt]={w_mean}\nVAL CM:\n{val_cm}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(fusion.state_dict(), best_path)
            print("✅ Saved new best fusion model")

    # Test
    fusion.load_state_dict(torch.load(best_path, map_location=device))

    if fusion_type == "concat":
        test_auc, test_f1, test_cm = evaluate_concat(
            img_enc, txt_enc, fusion, test_loader, device)
        print("\n=== FINAL TEST RESULTS (concat best) ===")
        print(
            f"TEST AUC: {test_auc:.4f} | TEST F1: {test_f1:.4f}\nTEST CM:\n{test_cm}")
        extra = ""
    else:
        test_auc, test_f1, test_cm, w_mean = evaluate_attn(
            img_enc, txt_enc, fusion, test_loader, device)
        print("\n=== FINAL TEST RESULTS (attention best) ===")
        print(
            f"TEST AUC: {test_auc:.4f} | TEST F1: {test_f1:.4f} | mean[w_img,w_txt]={w_mean}\nTEST CM:\n{test_cm}")
        extra = f"MEAN_WEIGHTS={w_mean}\n"

    with open(f"reports/fusion_{fusion_type}_metrics.txt", "w") as f:
        f.write(f"VAL_BEST_AUC={best_val_auc:.4f}\n")
        f.write(f"TEST_AUC={test_auc:.4f}\n")
        f.write(f"TEST_F1={test_f1:.4f}\n")
        f.write(f"TEST_CM=\n{test_cm}\n")
        f.write(extra)

    print(f"Saved metrics to reports/fusion_{fusion_type}_metrics.txt")


def main():
    print("Running CONCAT experiment...")
    run_experiment("concat")
    print("\nRunning ATTENTION experiment...")
    run_experiment("attention")


if __name__ == "__main__":
    main()
