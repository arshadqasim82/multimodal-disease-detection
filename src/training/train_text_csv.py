import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
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
class CSVTextDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len: int = 192):
        self.df = pd.read_csv(csv_path)

        required = ["text", "label"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        y = int(row["label"])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(y, dtype=torch.long),
        }
        return item


# -----------------------
# Model (ClinicalBERT + head)
# -----------------------
class BertBinaryClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.head(cls).squeeze(1)
        return logits


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, ps, preds = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
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
    return auc, f1, cm


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Patient-level splits from Phase 2
    train_csv = "splits/train.csv"
    val_csv = "splits/val.csv"
    test_csv = "splits/test.csv"

    # ClinicalBERT (good default). If your repo used a different one, swap it here.
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = CSVTextDataset(train_csv, tokenizer=tokenizer, max_len=192)
    val_ds = CSVTextDataset(val_csv, tokenizer=tokenizer, max_len=192)
    test_ds = CSVTextDataset(test_csv, tokenizer=tokenizer, max_len=192)

    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = BertBinaryClassifier(MODEL_NAME).to(device)

    # Class imbalance handling via pos_weight
    y_train = pd.read_csv(train_csv)["label"].values
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01)

    best_val_auc = -1.0
    best_path = Path("checkpoints/text_best_csv.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    patience = 2
    bad_epochs = 0

    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].float().to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_auc, val_f1, val_cm = eval_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | train_loss={total_loss/len(train_loader):.4f} | val_auc={val_auc:.4f} | val_f1={val_f1:.4f}")
        print("VAL CM:\n", val_cm)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"model_state": model.state_dict(),
                       "model_name": MODEL_NAME}, best_path)
            bad_epochs = 0
            print("✅ Saved new best text model:", best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"⏹️ Early stopping (patience={patience}).")
                break

    # Final test with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_auc, test_f1, test_cm = eval_epoch(model, test_loader, device)
    print("\n=== FINAL TEST (text-only, patient-level split) ===")
    print(f"TEST AUC: {test_auc:.4f} | TEST F1: {test_f1:.4f}")
    print("TEST CM:\n", test_cm)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/text_metrics_patient_split.txt", "w") as f:
        f.write(f"MODEL: {MODEL_NAME}\n")
        f.write(f"TEST AUC: {test_auc:.6f}\n")
        f.write(f"TEST F1: {test_f1:.6f}\n")
        f.write(f"TEST CM:\n{test_cm}\n")
        f.write(f"pos_weight (train): {float(pos_weight.item()):.6f}\n")

    print("Saved metrics to reports/text_metrics_patient_split.txt")


if __name__ == "__main__":
    train()
