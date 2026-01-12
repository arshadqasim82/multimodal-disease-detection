"""
Trains a text-only ClinicalBERT baseline on data/clinical_text.csv.

Outputs
-------
- reports/text_model_best.pt
- reports/text_metrics.txt

Evaluation
----------
- train/val/test split
- ROC-AUC and F1 with confusion matrix
"""

from src.models.text_encoder import ClinicalBertClassifier
from src.datasets.text_dataset import TextDatasetConfig, TextCSVDataset
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


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            probs.extend(p)
            labels.extend(y)

    auc = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return auc, f1, cm


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "emilyalsentzer/Bio_ClinicalBERT"

    ds = TextCSVDataset(TextDatasetConfig(
        csv_path="data/clinical_text.csv",
        tokenizer_name=model_name,
        max_length=192
    ))

    n_total = len(ds)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=8,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8,
                             shuffle=False, num_workers=2)

    model = ClinicalBertClassifier(model_name=model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_val_auc = -1.0
    Path("reports").mkdir(exist_ok=True)

    for epoch in range(5):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_auc, val_f1, val_cm = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1} | VAL AUC: {val_auc:.4f} | VAL F1: {val_f1:.4f}")
        print("VAL Confusion Matrix:\n", val_cm)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "reports/text_model_best.pt")
            print("✅ Saved new best text model")

    model.load_state_dict(torch.load(
        "reports/text_model_best.pt", map_location=device))
    test_auc, test_f1, test_cm = evaluate(model, test_loader, device)

    print("\n=== FINAL TEST RESULTS (best checkpoint) ===")
    print(f"TEST AUC: {test_auc:.4f} | TEST F1: {test_f1:.4f}")
    print("TEST Confusion Matrix:\n", test_cm)

    with open("reports/text_metrics.txt", "w") as f:
        f.write(f"VAL_BEST_AUC={best_val_auc:.4f}\n")
        f.write(f"TEST_AUC={test_auc:.4f}\n")
        f.write(f"TEST_F1={test_f1:.4f}\n")
        f.write(f"TEST_CM=\n{test_cm}\n")

    print("Saved metrics to reports/text_metrics.txt")


if __name__ == "__main__":
    main()
