from src.models.image_encoder import DenseNetImageEncoder
from src.datasets.image_dataset import ImageDatasetConfig, ImageFolderBinaryDataset
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
        for x, y, _ in loader:
            x = x.to(device)
            out = model(x)
            p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(y.numpy())

    auc = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return auc, f1, cm


def main():
    seed = 42
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ImageFolderBinaryDataset(ImageDatasetConfig(root_dir="data/images"))
    n_total = len(ds)

    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=16,
                             shuffle=False, num_workers=2)

    model = DenseNetImageEncoder(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    best_val_auc = -1.0
    Path("reports").mkdir(exist_ok=True)

    for epoch in range(5):
        model.train()
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        val_auc, val_f1, val_cm = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1} | VAL AUC: {val_auc:.4f} | VAL F1: {val_f1:.4f}")
        print("VAL Confusion Matrix:\n", val_cm)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "reports/image_model_best.pt")
            print("✅ Saved new best model")

    # Final test evaluation with best checkpoint
    model.load_state_dict(torch.load(
        "reports/image_model_best.pt", map_location=device))
    test_auc, test_f1, test_cm = evaluate(model, test_loader, device)

    print("\n=== FINAL TEST RESULTS (best checkpoint) ===")
    print(f"TEST AUC: {test_auc:.4f} | TEST F1: {test_f1:.4f}")
    print("TEST Confusion Matrix:\n", test_cm)

    # Save metrics summary
    with open("reports/image_metrics.txt", "w") as f:
        f.write(f"VAL_BEST_AUC={best_val_auc:.4f}\n")
        f.write(f"TEST_AUC={test_auc:.4f}\n")
        f.write(f"TEST_F1={test_f1:.4f}\n")
        f.write(f"TEST_CM=\n{test_cm}\n")

    print("Saved metrics to reports/image_metrics.txt")


if __name__ == "__main__":
    main()
