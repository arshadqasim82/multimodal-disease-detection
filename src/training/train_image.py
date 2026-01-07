from src.models.image_encoder import DenseNetImageEncoder
from src.datasets.image_dataset import ImageDatasetConfig, ImageFolderBinaryDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ImageFolderBinaryDataset(ImageDatasetConfig(root_dir="data/images"))
    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = DenseNetImageEncoder(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(3):  # small number for baseline
        model.train()
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                out = model(x)
                p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
                labels.extend(y.numpy())

        auc = roc_auc_score(labels, probs)
        preds = [1 if p > 0.5 else 0 for p in probs]
        f1 = f1_score(labels, preds)

        print(f"Epoch {epoch+1} | AUC: {auc:.4f} | F1: {f1:.4f}")

    # Save model
    Path("reports").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "reports/image_model.pt")
    print("Model saved to reports/image_model.pt")


if __name__ == "__main__":
    main()
