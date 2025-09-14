import json
from pathlib import Path
import random

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "labels.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True, parents=True)

CLASSES = ["Bohemian","Minimalist","Streetwear","Y2K","Cottagecore"]
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
PATIENCE = 3  # early stopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.image_path).convert("RGB")
        x = self.tf(img)
        y = CLASSES.index(row.style)
        return x, y

def build_model(num_classes=5):
    # ResNet18 is lightweight and great for beginners
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = True  # fine-tune everything (you can also freeze first)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in tqdm(loader, desc="Train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_true = [], []
    for xb, yb in tqdm(loader, desc="Val", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_true.extend(yb.cpu().tolist())
    f1 = f1_score(all_true, all_preds, average="macro")
    return running_loss/total, correct/total, f1, all_preds, all_true

def main():
    if not CSV_PATH.exists():
        raise SystemExit("labels.csv not found. Run data_prep.py and add some labels first.")

    df = pd.read_csv(CSV_PATH)
    # Keep only labeled rows (non-empty style and valid class)
    df = df[df["style"].isin(CLASSES)].copy()
    if len(df) < 50:
        print("Warning: very small labeled set. Label more images for better results.")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["style"]
    )

    train_ds = StyleDataset(train_df, augment=True)
    val_ds   = StyleDataset(val_df, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = -1.0
    patience_left = PATIENCE
    history = []

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        va_loss, va_acc, va_f1, preds, truth = evaluate(model, val_dl, criterion)
        print(f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1
        })
        # early stopping on F1
        if va_f1 > best_f1:
            best_f1 = va_f1
            patience_left = PATIENCE
            torch.save(model.state_dict(), ARTIFACTS / "best_resnet18.pt")
            with open(ARTIFACTS / "classes.json", "w") as f:
                json.dump(CLASSES, f)
            print("✅ Saved new best model.")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("⏹ Early stopping.")
                break

    pd.DataFrame(history).to_csv(ARTIFACTS / "training_history.csv", index=False)
    print("Done. Artifacts in:", ARTIFACTS)

if __name__ == "__main__":
    main()
