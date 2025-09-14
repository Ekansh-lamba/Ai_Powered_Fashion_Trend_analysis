import json
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "best_resnet18.pt"
CLASSES_PATH = ARTIFACTS / "classes.json"

IMG_SIZE = 224

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def load_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

def predict_csv(input_csv, out_csv):
    with open(CLASSES_PATH) as f:
        classes = json.load(f)
    model = load_model(len(classes))

    df = pd.read_csv(input_csv)
    outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = Image.open(row["image_path"]).convert("RGB")
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
        pred_idx = int(torch.argmax(logits, dim=1))
        outputs.append({
            "image_path": row["image_path"],
            "pred_style": classes[pred_idx],
            "prob": max(probs)
        })
    out_df = pd.DataFrame(outputs)
    out_df.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")

if __name__ == "__main__":
    # Example: predict on the *same* labels.csv (ignores your 'style' column)
    INPUT = ROOT / "data" / "labels.csv"
    OUT = ROOT / "artifacts" / "predictions.csv"
    predict_csv(INPUT, OUT)
