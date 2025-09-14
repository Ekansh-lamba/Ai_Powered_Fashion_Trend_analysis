import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_ROOT = ROOT / "data" / "H&M" / "images"
OUT_CSV = ROOT / "data" / "labels.csv"

# You can change/extend later
CLASSES = ["Bohemian","Minimalist","Streetwear","Y2K","Cottagecore"]

def main():
    if not IMG_ROOT.exists():
        raise SystemExit(f"Image root not found: {IMG_ROOT}")

    rows = []
    for p in IMG_ROOT.rglob("*.jpg"):
        # parent folder (e.g., 010) looks like an article group in your layout
        article_id = p.parent.name
        rows.append({
            "image_path": str(p.as_posix()),
            "article_id": article_id,
            "style": ""  # fill this manually for a subset to start
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path","article_id","style"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Template saved: {OUT_CSV}")
    print(f"Number of images found: {len(rows)}")
    print("Fill 'style' with one of:", CLASSES)

if __name__ == "__main__":
    main()
