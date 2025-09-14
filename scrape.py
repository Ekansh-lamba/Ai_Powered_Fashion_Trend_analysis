# scrape.py
import os, csv, time, re
from pathlib import Path
import requests
from bs4 import BeautifulSoup

OUT_DIR = Path("data/raw_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = Path("data/metadata.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

URLS = [
    # Add pages you have permission to scrape
    # "https://example.com/fashion/page1",
]

def safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', s)

def fetch_images_from(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    rows = []
    for img in soup.select("img"):
        src = img.get("src") or ""
        alt = img.get("alt") or ""
        if not src or src.startswith("data:"): 
            continue
        if src.startswith("//"): src = "https:" + src
        if src.startswith("/"):  src = url.rstrip("/") + src

        try:
            img_bytes = requests.get(src, timeout=20).content
            fname = safe_filename(src.split("/")[-1]) or f"img_{int(time.time()*1000)}.jpg"
            fpath = OUT_DIR / fname
            with open(fpath, "wb") as f:
                f.write(img_bytes)
            rows.append({"image_path": str(fpath), "source_url": url, "alt": alt, "label": ""})
            time.sleep(0.5)
        except Exception as e:
            print("Skip:", src, e)
    return rows

all_rows = []
for u in URLS:
    all_rows.extend(fetch_images_from(u))

if all_rows:
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "source_url", "alt", "label"])
        if write_header: w.writeheader()
        w.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows to {CSV_PATH}")
else:
    print("No rows scraped. Check URLs/selectors.")
