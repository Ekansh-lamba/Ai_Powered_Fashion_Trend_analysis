from pathlib import Path
import pandas as pd

CSV = Path("data/labels.csv")
CLASSES = ["Bohemian","Minimalist","Streetwear","Y2K","Cottagecore"]

df = pd.read_csv(CSV)

# normalize common issues
df["style"] = df["style"].astype(str).str.strip()
df["style"] = df["style"].replace({"bohemian":"Bohemian","minimalist":"Minimalist",
                                   "streetwear":"Streetwear","y2k":"Y2K","cottagecore":"Cottagecore"})

print("Total rows:", len(df))
print("Unique styles found:", sorted(df["style"].unique()))
print("\nCounts per class:")
print(df["style"].value_counts())

missing = df[~df["style"].isin(CLASSES)]
if not missing.empty:
    print("\nRows with invalid/empty style:")
    print(missing.head(10))
