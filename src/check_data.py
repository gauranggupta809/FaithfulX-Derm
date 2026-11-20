"""Run this FIRST. Verifies the dataset, split and lesion leakage."""
import argparse
from pathlib import Path
from common import load_meta

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--meta", default=None)
a = ap.parse_args()

df = load_meta(a.data_root, a.meta)
print("Total images:", len(df), "| unique lesions:", df["lesion_id"].nunique())
print("\nImages per split:\n", df["split"].value_counts())
print("\nClass x split:\n", df.pivot_table(index="class", columns="split", values="isic_id", aggfunc="count"))

sets = {s: set(g["lesion_id"]) for s, g in df.groupby("split")}
names = list(sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"Lesion overlap {names[i]} vs {names[j]}:", len(sets[names[i]] & sets[names[j]]))

missing = [p for p in df["image_path"] if not (Path(a.data_root) / p).exists()]
print("\nMissing image files:", len(missing))
if missing:
    print(missing[:5])
