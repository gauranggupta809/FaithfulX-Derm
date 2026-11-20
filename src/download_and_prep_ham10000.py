"""Download HAM10000 from Harvard Dataverse, extract images, and build master_metadata.csv
with a fixed lesion-level stratified 70/15/15 split (seed 42).
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import time
import subprocess

FILES = {
    "metadata": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/4338392",
        "filename": "HAM10000_metadata.tab",
        "expected_size": 800000
    },
    "part_1": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/3172585",
        "filename": "HAM10000_images_part_1.zip",
        "expected_size": 1360000000
    },
    "part_2": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/3172584",
        "filename": "HAM10000_images_part_2.zip",
        "expected_size": 1400000000
    }
}

def download_file(info, target_path):
    target_path = Path(target_path)
    url = info["url"]
    expected_size = info.get("expected_size", 0)

    if target_path.exists() and target_path.stat().st_size >= expected_size:
        print(f"File already completely downloaded: {target_path.name} ({target_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return

    print(f"Downloading from {url} to {target_path} ...", flush=True)
    cmd = ["curl", "-C", "-", "-L", "-o", str(target_path), url]

    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries} downloading {target_path.name} ...", flush=True)
            res = subprocess.run(cmd, check=True)
            if target_path.exists() and target_path.stat().st_size >= expected_size:
                print(f"Successfully downloaded {target_path.name} ({target_path.stat().st_size / 1e6:.1f} MB)", flush=True)
                break
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt} encountered error: {e}. Retrying in 5 seconds...", flush=True)
            time.sleep(5)
    else:
        raise RuntimeError(f"Failed to download {target_path.name} after {max_retries} attempts.")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path.name} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"Extracted {zip_path.name}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="skin_lesion_dataset", help="Target dataset root directory (<DATASET_ROOT>)")
    args = ap.parse_args()
    root_dir = Path(args.out_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = root_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)
    images_dir = root_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 1. Download metadata and archives
    for key, info in FILES.items():
        download_file(info, downloads_dir / info["filename"])

    # 2. Extract images
    extract_zip(downloads_dir / FILES["part_1"]["filename"], images_dir)
    extract_zip(downloads_dir / FILES["part_2"]["filename"], images_dir)

    all_images = list(images_dir.glob("*.jpg"))
    print(f"Total extracted images: {len(all_images)}")

    # 3. Read metadata
    meta_file = downloads_dir / FILES["metadata"]["filename"]
    # Check delimiter
    df_raw = pd.read_csv(meta_file, sep="\t" if meta_file.suffix == ".tab" else ",")
    # Strip quotes if present
    apply_fn = getattr(df_raw, "map", getattr(df_raw, "applymap", None))
    df_raw = apply_fn(lambda x: x.strip('\"') if isinstance(x, str) else x)

    # Standardize columns: isic_id, lesion_id, class, split, image_path
    df_raw["isic_id"] = df_raw["image_id"]
    df_raw["class"] = df_raw["dx"]
    df_raw["image_path"] = df_raw["isic_id"].apply(lambda x: f"images/{x}.jpg")

    # Verify all image paths exist
    missing = [p for p in df_raw["image_path"] if not (root_dir / p).exists()]
    if missing:
        print(f"WARNING: {len(missing)} missing files! First 5: {missing[:5]}")
    else:
        print("All 10,015 image files exist on disk.")

    # 4. Create Lesion-Level Stratified 70/15/15 Split (Seed 42)
    # Lesion level: group by lesion_id and pick majority class for lesion (consistent diagnosis)
    lesion_df = df_raw.groupby("lesion_id")["class"].agg(lambda s: s.mode()[0]).reset_index()
    print(f"Total unique lesions: {len(lesion_df)}")

    # 70% train, 30% temp
    train_lesions, temp_lesions = train_test_split(
        lesion_df,
        test_size=0.30,
        random_state=42,
        stratify=lesion_df["class"]
    )
    # Split temp into 50% val and 50% test (15% and 15% of total)
    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=0.50,
        random_state=42,
        stratify=temp_lesions["class"]
    )

    split_map = {}
    for lid in train_lesions["lesion_id"]:
        split_map[lid] = "train"
    for lid in val_lesions["lesion_id"]:
        split_map[lid] = "val"
    for lid in test_lesions["lesion_id"]:
        split_map[lid] = "test"

    df_raw["split"] = df_raw["lesion_id"].map(split_map)

    # Save master_metadata.csv
    master_csv = root_dir / "master_metadata.csv"
    cols = ["isic_id", "lesion_id", "class", "split", "image_path", "dx_type", "age", "sex", "localization"]
    df_raw[cols].to_csv(master_csv, index=False)
    print(f"Saved {master_csv} with {len(df_raw)} records.")

if __name__ == "__main__":
    main()
