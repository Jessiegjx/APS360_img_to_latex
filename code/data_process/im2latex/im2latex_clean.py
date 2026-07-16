'''
generate new csv files that only contain good image name and their labels.

'''

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

DATASETS = {
    "train": r"D:\APS360_proj\data\raw\im2Latex\im2latex_train.csv",
    "validate": r"D:\APS360_proj\data\raw\im2Latex\im2latex_validate.csv",
    "test": r"D:\APS360_proj\data\raw\im2Latex\im2latex_test.csv"
}

IMAGE_FOLDER = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"
OUT_FOLDER = r"D:\APS360_proj\data\processed\im2Latex"

os.makedirs(OUT_FOLDER, exist_ok=True)

MAX_WIDTH = 500
MAX_HEIGHT = 200
MAX_TOKENS = 200

def clean_dataset(name, csv_path):

    df = pd.read_csv(csv_path)

    valid_rows = []
    seen_images = set()

    removed = {
        "missing": 0,
        "corrupt": 0,
        "blank": 0,
        "duplicate": 0,
        "large": 0,
        "long_formula": 0
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):

        img_name = row["image"]
        img_path = os.path.join(IMAGE_FOLDER, img_name)

        if img_name in seen_images:
            removed["duplicate"] += 1
            continue

        if not os.path.exists(img_path):
            removed["missing"] += 1
            continue

        if len(str(row["formula"]).split()) > MAX_TOKENS:
            removed["long_formula"] += 1
            continue

        try:
            img = Image.open(img_path).convert("L")
            w, h = img.size

            if w > MAX_WIDTH or h > MAX_HEIGHT:
                removed["large"] += 1
                continue

            arr = np.array(img)

            if np.mean(arr) > 254:
                removed["blank"] += 1
                continue

        except:
            removed["corrupt"] += 1
            continue

        seen_images.add(img_name)
        valid_rows.append(row)


    clean_df = pd.DataFrame(valid_rows)

    clean_df.to_csv(
        os.path.join(OUT_FOLDER, f"{name}_clean.csv"),
        index=False
    )

    print("\n" + "="*50)
    print(name.upper())
    print("="*50)
    print("Original:", len(df))
    print("Remaining:", len(clean_df))

    for k,v in removed.items():
        print(f"Removed {k}: {v}")

for name, path in DATASETS.items():
    clean_dataset(name, path)