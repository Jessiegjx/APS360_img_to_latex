import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

DATASETS = {
    # "train": r"D:\APS360_proj\data\raw\im2Latex\im2latex_train.csv",
    # "validate": r"D:\APS360_proj\data\raw\im2Latex\im2latex_validate.csv",
    "test": r"D:\APS360_proj\data\raw\im2Latex\im2latex_test.csv"
}

IMAGE_FOLDER = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"

def analyze_images(name, csv_path):
    print("\n" + "="*60)
    print(name.upper())
    print("="*60)
    df = pd.read_csv(csv_path)
    
    missing = 0
    corrupt = 0
    blank = 0
    widths = []
    heights = []
    aspect_ratios = []

    top_whitespace = []
    left_whitespace = []
    right_whitespace = []
    bottom_whitespace = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(
            IMAGE_FOLDER,
            row["image"]
        )
        if not os.path.exists(img_path):
            missing += 1
            continue
        try:
            img = Image.open(img_path).convert("L")

            w, h = img.size

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w/h)
            # Check blank images
            arr = np.array(img)
            if np.mean(arr) > 254:
                blank += 1

            # Whitespace analysis
            # formula pixels are darker
            mask = arr < 250
            coords = np.argwhere(mask)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                top_whitespace.append(y_min)
                bottom_whitespace.append(h - y_max)

                left_whitespace.append(x_min)
                right_whitespace.append(w - x_max)

        except Exception:
            corrupt += 1


    print("\nImage problems")

    print("Missing:", missing)
    print("Corrupt:", corrupt)
    print("Blank:", blank)
    print("\nImage size")

    print(
        "Width:",
        "min =", min(widths),
        "max =", max(widths),
        "mean =", np.mean(widths)
    )

    print(
        "Height:",
        "min =", min(heights),
        "max =", max(heights),
        "mean =", np.mean(heights)
    )

    print("\nAspect ratio")

    print(
        "min =", min(aspect_ratios),
        "max =", max(aspect_ratios),
        "mean =", np.mean(aspect_ratios)
    )

    print("\nWhitespace (pixels)")

    print(
        "Top mean:",
        np.mean(top_whitespace)
    )

    print(
        "Bottom mean:",
        np.mean(bottom_whitespace)
    )

    print(
        "Left mean:",
        np.mean(left_whitespace)
    )

    print(
        "Right mean:",
        np.mean(right_whitespace)
    )


for name, path in DATASETS.items():
    analyze_images(name, path)