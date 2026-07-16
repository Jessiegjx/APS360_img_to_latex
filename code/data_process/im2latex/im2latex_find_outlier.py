import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

CSV = r"D:\APS360_proj\data\raw\im2Latex\im2latex_test.csv"
IMAGE_FOLDER = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"

df = pd.read_csv(CSV)

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_name = row["image"]
    formula = row["formula"]

    path = os.path.join(IMAGE_FOLDER, image_name)

    if not os.path.exists(path):
        continue

    try:
        img = Image.open(path)
        width, height = img.size

        if height > 200 or height < 20 or width > 1000:
            print("\nSuspicious image:")
            print("Image:", image_name)
            print("Size:", width, "x", height)
            print("Formula:", formula)

    except:
        pass