import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import Counter

IMAGE_FOLDER = r"D:\APS360_proj\data\raw\HME100k\images"

files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(".png")
]

print("="*60)
print("HME100K IMAGE STATISTICS")
print("="*60)
print("Total image files:", len(files))

widths=[]
heights=[]
ratios=[]
brightness=[]
contrast=[]
dark_pixels=[]
missing=0
corrupt=0

# for duplicate detection
image_sizes = Counter()

for filename in tqdm(files):

    path=os.path.join(
        IMAGE_FOLDER,
        filename
    )

    if not os.path.exists(path):
        missing+=1
        continue
    try:
        img=Image.open(path)
        # duplicate check by dimensions
        image_sizes[img.size]+=1
        w,h=img.size

        widths.append(w)
        heights.append(h)
        ratios.append(w/h)
        gray=np.array(
            img.convert("L")
        )

        brightness.append(
            np.mean(gray)
        )

        contrast.append(
            np.std(gray)
        )

        dark_pixels.append(
            np.mean(gray < 100)
        )

    except Exception:
        corrupt+=1


print("\nImage problems")
print("Missing:", missing)
print("Corrupt:", corrupt)

print("\nSize")

print(
    "Width:",
    "min", min(widths),
    "max", max(widths),
    "mean", np.mean(widths)
)

print(
    "Height:",
    "min", min(heights),
    "max", max(heights),
    "mean", np.mean(heights)
)


print("\nAspect ratio")

print(
    "min",
    min(ratios)
)

print(
    "max",
    max(ratios)
)

print(
    "mean",
    np.mean(ratios)
)

print("\nBrightness")

print(
    "min",
    min(brightness),
    "max",
    max(brightness),
    "mean",
    np.mean(brightness)
)

print("\nContrast")

print(
    "min",
    min(contrast),
    "max",
    max(contrast),
    "mean",
    np.mean(contrast)
)

print("\nDark pixel percentage")

print(
    "mean:",
    np.mean(dark_pixels)
)

print(
    "max:",
    max(dark_pixels)
)

print("\nDuplicate dimension check")

duplicate_sizes = {
    k:v for k,v in image_sizes.items()
    if v>1
}

print(
    "Repeated image dimensions:",
    len(duplicate_sizes)
)