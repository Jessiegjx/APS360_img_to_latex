import os
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

AIDA_ROOT = r"D:\APS360_proj\data\raw\AIDA"

BATCHES = range(1, 11)
SAMPLE_SIZE = 500      # Set to None to use every image

random.seed(42)

def analyze_batch(batch):

    folder = os.path.join(
        AIDA_ROOT,
        f"batch_{batch}",
        "background_images"
    )

    print("\n" + "="*60)
    print(f"BATCH {batch}")
    print("="*60)

    files = os.listdir(folder)
    if SAMPLE_SIZE is not None:
        files = random.sample(files, min(SAMPLE_SIZE, len(files)))

    missing = 0
    corrupt = 0

    widths = []
    heights = []
    ratios = []

    brightness = []
    contrast = []
    dark_pixels = []

    alpha_means = []

    for filename in tqdm(files):

        path = os.path.join(folder, filename)

        if not os.path.exists(path):
            missing += 1
            continue

        try:
            img = Image.open(path)

            w, h = img.size
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)

            arr = np.array(img)

            # RGB or RGBA
            if arr.ndim == 3 and arr.shape[2] == 4:
                alpha_means.append(np.mean(arr[:, :, 3]))
                rgb = arr[:, :, :3]
            elif arr.ndim == 3:
                rgb = arr
            else:
                rgb = np.stack([arr] * 3, axis=-1)

            gray = np.mean(rgb, axis=2)

            brightness.append(np.mean(gray))
            contrast.append(np.std(gray))
            dark_pixels.append(np.mean(gray < 100))

        except Exception:
            corrupt += 1

    print("\nImages analyzed:", len(files))

    print("\nImage problems")
    print("Missing:", missing)
    print("Corrupt:", corrupt)

    print("\nSize")
    print("Width:",
          "min", min(widths),
          "max", max(widths),
          "mean", np.mean(widths))
    print("Height:",
          "min", min(heights),
          "max", max(heights),
          "mean", np.mean(heights))

    print("\nAspect ratio")
    print("min", min(ratios),
          "max", max(ratios),
          "mean", np.mean(ratios))

    print("\nBrightness")
    print("min", min(brightness),
          "max", max(brightness),
          "mean", np.mean(brightness))

    print("\nContrast")
    print("min", min(contrast),
          "max", max(contrast),
          "mean", np.mean(contrast))

    print("\nDark pixel percentage")
    print("mean:", np.mean(dark_pixels))
    print("max:", max(dark_pixels))

    if alpha_means:
        print("\nAlpha channel")
        print("mean alpha:", np.mean(alpha_means))
    else:
        print("\nAlpha channel")
        print("No alpha channel found.")


for b in BATCHES:
    analyze_batch(b)