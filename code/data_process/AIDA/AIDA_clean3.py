'''
show before after image of 3 AIDA images
'''

import os
import random
import cv2
import matplotlib.pyplot as plt

IMAGE_FOLDER = r"D:\APS360_proj\data\raw\AIDA\batch_1\background_images"

NUM_IMAGES = 3
PADDING = 20
random.seed(52)

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

chosen = random.sample(image_files, NUM_IMAGES)

fig, axes = plt.subplots(NUM_IMAGES, 2, figsize=(12, 4 * NUM_IMAGES))

for i, filename in enumerate(chosen):

    path = os.path.join(IMAGE_FOLDER, filename)

    # ---------- Original ----------
    img = cv2.imread(path)
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------- Preprocessing ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, h=8)

    # Stronger local contrast (keeps handwriting bold)
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8,8)
    )
    gray = clahe.apply(gray)

    # Slightly brighten the whole image
    gray = cv2.convertScaleAbs(
        gray,
        alpha=1.08,   # contrast
        beta=12       # brightness
    )

    # Add white padding
    gray = cv2.copyMakeBorder(
        gray,
        PADDING,
        PADDING,
        PADDING,
        PADDING,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # ---------- Plot ----------
    axes[i,0].imshow(original)
    axes[i,0].set_title(f"Original\n{filename}")
    axes[i,0].axis("off")

    axes[i,1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[i,1].set_title("Processed")
    axes[i,1].axis("off")

plt.tight_layout()
plt.savefig("aida_preprocessing_examples.png", dpi=300)
plt.show()