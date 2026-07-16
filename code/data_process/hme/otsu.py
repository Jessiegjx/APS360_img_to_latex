import os
import random
import cv2
import matplotlib.pyplot as plt

IMAGE_FOLDER = r"D:\APS360_proj\data\raw\HME100k\images"

NUM_IMAGES = 5
PADDING = 20

random.seed(2)

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

chosen = random.sample(image_files, NUM_IMAGES)

fig, axes = plt.subplots(NUM_IMAGES, 2, figsize=(12, 3.5 * NUM_IMAGES))

for i, filename in enumerate(chosen):

    path = os.path.join(IMAGE_FOLDER, filename)

    img = cv2.imread(path)
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.fastNlMeansDenoising(gray, None, h=8)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = cv2.copyMakeBorder(
        binary,
        PADDING,
        PADDING,
        PADDING,
        PADDING,
        cv2.BORDER_CONSTANT,
        value=255
    )

    axes[i,0].imshow(original)
    axes[i,0].set_title(f"Original\n{filename}")
    axes[i,0].axis("off")

    axes[i,1].imshow(binary, cmap="gray", vmin=0, vmax=255)
    axes[i,1].set_title("Otsu Threshold")
    axes[i,1].axis("off")

plt.tight_layout()
plt.savefig("hme100k_otsu_threshold_examples.png", dpi=300)
plt.show()