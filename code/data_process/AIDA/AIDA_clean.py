import os
import random
from PIL import Image, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt

AIDA_ROOT = r"D:\APS360_proj\data\raw\AIDA"
OUTPUT_ROOT = r"D:\APS360_proj\data\processed\AIDA"

TRAIN_BATCHES = range(1, 9)
VAL_BATCH = [9]
TEST_BATCH = [10]

TARGET_SIZE = (512, 192)      # max (width, height)
PADDING = 10                  # pixels
NUM_EXAMPLES = 3

random.seed(42)


def preprocess(img):
    # RGB -> grayscale
    img = img.convert("L")

    # Improve contrast
    img = ImageOps.autocontrast(img)

    # Add white border
    img = ImageOps.expand(img, border=PADDING, fill=255)

    # Resize while preserving aspect ratio
    img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)

    # Paste onto white canvas
    canvas = Image.new("L", TARGET_SIZE, 255)

    x = (TARGET_SIZE[0] - img.width) // 2
    y = (TARGET_SIZE[1] - img.height) // 2

    canvas.paste(img, (x, y))

    return canvas


examples = []


def process_batches(split_name, batches):

    save_folder = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(save_folder, exist_ok=True)

    all_files = []

    for b in batches:
        folder = os.path.join(
            AIDA_ROOT,
            f"batch_{b}",
            "background_images"
        )

        for f in os.listdir(folder):
            all_files.append((folder, f))

    example_files = set(random.sample(all_files, NUM_EXAMPLES))

    for folder, filename in tqdm(all_files, desc=split_name):

        img_path = os.path.join(folder, filename)

        img = Image.open(img_path)

        processed = preprocess(img)

        processed.save(
            os.path.join(save_folder, filename)
        )

        if (folder, filename) in example_files:
            examples.append((img.copy(), processed.copy(), filename))


process_batches("train", TRAIN_BATCHES)
process_batches("validate", VAL_BATCH)
process_batches("test", TEST_BATCH)


# -------- Before / After figure --------

fig, axes = plt.subplots(NUM_EXAMPLES, 2, figsize=(10, 4 * NUM_EXAMPLES))

for i, (before, after, name) in enumerate(examples[:NUM_EXAMPLES]):

    axes[i,0].imshow(before)
    axes[i,0].set_title(f"Before\n{name}")
    axes[i,0].axis("off")

    axes[i,1].imshow(after, cmap="gray")
    axes[i,1].set_title("After")
    axes[i,1].axis("off")

plt.tight_layout()

save_path = os.path.join(
    OUTPUT_ROOT,
    "before_after_examples.png"
)

plt.savefig(save_path, dpi=300)
plt.close()

print("\nDone!")
print("Saved processed images to:", OUTPUT_ROOT)
print("Saved comparison figure to:", save_path)