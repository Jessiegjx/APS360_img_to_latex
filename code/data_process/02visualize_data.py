import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


CSV = r"data/raw/im2latex/im2latex_train.csv"

IMAGE_FOLDER = r"data/raw/im2latex/formula_images_processed"


df = pd.read_csv(CSV)


samples = df.sample(9)


fig, axes = plt.subplots(
    3,
    3,
    figsize=(14,8)
)


for ax, (_, row) in zip(
    axes.flatten(),
    samples.iterrows()
):

    path = os.path.join(
        IMAGE_FOLDER,
        row["image"]
    )

    img = Image.open(path)


    ax.imshow(img, cmap="gray")

    ax.set_title(
        row["formula"][:60],
        fontsize=7
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "report_figures/im2latex_samples.png",
    dpi=300
)

plt.show()