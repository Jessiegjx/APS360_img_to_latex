import os
import random
import cv2
import matplotlib.pyplot as plt

IMAGE_FOLDER = r"D:\APS360_proj\data\raw\HME100k\images"
NUM_IMAGES = 5
random.seed(2)

files=[
    f for f in os.listdir(IMAGE_FOLDER)
    if f.endswith(".png")
]

chosen=random.sample(
    files,
    NUM_IMAGES
)

fig,axes = plt.subplots(
    NUM_IMAGES,
    2,
    figsize=(12,15)
)

for i,filename in enumerate(chosen):
    path=os.path.join(
        IMAGE_FOLDER,
        filename
    )

    img=cv2.imread(path)

    gray=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # denoise
    gray=cv2.fastNlMeansDenoising(
        gray,
        None,
        h=8
    )

    # contrast enhancement
    clahe=cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8,8)
    )

    gray=clahe.apply(gray)

    # normalize background
    gray=cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # preserve bold equations
    gray=cv2.convertScaleAbs(
        gray,
        alpha=1.08,
        beta=10
    )

    # padding
    gray=cv2.copyMakeBorder(
        gray,
        20,
        20,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=255
    )

    axes[i,0].imshow(
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[i,0].set_title(
        "Original\n"+filename
    )
    axes[i,0].axis("off")

    axes[i,1].imshow(
        gray,
        cmap="gray"
    )

    axes[i,1].set_title(
        "Processed"
    )
    axes[i,1].axis("off")

plt.tight_layout()

plt.savefig(
    "hme100k_preprocessing_examples.png",
    dpi=300
)

plt.show()