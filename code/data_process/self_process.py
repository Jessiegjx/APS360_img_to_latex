import os
import cv2

INPUT=r"D:\APS360_proj\data\self_collected\self_images"
OUTPUT=r"D:\APS360_proj\data\self_collected\self_images_processed"

os.makedirs(OUTPUT,exist_ok=True)

for f in os.listdir(INPUT):

    path=os.path.join(INPUT,f)

    img=cv2.imread(path)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray=cv2.fastNlMeansDenoising(
        gray,
        None,
        h=8
    )

    clahe=cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8,8)
    )

    gray=clahe.apply(gray)

    gray=cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    gray=cv2.convertScaleAbs(
        gray,
        alpha=1.08,
        beta=10
    )

    gray=cv2.copyMakeBorder(
        gray,
        20,20,20,20,
        cv2.BORDER_CONSTANT,
        value=255
    )

    cv2.imwrite(
        os.path.join(OUTPUT,f),
        gray
    )

print("done")