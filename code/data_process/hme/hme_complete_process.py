import os
import cv2
from tqdm import tqdm

INPUT_DIR=r"D:\APS360_proj\data\raw\HME100k\images"
OUTPUT_DIR=r"D:\APS360_proj\data\processed\hme100k_processed"

os.makedirs(OUTPUT_DIR,exist_ok=True)

files=[f for f in os.listdir(INPUT_DIR) if f.endswith(".png")]

for filename in tqdm(files):
    path=os.path.join(INPUT_DIR,filename)
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.fastNlMeansDenoising(gray,None,h=8)
    clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
    gray=clahe.apply(gray)
    gray=cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
    gray=cv2.convertScaleAbs(gray,alpha=1.08,beta=10)

    gray=cv2.copyMakeBorder(
        gray,
        20,20,20,20,
        cv2.BORDER_CONSTANT,
        value=255
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR,filename),gray)

print("Done.")