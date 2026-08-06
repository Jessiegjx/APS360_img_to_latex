import os
from PIL import Image

folder = r"D:\APS360_proj\data\self_collected\self_images_processed"

ratios = []

for f in os.listdir(folder):
    if f.endswith(".jpg") or f.endswith(".png"):
        img = Image.open(os.path.join(folder,f))
        w,h = img.size
        ratios.append(w/h)

print(sum(ratios)/len(ratios))