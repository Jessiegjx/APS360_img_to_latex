import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json

class HME100kDataset(Dataset):
    def __init__(self, txt_path, image_folder, tokenizer_path, processed=False):
        self.image_folder = image_folder
        self.samples = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img, formula = line.split("\t", 1)
                    self.samples.append((img, formula))

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            self.tokenizer = json.load(f)["vocab"]

        self.target_h = 160
        self.target_w = 480

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.485],
                std=[0.229,0.224,0.225]
            )
        ])

        self.pad = self.tokenizer["<PAD>"]
        self.bos = self.tokenizer["<BOS>"]
        self.eos = self.tokenizer["<EOS>"]
        self.unk = self.tokenizer["<UNK>"]

    def resize_and_pad(self, img):
        ratio = min(self.target_w/img.width, self.target_h/img.height)
        new_w = int(img.width*ratio)
        new_h = int(img.height*ratio)

        img = img.resize((new_w,new_h), Image.LANCZOS)

        canvas = Image.new(
            "RGB",
            (self.target_w,self.target_h),
            (255,255,255)
        )
        canvas.paste(img,(0,0))
        return canvas

    def tokenize(self, formula):
        tokens = formula.split()
        ids = [self.bos]

        for t in tokens:
            ids.append(self.tokenizer.get(t,self.unk))

        ids.append(self.eos)
        return torch.tensor(ids,dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        img_path, formula = self.samples[idx]

        full_path = os.path.join(
            self.image_folder,
            os.path.basename(img_path)
        )

        img = Image.open(full_path).convert("RGB")
        img = self.resize_and_pad(img)
        img = self.transform(img)

        caption = self.tokenize(formula)
        return img, caption
# old not resized version    
# import os
# import torch
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# import json


# class HME100kDataset(Dataset):

#     def __init__(
#         self,
#         txt_path,
#         image_folder,
#         tokenizer_path,
#         processed=False
#     ):

#         self.image_folder = image_folder

#         # read train.txt
#         self.samples = []

#         with open(txt_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()

#                 if not line:
#                     continue

#                 img, formula = line.split("\t",1)
#                 self.samples.append((img, formula))


#         with open(tokenizer_path,"r",encoding="utf-8") as f:
#             self.tokenizer=json.load(f)["vocab"]

#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485,0.456,0.406],
#                 std=[0.229,0.224,0.225]
#             )
#         ])

#         self.pad=self.tokenizer["<PAD>"]
#         self.bos=self.tokenizer["<BOS>"]
#         self.eos=self.tokenizer["<EOS>"]
#         self.unk=self.tokenizer["<UNK>"]


#     def tokenize(self,formula):
#         tokens=formula.split()
#         ids=[self.bos]

#         for t in tokens:
#             ids.append(self.tokenizer.get(t, self.unk))
#         ids.append(self.eos)

#         return torch.tensor(ids)

#     def __len__(self):
#         return len(self.samples)


#     def __getitem__(self,idx):
#         img_path, formula = self.samples[idx]
#         full_path=os.path.join(
#             self.image_folder,
#             os.path.basename(img_path)
#         )

#         # grayscale HME -> RGB for ResNet
#         img=Image.open(full_path).convert("RGB")
#         img=self.transform(img)
#         caption=self.tokenize(formula)

#         return img, caption