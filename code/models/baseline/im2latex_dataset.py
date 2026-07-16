import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Im2LatexDataset(Dataset):

    def __init__(self, csv_path, image_folder, tokenizer_path):

        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            self.tokenizer = json.load(f)["vocab"]

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])

        self.pad = self.tokenizer["<PAD>"]
        self.bos = self.tokenizer["<BOS>"]
        self.eos = self.tokenizer["<EOS>"]
        self.unk = self.tokenizer["<UNK>"]


    def tokenize(self, formula):

        tokens = formula.split()

        ids = [self.bos]

        for t in tokens:
            ids.append(
                self.tokenizer.get(t, self.unk)
            )

        ids.append(self.eos)

        return torch.tensor(ids)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.image_folder,
            row["image"]
        )

        img = Image.open(img_path).convert("L")

        img = self.transform(img)

        caption = self.tokenize(
            row["formula"]
        )

        return img, caption