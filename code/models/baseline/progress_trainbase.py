import os
import json
import random
import torch
import sys
from pathlib import Path

# sys.path.append(r"D:\APS360_proj\data\processed\im2Latex")

from torch.utils.data import DataLoader, Subset
from models.transformer.im2latex_dataset import Im2LatexDataset
from models.baseline.collate import collate_fn
from models.baseline.cnn_encoder import CNNEncoder
from models.baseline.rnn_decoder import RNNDecoder
from models.baseline.traing_baseline import train, plot_loss, plot_accuracy


random.seed(42)
torch.manual_seed(42)


TRAIN_CSV = r"D:\APS360_proj\data\processed\im2Latex\train_clean.csv"
VAL_CSV = r"D:\APS360_proj\data\processed\im2Latex\validate_clean.csv"

IMAGE_DIR = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"
TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"

TRAIN_SIZE = 10000
VAL_SIZE = 1000

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", DEVICE)


train_dataset = Im2LatexDataset(
    TRAIN_CSV,
    IMAGE_DIR,
    TOKENIZER
)

val_dataset = Im2LatexDataset(
    VAL_CSV,
    IMAGE_DIR,
    TOKENIZER
)


train_indices = random.sample(
    range(len(train_dataset)),
    TRAIN_SIZE
)

val_indices = random.sample(
    range(len(val_dataset)),
    VAL_SIZE
)


train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)


train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)


with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)["vocab"]


encoder = CNNEncoder()

decoder = RNNDecoder(
    vocab_size=len(vocab),
    embed_size=256,
    hidden_size=256
)


train_losses, val_losses, token_accs, exact_accs = train(
    encoder,
    decoder,
    train_loader,
    val_loader,
    vocab,
    epochs=EPOCHS,
    lr=LR,
    device=DEVICE
)


plot_loss(train_losses, val_losses)
plot_accuracy(token_accs, exact_accs)