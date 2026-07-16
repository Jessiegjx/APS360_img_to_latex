import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset


from im2latex_dataset import Im2LatexDataset
from collate_trans import collate_transformer

from resnet_encoder import ResNetEncoder
from transformer_decoder import TransformerDecoder



# =====================================================
# Random seed
# =====================================================

random.seed(42)
torch.manual_seed(42)



# =====================================================
# Paths
# =====================================================

TRAIN_CSV = r"D:\APS360_proj\data\processed\im2Latex\train_clean.csv"

VAL_CSV = r"D:\APS360_proj\data\processed\im2Latex\validate_clean.csv"

IMAGE_DIR = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"

TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"
CHECKPOINT = "transformer_best.pth"



# =====================================================
# Hyperparameters
# =====================================================

TRAIN_SIZE = 10000
VAL_SIZE = 1000

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)

# =====================================================
# Load vocabulary
# =====================================================

with open(TOKENIZER,"r",encoding="utf-8") as f:

    vocab=json.load(f)["vocab"]


PAD_IDX = vocab["<PAD>"]



# =====================================================
# Dataset
# =====================================================

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



train_indices=random.sample(
    range(len(train_dataset)),
    TRAIN_SIZE
)


val_indices=random.sample(
    range(len(val_dataset)),
    VAL_SIZE
)



train_dataset=Subset(
    train_dataset,
    train_indices
)


val_dataset=Subset(
    val_dataset,
    val_indices
)




# =====================================================
# Data loaders
# =====================================================

train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_transformer,
    num_workers=0,
    pin_memory=True
)



val_loader=DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_transformer,
    num_workers=0,
    pin_memory=True
)



# =====================================================
# Metrics
# =====================================================


def token_accuracy(
    outputs,
    targets,
    pad_idx
):

    preds=outputs.argmax(-1)


    mask=targets!=pad_idx


    correct=(
        (preds==targets)
        &
        mask
    ).sum().item()


    total=mask.sum().item()


    return correct/max(total,1)




def exact_match(
    outputs,
    targets,
    pad_idx
):

    preds=outputs.argmax(-1)


    total=targets.size(0)

    correct=0


    for p,t in zip(preds,targets):

        mask=t!=pad_idx


        if torch.equal(
            p[mask],
            t[mask]
        ):
            correct+=1


    return correct/total



# =====================================================
# Train one epoch
# =====================================================


def train_epoch(
    encoder,
    decoder,
    loader,
    optimizer,
    criterion
):

    encoder.train()
    decoder.train()

    total_loss = 0
    total_token = 0
    total_exact = 0


    for images,captions,padding_mask in tqdm(loader):

        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)


        optimizer.zero_grad()


        image_features = encoder(images)


        decoder_input = captions[:,:-1]

        target = captions[:,1:]


        output = decoder(
            image_features,
            decoder_input,
            padding_mask[:,:-1]
        )


        loss = criterion(
            output.reshape(-1, output.size(-1)),
            target.reshape(-1)
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        total_token += token_accuracy(
            output,
            target,
            PAD_IDX
        )


        total_exact += exact_match(
            output,
            target,
            PAD_IDX
        )


    return (
        total_loss/len(loader),
        total_token/len(loader),
        total_exact/len(loader)
    )


# =====================================================
# Validation
# =====================================================


def validate(
    encoder,
    decoder,
    loader,
    criterion
):

    encoder.eval()
    decoder.eval()


    total_loss=0

    token_acc=0

    exact_acc=0



    with torch.no_grad():


        for images,captions,padding_mask in loader:


            images=images.to(DEVICE)

            captions=captions.to(DEVICE)

            padding_mask=padding_mask.to(DEVICE)



            image_features=encoder(images)



            decoder_input=captions[:,:-1]


            target=captions[:,1:]



            output=decoder(
                image_features,
                decoder_input,
                padding_mask[:,:-1]
            )



            loss=criterion(
                output.reshape(
                    -1,
                    output.size(-1)
                ),

                target.reshape(-1)
            )



            total_loss+=loss.item()



            token_acc+=token_accuracy(
                output,
                target,
                PAD_IDX
            )


            exact_acc+=exact_match(
                output,
                target,
                PAD_IDX
            )



    n=len(loader)


    return (
        total_loss/n,
        token_acc/n,
        exact_acc/n
    )



# =====================================================
# Models
# =====================================================


encoder=ResNetEncoder(
    embed_dim=256
).to(DEVICE)



decoder=TransformerDecoder(
    vocab_size=len(vocab),
    embed_dim=256,
    heads=8,
    layers=3
).to(DEVICE)



# Freeze ResNet initially

for p in encoder.backbone.parameters():

    p.requires_grad=False




criterion=nn.CrossEntropyLoss(
    ignore_index=PAD_IDX
)



optimizer=optim.AdamW(
    list(decoder.parameters())
    +
    list(encoder.projection.parameters())
    +
    [encoder.pos_embedding],

    lr=LR
)



# =====================================================
# Training
# =====================================================

train_losses=[]
val_losses=[]

train_token_acc=[]
val_token_acc=[]

train_exact_acc=[]
val_exact_acc=[]



best=float("inf")



for epoch in range(EPOCHS):


    train_loss, train_token, train_exact = train_epoch(
        encoder,
        decoder,
        train_loader,
        optimizer,
        criterion
    )


    val_loss,v_token,v_exact=validate(
        encoder,
        decoder,
        val_loader,
        criterion
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_token_acc.append(train_token)
    val_token_acc.append(v_token)

    train_exact_acc.append(train_exact)
    val_exact_acc.append(v_exact)



    print(
        f"""
Epoch {epoch+1}

Train loss:
{train_loss:.4f}

Validation loss:
{val_loss:.4f}

Token accuracy:
{v_token:.3f}

Exact match:
{v_exact:.3f}

"""
    )



    if val_loss < best:

        best=val_loss

        torch.save(
            {
            "encoder":encoder.state_dict(),
            "decoder":decoder.state_dict()
            },

            CHECKPOINT
        )

        print("saved best model")


# =====================================================
# Plot 1: Loss
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    train_losses,
    label="Training loss"
)

plt.plot(
    val_losses,
    label="Validation loss"
)


plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")

plt.title(
    "CNN-Transformer Training and Validation Loss"
)

plt.legend()

plt.grid()

plt.savefig(
    "transformer_loss.png",
    dpi=300
)

plt.show()



# =====================================================
# Plot 2: Accuracy
# =====================================================

plt.figure(figsize=(8,5))

# Token accuracy

plt.plot(
    train_token_acc,
    label="Train token accuracy"
)

plt.plot(
    val_token_acc,
    label="Validation token accuracy"
)

# Exact match

plt.plot(
    train_exact_acc,
    label="Train exact match"
)

plt.plot(
    val_exact_acc,
    label="Validation exact match"
)



plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNN-Transformer Token Accuracy and Exact Match"
)

plt.legend()

plt.grid()

plt.savefig(
    "transformer_accuracy.png",
    dpi=300
)


plt.show()