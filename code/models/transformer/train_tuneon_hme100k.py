import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
# from im2latex_dataset import Im2LatexDataset
from collate_trans import collate_transformer
from resnet_encoder import ResNetEncoder
from transformer_decoder import TransformerDecoder
from torch.utils.data import random_split

# from hme100k_dataset import HME100kDataset
# from torch.utils.data import ConcatDataset

from hme100k_dataset import HME100kDataset
from torch.utils.data import random_split, Subset

def main():
    # ====================================================
    # Random seed
    # =====================================================
    random.seed(42)
    torch.manual_seed(42)

    # =====================================================
    # Paths
    # =====================================================
    # TRAIN_CSV = r"D:\APS360_proj\data\processed\im2Latex\train_clean.csv"
    # VAL_CSV = r"D:\APS360_proj\data\processed\im2Latex\validate_clean.csv"
    # IMAGE_DIR = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"
    TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"
    # CHECKPOINT = r"D:\APS360_proj\im2latex_best.pth" 
    CHECKPOINT = r"D:\APS360_proj\hme100k_finetuned_best.pth"

    HME_TRAIN = r"D:\APS360_proj\data\raw\HME100k\splits\train.txt"
    HME_VAL = r"D:\APS360_proj\data\raw\HME100k\splits\val.txt"
    HME_IMAGE_DIR = r"D:\APS360_proj\data\processed\hme100k_processed"
    
    # =====================================================
    # Hyperparameters
    # =====================================================

    # TRAIN_SIZE = 75000
    # VAL_SIZE = 8300
    BATCH_SIZE = 24
    EPOCHS = 30
    LR = 1e-5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", DEVICE)

    # =====================================================
    # Load vocabulary
    # =====================================================

    with open(TOKENIZER,"r",encoding="utf-8") as f:
        vocab=json.load(f)["vocab"]

    PAD_IDX = vocab["<PAD>"]

    # for testing
    id_to_token = {v:k for k,v in vocab.items()}


    # =====================================================
    # Dataset
    # =====================================================

    # train_dataset = Im2LatexDataset(TRAIN_CSV, IMAGE_DIR, TOKENIZER)
    # val_dataset = Im2LatexDataset(VAL_CSV,IMAGE_DIR,TOKENIZER )

    train_dataset = HME100kDataset(
        txt_path=HME_TRAIN,
        image_folder=HME_IMAGE_DIR,
        tokenizer_path=TOKENIZER
    )
    val_dataset = HME100kDataset(
        txt_path=HME_VAL,
        image_folder=HME_IMAGE_DIR,
        tokenizer_path=TOKENIZER
    )

    # =====================================================
    # Data loaders
    # =====================================================
    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_transformer,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

   
    val_loader=DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_transformer,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
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

    def exact_match(outputs, targets, pad_idx):
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
    def train_epoch(encoder, decoder, loader, optimizer, criterion):
        encoder.train()
        decoder.train()

        total_loss = 0
        total_token = 0
        total_exact = 0

        for images, captions, padding_mask in tqdm(loader):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)

            optimizer.zero_grad()

            decoder_input = captions[:, :-1]
            target = captions[:, 1:]

            # check input
            if torch.isnan(images).any():
                print("IMAGE NAN")
                break

            if torch.isnan(captions.float()).any():
                print("CAPTION NAN")
                break

            valid_tokens = (target != PAD_IDX).sum(dim=1)

            if (valid_tokens == 0).any():
                print("EMPTY TARGET")
                print(captions[valid_tokens == 0])
                break

            # forward pass (NO torch.no_grad here)
            image_features = encoder(images)

            if torch.isnan(image_features).any():
                print("IMAGE FEATURES NAN")
                break

            output = decoder(
                image_features,
                decoder_input,
                padding_mask[:, :-1]
            )

            if torch.isinf(output).any():
                print("Inf in decoder output")
                break

            if torch.isnan(output).any():
                print("NaN in decoder output")
                break

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                target.reshape(-1)
            )

            if torch.isnan(loss):
                print("====== NaN LOSS ======")
                print("Valid tokens:", (target != PAD_IDX).sum().item())
                print("Output has NaN:", torch.isnan(output).any().item())
                print("Image features have NaN:", torch.isnan(image_features).any().item())
                print("Captions:")
                print(captions)
                break

            # backward
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) +
                list(encoder.projection.parameters())
                + 
                list(encoder.backbone[7].parameters())
                ,
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            total_token += token_accuracy(output, target, PAD_IDX)
            total_exact += exact_match(output, target, PAD_IDX)

        return (
            total_loss / len(loader),
            total_token / len(loader),
            total_exact / len(loader)
        )

    # =====================================================
    # Validation
    # =====================================================
    def validate(encoder, decoder, loader, criterion,):
        encoder.eval()
        decoder.eval()
        total_loss=0
        token_acc=0
        exact_acc=0

        with torch.no_grad():
            # with torch.amp.autocast("cuda", enabled=DEVICE=="cuda"):
            for images,captions,padding_mask in loader:

                images=images.to(DEVICE)
                captions=captions.to(DEVICE)
                padding_mask=padding_mask.to(DEVICE)

                # image_features=encoder(images)
                # output=decoder(
                #     image_features,
                #     decoder_input,
                #     padding_mask[:,:-1]
                # )
                # loss=criterion(output.reshape(-1, output.size(-1)),target.reshape(-1))
                # with torch.amp.autocast("cuda", enabled=DEVICE=="cuda"):
            
                image_features=encoder(images)
                decoder_input=captions[:,:-1]
                target=captions[:,1:]

                output=decoder(
                    image_features,
                    decoder_input,
                    padding_mask[:,:-1]
                )

                loss=criterion(
                    output.reshape(-1, output.size(-1)),
                    target.reshape(-1)
                )

                total_loss+=loss.item()
                token_acc+=token_accuracy(output, target, PAD_IDX)
                exact_acc+=exact_match(output, target, PAD_IDX)

        n=len(loader)
        return (total_loss/n, token_acc/n, exact_acc/n )

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

    # =====================================================
    # Load previous checkpoint if exists
    # =====================================================

    best = float("inf")
    checkpoint = None

    if os.path.exists(CHECKPOINT):
        print("Loading checkpoint:", CHECKPOINT)
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
        encoder.load_state_dict(checkpoint["encoder"], strict=False)
        decoder.load_state_dict(checkpoint["decoder"])

        if "best" in checkpoint:
            best = checkpoint["best"]
            # best = float("inf") # force save the first run DELETE LATER

        print("Checkpoint loaded")
        print("Previous best validation loss:", best)

    else: 
        print("No checkpoint found. Training from scratch.")

    #===============================================================================
    # Freeze ResNet initially !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for p in encoder.backbone.parameters():
        p.requires_grad=False  # this freeze
        # p.requires_grad=True  # unfreeze if want to improve vision feature
        
    # Unfreeze only layer4
    for p in encoder.backbone[7].parameters():
        p.requires_grad = True

    criterion=nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.03)

    # optimizer for frozen layer4
    # optimizer=optim.AdamW(
    #     list(decoder.parameters())+
    #     list(encoder.projection.parameters()),
    #     lr=LR
    # )
    # diff learning speed for diff part of network
    optimizer = optim.AdamW(
    [
        {"params": decoder.parameters(), "lr": LR},
        {"params": encoder.projection.parameters(), "lr": LR},
        {"params": encoder.backbone[7].parameters(), "lr": LR * 0.1},
    ],
    weight_decay=1e-4,)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )

    # Load optimizer state
    if checkpoint is not None and "optimizer" in checkpoint:
        # optimizer.load_state_dict(checkpoint["optimizer"])
        print("Optimizer state restart")
    # load sceduler
    # if checkpoint is not None and "scheduler" in checkpoint:
    #     # scheduler.load_state_dict(checkpoint["scheduler"])
    #     # print("Scheduler state loaded")
    # else:
    #     print("No scheduler state found, starting scheduler fresh")

   
    # =====================================================
    # Training
    # =====================================================

    train_losses=[]
    val_losses=[]

    train_token_acc=[]
    val_token_acc=[]

    train_exact_acc=[]
    val_exact_acc=[]

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
        scheduler.step(val_loss) # update lr 

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_token_acc.append(train_token)
        val_token_acc.append(v_token)

        train_exact_acc.append(train_exact)
        val_exact_acc.append(v_exact)

        print(f"""Epoch {epoch+1}
        Train loss:              {train_loss:.4f}
        Validation loss:         {val_loss:.4f}
        Train token accuracy:    {train_token:.4f}
        Validation token accuracy:{v_token:.4f}
        Train exact match:       {train_exact:.4f}
        Validation exact match:  {v_exact:.4f}
        """)
        print("LR:", optimizer.param_groups[0]["lr"])

        # save better loss to path
        if val_loss < best:
            best=val_loss
            torch.save(
                {
                "encoder":encoder.state_dict(),
                "decoder":decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best
                },
                CHECKPOINT
            )
            print("saved best model")

        # save the latest weights regarless in another one
        torch.save(
            {
                "encoder":encoder.state_dict(),
                "decoder":decoder.state_dict(),
                "optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "best":best,
                "epoch": epoch+1
            },
            r"D:\APS360_proj\hme100k_latest.pth"
        )

        # save data to csv
        history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_token": train_token_acc,
            "val_token": val_token_acc,
            "train_exact": train_exact_acc,
            "val_exact": val_exact_acc
        }

        with open("hme_history.json","w") as f:
            json.dump(history,f)


    # =====================================================
    # Plot 1: Loss
    # =====================================================

    plt.figure(figsize=(8,5))

    plt.plot(train_losses, label="Training loss")
    plt.plot( val_losses, label="Validation loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("CNN-Transformer Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("transformer_loss.png",dpi=300)
    plt.show()

    # =====================================================
    # Plot 2: Accuracy
    # =====================================================
    plt.figure(figsize=(8,5))

    # Token accuracy
    plt.plot(train_token_acc, label="Train token accuracy")
    plt.plot(val_token_acc, label="Validation token accuracy")

    # Exact match
    plt.plot( train_exact_acc, label="Train exact match")
    plt.plot( val_exact_acc, label="Validation exact match")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN-Transformer Token Accuracy and Exact Match")
    plt.legend()
    plt.grid()
    plt.savefig( "transformer_accuracy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()