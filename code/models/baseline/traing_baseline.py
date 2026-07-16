import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


def token_accuracy(outputs, targets, pad_idx):

    preds = outputs.argmax(dim=-1)

    mask = targets != pad_idx

    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()

    if total == 0:
        return 0

    return correct / total



def exact_match(outputs, targets, pad_idx):

    preds = outputs.argmax(dim=-1)

    matches = 0

    for pred, target in zip(preds, targets):

        mask = target != pad_idx

        if torch.equal(pred[mask], target[mask]):
            matches += 1

    return matches / targets.size(0)



def train_epoch(
    encoder,
    decoder,
    loader,
    optimizer,
    criterion,
    device
):

    encoder.train()
    decoder.train()

    total_loss = 0

    for imgs, captions, lengths in tqdm(loader):

        imgs = imgs.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        features = encoder(imgs)

        outputs = decoder(
            features,
            captions[:, :-1]
        )

        targets = captions[:, 1:]

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss / len(loader)



def validate(
    encoder,
    decoder,
    loader,
    criterion,
    pad_idx,
    device
):

    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_token_acc = 0
    total_exact_acc = 0


    with torch.no_grad():

        for imgs, captions, lengths in loader:

            imgs = imgs.to(device)
            captions = captions.to(device)


            features = encoder(imgs)


            outputs = decoder(
                features,
                captions[:, :-1]
            )


            targets = captions[:, 1:]


            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )


            total_loss += loss.item()


            total_token_acc += token_accuracy(
                outputs,
                targets,
                pad_idx
            )


            total_exact_acc += exact_match(
                outputs,
                targets,
                pad_idx
            )


    return (
        total_loss / len(loader),
        total_token_acc / len(loader),
        total_exact_acc / len(loader)
    )



def train(
    encoder,
    decoder,
    train_loader,
    val_loader,
    vocab,
    epochs=10,
    lr=1e-3,
    device="cuda"
):

    encoder.to(device)
    decoder.to(device)


    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab["<PAD>"]
    )


    optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=lr
    )


    best_val = float("inf")


    train_losses = []
    val_losses = []

    token_accs = []
    exact_accs = []


    for epoch in range(epochs):


        train_loss = train_epoch(
            encoder,
            decoder,
            train_loader,
            optimizer,
            criterion,
            device
        )


        val_loss, val_token_acc, val_exact_acc = validate(
            encoder,
            decoder,
            val_loader,
            criterion,
            vocab["<PAD>"],
            device
        )


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        token_accs.append(val_token_acc)
        exact_accs.append(val_exact_acc)



        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict()
                },
                "baseline_best.pth"
            )



        print(
            f"Epoch {epoch+1}: "
            f"train loss={train_loss:.4f}, "
            f"val loss={val_loss:.4f}, "
            f"token acc={val_token_acc:.3f}, "
            f"exact={val_exact_acc:.3f}"
        )


    return (
        train_losses,
        val_losses,
        token_accs,
        exact_accs
    )



def plot_loss(train_losses, val_losses):

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



def plot_accuracy(token_accs, exact_accs):

    plt.plot(token_accs, label="Token Accuracy")
    plt.plot(exact_accs, label="Exact Match")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    