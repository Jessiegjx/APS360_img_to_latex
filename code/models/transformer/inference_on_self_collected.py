import json
import torch
import matplotlib.pyplot as plt

from self_collected_dataset import SelfCollectedDataset
from resnet_encoder import ResNetEncoder
from transformer_decoder import TransformerDecoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

TXT = r"D:\APS360_proj\data\self_collected\self_labels.txt"
IMAGE_DIR = r"D:\APS360_proj\data\self_collected\self_images_processed"
TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"

# CHECKPOINT = r"D:\APS360_proj\saved_paths\hme100k_finetuned_best_170epoch_final.pth"
CHECKPOINT = r"D:\APS360_proj\saved_paths\im2latex_transformer_best_after140epoch.pth"

MAX_LEN = 100


# -------------------------------------------------------
# Load vocabulary
# -------------------------------------------------------

with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)["vocab"]

itos = {v:k for k,v in vocab.items()}

PAD = vocab["<PAD>"]
BOS = vocab["<BOS>"]
EOS = vocab["<EOS>"]

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

dataset = SelfCollectedDataset(
    TXT,
    IMAGE_DIR,
    TOKENIZER
)

# -------------------------------------------------------
# Models
# -------------------------------------------------------

encoder = ResNetEncoder(
    embed_dim=256
).to(DEVICE)

decoder = TransformerDecoder(
    vocab_size=len(vocab),
    embed_dim=256,
    heads=8,
    layers=3
).to(DEVICE)

checkpoint=torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()
print("Loaded checkpoint.")

# -------------------------------------------------------
# Pick example
# -------------------------------------------------------
# idx = 5   # change 0-19
# image, caption = dataset[idx]
# name = dataset.samples[idx][0]
# image = image.unsqueeze(0).to(DEVICE)

# -------------------------------------------------------
# Ground truth
# -------------------------------------------------------
# ground_truth=[]

# for token in caption.tolist():
#     if token in [PAD,BOS]:
#         continue
#     if token == EOS:
#         break
#     ground_truth.append(
#         itos[token]
#     )
# ground_truth=" ".join(ground_truth)

# -------------------------------------------------------
# Beam search
# -------------------------------------------------------

@torch.no_grad()
def generate_beam_search(image, beam_size=3, alpha=0.1):

    memory=encoder(image)

    print("memory:",memory.shape)
    print("mean:",memory.mean())
    print("std:",memory.std())


    beams=[([BOS],0.0)]
    finished=[]


    for _ in range(MAX_LEN):
        candidates=[]
        for seq,score in beams:
            if seq[-1]==EOS:
                finished.append((seq,score))
                continue

            tokens=torch.tensor(
                [seq],
                device=DEVICE
            )
            mask=tokens==PAD


            logits=decoder(
                memory,
                tokens,
                mask
            )[:,-1]

            log_probs=torch.log_softmax(
                logits,
                dim=-1
            )
            log_probs[:,PAD] = -float("inf")
            log_probs[:,BOS] = -float("inf")

            vals,ids=torch.topk(
                log_probs,
                beam_size
            )

            for v,i in zip(vals[0],ids[0]):
                candidates.append((seq+[i.item()], score+v.item()))

        if not candidates:
            break

        beams=sorted(
            candidates,
            key=lambda x:x[1]/len(x[0])**alpha,
            reverse=True
        )[:beam_size]

    finished+=beams

    seq,_=max(
        finished,
        key=lambda x:x[1]/len(x[0])**alpha)

    return " ".join(
        itos[t]
        for t in seq[1:]
        if t not in [EOS,PAD]
    )

# prediction=generate_beam_search(
#     image,
#     beam_size=3,
#     alpha=0.1
# )

# -------------------------------------------------------
# Display
# -------------------------------------------------------

# plt.figure(figsize=(12,3))

# # img_show = image.squeeze()[0].cpu()
# img_show = image.squeeze().permute(1,2,0).cpu()

# # plt.imshow(
# #     img_show,
# #     cmap="gray"
# # )
# # plt.title(name)
# # plt.axis("off")
# # plt.show()

# print("IMAGE:")
# print(name)

# print("\nGROUND TRUTH\n")
# print(ground_truth)

# print("\nPREDICTION\n")
# print(prediction)

# -------------------------------------------------------
# Run inference on all self-collected samples
# -------------------------------------------------------

for idx in range(len(dataset)):

    image, caption = dataset[idx]
    name = dataset.samples[idx][0]

    image = image.unsqueeze(0).to(DEVICE)


    # -----------------------------
    # Ground truth
    # -----------------------------
    ground_truth=[]

    for token in caption.tolist():
        if token in [PAD,BOS]:
            continue
        if token == EOS:
            break

        ground_truth.append(
            itos[token]
        )

    ground_truth = " ".join(ground_truth)


    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = generate_beam_search(
        image,
        beam_size=3,
        alpha=0.1
    )


    # -----------------------------
    # Print
    # -----------------------------
    print("="*80)
    print("IMAGE:", name)

    print("\nGROUND TRUTH:")
    print(ground_truth)

    print("\nPREDICTION:")
    print(prediction)

    print()