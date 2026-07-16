import json
import torch
import matplotlib.pyplot as plt

from im2latex_dataset import Im2LatexDataset
from resnet_encoder import ResNetEncoder
from transformer_decoder import TransformerDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

CSV = r"D:\APS360_proj\data\processed\im2Latex\validate_clean.csv"

IMAGE_DIR = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"

TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"

CHECKPOINT = r"D:\APS360_proj\code\transformer_best.pth"

MAX_LEN = 150

# -------------------------------------------------------
# Load vocabulary
# -------------------------------------------------------

with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)["vocab"]

itos = {v: k for k, v in vocab.items()}

PAD = vocab["<PAD>"]
BOS = vocab["<BOS>"]
EOS = vocab["<EOS>"]

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

dataset = Im2LatexDataset(
    CSV,
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

checkpoint = torch.load(
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

idx = 1      # change this

image, caption = dataset[idx]

image = image.unsqueeze(0).to(DEVICE)

# -------------------------------------------------------
# Decode ground truth
# -------------------------------------------------------

ground_truth = []

for token in caption.tolist():

    if token == BOS or token == PAD:
        continue

    if token == EOS:
        break

    ground_truth.append(itos[token])

ground_truth = " ".join(ground_truth)

# -------------------------------------------------------
# Greedy Transformer decoding
# -------------------------------------------------------

@torch.no_grad()
def generate(image):

    memory = encoder(image)

    generated = [BOS]

    for _ in range(MAX_LEN):

        tokens = torch.tensor(
            [generated],
            device=DEVICE
        )

        padding_mask = (
            tokens == PAD
        )

        output = decoder(
            memory,
            tokens,
            padding_mask
        )

        next_token = output[:, -1].argmax(-1).item()

        if next_token == EOS:
            break

        generated.append(next_token)

    prediction = []

    for t in generated[1:]:

        if t == EOS:
            break

        prediction.append(itos[t])

    return " ".join(prediction)

prediction = generate(image)

# -------------------------------------------------------
# Display
# -------------------------------------------------------

# plt.figure(figsize=(12,3))

# plt.imshow(
#     image.squeeze().cpu(),
#     cmap="gray"
# )

# plt.axis("off")

# plt.show()

print("="*80)
print("GROUND TRUTH\n")
print(ground_truth)

print("\n"+"="*80)
print("PREDICTION\n")
print(prediction)
print("="*80)