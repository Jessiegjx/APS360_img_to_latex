import json
import torch
import matplotlib.pyplot as plt

from models.transformer.im2latex_dataset import Im2LatexDataset
from models.baseline.cnn_encoder import CNNEncoder
from models.baseline.rnn_decoder import RNNDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV = r"D:\APS360_proj\data\processed\im2Latex\validate_clean.csv"
IMAGE_DIR = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"
TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"
CHECKPOINT = r"D:\APS360_proj\code\baseline_best.pth"

MAX_LEN = 150

# -----------------------------------------------------
# Load tokenizer
# -----------------------------------------------------

with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)["vocab"]

itos = {v: k for k, v in vocab.items()}

PAD = vocab["<PAD>"]
BOS = vocab["<BOS>"]
EOS = vocab["<EOS>"]

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------

dataset = Im2LatexDataset(
    CSV,
    IMAGE_DIR,
    TOKENIZER
)

# -----------------------------------------------------
# Models
# -----------------------------------------------------

encoder = CNNEncoder().to(DEVICE)

decoder = RNNDecoder(
    vocab_size=len(vocab),
    embed_size=256,
    hidden_size=256
).to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

encoder.load_state_dict(ckpt["encoder"])
decoder.load_state_dict(ckpt["decoder"])

encoder.eval()
decoder.eval()

# for idx in [0,50,100]:

#     image,_ = dataset[idx]
#     image = image.unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         feat = encoder(image)

#     print(idx)
#     print(feat[0,:20])
#     print()


# -----------------------------------------------------
# Choose one example
# -----------------------------------------------------

idx = 0          # change this to inspect different samples

image, target_ids = dataset[idx]

image = image.unsqueeze(0).to(DEVICE)

# -----------------------------------------------------
# Decode ground truth
# -----------------------------------------------------

gt = []

for t in target_ids.tolist():

    if t in [PAD, BOS]:
        continue

    if t == EOS:
        break

    gt.append(itos[t])

ground_truth = " ".join(gt)

# -----------------------------------------------------
# Greedy decoding
# -----------------------------------------------------

with torch.no_grad():

    features = encoder(image)

    hidden = features.unsqueeze(0)

    token = torch.tensor([[BOS]], device=DEVICE)

    prediction = []

    for _ in range(MAX_LEN):

        emb = decoder.embedding(token)

        out, hidden = decoder.rnn(emb, hidden)

        logits = decoder.fc(out.squeeze(1))

        token = logits.argmax(dim=1, keepdim=True)

        token_id = token.item()

        if token_id == EOS:
            break

        prediction.append(itos[token_id])

predicted = " ".join(prediction)

# -----------------------------------------------------
# Display
# -----------------------------------------------------

print("=" * 80)
print("GROUND TRUTH:\n")
print(ground_truth)

print("\n" + "=" * 80)
print("PREDICTION:\n")
print(predicted)
print("=" * 80)

plt.imshow(image.squeeze().cpu(), cmap="gray")
plt.axis("off")
plt.show()
