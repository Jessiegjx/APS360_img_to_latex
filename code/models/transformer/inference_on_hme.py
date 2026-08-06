import json
import torch
import matplotlib.pyplot as plt

from im2latex_dataset import Im2LatexDataset
from resnet_encoder import ResNetEncoder
from transformer_decoder import TransformerDecoder
from hme100k_dataset import HME100kDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

TXT = r"D:\APS360_proj\data\raw\HME100k\splits\test.txt"
IMAGE_DIR = r"D:\APS360_proj\data\processed\hme100k_processed"
TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"
# CHECKPOINT = r"D:\APS360_proj\hme100k_finetuned_best.pth"
CHECKPOINT = r"D:\APS360_proj\saved_paths\hme100k_finetuned_best_170epoch_final.pth"
# CHECKPOINT = r"D:\APS360_proj\saved_paths\im2latex_transformer_best_after140epoch.pth"

checkpoint=torch.load(CHECKPOINT)

print(checkpoint.keys())
print(checkpoint.get("epoch"))

MAX_LEN = 100

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

dataset = HME100kDataset(
    txt_path=TXT,
    image_folder=IMAGE_DIR,
    tokenizer_path=TOKENIZER
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

idx = 106     # change this
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
    # memory = torch.zeros_like(encoder(image))
    # memory = torch.randn_like(encoder(image))
    print(memory.shape)
    print(memory.mean())
    print(memory.std())

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
        probs = torch.softmax(output[:, -1], dim=-1)
        values, indices = torch.topk(probs, 5)

        print("Current sequence:")
        print(" ".join(itos[t] for t in generated))

        print("Top 5 predictions:")
        for v, idx in zip(values[0], indices[0]):
            print(itos[idx.item()], f"{v.item():.4f}")
        print()

        next_token = output[:, -1].argmax(-1).item()
        print(len(generated), itos[next_token])

        if next_token == EOS:
            break
        generated.append(next_token)

    prediction = []
    for t in generated[1:]:
        if t == EOS:
            break
        prediction.append(itos[t])
    return " ".join(prediction)

# prediction = generate(image)

# -------------------------------------------------------
# Beam search decoding
# -------------------------------------------------------

@torch.no_grad()
def generate_beam_search(image, beam_size=5, alpha=0.7):
    memory = encoder(image)
    print(memory.shape)
    print(memory.mean())
    print(memory.std())
    beams = [([BOS], 0.0)]
    finished = []

    for _ in range(MAX_LEN):
        candidates = []

        for seq, score in beams:
            if seq[-1] == EOS:
                finished.append((seq, score))
                continue

            tokens = torch.tensor([seq], device=DEVICE)
            mask = tokens == PAD

            logits = decoder(memory,tokens, mask)[:, -1]
            log_probs = torch.log_softmax(logits, -1)
            vals, ids = torch.topk( log_probs,beam_size)

            for v, i in zip(vals[0], ids[0]):
                candidates.append(( seq + [i.item()], score + v.item() ))

        if not candidates:
            break

        beams = sorted(
            candidates,
            key=lambda x: x[1] / len(x[0])**alpha,
            reverse=True
        )[:beam_size]

    finished += beams
    seq, _ = max(finished, key=lambda x: x[1] / len(x[0])**alpha)

    return " ".join(
        itos[t] for t in seq[1:]
        if t not in [EOS, PAD]
    )

prediction = generate_beam_search(image, beam_size=3, alpha = 0.1)

# -------------------------------------------------------
# Display
# -------------------------------------------------------

img_show = image.squeeze(0).cpu()

# undo normalization first
mean = torch.tensor([0.485,0.456,0.485]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

img_show = img_show * std + mean

# CHW -> HWC
img_show = img_show.permute(1,2,0)

img_show = img_show.clamp(0,1)

plt.figure(figsize=(12,3))
plt.imshow(img_show)

plt.axis("off")
plt.show()

print("="*80)
print("GROUND TRUTH\n")
print(ground_truth)

# print("\n"+"="*80)
print("PREDICTION\n")
print(prediction)
# print("="*80)