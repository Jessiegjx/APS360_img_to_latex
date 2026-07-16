import torch
import json
import matplotlib.pyplot as plt

from models.transformer.im2latex_dataset import Im2LatexDataset
from models.baseline.cnn_encoder import CNNEncoder
from models.baseline.rnn_decoder import RNNDecoder


# =====================================================
# Paths
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV = r"D:\APS360_proj\data\processed\im2Latex\validate_clean.csv"
IMAGE_DIR = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"
TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"

CHECKPOINT = r"D:\APS360_proj\code\baseline_best.pth"


print("Using:", DEVICE)


# =====================================================
# Load tokenizer
# =====================================================

with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)["vocab"]

itos = {v:k for k,v in vocab.items()}


PAD = vocab["<PAD>"]
BOS = vocab["<BOS>"]
EOS = vocab["<EOS>"]


# =====================================================
# Load dataset
# =====================================================

dataset = Im2LatexDataset(
    CSV,
    IMAGE_DIR,
    TOKENIZER
)

print("\nDataset size:", len(dataset))


# =====================================================
# Load models
# =====================================================

encoder = CNNEncoder().to(DEVICE)

decoder = RNNDecoder(
    vocab_size=len(vocab),
    embed_size=256,
    hidden_size=256
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])


encoder.eval()
decoder.eval()


print("\nModels loaded successfully")


# =====================================================
# 1. Check images
# =====================================================

indices = [0,50,100]


print("\n==============================")
print("IMAGE STATISTICS")
print("==============================")


fig, axes = plt.subplots(
    1,
    len(indices),
    figsize=(12,4)
)


for i,idx in enumerate(indices):

    img, target = dataset[idx]

    print(
        f"""
Index {idx}
shape : {img.shape}
min   : {img.min()}
max   : {img.max()}
mean  : {img.mean()}
std   : {img.std()}
"""
    )


    axes[i].imshow(
        img.squeeze(),
        cmap="gray"
    )

    axes[i].set_title(idx)
    axes[i].axis("off")


plt.show()



# =====================================================
# 2. Check CNN feature outputs
# =====================================================


print("\n==============================")
print("CNN FEATURES")
print("==============================")


features = []


with torch.no_grad():

    for idx in indices:

        img,_ = dataset[idx]

        img = img.unsqueeze(0).to(DEVICE)

        feat = encoder(img)

        features.append(feat)

        print(
            f"""
Index {idx}
feature shape:
{feat.shape}

first 20 values:
{feat[0,:20]}
"""
        )



# =====================================================
# 3. Feature distances
# =====================================================

print("\n==============================")
print("FEATURE DISTANCES")
print("==============================")


for i in range(len(features)):

    for j in range(i+1,len(features)):

        distance = torch.norm(
            features[i]-features[j]
        )

        print(
            f"{indices[i]} vs {indices[j]}:"
            f" {distance.item():.6f}"
        )



# =====================================================
# 4. Test predictions
# =====================================================

def decode(ids):

    result=[]

    for i in ids:

        if i == EOS:
            break

        if i not in [PAD, BOS]:
            result.append(itos[i])

    return " ".join(result)



def generate(img,max_len=100):

    img=img.unsqueeze(0).to(DEVICE)


    with torch.no_grad():

        feature = encoder(img)

        hidden = feature.unsqueeze(0)


        token = torch.tensor(
            [[BOS]],
            device=DEVICE
        )


        output=[]


        for _ in range(max_len):

            emb = decoder.embedding(token)

            out,hidden = decoder.rnn(
                emb,
                hidden
            )

            logits = decoder.fc(
                out.squeeze(1)
            )


            token = logits.argmax(
                dim=1,
                keepdim=True
            )


            token_id = token.item()


            if token_id==EOS:
                break


            output.append(token_id)


    return decode(output)



print("\n==============================")
print("PREDICTIONS")
print("==============================")


for idx in indices:

    img,target = dataset[idx]


    print("\nINDEX:",idx)


    print("\nGROUND TRUTH:")
    print(decode(target))


    print("\nPREDICTION:")
    print(generate(img))


    print("-"*60)