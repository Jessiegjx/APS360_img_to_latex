import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sys
sys.path.append(r"D:\APS360_proj\code")
from im2latex_dataset import Im2LatexDataset
from models.transformer.hme100k_dataset import HME100kDataset
from models.transformer.self_collected_dataset import SelfCollectedDataset
from collate import collate_fn
from models.baseline.baseline_CNN_RNN import CNN_RNN


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = r"D:\APS360_proj\data\processed\tokenizer.json"
CHECKPOINT = r"D:\APS360_proj\saved_paths\baseline_best.pth"
MAX_LEN = 100


# CHANGE THIS DATASET BLOCK ONLY

CSV = r"D:\APS360_proj\data\processed\im2Latex\test_clean_fixed.csv"
IMAGE_DIR = r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"
dataset = Im2LatexDataset(CSV, IMAGE_DIR, TOKENIZER)

# TXT = r"D:\APS360_proj\data\raw\HME100k\splits\test.txt"
# IMAGE_DIR = r"D:\APS360_proj\data\processed\hme100k_processed"
# dataset = HME100kDataset(TXT, IMAGE_DIR, TOKENIZER)

# TXT = r"D:\APS360_proj\data\self_collected\self_labels.txt"
# IMAGE_DIR = r"D:\APS360_proj\data\self_collected\self_images_processed"
# dataset = SelfCollectedDataset(TXT, IMAGE_DIR, TOKENIZER)


loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)


with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)["vocab"]

itos = {v:k for k,v in vocab.items()}
PAD, BOS, EOS = vocab["<PAD>"], vocab["<BOS>"], vocab["<EOS>"]


model = CNN_RNN(len(vocab)).to(DEVICE)
state = torch.load(CHECKPOINT, map_location=DEVICE)


model.encoder.load_state_dict(
    state["encoder"]
)

model.decoder.load_state_dict(
    state["decoder"]
)
model.eval()


def normalize(x):
    return "".join(x).replace(" ","")


@torch.no_grad()
def generate(img):
    hidden = model.encoder(img).unsqueeze(0)
    token = torch.tensor([[BOS]], device=DEVICE)
    out = []

    for _ in range(MAX_LEN):

        emb = model.decoder.embedding(token[:, -1:])
        y, hidden = model.decoder.rnn(emb, hidden)

        nxt = model.decoder.fc( y[:, -1]).argmax(-1).item()

        if nxt == EOS:
            break

        if nxt != PAD:
            out.append(itos[nxt])

        token = torch.cat([token, torch.tensor([[nxt]], device=DEVICE)], 1)

    return out



preds, refs = [], []
norm_preds, norm_refs = [], []

exact = exact_norm = 0


for img, cap, _ in tqdm(loader):
    img = img.to(DEVICE)
    if img.shape[1] == 3:
        img = img.mean(dim=1, keepdim=True)

    pred = generate(img)
    ref = [
        itos[t.item()]
        for t in cap[0]
        if t.item() not in [PAD,BOS,EOS]
    ]

    preds.append(pred)
    refs.append([ref])

    pn, rn = normalize(pred), normalize(ref)

    norm_preds.append(list(pn))
    norm_refs.append([list(rn)])

    exact += pred == ref
    exact_norm += pn == rn


print("="*40)
print("Samples:", len(preds))
print(f"BLEU tok: {corpus_bleu(refs,preds,smoothing_function=SmoothingFunction().method1)*100:.2f}")
print(f"BLEU norm: {corpus_bleu(norm_refs,norm_preds,smoothing_function=SmoothingFunction().method1)*100:.2f}")
print(f"Exact: {exact/len(preds)*100:.2f}%")
print(f"Exact norm: {exact_norm/len(preds)*100:.2f}%")
print("="*40)