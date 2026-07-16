import os
import json
import csv
from collections import Counter

IM2LATEX_CSV = r"D:\APS360_proj\data\raw\im2Latex\im2latex_train.csv"
HME_FILE = r"D:\APS360_proj\data\raw\HME100k\train.txt"
SAVE_PATH = r"D:\APS360_proj\data\processed\tokenizer.json"
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

tokens = Counter()

# -------------------------
# HME100K
# -------------------------

print("Reading HME100K...")

hme_count = 0
with open(HME_FILE, "r", encoding="utf-8") as f:
    for line in f:
        _, latex = line.strip().split("\t")
        if latex:
            tokens.update(latex.split())
            hme_count += 1

print("HME samples:", hme_count)

# -------------------------
# IM2LATEX
# -------------------------

print("Reading IM2LATEX...")
im_count = 0

with open(IM2LATEX_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        latex = row["formula"]

        if latex:
            tokens.update(latex.split())
            im_count += 1


print("IM2LATEX samples:", im_count)

# -------------------------
# Build vocabulary
# -------------------------
vocab = {}

for t in SPECIAL_TOKENS:
    vocab[t] = len(vocab)

for t, _ in tokens.most_common():
    if t not in vocab:
        vocab[t] = len(vocab)

print("\nVocabulary size:", len(vocab))

print("\nTop tokens:")
for t, c in tokens.most_common(30):
    print(t, c)


# -------------------------
# Save
# -------------------------

os.makedirs(
    os.path.dirname(SAVE_PATH),
    exist_ok=True
)

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "vocab": vocab,
            "token_counts": tokens
        },
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nSaved:", SAVE_PATH)