import numpy as np
from collections import Counter

LABEL_FILE = r"D:\APS360_proj\data\raw\HME100k\train.txt"
latex_lengths=[]
token_counter=Counter()
image_names=[]

missing=0
duplicates=0

seen=set()

with open(
    LABEL_FILE,
    "r",
    encoding="utf-8"
) as f:
    
    for line in f:
        image, latex = line.strip().split("\t")

        # duplicate image ids
        if image in seen:
            duplicates += 1

        seen.add(image)

        image_names.append(image)

        if latex.strip()=="":
            missing+=1
            continue

        tokens=latex.split()
        latex_lengths.append(
            len(tokens)
        )

        token_counter.update(tokens)

print("="*60)
print("HME100K TEXT STATISTICS")
print("="*60)

print("\nTotal labels:")
print(len(image_names))

print("\nMissing latex:")
print(missing)

print("\nDuplicate image entries:")
print(duplicates)


print("\nFormula length")

print(
    "Min:",
    min(latex_lengths)
)

print(
    "Max:",
    max(latex_lengths)
)

print(
    "Mean:",
    np.mean(latex_lengths)
)

print(
    "Median:",
    np.median(latex_lengths)
)

print("\nVocabulary size")

print(
    len(token_counter)
)

print("\nMost common tokens")
for token,count in token_counter.most_common(50):
    print(
        token,
        count
    )