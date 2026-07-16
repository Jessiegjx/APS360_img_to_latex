import os
import ijson
import numpy as np
import re

AIDA_ROOT = r"D:\APS360_proj\data\raw\AIDA"

SPLITS = {
    "train": range(1,9),
    "validate": [9],
    "test": [10]
}

def latex_tokenize(s):
    return re.findall(r'\\[a-zA-Z]+|\\.|[{}_^]|[0-9]+|[a-zA-Z]|[^\s]', s)

def analyze_split(name, batches):
    print("\n" + "="*60)
    print(name.upper())
    print("="*60)

    count = 0
    latex_lengths = []
    widths = []
    heights = []
    depths = []
    fonts = set()
    missing_latex = 0

    for b in batches:
        json_path = os.path.join(
            AIDA_ROOT,
            f"batch_{b}",
            "JSON",
            f"kaggle_data_{b}.json"
        )

        print("Reading:", json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            for item in ijson.items(f, "item"):

                count += 1

                if not item["latex"]:
                    missing_latex += 1

                # latex_lengths.append(len(item["latex"].split()))
                latex_lengths.append(len(latex_tokenize(item["latex"])))

                img = item["image_data"]

                widths.append(img["width"])
                heights.append(img["height"])
                depths.append(img["depth"])

                fonts.add(item["font"])

    print("\nSamples:", count)

    print("\nMissing")
    print("Missing latex:", missing_latex)

    print("\nImage size")
    print("Width:", min(widths), max(widths), np.mean(widths))
    print("Height:", min( heights), max(heights), np.mean(heights))
    print("Depth:", set(depths))

    print("\nFormula length")
    print("Min:", min(latex_lengths))
    print("Max:", max(latex_lengths))
    print("Mean:", np.mean(latex_lengths))
    print("Median:", np.median(latex_lengths))

    print("\nFonts:", len(fonts))


for split, batches in SPLITS.items():
    analyze_split(split, batches)