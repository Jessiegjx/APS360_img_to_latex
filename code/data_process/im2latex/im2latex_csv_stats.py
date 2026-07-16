'''
author: Jessie Guo
Date 7.13 2026

Purpose: for both im2latex & HME100k

count images
check corrupt images
image size statistics
missing labels
duplicated filenames
output a csv summary

'''
import pandas as pd


DATASETS = {
    "train": r"D:\APS360_proj\data\raw\im2Latex\im2latex_train.csv",
    "validate": r"D:\APS360_proj\data\raw\im2Latex\im2latex_validate.csv",
    "test": r"D:\APS360_proj\data\raw\im2Latex\im2latex_test.csv"
}


def analyze_csv(name, path):

    print("\n" + "="*60)
    print(name.upper())
    print("="*60)

    df = pd.read_csv(path)
    print("Samples:", len(df))
    print("\nMissing values")
    print(
        "Missing formulas:",
        df["formula"].isna().sum()
    )
    print(
        "Missing image names:",
        df["image"].isna().sum()
    )


    print("\nDuplicates")

    print(
        "Duplicate images:",
        df["image"].duplicated().sum()
    )
    print(
        "Duplicate formulas:",
        df["formula"].duplicated().sum()
    )
    lengths = df["formula"].dropna().apply(
        lambda x: len(x.split())
    )


    print("\nFormula sequence length")

    print(
        "Min:",
        lengths.min()
    )
    print(
        "Max:",
        lengths.max()
    )
    print(
        "Mean:",
        lengths.mean()
    )
    print(
        "Median:",
        lengths.median()
    )
    print(
        "Very long formulas (>200):",
        (lengths > 200).sum()
    )

for name,path in DATASETS.items():
    analyze_csv(name,path)