import pandas as pd

CSV=r"D:\APS360_proj\data\processed\im2Latex\test_clean.csv"

df=pd.read_csv(CSV)

df=df.dropna()

df.to_csv(
    r"D:\APS360_proj\data\processed\im2Latex\test_clean_fixed.csv",
    index=False
)

print("Remaining samples:",len(df))