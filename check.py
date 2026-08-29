import pandas as pd
df = pd.read_parquet("data/clean/transfers_clean_v1.parquet")
print(df[df["season"] == 2022]["league"].value_counts())