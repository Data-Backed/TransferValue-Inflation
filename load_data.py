import pandas as pd
from pathlib import Path

leagues = ["premier_league", "laliga", "bundesliga", "serie_a", "ligue_1"]
base = Path("transfermarkt-data")

dfs = []
for league in leagues:
    for f in (base / league).glob("*.csv"):
        season_df = pd.read_csv(f)
        season_df["league"] = league
        season_df["season_file"] = f.stem
        dfs.append(season_df)

df = pd.concat(dfs, ignore_index=True)
print(df.shape)
print(df.columns.tolist())
print(df.head())

print(df[["fee", "market_value"]].describe())
print(df["window"].value_counts())
print(df["movement"].value_counts())
print(df["is_loan"].value_counts())
print(df[["position", "pos"]].drop_duplicates().head(20))
print(df["fee"].dropna().unique()[:20])  # see raw fee format

