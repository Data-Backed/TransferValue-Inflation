import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import process, fuzz

from position_groups import add_position_group

LEAGUE_SLUGS = ["eng", "esp", "ger", "ita", "fra"]


def load_and_stack(stat_type):
    dfs = []
    for slug in LEAGUE_SLUGS:
        path = Path(f"data/raw/fbref_{stat_type}_{slug}.parquet")
        if path.exists():
            dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True)

standard = load_and_stack("standard")
misc = load_and_stack("misc")
keeper = load_and_stack("keeper")

print(f"standard: {standard.shape}, misc: {misc.shape}, keeper: {keeper.shape}")

standard["nineties"] = standard["Playing Time_90s"].replace(0, np.nan)
standard["goals_p90"] = standard["Performance_Gls"] / standard["nineties"]
standard["assists_p90"] = standard["Performance_Ast"] / standard["nineties"]
standard["attacking_output"] = standard["goals_p90"].fillna(0) + standard["assists_p90"].fillna(0)

misc["nineties"] = misc["90s"].replace(0, np.nan)
misc["tklw_p90"] = misc["Performance_TklW"] / misc["nineties"]
misc["int_p90"] = misc["Performance_Int"] / misc["nineties"]
misc["defensive_output"] = misc["tklw_p90"].fillna(0) + misc["int_p90"].fillna(0)

keeper["keeping_output"] = (
    keeper["Performance_Save%"].fillna(0) + keeper["Performance_CS%"].fillna(0)
)

stats = standard[["league", "season", "team", "player", "pos", "age",
                   "attacking_output"]].merge(
    misc[["league", "season", "team", "player", "defensive_output"]],
    on=["league", "season", "team", "player"], how="outer"
).merge(
    keeper[["league", "season", "team", "player", "keeping_output"]],
    on=["league", "season", "team", "player"], how="outer"
)

print(f"Combined FBref stats table: {stats.shape}")

FBREF_POS_MAP = {
    "GK": "GK",
    "DF": "DEF", "DF,MF": "DEF", "MF,DF": "DEF",
    "MF": "MID", "MF,FW": "MID", "FW,MF": "MID",
    "FW": "FWD", "DF,FW": "FWD",
}
stats["position_group"] = stats["pos"].map(FBREF_POS_MAP)
stats["position_group"] = stats["position_group"].fillna("MID")


def fbref_season_to_year(s):
    s = str(s)
    if len(s) == 4:
        prefix = "20"
        return int(prefix + s[:2])
    return np.nan

stats["season"] = stats["season"].apply(fbref_season_to_year)


def pctile(s):
    return s.rank(pct=True).astype("float64")

stats["composite_score"] = np.full(len(stats), np.nan, dtype="float64")

for grp, col in [("FWD", "attacking_output"), ("MID", "attacking_output"),
                  ("DEF", "defensive_output"), ("GK", "keeping_output")]:
    mask = stats["position_group"] == grp
    stats.loc[mask, "composite_score"] = (
        stats.loc[mask]
        .groupby("season")[col]
        .transform(pctile)
    )

print("Composite score coverage:")
print(stats.groupby("position_group")["composite_score"].apply(lambda x: x.notna().mean()))

stats_final = stats[["league", "season", "player", "position_group", "composite_score"]].dropna(subset=["composite_score"])
stats_final.to_parquet("data/clean/fbref_composite_scores.parquet")
print(f"Saved {stats_final.shape[0]} player-season composite scores")

df = pd.read_parquet("data/clean/transfers_clean_v1.parquet")
df = add_position_group(df)

THRESHOLD = 85

matched_rows = []
unmatched_players = set()

for season, season_df in df.groupby("season"):
    fbref_season = stats_final[stats_final["season"] == season]
    if fbref_season.empty:
        continue
    choices = fbref_season["player"].tolist()

    for idx, row in season_df.iterrows():
        result = process.extractOne(
            row["player_name"], choices, scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= THRESHOLD:
            matched_name, score, _ = result
            match_row = fbref_season[fbref_season["player"] == matched_name].iloc[0]
            matched_rows.append({
                "transfer_index": idx,
                "matched_fbref_name": matched_name,
                "match_score": score,
                "composite_score": match_row["composite_score"],
            })
        else:
            unmatched_players.add(row["player_name"])

matches_df = pd.DataFrame(matched_rows).set_index("transfer_index")
print(f"\nMatched {len(matches_df)} of {len(df)} transfer rows "
      f"({len(matches_df)/len(df):.1%})")
print(f"{len(unmatched_players)} unique unmatched player names")

df = df.join(matches_df[["composite_score", "match_score"]])
df.to_parquet("data/clean/transfers_with_stats_v1.parquet")
print("\nSaved: data/clean/transfers_with_stats_v1.parquet")
print(f"Rows with a composite_score: {df['composite_score'].notna().sum()} / {len(df)}")
