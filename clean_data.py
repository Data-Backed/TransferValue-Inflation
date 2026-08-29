"""
clean_data.py

Reads the raw concatenated transfer data (from load_data.py) and produces
a cleaned dataset ready for modeling. Every major decision is logged to
cleaning_log.txt so it can go straight into the README limitations section.

Run this AFTER load_data.py has produced data/raw/all_transfers_raw.parquet.
If you haven't saved that yet, this script will just re-load from the CSVs
directly (slower, but works standalone).
"""

import pandas as pd
from pathlib import Path

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

raw_path = Path("data/raw/all_transfers_raw.parquet")

if raw_path.exists():
    df = pd.read_parquet(raw_path)
    log(f"Loaded cached raw data: {df.shape}")
else:
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
    log(f"Loaded fresh from CSVs: {df.shape}")

start_rows = len(df)

# ---------------------------------------------------------------------------
# 2. Restrict to season range (edit YEAR_MIN if you want 2000 vs 2010 scope)
# ---------------------------------------------------------------------------

YEAR_MIN = 2000
df["season"] = pd.to_numeric(df["season"], errors="coerce")
df = df[df["season"] >= YEAR_MIN]
log(f"After restricting to season >= {YEAR_MIN}: {len(df)} rows "
    f"(dropped {start_rows - len(df)})")

# ---------------------------------------------------------------------------
# 3. Exclude loans
# ---------------------------------------------------------------------------

before = len(df)
df = df[df["is_loan"] == 0].copy()
log(f"After excluding loans: {len(df)} rows (dropped {before - len(df)})")

# ---------------------------------------------------------------------------
# 4. Handle zero / missing fees
# ---------------------------------------------------------------------------
# A fee of 0 is ambiguous: could be a genuine free transfer, or a value the
# scraper couldn't resolve. We treat rows with fee == 0 AND a populated
# market_value as "likely free transfer, keep but flag" and rows with fee
# missing entirely as "unknown, drop from model but keep in a side table
# for reference."

df["fee_confidence"] = "confirmed"
df.loc[df["fee"] == 0, "fee_confidence"] = "likely_free"
df.loc[df["fee"].isna(), "fee_confidence"] = "unknown"

n_unknown = (df["fee_confidence"] == "unknown").sum()
n_free = (df["fee_confidence"] == "likely_free").sum()
log(f"Fee confidence breakdown: "
    f"{(df['fee_confidence']=='confirmed').sum()} confirmed, "
    f"{n_free} likely free (fee=0), {n_unknown} unknown (fee missing)")

# Split off rows with no usable fee at all - keep them saved separately
# in case you want them later, but they don't belong in the fee model.
df_no_fee = df[df["fee_confidence"] == "unknown"].copy()
df = df[df["fee_confidence"] != "unknown"].copy()
log(f"Dropped {len(df_no_fee)} rows with no usable fee from modeling set "
    f"(saved separately)")

# For the model, we generally want ACTUAL paid fees only (drop free
# transfers too), but keep the flag so you can decide later.
MODEL_FEES_ONLY = True
if MODEL_FEES_ONLY:
    before = len(df)
    df = df[df["fee_confidence"] == "confirmed"].copy()
    log(f"MODEL_FEES_ONLY=True: dropped {before - len(df)} free-transfer "
        f"rows from modeling set (fee_confidence == 'likely_free')")

# ---------------------------------------------------------------------------
# 5. Deduplicate cross-listed transfers
# ---------------------------------------------------------------------------
# The same transfer can appear as an "out" row (from the selling club) and
# an "in" row (to the buying club) when both clubs are in our 5 leagues.
# We keep one row per (player_id, season, fee, dealing_club) combination,
# preferring the "in" record since it's typically logged by the buying
# club with more complete info.

before = len(df)
df = df.sort_values("movement", ascending=False)  # "out" < "in" alphabetically is wrong; sort explicitly
df["movement_rank"] = df["movement"].map({"in": 0, "out": 1})
df = df.sort_values("movement_rank")
df = df.drop_duplicates(subset=["player_id", "season", "fee", "dealing_club"], keep="first")
df = df.drop(columns=["movement_rank"])
log(f"After deduplicating cross-listed transfers: {len(df)} rows "
    f"(dropped {before - len(df)})")

# ---------------------------------------------------------------------------
# 6. Standardize position
# ---------------------------------------------------------------------------
# Use the full-text `position` column as primary. Where it's missing or is
# one of the vague fallback labels, fall back to `pos`.

VAGUE_LABELS = {"Midfielder", "Defender", "Forward", "Attacker"}
df["position_clean"] = df["position"]
mask_vague = df["position"].isin(VAGUE_LABELS) | df["position"].isna()
df.loc[mask_vague, "position_clean"] = df.loc[mask_vague, "pos"]

n_still_vague = df["position_clean"].isin(VAGUE_LABELS).sum()
log(f"Position standardized. {n_still_vague} rows still have only a "
    f"coarse position label after fallback.")

# ---------------------------------------------------------------------------
# 7. Age sanity check
# ---------------------------------------------------------------------------

df["age"] = pd.to_numeric(df["age"], errors="coerce")
before = len(df)
df = df[(df["age"] >= 15) & (df["age"] <= 42)]
log(f"After dropping implausible ages (<15 or >42): {len(df)} rows "
    f"(dropped {before - len(df)})")

# ---------------------------------------------------------------------------
# 8. Outlier flag (don't drop, just flag)
# ---------------------------------------------------------------------------

q_low = df.groupby(["season", "position_clean"])["fee"].transform(
    lambda x: x.quantile(0.01))
q_high = df.groupby(["season", "position_clean"])["fee"].transform(
    lambda x: x.quantile(0.99))
df["fee_outlier_flag"] = (df["fee"] < q_low) | (df["fee"] > q_high)
log(f"Flagged {df['fee_outlier_flag'].sum()} rows as outliers "
    f"(top/bottom 1% within season-position group) - not dropped, "
    f"just flagged for sensitivity checks later.")

# ---------------------------------------------------------------------------
# 9. Save cleaned data + log
# ---------------------------------------------------------------------------

Path("data/clean").mkdir(parents=True, exist_ok=True)
df.to_parquet("data/clean/transfers_clean_v1.parquet")
df_no_fee.to_parquet("data/clean/transfers_no_fee_reference.parquet")

log(f"\nFinal cleaned dataset: {df.shape}")
log(f"Columns: {df.columns.tolist()}")

with open("cleaning_log.md", "w") as f:
    f.write("# Cleaning Log\n\n")
    f.write("\n".join(f"- {line}" for line in log_lines))

print("\nSaved: data/clean/transfers_clean_v1.parquet")
print("Saved: cleaning_log.md")
