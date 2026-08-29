import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from unidecode import unidecode
from pathlib import Path

ALL_LEAGUES = ["premier_league", "laliga", "bundesliga", "serie_a", "ligue_1"]

df = pd.read_parquet("data/clean/transfers_with_stats_v1.parquet")

POSITION_GROUP_MAP = {
    "Goalkeeper": "GK",
    "Centre-Back": "DEF", "Left-Back": "DEF", "Right-Back": "DEF",
    "Defender": "DEF", "Sweeper": "DEF",
    "Defensive Midfield": "MID", "Central Midfield": "MID",
    "Attacking Midfield": "MID", "Left Midfield": "MID",
    "Right Midfield": "MID", "Midfielder": "MID",
    "Left Winger": "FWD", "Right Winger": "FWD", "Centre-Forward": "FWD",
    "Second Striker": "FWD", "Striker": "FWD", "Forward": "FWD", "Other": "MID",
}

model_cols = ["fee", "age", "position_clean", "window", "season", "league"]
df_model = df.dropna(subset=model_cols).copy()
df_model["league"] = pd.Categorical(df_model["league"], categories=ALL_LEAGUES)
df_model["position_group"] = df_model["position_clean"].map(POSITION_GROUP_MAP).fillna("MID")

MIN_TRANSFERS_PER_YEAR = 30
year_counts = df_model["season"].value_counts()
reliable_years = year_counts[year_counts >= MIN_TRANSFERS_PER_YEAR].index
df_model = df_model[df_model["season"].isin(reliable_years)].copy()

df_model["has_stats"] = df_model["composite_score"].notna().astype(int)
df_model["composite_score_filled"] = df_model["composite_score"].fillna(0.5)

df_model["log_fee"] = np.log(df_model["fee"])
df_model["age_sq"] = df_model["age"] ** 2
df_model["season_cat"] = df_model["season"].astype(int).astype(str)

pos_counts = df_model["position_clean"].value_counts()
rare_positions = pos_counts[pos_counts < 30].index
df_model["position_model"] = df_model["position_clean"].where(
    ~df_model["position_clean"].isin(rare_positions), "Other"
)

BASE_YEAR = str(int(df_model["season"].min()))
MIN_YEAR = int(df_model["season"].min())
MAX_YEAR = int(df_model["season"].max())

formula_index = (
    "log_fee ~ age + age_sq + C(position_model) + C(window) + C(league) "
    f"+ C(season_cat, Treatment(reference=\x27{BASE_YEAR}\x27))"
)
print("Fitting base index model...")
model_index = smf.ols(formula=formula_index, data=df_model).fit()
print(f"Base index model R-squared: {model_index.rsquared:.3f}")

AGE_COEF = model_index.params["age"]
AGE_SQ_COEF = model_index.params["age_sq"]

year_effects = {}
for name, coef in model_index.params.items():
    if "season_cat" in name:
        year = name.split("T.")[-1].rstrip("]")
        year_effects[int(year)] = coef
year_effects[int(BASE_YEAR)] = 0.0
index_df = pd.DataFrame(sorted(year_effects.items()), columns=["year", "log_coef"])
index_df["index"] = np.exp(index_df["log_coef"])


def get_index(year):
    row = index_df.loc[index_df["year"] == year, "index"]
    return row.values[0] if not row.empty else None


formula_quality = (
    "log_fee ~ age + age_sq + composite_score_filled + has_stats "
    "+ C(position_model) + C(window) + C(league) "
    f"+ C(season_cat, Treatment(reference=\x27{BASE_YEAR}\x27))"
)
print("Fitting quality model...")
model_quality = smf.ols(formula=formula_quality, data=df_model).fit()
print(f"Quality model R-squared: {model_quality.rsquared:.3f}")

COMPOSITE_COEF = model_quality.params["composite_score_filled"]


def find_player_transfers(name, df_source=df):
    name_clean = unidecode(name).lower()
    normalized = df_source["player_name"].apply(lambda x: unidecode(str(x)).lower())
    return df_source[normalized.str.contains(name_clean, na=False)].sort_values("season", ascending=False)


def get_candidates_table(name):
    matches = find_player_transfers(name)
    if matches.empty:
        return None
    candidates = matches.groupby("player_id").agg(
        player_name=("player_name", "first"),
        nationality=("nationality", "first"),
        position=("position_clean", "first"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        n_transfers=("player_id", "count"),
    ).reset_index().sort_values("last_season", ascending=False)
    return candidates


def show_candidates(name):
    candidates = get_candidates_table(name)
    if candidates is None:
        print(f"No candidates found for \x27{name}\x27.")
        return None
    print(f"\nFound {len(candidates)} distinct player(s) matching \x27{name}\x27:\n")
    for _, row in candidates.iterrows():
        pid = row["player_id"]
        pname = row["player_name"]
        nat = row["nationality"]
        pos = row["position"]
        first_yr = int(row["first_season"])
        last_yr = int(row["last_season"])
        n_tx = row["n_transfers"]
        print(f"  player_id={pid}  |  {pname}  ({nat}, {pos})  |  "
              f"active {first_yr}-{last_yr}, {n_tx} transfer(s) on record")
    return candidates


def find_nearest_record(player_name, near_year, player_id=None):
    if player_id is not None:
        matches = df[df["player_id"] == player_id].copy()
    else:
        matches = find_player_transfers(player_name)
        if not matches.empty:
            n_distinct = matches["player_id"].nunique()
            if n_distinct > 1:
                print(f"NOTE: \x27{player_name}\x27 matches {n_distinct} distinct players. "
                      f"Defaulting to nearest-year match - use interactive_predict() to choose.")
    if matches.empty:
        return None
    matches = matches.copy()
    matches["year_dist"] = (matches["season"] - near_year).abs()
    return matches.sort_values("year_dist").iloc[0]


LEAGUE_SLUGS = ["eng", "esp", "ger", "ita", "fra"]
SLUG_TO_LEAGUE = {
    "eng": "premier_league", "esp": "laliga", "ger": "bundesliga",
    "ita": "serie_a", "fra": "ligue_1",
}

def load_fbref(stat_type):
    dfs = []
    for slug in LEAGUE_SLUGS:
        path = Path(f"data/raw/fbref_{stat_type}_{slug}.parquet")
        if path.exists():
            d = pd.read_parquet(path)
            d["league"] = SLUG_TO_LEAGUE[slug]
            dfs.append(d)
    combined = pd.concat(dfs, ignore_index=True)

    def fbref_season_to_year(s):
        s = str(s)
        return int("20" + s[:2]) if len(s) == 4 else np.nan
    combined["season"] = combined["season"].apply(fbref_season_to_year)
    return combined


FBREF_POS_MAP = {
    "GK": "GK", "DF": "DEF", "DF,MF": "DEF", "MF,DF": "DEF",
    "MF": "MID", "MF,FW": "MID", "FW,MF": "MID",
    "FW": "FWD", "DF,FW": "FWD",
}

print("Loading FBref stat tables for ranking...")
fbref_standard = load_fbref("standard")
fbref_misc = load_fbref("misc")
fbref_keeper = load_fbref("keeper")

fbref_standard["fbref_position_group"] = fbref_standard["pos"].map(FBREF_POS_MAP).fillna("MID")
fbref_standard["nineties"] = fbref_standard["Playing Time_90s"].replace(0, np.nan)
fbref_standard["goals_p90"] = fbref_standard["Performance_Gls"] / fbref_standard["nineties"]
fbref_standard["assists_p90"] = fbref_standard["Performance_Ast"] / fbref_standard["nineties"]
fbref_standard["attacking_output"] = fbref_standard["goals_p90"].fillna(0) + fbref_standard["assists_p90"].fillna(0)
fbref_standard_reliable = fbref_standard[fbref_standard["Playing Time_90s"] >= 10]

fbref_misc["has_defensive_data"] = fbref_misc["Performance_TklW"].notna() & fbref_misc["Performance_Int"].notna()
fbref_misc["fbref_position_group"] = fbref_misc["pos"].map(FBREF_POS_MAP).fillna("MID")
fbref_misc["nineties"] = fbref_misc["90s"].replace(0, np.nan)
fbref_misc["tklw_p90"] = fbref_misc["Performance_TklW"] / fbref_misc["nineties"]
fbref_misc["int_p90"] = fbref_misc["Performance_Int"] / fbref_misc["nineties"]
fbref_misc["defensive_output"] = fbref_misc["tklw_p90"] + fbref_misc["int_p90"]
fbref_misc_reliable = fbref_misc[(fbref_misc["90s"] >= 10) & (fbref_misc["has_defensive_data"])]

fbref_keeper["has_keeper_data"] = fbref_keeper["Performance_Save%"].notna() & fbref_keeper["Performance_CS%"].notna()
fbref_keeper["fbref_position_group"] = fbref_keeper["pos"].map(FBREF_POS_MAP).fillna("GK")
fbref_keeper["keeping_output"] = fbref_keeper["Performance_Save%"] + fbref_keeper["Performance_CS%"]
fbref_keeper_reliable = fbref_keeper[(fbref_keeper["Playing Time_90s"] >= 5) & (fbref_keeper["has_keeper_data"])]


def find_nearest_in_table(table, player_name, near_year):
    name_clean = unidecode(player_name).lower()
    normalized = table["player"].apply(lambda x: unidecode(str(x)).lower())
    matches = table[normalized.str.contains(name_clean, na=False)].copy()
    if matches.empty:
        return None
    matches["year_dist"] = (matches["season"] - near_year).abs()
    return matches.sort_values("year_dist").iloc[0]


def find_nearest_fbref_record(player_name, near_year):
    return find_nearest_in_table(fbref_standard, player_name, near_year)


def _get_metric_row_and_pool(player_name, near_year, position_group, era_years=None):
    if position_group in ("FWD", "MID"):
        row = find_nearest_in_table(fbref_standard, player_name, near_year)
        if row is None:
            return None, None, None
        pool = fbref_standard_reliable[fbref_standard_reliable["fbref_position_group"] == position_group]
        if era_years is not None:
            pool = pool[pool["season"].isin(era_years)]
        return row, row["attacking_output"], pool["attacking_output"]
    elif position_group == "DEF":
        tracked = fbref_misc[fbref_misc["has_defensive_data"]]
        row = find_nearest_in_table(tracked, player_name, near_year)
        if row is None:
            return None, None, None
        pool = fbref_misc_reliable[fbref_misc_reliable["fbref_position_group"] == "DEF"]
        if era_years is not None:
            pool = pool[pool["season"].isin(era_years)]
        return row, row["defensive_output"], pool["defensive_output"]
    elif position_group == "GK":
        tracked = fbref_keeper[fbref_keeper["has_keeper_data"]]
        row = find_nearest_in_table(tracked, player_name, near_year)
        if row is None:
            return None, None, None
        pool = fbref_keeper_reliable
        if era_years is not None:
            pool = pool[pool["season"].isin(era_years)]
        return row, row["keeping_output"], pool["keeping_output"]
    else:
        return None, None, None


def get_percentile(player_name, near_year, position_group, era_years=None):
    row, player_value, pool_values = _get_metric_row_and_pool(player_name, near_year, position_group, era_years)
    if row is None or player_value is None or pd.isna(player_value) or len(pool_values) == 0:
        return 0.5, 0, False
    percentile = (pool_values < player_value).mean()
    return percentile, len(pool_values), True


def get_era_window(target_year, half_width, min_year=MIN_YEAR, max_year=MAX_YEAR):
    start = target_year - half_width
    end = target_year + half_width
    if end > max_year:
        shortfall = end - max_year
        end = max_year
        start = max(start - shortfall, min_year)
    if start < min_year:
        shortfall = min_year - start
        start = min_year
        end = min(end + shortfall, max_year)
    return range(start, end + 1)


OUTLIER_THRESHOLD = 0.95
MIN_PLAUSIBLE_AGE = 15
MAX_PLAUSIBLE_AGE = 42


def get_alltime_fee_leaderboard(position_group, target_year):
    pool = df_model[df_model["position_group"] == position_group].copy()
    idx_target = get_index(target_year)
    pool["fee_adjusted"] = pool.apply(
        lambda r: r["fee"] * (idx_target / get_index(int(r["season"]))), axis=1
    )
    return pool.sort_values("fee_adjusted", ascending=False)["fee_adjusted"].reset_index(drop=True)


def percentile_to_rank(percentile):
    if percentile >= 0.99:
        return 1
    rank = 7 - (percentile - 0.95) / (0.99 - 0.95) * 6
    return max(1, round(rank))


def predict_no_anchor(player_name, target_year, near_year, verbose=True):
    row = find_nearest_fbref_record(player_name, near_year)
    if row is None:
        if verbose:
            print(f"No player found matching \x27{player_name}\x27 in either the transfer data or FBref stats.")
        return None

    fbref_player_name = row["player"]
    player_season = int(row["season"])
    position_group = row["fbref_position_group"]

    HALF_WIDTH = 3
    era_years = get_era_window(target_year, HALF_WIDTH)

    era_percentile, era_pool_size, has_real_data = get_percentile(player_name, near_year, position_group, era_years=era_years)

    if not has_real_data:
        if verbose:
            print("There is no available transfer data or match stats for this player, unable to calculate fee")
        return None

    used_alltime = era_percentile >= OUTLIER_THRESHOLD

    if used_alltime:
        final_percentile, alltime_pool_size, _ = get_percentile(player_name, near_year, position_group, era_years=None)
        leaderboard = get_alltime_fee_leaderboard(position_group, target_year)
        rank = percentile_to_rank(final_percentile)
        rank = min(rank, len(leaderboard))
        pred = leaderboard.iloc[rank - 1]

        if verbose:
            print(f"\n[No fee on record - ALL-TIME LEADERBOARD fallback]")
            print(f"FBref record: {fbref_player_name}, season {player_season}, position group: {position_group}")
            print(f"Era percentile (vs {era_pool_size} player-seasons in era): {era_percentile:.4f}")
            print(f"OUTLIER DETECTED - escalated to all-time comparison")
            print(f"All-time percentile (vs {alltime_pool_size} player-seasons, all years): {final_percentile:.4f}")
            print(f"Mapped to leaderboard rank #{rank} of {len(leaderboard)} (all-time fees, inflation-adjusted to {target_year})")
            print(f"--- Estimated value in {target_year} (LOWER CONFIDENCE - no fee anchor) ---")
            print(f"EUR{pred:,.0f}  (rank #{rank} most expensive {position_group} transfer ever, in {target_year} terms)")

        return pred

    else:
        fee_pool = df_model[
            (df_model["season"].isin(era_years)) &
            (df_model["position_group"] == position_group)
        ]
        extra_width = HALF_WIDTH
        while len(fee_pool) < 20 and extra_width < 15:
            extra_width += 2
            era_years = get_era_window(target_year, extra_width)
            fee_pool = df_model[
                (df_model["season"].isin(era_years)) &
                (df_model["position_group"] == position_group)
            ]
        if len(fee_pool) < 3:
            print(f"Not enough real transfers found for position group {position_group} near {target_year}.")
            return None

        min_tail_fraction = 3 / len(fee_pool)
        effective_percentile = min(era_percentile, 1 - min_tail_fraction)
        threshold = np.quantile(fee_pool["fee"], effective_percentile)
        tail = fee_pool.loc[fee_pool["fee"] >= threshold, "fee"]
        pred = tail.mean()

        if verbose:
            print(f"\n[No fee on record - era-relative fallback]")
            print(f"FBref record: {fbref_player_name}, season {player_season}, position group: {position_group}")
            print(f"Era percentile (vs {era_pool_size} player-seasons in era): {era_percentile:.4f}")
            print(f"Era window: {min(era_years)}-{max(era_years)} ({len(fee_pool)} real transfers, same position group)")
            print(f"Effective percentile used (capped for tail sample size): {effective_percentile:.4f}")
            print(f"Percentile boundary fee: EUR{threshold:,.0f}  |  Tail size: {len(tail)} transfers")
            print(f"--- Estimated value in {target_year} (LOWER CONFIDENCE - no fee anchor) ---")
            print(f"EUR{pred:,.0f}  (mean fee among transfers at/above this percentile)")

        return pred


def predict_value(player_name, target_year, source_year=None, player_id=None, verbose=True):
    if target_year < MIN_YEAR or target_year > MAX_YEAR:
        print(f"target_year {target_year} is outside the fitted index range ({MIN_YEAR}-{MAX_YEAR}).")
        return None

    anchor_year = source_year if source_year is not None else target_year
    anchor_row = find_nearest_record(player_name, anchor_year, player_id=player_id)

    if anchor_row is None:
        near_year = source_year if source_year is not None else target_year
        return predict_no_anchor(player_name, target_year, near_year, verbose=verbose)

    anchor_name = anchor_row["player_name"]
    anchor_id = anchor_row["player_id"]
    known_year = int(anchor_row["season"])
    known_age = anchor_row["age"]
    known_fee = anchor_row["fee"]
    composite_score = anchor_row["composite_score"]
    has_stats = pd.notna(composite_score)
    composite_score_filled = composite_score if has_stats else 0.5

    if source_year is None:
        source_year = known_year

    if source_year < MIN_YEAR or source_year > MAX_YEAR:
        print(f"source_year {source_year} is outside the fitted index range ({MIN_YEAR}-{MAX_YEAR}).")
        return None

    interpolated = source_year != known_year
    age_at_source = known_age + (source_year - known_year)

    if age_at_source < MIN_PLAUSIBLE_AGE or age_at_source > MAX_PLAUSIBLE_AGE:
        print(f"Player would be {age_at_source:.0f} in {source_year}, unable to calculate fee")
        return None

    if interpolated:
        idx_known = get_index(known_year)
        idx_source = get_index(source_year)
        log_fee_source = (
            np.log(known_fee)
            + AGE_COEF * (age_at_source - known_age)
            + AGE_SQ_COEF * (age_at_source ** 2 - known_age ** 2)
            + np.log(idx_source / idx_known)
        )
        source_fee = np.exp(log_fee_source)
    else:
        source_fee = known_fee

    idx_source = get_index(source_year)
    idx_target = get_index(target_year)
    base_prediction = source_fee * (idx_target / idx_source)

    if has_stats:
        quality_premium = np.exp(COMPOSITE_COEF * (composite_score_filled - 0.5))
    else:
        quality_premium = 1.0

    final_prediction = base_prediction * quality_premium

    if verbose:
        interp_note = f" (interpolated from real {known_year} record, age {known_age})" if interpolated else " (real record)"
        stats_note = f"{composite_score_filled:.2f} percentile" if has_stats else "no stats available"
        print(f"\n[Fee anchor found]")
        print(f"Anchor: {anchor_name} (player_id={anchor_id}), age {known_age} in {known_year}, "
              f"fee EUR{known_fee:,.0f}, composite score: {stats_note}")
        print(f"Source snapshot: age {age_at_source} in {source_year}{interp_note}")
        print(f"Method A (index only):        EUR{base_prediction:,.0f}")
        print(f"Quality premium multiplier:   x{quality_premium:.3f}")
        print(f"--- Final predicted value in {target_year} ---")
        print(f"EUR{final_prediction:,.0f}")

    return final_prediction


def interactive_predict(player_name, target_year, source_year=None):
    """
    Checks if the name is ambiguous first. If more than one distinct
    player matches, shows a numbered list and asks you to pick one
    before running the prediction.
    """
    candidates = get_candidates_table(player_name)

    if candidates is None:
        return predict_value(player_name, target_year, source_year)

    if len(candidates) == 1:
        chosen_id = candidates.iloc[0]["player_id"]
        return predict_value(player_name, target_year, source_year, player_id=chosen_id)

    print(f"\n\x27{player_name}\x27 matches {len(candidates)} different players:\n")
    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        pname = row["player_name"]
        nat = row["nationality"]
        pos = row["position"]
        first_yr = int(row["first_season"])
        last_yr = int(row["last_season"])
        print(f"  [{i}] {pname}  ({nat}, {pos})  |  active {first_yr}-{last_yr}")

    while True:
        choice = input(f"\nWhich player? (1-{len(candidates)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            chosen_id = candidates.iloc[int(choice) - 1]["player_id"]
            break
        print("Invalid choice, try again.")

    return predict_value(player_name, target_year, source_year, player_id=chosen_id)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Interactive player valuation. Type \x27quit\x27 as a name to exit.")
    while True:
        print("\n" + "=" * 60)
        name = input("Player name: ").strip()
        if name.lower() == "quit":
            break
        try:
            tgt = int(input("Target year: ").strip())
        except ValueError:
            print("Invalid year, try again.")
            continue
        src_input = input("Source year (press Enter to skip): ").strip()
        src = int(src_input) if src_input else None

        interactive_predict(name, tgt, src)
