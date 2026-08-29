import pandas as pd

POSITION_GROUP_MAP = {
    "Goalkeeper": "GK",

    "Centre-Back": "DEF",
    "Left-Back": "DEF",
    "Right-Back": "DEF",
    "Defender": "DEF",
    "Sweeper": "DEF",

    "Defensive Midfield": "MID",
    "Central Midfield": "MID",
    "Attacking Midfield": "MID",
    "Left Midfield": "MID",
    "Right Midfield": "MID",
    "Midfielder": "MID",

    "Left Winger": "FWD",
    "Right Winger": "FWD",
    "Centre-Forward": "FWD",
    "Second Striker": "FWD",
    "Striker": "FWD",
    "Forward": "FWD",
    "Other": "MID",
}


def add_position_group(df, position_col="position_clean"):
    df = df.copy()
    df["position_group"] = df[position_col].map(POSITION_GROUP_MAP)
    unmapped = df["position_group"].isna().sum()
    if unmapped:
        print(f"Warning: {unmapped} rows have an unmapped position label. "
              f"Unmapped values: {df.loc[df['position_group'].isna(), position_col].unique()}")
    return df


if __name__ == "__main__":
    df = pd.read_parquet("data/clean/transfers_clean_v1.parquet")
    df = add_position_group(df)
    print(df["position_group"].value_counts())
