import soccerdata as sd
import pandas as pd
from pathlib import Path

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

SEASONS = [str(y) for y in range(2000, 2026)]

Path("data/raw").mkdir(parents=True, exist_ok=True)

all_stats = []

for league in LEAGUES:
    print(f"\nFetching {league}...")
    try:
        fbref = sd.FBref(leagues=league, seasons=SEASONS)

        standard = fbref.read_player_season_stats(stat_type="standard")
        keeper = fbref.read_player_season_stats(stat_type="keeper")
        misc = fbref.read_player_season_stats(stat_type="misc")

        standard = standard.reset_index()
        keeper = keeper.reset_index()
        misc = misc.reset_index()

        for d in (standard, keeper, misc):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = ["_".join(filter(None, col)).strip() for col in d.columns]

        standard["league"] = league
        keeper["league"] = league
        misc["league"] = league

        all_stats.append({"league": league, "standard": standard,
                           "keeper": keeper, "misc": misc})

    except Exception as e:
        print(f"Failed to fetch {league}: {e}")
        continue

for entry in all_stats:
    league_slug = entry["league"].split("-")[0].lower()
    entry["standard"].to_parquet(f"data/raw/fbref_standard_{league_slug}.parquet")
    entry["keeper"].to_parquet(f"data/raw/fbref_keeper_{league_slug}.parquet")
    entry["misc"].to_parquet(f"data/raw/fbref_misc_{league_slug}.parquet")
    print(f"Saved stats for {entry['league']}")

print("\nDone. Check data/raw/ for fbref_*.parquet files.")
