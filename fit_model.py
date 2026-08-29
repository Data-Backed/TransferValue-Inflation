import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_parquet("data/clean/transfers_with_stats_v1.parquet")
print(f"Loaded {df.shape[0]} rows for modeling")

model_cols = ["fee", "age", "position_clean", "window", "season", "league", "market_value"]
df_model = df.dropna(subset=model_cols).copy()
print(f"After dropping rows missing model fields: {df_model.shape[0]} rows")

MIN_TRANSFERS_PER_YEAR = 30  # years with fewer confirmed transfers are excluded as unreliable

year_counts = df_model["season"].value_counts()
reliable_years = year_counts[year_counts >= MIN_TRANSFERS_PER_YEAR].index
excluded_years = sorted(set(df_model["season"].unique()) - set(reliable_years))
print(f"Excluding {len(excluded_years)} years with fewer than {MIN_TRANSFERS_PER_YEAR} "
      f"transfers: {excluded_years}")

df_model = df_model[df_model["season"].isin(reliable_years)].copy()
print(f"After excluding thin years: {df_model.shape[0]} rows")

df_model["has_stats"] = df_model["composite_score"].notna().astype(int)
df_model["composite_score_filled"] = df_model["composite_score"].fillna(0.5)

df_model["log_fee"] = np.log(df_model["fee"])
df_model["age_sq"] = df_model["age"] ** 2
df_model["log_market_value"] = np.log(df_model["market_value"])
df_model["season_cat"] = df_model["season"].astype(int).astype(str)

pos_counts = df_model["position_clean"].value_counts()
rare_positions = pos_counts[pos_counts < 30].index
df_model["position_model"] = df_model["position_clean"].where(
    ~df_model["position_clean"].isin(rare_positions), "Other"
)

BASE_YEAR = str(int(df_model["season"].min()))

formula = (
    "log_fee ~ age + age_sq + log_market_value + composite_score_filled + has_stats "
    "+ C(position_model) + C(window) + C(league) "
    f"+ C(season_cat, Treatment(reference=\x27{BASE_YEAR}\x27))"
)

print(f"\nFitting model with base year {BASE_YEAR}...")
model = smf.ols(formula=formula, data=df_model).fit()

print(model.summary())

year_effects = {}
for name, coef in model.params.items():
    if "season_cat" in name:
        year = name.split("T.")[-1].rstrip("]")
        year_effects[int(year)] = coef
year_effects[int(BASE_YEAR)] = 0.0

index_df = pd.DataFrame(sorted(year_effects.items()), columns=["year", "log_coef"])
index_df["index"] = np.exp(index_df["log_coef"])
index_df = index_df.sort_values("year")

print("\nInflation index by year (1.0 = base year level):")
print(index_df.to_string(index=False))

Path("data/clean").mkdir(parents=True, exist_ok=True)
index_df.to_csv("data/clean/inflation_index.csv", index=False)
print("\nSaved: data/clean/inflation_index.csv")

plt.figure(figsize=(10, 6))
plt.plot(index_df["year"], index_df["index"], marker="o")
plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
plt.title(f"Football Transfer Fee Inflation Index (base year = {BASE_YEAR})")
plt.xlabel("Year")
plt.ylabel("Index (relative fee level)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/clean/inflation_index.png", dpi=150)
print("Saved: data/clean/inflation_index.png")

print("\nModel R-squared:", round(model.rsquared, 3))
print("Done.")
