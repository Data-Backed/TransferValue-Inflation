import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/clean/transfers_with_stats_v1.parquet")

model_cols = ["fee", "age", "position_clean", "window", "season", "league", "market_value"]
df_model = df.dropna(subset=model_cols).copy()

MIN_TRANSFERS_PER_YEAR = 30
year_counts = df_model["season"].value_counts()
reliable_years = year_counts[year_counts >= MIN_TRANSFERS_PER_YEAR].index
df_model = df_model[df_model["season"].isin(reliable_years)].copy()

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

# Split BEFORE fitting - train on 85%, test on the held-out 15%
train_df, test_df = train_test_split(df_model, test_size=0.15, random_state=42)
print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

formula = (
    "log_fee ~ age + age_sq + log_market_value + composite_score_filled + has_stats "
    "+ C(position_model) + C(window) + C(league) "
    f"+ C(season_cat, Treatment(reference=\x27{BASE_YEAR}\x27))"
)

print("\nFitting on training set only...")
model = smf.ols(formula=formula, data=train_df).fit()
print(f"Training R-squared: {model.rsquared:.3f}")

# Predict on the held-out test set
# Some test rows may have a season_cat/position_model/league category the
# training set never saw - drop those rather than error out (rare edge case)
valid_categories = {
    "season_cat": set(train_df["season_cat"].unique()),
    "position_model": set(train_df["position_model"].unique()),
    "league": set(train_df["league"].unique()),
    "window": set(train_df["window"].unique()),
}
mask = (
    test_df["season_cat"].isin(valid_categories["season_cat"]) &
    test_df["position_model"].isin(valid_categories["position_model"]) &
    test_df["league"].isin(valid_categories["league"]) &
    test_df["window"].isin(valid_categories["window"])
)
test_df_valid = test_df[mask].copy()
dropped = len(test_df) - len(test_df_valid)
if dropped:
    print(f"Dropped {dropped} test rows with categories unseen in training")

pred_log_fee = model.predict(test_df_valid)
actual_log_fee = test_df_valid["log_fee"]

pred_fee = np.exp(pred_log_fee)
actual_fee = test_df_valid["fee"]

# Metrics on log scale (standard for skewed fee data)
rmse_log = np.sqrt(np.mean((pred_log_fee - actual_log_fee) ** 2))
mae_log = np.mean(np.abs(pred_log_fee - actual_log_fee))

# Metrics on original EUR scale (more interpretable)
mape = np.mean(np.abs(pred_fee - actual_fee) / actual_fee) * 100
median_ape = np.median(np.abs(pred_fee - actual_fee) / actual_fee) * 100

print(f"\n--- Holdout validation results (n={len(test_df_valid)}) ---")
print(f"RMSE (log scale):        {rmse_log:.3f}")
print(f"MAE (log scale):         {mae_log:.3f}")
print(f"Mean abs % error:        {mape:.1f}%")
print(f"Median abs % error:      {median_ape:.1f}%")

# Show a sample of predictions vs actuals for a gut check
sample = test_df_valid[["player_name", "season", "fee"]].copy()
sample["predicted_fee"] = pred_fee.values
sample["pct_error"] = ((sample["predicted_fee"] - sample["fee"]) / sample["fee"] * 100).round(1)
print("\nSample of 15 predictions vs actuals:")
print(sample.sample(15, random_state=1).to_string(index=False))
