# Football Transfer Fee Inflation & Era-Translation Model

This is a hedonic pricing model that (1) measures football transfer fee inflation across the top 5 European leagues (2000–2025) and (2) predicts what any player's transfer fee would be in a different year — accounting for market inflation, age, position, and performance quality.

Built through an actuarial lense: it separates price-level trend from player quality, explicitly stating its sample-size limitations, and only produces a fee if there is data to provide one.

---

# 1. What this project does

- **Inflation index**: 
controls for player agee, position,league, and transfer window when quantifying how football transfers have changed over time
- **Era-translation engine**: `predict_value(player_name, target_year, source_year)` - input a player and the year you want to value them in. It works for players with a real recorded transfer fee and, with lower confidence, players who have never had a transfer/only had moves outside the top 5 leagues

---

## 2. Data Sources

| Source | What it provides | Coverage |
|---|---|---|
| [`eordo/transfermarkt-data`](https://github.com/eordo/transfermarkt-data) (GitHub) | Transfer fees, player age, position, market value, transfer window | Premier League, La Liga, Bundesliga, Serie A, Ligue 1 — 1992–2025 |
| FBref, via the `soccerdata` Python library | Goals, assists, tackles, interceptions, save %, clean sheet % | Same 5 leagues, 2000–2025 |

Both sources required cleaning quirks worth knowing about if you extend this project:
- The Transfermarkt repo's Spanish league folder is named `laliga` (no underscore) — an early version of this pipeline missed this and silently ran with only 4 leagues for a period before it was caught and fixed.
- FBref requires a Chrome-driven scraper (`seleniumbase`) rather than plain HTTP requests, due to bot-protection measures on the site.

---

## 3. Data Cleaning & Scope Decisions

Starting from ~93,500 raw Transfermarkt transfer records:

| Step | Rows remaining | Notes |
|---|---|---|
| Restrict to season ≥ 2000 | ~78,500 | |
| Exclude loans | ~45,700 | Loans aren't "sales" and would distort fee modeling |
| Exclude free transfers (fee = 0) | ~20,800 confirmed-fee rows | Kept separately, not discarded |
| Deduplicate cross-listed transfers | ~20,832 | Same transfer often logged once per club |
| Drop years with less than 30 confirmed transfers | final modeling set | 2001–2003 excluded — sample too thin for stable year coefficients |

**Fee confidence tiers**: every row is tagged `confirmed`, `likely_free` (fee = 0), or `unknown` (fee missing). Only `confirmed` rows are used in the fee model.

**Name matching**: accents have no effect on the player searching (`unidecode`) so "Mbappé" and "Mbappe" match — without this, any accented name silently failed to match.

---

## 4. Methodology

### 4a. The Inflation Index

A hedonic regression, fit on ~20,800 confirmed transfers:

```
log(fee) ~ age + age² + position + league + transfer_window + year fixed effects
```

The exponentiated year coefficients form the index — the relative price level of football transfers by year, holding player characteristics constant. This is different from simply averaging fees by year, which would conflate genuine inflation with "which players happened to move that year."

**R² = 0.676** for the full model (which also includes `market_value` as a predictor).

A separate, leaner version of this index — used inside the era-translation engine — deliberately **excludes `market_value`**. Early in development, including `market_value` alongside year fixed effects corrupted the index: since `market_value` itself trends upward with inflation, controlling for it strips out most of the genuine trend signal from the year coefficients, producing a nonsensical "deflating" index for elite players. Removing it costs R² (down to 0.193) but keeps the index directionally correct — a deliberate accuracy-for-correctness tradeoff.

### 4b. Era-Translation — Two Paths

**Path 1: Fee-anchored (used whenever a real transfer fee exists)**
1. Find the player's transfer record nearest to the requested `source_year`.
2. If `source_year` doesn't exactly match a real record, interpolate: age the player forward/backward using the model's age curve, and adjust for the price-level difference between the known year and the requested source year.
3. Scale the resulting fee from `source_year` to `target_year` using the index ratio.
4. Apply a **quality premium**: a multiplier derived from the player's composite performance score (see 4c), so two players with the same real fee but different quality profiles translate differently across eras.

**Path 2: No-anchor fallback (players with no fee on record anywhere)**
1. Rank the player's raw performance output against real players in the same position group, within a multi-year "era window" (target year ± 3, shifted to stay inside the dataset near either edge).
2. **If below the 95th percentile for their era**: take the mean fee among real transfers at or above that percentile, within the era window.
3. **If at or above the 95th percentile (a genuine outlier)**: escalate to an all-time comparison - rank the player against every player-season in the same position group across the full dataset, not just their own era. Map that all-time percentile onto a literal rank in an inflation-adjusted "most expensive transfer ever" leaderboard for that position (99th percentile → rank #1, 95th percentile → roughly rank #7, interpolated in between). The outlier gets the *maximum* fee in the qualifying tier, not the average - "the best gets the biggest," not a diluted mean.
4. **If no tracked performance data exists at all** (see limitations — this affects most defenders before ~2013): refuses to produce a number rather than defaulting to a misleading neutral guess.

### 4c. Composite Performance Score

Computed as a **percentile rank within the player's own season and position group** — not a raw stat — so it's automatically comparable across eras regardless of how playing styles or stat magnitudes have shifted:

| Position group | Metric |
|---|---|
| FWD / MID | Goals + assists per 90 |
| DEF | Tackles won + interceptions per 90 |
| GK | Save % + clean sheet % |

Because this is a *percentile within season*, its distribution is stable across years by construction — which is what makes it safe to include alongside year fixed effects without contaminating the index (unlike `market_value`).

---

## 5. Validation

Holdout test (15% of confirmed transfers, held out before fitting):

- **Median absolute % error: 40.5%** — the more reliable summary statistic for this kind of noisy, negotiation-driven market.
- **Mean absolute % error: 555.1%** — inflated by *retransformation bias*, a known property of log-linear regression where percentage errors on small-fee transfers appear disproportionately large even when the absolute euro miss is modest. Not indicative of the model being unreliable at the top end.

---

## 6. Known Limitations

Documented explicitly rather than papered over — these are real, specific gaps found during testing, not hypothetical caveats:

- **Pre-2013 defensive data doesn't exist.** FBref's tackle/interception tracking effectively starts around the Opta-data era (~2013). Defenders whose whole career predates this (e.g. Jamie Carragher) have no performance data at all, and the model correctly refuses to estimate their value rather than guessing.
- **The defensive metric underrates elite positional defenders.** Tackles + interceptions rewards *committed* defending, not prevention. Virgil van Dijk's peak 2017 season scores at only the 11th percentile on this metric despite being widely regarded as one of the best defenders of his generation — his game relies on positioning and aerial control rather than high tackle volume. This is a real, named blind spot, not a bug.
- **Some well-known players are simply absent.** Players like Antoine Griezmann and Lionel Messi have gaps or complete absences in the confirmed-fee dataset — usually because their major moves were free transfers, loans, or had undisclosed fees. Messi is handled via the no-anchor fallback; some others are not recoverable without additional data sources.
- **Position mapping in the no-anchor fallback is an approximation.** FBref's broad position groups (GK/DEF/MID/FWD) are mapped to a single representative Transfermarkt-style label (e.g. all midfielders → "Central Midfield") for prediction purposes, rather than the player's true specific role.
- **Ambiguous names are resolved interactively.** A search for a common surname (e.g. "Silva", which has 34 matches, or Rodri, with 47 matches) won't pick the name that happens to be sorted first. `predict_value.py` checks every name against the full list of distinct players (grouped by Transfermarkt's `player_id`) before running the valuation: if only one match exists, it proceeds automatically; if there are more than one match, it prints a numbered list of every candidate - name, nationality, position, years active - and asks which player you wanted before producing a result. This was an early bug (a search for "Rodri" originally retuened Chema Rodriguez instead of the former Manchester City Midfielder) that is now fixed.
- **Non-Latin-script name transliteration isn't fully covered.** Accent-stripping fixes Mbappé-style cases but doesn't resolve inconsistent Latin romanizations of names originally in other scripts.
- **Implausible age interpolation is refused, not guessed.** If interpolating a player's age to a requested `source_year` would put them under 15 or over 42, the function refuses outright rather than returning a nonsense figure.

---

## 7. Worked Examples

**Neymar, 2017 real transfer → predicted 2025 value**
Anchored on his real €222M PSG move; index-scaled and quality-premium-adjusted for 2025's price level.

**Lionel Messi, 2025 valuation (no fee anchor)**
Messi's major moves were free tranfers, meaning there is no transfer data for him. His 2015 performance profile is ranked against the all-time forward pool, detected as a genuine outlier (>95th percentile even against all-time peers), and valued via the inflation-adjusted all-time leaderboard rather than a diluted era-average.

**Jamie Carragher, 2025 valuation — refused**
No transfer fee on record (homegrown academy product) and no tracked defensive data exists for his entire career (2000–2013, predating detailed event tracking). The model correctly declines to produce a number: *"There is no available transfer data or match stats for this player, unable to calculate fee."*

**Kylian Mbappé, hypothetical 2007 source year — refused**
Requesting his real 2018 anchor interpolated back to 2007 would imply an age of 8. The model refuses: *"Player would be 8 in 2007, unable to calculate fee."*

---

## 8. How to Run

```bash
pip install pandas numpy statsmodels matplotlib scikit-learn soccerdata rapidfuzz unidecode
```

Run in order:

```bash
python load_data.py       # pulls raw Transfermarkt CSVs into one table
python clean_data.py      # filters loans/frees, dedupes, flags fee confidence
python fetch_stats.py     # scrapes FBref performance data (slow - Chrome required)
python merge_stats.py     # fuzzy-matches players, builds composite scores
python fit_model.py       # fits the headline inflation index + chart
python predict_value.py   # the era-translation engine - edit the test cases at the bottom
```

`fetch_stats.py` requires Google Chrome installed locally (used by `soccerdata`'s scraper to bypass bot detection).

**Note on `predict_value.py`**: running it starts an interactive prompt - type a player name, a target year, and a source year, If the name matches more than one player (e.g "Silva", "Rodri"), you'll be shown a numbered list of candidated - with nationality, position, and years active - and asked to pick the right one before tha valuation runs. Type `quit` to exit.

---

## 9. Tech Stack

- **Data**: `pandas`, `pyarrow` (Parquet storage)
- **Modeling**: `statsmodels` (OLS hedonic regression)
- **Scraping**: `soccerdata` (FBref, via Selenium/Chrome)
- **Matching**: `rapidfuzz` (fuzzy name matching), `unidecode` (accent normalization)
- **Validation**: `scikit-learn` (train/test split)
- **Visualization**: `matplotlib`

## 10. Attribution

- Transfer data: [eordo/transfermarkt-data](https://github.com/eordo/transfermarkt-data), sourced from Transfermarkt.
- Performance data: [FBref](https://fbref.com), via the [soccerdata](https://github.com/probberechts/soccerdata) library.
