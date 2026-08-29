# Football Transfer Fee Inflation Project — 4-Week Sprint Version
**Target completion: before September 5, 2026**

## Key change: top 5 leagues stays in scope

Leagues aren't actually the expensive part of the project — scraping is roughly equal effort for 1 league or 5, since it's automated either way. The trick is **not building a scraper from scratch**: use an existing pre-scraped Transfermarkt dataset (Kaggle has a well-maintained one covering fees, ages, positions, and appearances across the top 5 leagues, updated regularly). That removes almost all of Week 1's risk and buys back the time the extra leagues would otherwise cost. To pay for it, cut modeling depth instead — that's the tradeoff below.

## What gets cut vs. the full version

| Full version | Sprint version |
|---|---|
| Custom scraper, 2000–present | **Existing dataset (Kaggle/Transfermarkt), 2010–present, top 5 leagues kept** |
| Advanced stats (xG, progressive passes) | Basic stats only: goals, assists, appearances, minutes |
| Two-part hurdle model for elite tier | Single GLM with log-link (skip the tail-splitting) |
| Bayesian hierarchical credibility model | Simple regularization (ridge) as a credibility proxy |
| Interactive Streamlit app | Static notebook + 3-4 clean charts (app only if time remains) |
| 5 anchor case studies | **2 case studies**: Neymar and Elliot Anderson only |
| Formal backtesting across tiers | Single holdout test, no tier-by-tier breakdown |

Nothing here weakens the *core idea* (hedonic index + era-translation) — it just swaps custom scraping for an existing dataset and drops modeling polish/extensions.

---

## Week 1 (Aug 7–14): Data

- **Day 1**: Source an existing top-5-league Transfermarkt dataset (Kaggle — search "Transfermarkt transfers" or "football transfer fees dataset"; several are maintained through 2024/2025). Download and do a first-pass inspection of coverage and fields.
- **Day 2**: Check whether it already includes Neymar's 2013 and 2017 moves and Elliot Anderson's 2024 move — if any anchor case is missing, patch it in manually (a handful of rows, quick to add by hand rather than scraping).
- **Day 3**: Merge in basic performance stats from FBref for the seasons/players you'll actually model (goals, assists, minutes, appearances — skip xG, it's not in scope this round). You don't need this for every row in a 5-league dataset — a representative sample (e.g. top 2 divisions' worth of transfers per season, or all transfers above a fee floor) keeps this from becoming its own multi-week task.
- **Day 4–5**: Clean into one master table: player, age_at_transfer, position, fee, season, league, basic stats. Standardize currency to EUR, exclude loans/frees, flag estimated fees. Multi-league data means more duplicate-name and currency edge cases than the single-league version — budget real time here.
- **Day 6–7 (buffer)**: Fill gaps, resolve name-matching issues across leagues, sanity-check for outlier errors.

**Deliverable**: one clean CSV, ideally 2,000+ transfer rows across the top 5 leagues, 2010–2026, including Neymar's and Anderson's transfer seasons.

---

## Week 2 (Aug 15–21): Index + Model

- **Day 1–2**: Fit the hedonic GLM: `log(fee) ~ age + age² + position + goals + assists + appearances + year_FE`. Use `statsmodels`.
- **Day 3**: Extract year fixed effects → plot as your inflation index. This is your headline chart — get it right.
- **Day 4**: Sanity-check the index against known narrative spikes (2017 Neymar/Mbappé window, 2020 COVID dip).
- **Day 5**: Fit with ridge regularization instead of plain OLS if any position/age cells are thin — simple `sklearn.Ridge` on the log-fee target is enough; skip full Bayesian credibility.
- **Day 6–7 (buffer)**: Re-run and debug. This is the highest-risk phase — protect the buffer days.

**Deliverable**: fitted model + inflation index chart + written model summary (2-3 paragraphs).

---

## Week 3 (Aug 22–28): Era-Translation + Case Studies

- **Day 1–2**: Build the translation as a **general function**, not two one-off scripts: `predict_value(player_name, target_year)`. Internally it: (a) looks up the player's attributes/stats from your dataset (or accepts manual stat entry if the player isn't in it), (b) standardizes their stats against era peers, (c) runs both the simple CPI-deflation method and the full re-estimation method, (d) returns fee + interval for both. Building it generally from the start costs almost nothing extra over hardcoding two cases, and it's what makes the project actually usable rather than just a demo.
- **Day 3**: Call `predict_value("Neymar", 2026)` — compare both methods, sanity check against his actual 2017 PSG fee (€222m) as an anchor.
- **Day 4**: Call `predict_value("Elliot Anderson", 2014)` — same comparison.
- **Day 5**: Bootstrap a rough 90% interval inside the function (resample residuals, refit, or just use the GLM's built-in prediction interval) so every call returns an interval, not just the two anchor cases.
- **Day 6–7 (buffer)**: Test the function on 3–4 *extra* players you didn't originally plan for (pick ones already in your dataset) to confirm it genuinely generalizes, not just to your two hand-picked examples. Write up the anchor case study narrative.

**Deliverable**: a working `predict_value(name, year)` function that runs on any player in your dataset, demonstrated on Neymar and Anderson plus a few spot-checks.

---

## Week 4 (Aug 29–Sep 4): Validation + Write-Up

- **Day 1**: Single holdout test — hold out ~15% of transfers, check predicted vs. actual fee, report RMSE on log(fee). Skip tier-by-tier breakdown.
- **Day 2**: Write the README: problem statement, method (hedonic index + era-translation), assumptions/limitations (undisclosed fees, basic stats only, out-of-dataset players need manual stat entry, survivorship bias), and the two case studies as the hook.
- **Day 3**: Wrap `predict_value()` in a minimal interface — either a clean notebook cell with a name/year prompt, or (if ahead of schedule) a one-page Streamlit form: text input for player name, number input for target year, output = fee + interval + a small chart. This is the difference between "a model" and "a tool," and it's low-cost if the function from Week 3 is already general.
- **Day 4**: Clean up notebook/repo.
- **Day 5 (buffer/submit)**: Final polish, push to GitHub, done — with a day to spare before Sept 5.

**Deliverable**: GitHub repo, README, notebook (or small app) where any player in the dataset can be entered with a target year to get a valuation, demonstrated on the two case studies.

---

## What to explicitly cut if you fall behind

If week 2 or 3 overruns, cut in this order (least damaging to the core story first):
1. Performance-stat sample size — restrict basic stats merge to a subset (e.g. transfers above €5m) rather than all 5-league rows.
2. Ridge/credibility step — plain OLS is fine, just note the limitation.
3. Bootstrap intervals — report point estimates with a caveat instead.
4. **Only as a last resort**: drop to top 5 leagues' year fixed effects but restrict the *performance-model* fitting to PL-only, and note this as a scope limitation in the README rather than reworking the whole pipeline late. Since the dataset already covers all 5 leagues, you keep the leagues in the index even if the deeper model narrows.

**Never cut**: the two case studies, the inflation index chart, and the general `predict_value()` function — those are the entire point of the project and what makes it presentable *and* usable.

---

## Tech stack (unchanged, kept minimal)

- `pandas`, `statsmodels`, `sklearn` (Ridge only, skip PyMC)
- One Jupyter notebook, not a multi-file pipeline
- GitHub repo with README as the primary deliverable
