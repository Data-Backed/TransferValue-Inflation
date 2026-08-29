# Cleaning Log

- Loaded fresh from CSVs: (93483, 17)
- After restricting to season >= 2000: 78460 rows (dropped 15023)
- After excluding loans: 45693 rows (dropped 32767)
- Fee confidence breakdown: 20845 confirmed, 16061 likely free (fee=0), 8787 unknown (fee missing)
- Dropped 8787 rows with no usable fee from modeling set (saved separately)
- MODEL_FEES_ONLY=True: dropped 16061 free-transfer rows from modeling set (fee_confidence == 'likely_free')
- After deduplicating cross-listed transfers: 20832 rows (dropped 13)
- Position standardized. 9 rows still have only a coarse position label after fallback.
- After dropping implausible ages (<15 or >42): 20832 rows (dropped 0)
- Flagged 505 rows as outliers (top/bottom 1% within season-position group) - not dropped, just flagged for sensitivity checks later.
- 
Final cleaned dataset: (20832, 20)
- Columns: ['season', 'league', 'club', 'window', 'movement', 'player_name', 'player_id', 'age', 'nationality', 'position', 'pos', 'market_value', 'dealing_club', 'dealing_country', 'fee', 'is_loan', 'season_file', 'fee_confidence', 'position_clean', 'fee_outlier_flag']