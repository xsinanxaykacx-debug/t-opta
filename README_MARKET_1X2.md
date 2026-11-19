# 1X2 Market Favorite Bot

## Historical market analysis

Run:

```bash
python market_1x2_analysis.py xgo.csv --out results_market_1x2
```

The historical analysis reads one complete H/D/A odds triplet per match (Avg first, then BbAv, then B365). Each raw implied probability is `1 / decimal odds`; probabilities are not normalized and bookmaker margin is not removed. Final goals are used only to label the actual result.

The report is chronological: 70% discovery, 15% validation, final 15% sealed and unscored. It writes:
- `market_1x2_probability_bands.csv`: 60.00–70.99%, 71.00–80.99%, 81.00–90.99%, 91.00–100.00%.
- `market_1x2_favorite_types.csv`: MS1/MSX/MS2 accuracy within each band.
- `market_1x2_segments.csv`: league, season and league-season splits.
- `market_1x2_outcome_calibration.csv` and `market_1x2_calibration_summary.csv`.
- `market_1x2_coupon_simulations.csv`: validation group simulations using unique same-day fixture combinations only.
- `market_1x2_summary.json`: selected candidate bands and aggregate results.

Candidate selections are made independently for every probability band and favorite type (MS1, MSX, MS2), using discovery only: each band/type pair must have at least 50 matches, at least 75% favorite hit rate, and a Wilson 95% confidence-interval lower bound of at least 75%. Only qualifying band/type pairs are evaluated on validation and eligible for upcoming picks. A strong MS1 segment cannot make a weak MSX or MS2 segment eligible. The final 15% holdout is not used.

## Automatic upcoming fixtures and odds from Mackolik

The bot uses **Mackolik as its sole live fixture and odds source**. No separate odds provider, API key, or manual upcoming CSV is used.

Run the complete bot on Windows:

```bat
run_bot.bat
```

The launcher:
1. Prepares the Python environment and installs requirements.
2. Runs the historical analysis once, then reuses its summary on later launches.
3. Reads upcoming football fixtures from Mackolik's public football livescores feed (today plus the next six calendar days).
4. Opens each listed Mackolik match's İddaa page and reads the first/current **Maç Sonucu** 1-X-2 prices.
5. Writes the captured data to `mackolik_upcoming.csv`.
6. Sends those rows directly to the existing odds-only prediction engine.
7. Opens `results_upcoming` with the individual picks, all market favorites and same-day coupon outputs.

The source fields saved with each match include the Mackolik match ID, page URL, local kickoff time and odds-capture timestamp. Matches are included only when they are upcoming, listed as pre-match, and have a complete 1/X/2 price row. Matches without a published 1X2 market are skipped rather than filled with guessed prices.

You can run the provider by itself:

```bash
python mackolik_provider.py --days 7 --out mackolik_upcoming.csv
```

The provider uses Mackolik's currently exposed web feed and match-page markup, not a documented/contracted public API. Mackolik can change these endpoints or page structures; if that happens, the bot reports the retrieval/parsing failure instead of silently switching to another source.

## Prediction and coupon rules

The picker:
- Calculates the market favorite from raw 1/odds only.
- Keeps only bands selected by the historical discovery gate.
- Requires ten eligible fixtures from one calendar date for the three 3-leg coupons.
- Never combines fixtures from different dates in a coupon group.
- If no date has ten eligible fixtures, exports the available individual picks and reports that no same-day coupon group is available.
- Calculates combined decimal odds and potential return at 100 TL per coupon.

Outputs include `mackolik_upcoming.csv`, `results_upcoming/all_upcoming_market_favorites.csv`, `upcoming_market_picks.csv`, `upcoming_coupons.csv`, `upcoming_coupon_summary.csv` and `upcoming_prediction_summary.json`.

## Important limitations

Historical validation was performed on the historical archive's odds source, not on Mackolik odds. The market source and price construction differ, so the historical hit rates are **not proof** that the same hit rates will hold for Mackolik. Mackolik odds are now used for live picks; they must be tracked prospectively and evaluated separately before claiming a 7/10 or 8/10 success rate. The capture timestamp is saved so pre-kickoff use can be audited. No prediction or coupon is guaranteed.
