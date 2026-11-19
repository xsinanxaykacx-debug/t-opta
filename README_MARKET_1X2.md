# 1X2 Market Favorite Bot

## 1. Build the historical market report

```bash
python market_1x2_analysis.py xgo.csv --out results_market_1x2
```

This reads one complete H/D/A odds triplet per match (Avg first, then BbAv, then B365). Each raw implied probability is `1 / decimal odds`; probabilities are not normalized and bookmaker margin is not removed. Final goals are used only to label the actual result.

The report is chronological: 70% discovery, 15% validation, final 15% sealed and unscored. It writes:
- `market_1x2_probability_bands.csv`: 60.00–70.99%, 71.00–80.99%, 81.00–90.99%, 91.00–100.00%.
- `market_1x2_favorite_types.csv`: MS1/MSX/MS2 accuracy within each band.
- `market_1x2_segments.csv`: league, season and league-season splits.
- `market_1x2_outcome_calibration.csv` and `market_1x2_calibration_summary.csv`.
- `market_1x2_coupon_simulations.csv`: up to 50 historical validation group simulations. Only unique fixture combinations are evaluated; if fewer than 50 distinct same-day groups exist, all available groups are reported and the shortfall is explicit. Duplicate resampling is prohibited. Historical coupons use the same probability ranking and interleaved 3-leg distribution as the upcoming picker.
- `market_1x2_summary.json`: selected candidate bands and aggregate results.

Candidate selections are made independently for every probability band and favorite type (MS1, MSX, MS2), using discovery only: each band/type pair must have at least 50 matches, at least 75% favorite hit rate, and a Wilson 95% confidence-interval lower bound of at least 75%. Only the qualifying band/type pairs are evaluated on validation and eligible for upcoming picks. A strong MS1 segment cannot make a weak MSX or MS2 segment eligible. The final 15% holdout is not used.

## 2. Prepare upcoming fixtures

Create a CSV with one row per upcoming match and at least one complete odds triplet. Accepted column triplets include:
- `AvgH, AvgD, AvgA`
- `BbAvH, BbAvD, BbAvA`
- `B365H, B365D, B365A`

Optional fixture columns such as `Date`, `HomeTeam`, and `AwayTeam` are preserved. Do not add post-match fields; they are not needed.

## 3. Generate today's picks and coupons

```bash
python market_1x2_predict.py upcoming.csv --model-summary results_market_1x2/market_1x2_summary.json --out results_upcoming
```

The picker:
- Calculates the market favorite from raw 1/odds only.
- Keeps only bands selected by the historical discovery gate.
- Requires ten eligible fixtures from the same calendar date; fixtures from different dates are never mixed in one coupon group.
- If multiple dates qualify, chooses the date whose ten strongest favorites have the highest mean raw implied probability.
- Distributes the top nine from that date across three 3-leg coupons; the tenth pick is held aside.
- If no date has ten eligible fixtures, it reports that no same-day coupon group is available and does not fabricate one.
- Calculates each coupon's combined decimal odds and potential return at 100 TL per coupon.

Outputs are `all_upcoming_market_favorites.csv`, `upcoming_market_picks.csv`, `upcoming_coupons.csv`, `upcoming_coupon_summary.csv`, and `upcoming_prediction_summary.json`.

## Limitations

The historical archive has no odds-capture timestamps, so it cannot prove that historical odds were captured before kickoff. Upcoming inputs must therefore be supplied from a pre-kickoff source. The market favorite is a descriptive market selection, not a guarantee; band hit rates and coupon outcomes can change across seasons and samples.
