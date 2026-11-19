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

## Automatic upcoming fixtures, odds and web dashboard from Mackolik

The bot uses **Maçkolik as its only live fixture and odds source**. There is no separate provider, API key, or manual `upcoming.csv` entry.

Run:

```bat
run_bot.bat
```

The launcher prepares the Python environment, performs the historical analysis once, starts the local T-OPTA Tahminkâr website, and opens `http://127.0.0.1:8765`.

The website:
1. Reads today's football fixtures directly from Maçkolik's public livescores JSON feed.
2. Resolves competition names from Maçkolik's competition catalogue.
3. Opens each pre-match fixture's Maçkolik İddaa detail page and reads the first/current **Maç Sonucu** 1-X-2 price row.
4. Calculates raw implied probabilities as `1 / odds`; it does not normalize probabilities or remove the margin.
5. Applies the existing discovery-selected band + favorite-type gates and displays every fetched match, including those outside the selection criteria.
6. If fewer than ten qualifying matches exist, it displays and exports all qualifying picks and explicitly reports the count (for example, "10 uygun maç yok; 6 uygun seçim bulundu"). It does not invent filler matches.
7. When ten qualifying fixtures exist on the same calendar day, it creates the existing three 3-leg coupon groups.

The dashboard refresh button repeats the Maçkolik fetch. The captured source URL and odds timestamp are included in the generated CSV output. The bot writes the detailed files under `results_upcoming`, including `upcoming_market_picks.csv`, `all_upcoming_market_favorites.csv`, `upcoming_coupons.csv`, `upcoming_coupon_summary.csv`, and `upcoming_prediction_summary.json`.

There is no fallback fixture list, fake odds, fabricated expected score, or browser-side substitute calculation. If Maçkolik is unavailable or a page has no complete 1-X-2 market, the dashboard reports the error or skips that match; it never fabricates a prediction. The odds-only model does not predict exact scores.

The provider uses Maçkolik's currently exposed public web feed and match-page markup, not a documented service contract. Maçkolik can change those structures; failures are surfaced instead of silently switching providers.

## Important limitations

Historical validation used the odds source stored in the historical archive, not Maçkolik. Therefore historical hit rates are not proof that the same rates transfer to Maçkolik. Track the Maçkolik picks prospectively before making any success-rate claim. No prediction or coupon is guaranteed.
