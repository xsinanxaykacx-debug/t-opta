# T-OPTA Checkpoint — 2026-11-18

## Current state
- Repository: https://github.com/xsinanxaykacx-debug/t-opta
- Main branch includes UTC-day leakage fixes, reliability diagnostics, SciPy dependency, selective-accuracy reporting, and a validation gate that keeps the final holdout sealed.
- Main BOT.py blob: `8402945bd4692be8d134bed0d0a0460e58a3e47b`
- Latest merged implementation commit: `3e4c64034cfed668f0c99d5425658e70ae9b779a`

## Data and split
- Local source dataset used for validation exploration: `xg18_with_all_league_odds.csv`
- Rows after required-field filtering: 15,734
- Raw row-based split (8,500 / 1,500 / 5,734) was unsafe because both boundaries cut through a UTC day.
- Corrected whole-day split on this dataset:
  - Train: 8,495
  - Validation: 1,493
  - Final holdout: 5,746
- Validation subdivisions:
  - Calibration: 892
  - Policy Select: 288
  - Policy Check: 313
- Final holdout date range: 2025-01-11 through 2026-03-14.
- The final holdout has NOT been scored.

## Methodology fixes
- Feature generation batches all fixtures by UTC calendar day before adding that day's completed xG to team/league history.
- Current-match Home_xG/Away_xG are not direct model inputs; only prior-date xG contributes to pre-match Poisson lambdas.
- FTHG/FTAG are target labels only.
- Train/validation/test and calibration/policy-select/policy-check splits are aligned to complete UTC days.
- Market probabilities are de-vigged from a complete Over/Under pair, preferring Avg and falling back to B365 as a complete pair.
- Odds capture timestamps are absent; pre-kickoff availability cannot be independently proven from this CSV.
- Reliability bins report market probability, model probability, empirical frequency, market/model bias, and model-minus-market edge.
- Poisson expected total goals are exported as `lambda_total` / `expected_total_goals`.

## Validation-only findings (not final-test results)
- Hybrid model on Policy Check, same 261 odds-valid rows:
  - Accuracy: 55.17%
  - Log loss: 0.6870
  - Brier: 0.2472
- De-vig market baseline on the same 261 rows:
  - Accuracy: 57.09%
  - Log loss: 0.6727
  - Brier: 0.2402
- Policy Select chose edge threshold 0.05 (86 bets, 59.30% hit rate, -5.35% ROI).
- Policy Check at that threshold: 91 bets, 50.55% hit rate, -19.80% ROI.
- Fixed 0.08 edge on Policy Check: 53 bets, 56.60% hit rate, -11.75% ROI.
- Selective confidence pmax >= 0.65 on Policy Check: 89 picks, 59.55% accuracy.
- No current strategy meets the 65% hit-rate and positive-ROI validation requirements.
- Current conclusion: NO BET; continue model development using Train/Validation only.

## Additional train-only model experiments (2026-11-18)
- Internal chronological OOS was carved entirely from Train (not Validation or Final Test): fit 5,936; calibration 748; policy-select 272; policy-check 261; internal OOS 1,278 (1,086 odds-valid).
- De-vig market raw baseline on internal OOS: 61.05% accuracy, log loss 0.6628, Brier 0.2352, AUC 0.6140.
- Candidate calibrated models on the same 1,086 odds-valid rows:
  - Logistic market+Poisson: 57.46% accuracy, log loss 0.6699, Brier 0.2386.
  - Extended logistic: 56.63%, log loss 0.6733, Brier 0.2398.
  - HistGradientBoosting: 60.04%, log loss 0.6693, Brier 0.2380.
  - LightGBM: 58.75%, log loss 0.6716, Brier 0.2393.
  - XGBoost: 59.85%, log loss 0.6835, Brier 0.2407.
  - CatBoost: 60.04%, log loss 0.6758, Brier 0.2394.
- None beat the raw market baseline on the same rows. Edge thresholds with at least 50 bets were negative ROI for all tested candidates; small positive samples were rejected as too small.
- Market-only confidence strategy was also inconsistent: internal OOS at pmax >= 0.625 had 243 picks, 71.60% hit and +0.32% ROI; Policy Check at >= 0.60 had 69 picks, 65.22% hit but -5.83% ROI. At >= 0.625 Policy Check had only 42 picks and -8.07% ROI.
- These experiments used Train-only internal splits plus the already-used validation segments. They do not access or score Final Test.

## Safety gate
- Final holdout remains sealed unless Policy Check has at least 50 bets, hit rate >= 65%, and ROI > 0%.
- Do not tune thresholds or features using final holdout outcomes.
- If validation does not pass, do not run or inspect the final holdout.

## Code / tests
- `requirements.txt` now includes `scipy>=1.13,<2.0`.
- `tests/test_bot.py` covers same-day leakage, current-match xG exclusion, Poisson expected goals, UTC-day splits, complete-pair de-vigging, reliability bins, and policy gate behavior.
- GitHub Actions workflow added at `.github/workflows/ci.yml`; the connected status API has not shown a Python Actions run. The only reported status is the unrelated Vercel deployment rate-limit failure.
- Equivalent Python logic checks run in this session passed; the exact repository pytest suite has not yet been confirmed green.

## Fresh-data source check
- Football-Data.co.uk lists 2026/27 CSV results/odds and was updated 2026-11-16; it also distinguishes earlier market odds from closing odds. Source: https://football-data.co.uk/downloadm.php
- Its terms describe the data as for private individuals and exclude commercial/data-training products using automated bots/scrapers/AI. Do not ingest it into this automated model until permitted use is clarified. Source: https://football-data.co.uk/data
- Understat currently covers xG for six main leagues, not a like-for-like replacement for the current 18-league xG dataset. Source: https://understat.com/

## Next steps
1. Resolve why GitHub Actions is not reporting a run and obtain a real `pytest -q` result.
3. Improve the model on Train/Validation only; current hybrid does not beat market baseline.
4. Establish a fresh Policy Select / Policy Check protocol on new data.
5. Open a new final holdout only after the new validation gate passes.

## Odds-only high-odds model — 2026-11-18
- Created isolated module `odds_only_bot.py`; the existing hybrid `BOT.py` was not overwritten.
- Predictor inputs are exclusively derived from a complete pre-match Over/Under 2.5 odds pair: de-vigged Over probability, bookmaker margin, log Over odds, and log Under odds.
- No xG, goals history, shots, possession, corners, squad value, team names, or match-result fields enter the predictor matrix. `FTHG` and `FTAG` are used only to construct the Over 2.5 target label and score predictions.
- Odds source selection: complete Avg pair first; if unavailable, complete B365 pair. Never mix one side from Avg with the other from B365.
- Evaluation uses a chronological split on complete UTC dates. The last 15% is explicitly sealed and not evaluated. The first 70% is used for fitting and the following 15% for development validation.
- High-odds directional results are reported for the model-selected side only when its odds are between 1.60 and 2.50. Results are also broken into 1.60–1.80, 1.80–2.00, 2.00–2.25, and 2.25–2.50 bands.
- ROI is calculated per individual bet (win: odds minus one; loss: minus one unit), not by multiplying hit rate by average odds.
- Fixed expected-return filters of 0%, 2%, and 5% are reported separately; they are not tuned on the sealed holdout.
- Added `tests/test_odds_only_bot.py` for odds-only feature isolation, complete-pair fallback, whole-day chronological splits, exact per-bet ROI, and 1.60–2.50 eligibility.
- Critical limitation: the source CSV has no odds capture timestamps. Avg/B365 fields cannot yet be independently proven to have been captured before kickoff. The script prints this warning and must not be described as verified pre-match until the data provenance is confirmed.
- Execution status: source and tests committed to branch; full repository pytest / real-data validation pending CI and dataset availability. No accuracy or profitability claim is made yet.
- Branch: `feature/odds-only-high-odds-model`.

## League / season diagnostics — 2026-11-18
- Added validation-only descriptive breakdowns by league, season, and league-season in `validation_league_season_diagnostics.csv`.
- Football season boundaries use July 1: dates from July onward are assigned to the season starting in that calendar year; earlier dates are assigned to the season starting in the prior year.
- Each segment reports model directional accuracy, market directional accuracy, number of selected-side bets in 1.60–2.50, wins, hit rate, average odds, average per-bet break-even rate, and exact per-bet ROI.
- Segments with fewer than 50 eligible bets are explicitly flagged as insufficient sample; no league/season is selected or tuned from these descriptive results.
- Added a unit test for season boundaries, league-season grouping, small-sample flagging, and exact ROI.
- CI was observed installing dependencies and entering pytest; the real-data validation/artifacts are not yet confirmed. New commits will trigger a fresh run.
- This remains development validation only. Final holdout remains sealed, and no performance claim is made.

- Extended each segment with a like-for-like market-direction baseline: selected-side bet count, hit rate, average odds, break-even hit rate, and ROI. This allows direct model-versus-market comparison within the same league/season group rather than reporting model results alone.


## 1X2 bookmaker-market diagnostics — 2026-11-19
- Added isolated `market_1x2_analysis.py` and tests in `tests/test_market_1x2_analysis.py`.
- The analysis uses a complete H/D/A odds triplet from one source only; source preference is Avg, then BbAv, then B365. It never mixes home/draw/away prices across bookmakers.
- Raw implied probabilities are de-vigged by dividing each inverse price by the sum of all three inverse prices.
- The market favorite is the highest de-vigged probability. Reports separate favorite hit rate, predicted Home/Draw/Away accuracy, actual outcome counts, multiclass Brier score, and confidence bands (50–60%, 60–70%, 70–80%, 80%+).
- Reports are split chronologically by complete UTC dates: first 70% discovery, next 15% validation, last 15% sealed holdout. The script records holdout row count/date range only and explicitly does not calculate holdout metrics.
- League, season (July boundary), and league-season descriptive tables are emitted separately for discovery and validation.
- FTHG/FTAG are used only to construct the final H/D/A outcome label. No target-match xG, shots, possession, corners, team stats, or model outputs enter market probabilities.
- The odds archive has no capture timestamps; the script explicitly reports that pre-kickoff availability cannot be independently verified. No claim of verified pre-match odds is allowed until provenance/timestamps are confirmed.
- CI workflow now runs the 1X2 diagnostics and uploads both odds-only and 1X2 reports. Awaiting the new run on the updated branch before claiming tests or dataset analysis passed.
- PR #5 remains draft/open; do not merge or modify main/BOT.py.


## 1X2 analysis implementation refinement — 2026-11-19
- 1X2 odds selection now falls back row-by-row in source priority order Avg -> BbAv -> B365, but each row always uses a complete H/D/A triplet from exactly one source. No cross-bookmaker mixing.
- CSV is read with the same latin-1 encoding used by the existing odds-only pipeline.
- Class diagnostics now include per-outcome predicted precision, actual-outcome recall, and F1 for Home, Draw, and Away; segment output includes league, season, and league-season slices for both discovery and validation.
- The workflow was split into independent full-test and market-analysis jobs so a slow full test suite does not prevent the market report job from starting.
- Latest implementation branch head before this checkpoint update: `7c4cc3cadae9bf14b1105e16684912e27840057a`. PR #5 remains draft/open and targets main; main and `BOT.py` remain untouched.
- CI status at checkpoint time: the earlier combined workflow run `44462383373` was still in progress at the full pytest step; the latest split workflow jobs were queued. No 1X2 data report or test pass is claimed until the run on the final head completes.

## 1X2 outcome probability calibration — 2026-11-19
- Added `market_1x2_outcome_calibration.csv` to report the calibration of Home, Draw, and Away probabilities separately, rather than evaluating only the most-likely market favorite.
- For each outcome, discovery and validation now report 10 probability bands (0–10% through 90–100%), sample count, mean predicted probability, observed outcome frequency, calibration error, and one-vs-rest Brier score.
- The analysis remains strictly chronological; the final 15% holdout remains sealed and is not included in calibration.
- Added a unit test checking all three outcomes and both development partitions are represented and that every match contributes exactly one bin per outcome.
- Workflow artifact upload already includes the entire `results_market_1x2/` directory, so the new calibration CSV will be included automatically.
- Implementation commits: `53100c3171488a561dddbecde4434f5af8728847` and `b54de289e8cb1356f72d65353f098fd8958de4fa`. Awaiting CI on the latest branch head; no calibration results are claimed until the new artifact is produced.
