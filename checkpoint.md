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


## 1X2 calibration uncertainty refinement — 2026-11-19
- Latest branch head `b668d5dca76ee1051292402f1945afba273851e2` passed all three GitHub Actions workflows: `tests`, `Python tests`, and `Odds-only model validation`; both the complete test job and real-data market-analysis job completed successfully.
- Inspected the workflow artifact `14511972223`. The 1X2 validation partition contains 2,112 matches; discovery contains 9,933. The sealed 2,124-match holdout remains unscored.
- Validation reliability observations: Draw probabilities in the 20–30% band: 1,679 matches, predicted mean 26.09%, observed 24.96%. Away probabilities in 60–70%: 67 matches, predicted 64.29%, observed 79.10%; 70–80%: 30 matches, predicted 74.39%, observed 90.00%. These higher-away bins are small and must not be treated as a proven edge.
- Added Wilson 95% confidence intervals for observed outcome frequencies in each probability bin and a sample-weighted calibration summary per outcome/partition. Summary includes weighted absolute/signed calibration error and a maximum absolute bin error restricted to bins with at least 30 matches, to reduce small-bin overinterpretation.
- Added tests for confidence-interval containment, summary coverage, and match accounting. Awaiting CI on these new commits; no new results are claimed until CI and the regenerated artifact succeed.
- The odds timestamp limitation remains: pre-kickoff capture cannot be independently verified. No ROI/edge claim; no holdout scoring; PR #5 remains draft/open; main and legacy `BOT.py` untouched.


## Favorite probability-band uncertainty — 2026-11-19
- Latest CI on commit `5dc5f6dce6e6b7b7b6f1fbdf9adcbf217c454d0d` passed all three workflows (six jobs total); the full test suite and both real-data analysis steps completed successfully.
- Inspected artifact `14516180942` from run `44497751703`. 1X2 validation: 2,112 matches, favorite hit rate 52.75%. Favorite probability bands: 50–60% n=484, 56.41%; 60–70% n=247, 72.06%; 70–80% n=131, 78.63%; 80–100% n=41, 87.80%. These are descriptive validation results, not evidence of positive betting ROI.
- The Over/Under directional model's 1.60–2.50 validation slice had 1,419 bets, 53.77% hit rate, and -7.05% ROI. EV-filtered selections produced no bets. This candidate is not profitable on the current validation sample.
- Added Wilson 95% confidence intervals to each favorite-probability band, plus the observed-minus-mean-market-confidence gap and its interval, and favorite win counts. Added tests checking interval containment and gap arithmetic.
- The new report makes uncertainty explicit; it does not use or score the sealed 2,124-match holdout. Odds capture timestamps remain unavailable, so pre-kickoff availability is not independently verified.
- PR #5 remains draft/open; do not merge, do not modify main or legacy `BOT.py`.

## Raw 1X2 favorite bands and coupon validation — 2026-11-19
- Replaced the previous de-vigged 1X2 probabilities with the user's requested raw implied probabilities: each outcome is exactly `1 / decimal_odds`; no normalization and no margin removal is applied. The three raw percentages may sum above 100%.
- Complete triplet source priority remains Avg -> BbAv -> B365, row-by-row, without mixing bookmakers.
- Favorite probability bands are now exactly 60.00–70.99%, 71.00–80.99%, 81.00–90.99%, and 91.00–100.00%. Reports include sample count, wins, hit rate, Wilson 95% interval, and favorite type (MS1/MSX/MS2) split within each band.
- Discovery and validation remain chronological (whole UTC dates); the final 15% / 2,124-match holdout is still sealed and unscored.
- The first real-data run of the raw-probability implementation completed its market-analysis job and produced 14,169 complete Avg H/D/A triplets: discovery 9,933 and validation 2,112. Validation overall favorite hit rate was 52.75% (not a 65%+ overall system).
- Preliminary raw-band validation results from that run: 60.00–70.99%: 202/310 = 65.16%; 71.00–80.99%: 122/159 = 76.73%; 81.00–90.99%: 64/71 = 90.14%; 91.00–100.00%: 7/9 = 77.78%. The final band has only 9 matches and is not reliable evidence; its Wilson interval is wide. The strongest candidate bands were selected from discovery for the coupon evaluation: 71.00–90.99%.
- Added 50 reproducible 10-match validation simulations. Each group is split into three 3-leg coupons (one match unused), 100 TL stake per coupon, actual historical odds of each selected favorite used for payout. Current code restricts the candidate pool to the discovery-selected 71.00–90.99% bands; the updated artifact for this stricter policy is still pending.
- Added output `market_1x2_favorite_types.csv` and `market_1x2_coupon_simulations.csv`; `market_1x2_probability_bands.csv` now contains the four requested raw bands.
- The earlier broad >=60% exploratory simulation (not the final candidate policy) yielded 7.22/10 mean hits, 37/50 groups with >=7 hits, 24/50 with >=8, and at least one coupon hit in 39/50; this is not to be used as the final coupon result because the selection policy was subsequently tightened to 71.00–90.99%.
- Latest code commit at checkpoint: `779554734579e7b37c2ff127bbb7a25bcd48e449`. GitHub Actions for the latest head are queued; complete pytest and the stricter-band coupon artifact are not yet confirmed green.
- Odds capture timestamps are absent; pre-kickoff capture cannot be independently verified. No holdout scoring, no claim of guaranteed 7–8/10, no PR merge, and no changes to main or legacy `BOT.py`.

## Upcoming 1X2 picker and final validation state — 2026-11-19
- Added `market_1x2_predict.py`: reads upcoming fixtures from a CSV, requires a complete H/D/A odds triplet, calculates raw `1 / odds` probabilities, identifies MS1/MSX/MS2, and marks fixtures only when their band passed the historical discovery gate. No FTHG/FTAG or other post-match data is required for upcoming predictions.
- The picker ranks eligible upcoming fixtures by raw favorite probability, selects up to 10, distributes the top nine across three 3-leg coupons (ranks 1/4/7, 2/5/8, 3/6/9), leaves rank 10 aside, and reports combined odds / potential return for 100 TL per coupon. It refuses to create incomplete coupons if fewer than 10 eligible fixtures exist.
- Added `tests/test_market_1x2_predict.py` and `README_MARKET_1X2.md` with the input format and command examples.
- Candidate selection is now dynamic and discovery-only: a band must have at least 50 discovery matches and at least 75% favorite hit rate. The selected bands are then evaluated on the separate validation partition. For the current data, preliminary discovery metrics select 71.00–80.99% and 81.00–90.99%; the 91%+ band is too small in discovery.
- Full pytest passed on earlier implementation commit `ff700fbf410e5cde04d2fab509aee974a2d42a51`: 79 passed in 270.30 seconds. That run preceded the dynamic band gate and the new upcoming-picker tests, so it is not the final-head test result.
- The latest code/workflow head is `963a5a858f1e48bcabdc8cdce3e567ad393dc9e7`. Latest-head GitHub Actions are queued; the final dynamic-band coupon artifact and full tests including the new upcoming-picker tests are not yet confirmed.
- PR #5 remains open/draft and unmerged. Main and legacy `BOT.py` remain untouched. Final 15% holdout remains sealed. Historical odds timestamps remain unavailable, so historical pre-kickoff capture is not independently verifiable.
