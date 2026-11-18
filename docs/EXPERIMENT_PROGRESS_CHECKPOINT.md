# T-OPTA Experiment Progress Checkpoint

Updated: 2026-11-18
Active branch: `diagnostics/over-under-market-analysis`
Repository: `xsinanxaykacx-debug/t-opta`

## Goal and acceptance criteria

- User goal: a practical football Over/Under 2.5 prediction bot that correctly predicts at least 6–7 of 10 matches.
- Do not declare completion until independently evaluated results support the target.
- Accuracy alone is insufficient; ROI, calibration, market comparison, sample size and temporal validity must also be reported.
- Never use closing odds or any feature unavailable at prediction time.
- Keep the Final Test sealed. Do not tune features, thresholds, calibration, model selection or policies against it.
- Do not merge this branch to `main` until tests, diagnostics, artifacts, leakage checks and explicit merge conditions are satisfied.

## Previously recorded baseline (user checkpoint)

- Final Test accuracy: 66%.
- Final Test ROI: -5.90%.
- This is a historical checkpoint result supplied by the user, not a new evaluation in this session.
- The Final Test remains sealed and must not be re-opened for model or threshold tuning.

## Current pipeline

- `BOT.py`: chronological Over/Under 2.5 pipeline.
- `diagnostics.py`: independent diagnostics for directional ROI, market/model Brier and log loss, odds buckets, edge buckets and reliability bins.
- Data split in current run:
  - TRAIN: 8,500 matches (2022-07-15 to 2024-09-22)
  - CALIBRATION: 900 matches
  - POLICY SELECT: 300 matches (2024-11-30 to 2024-12-15)
  - POLICY CHECK: 300 matches (2024-12-15 to 2025-01-11)
  - Final Test begins after the policy-check period and is excluded from policy-only execution.
- Policy-only mode truncates the dataset to TRAIN + validation rows before feature generation. It does not create, summarize, score or evaluate a Final Test dataframe.
- Command: `python BOT.py xgo.csv --policy-only`.

## Current POLICY SELECT / POLICY CHECK findings

Latest completed diagnostics run: GitHub Actions run `44339665876`, workflow `diagnostics-report` run #7, commit `b3194f1782933c78850638c6a9ed5fe1ca22b089`.
Status: SUCCESS. Five diagnostic CSVs uploaded as artifact `t-opta-diagnostics-44339665876`.

POLICY SELECT:
- 300 matches; 241 with valid market odds.
- Edge threshold 0.03: 153 bets, 57.5% hit rate, -6.33% ROI.
- Edge threshold 0.05: 100 bets, 55.0% hit rate, -11.71% ROI.
- Edge threshold 0.08: 48 bets, 75.0% hit rate, +16.10% ROI.
- Edge threshold 0.10: 18 bets, 77.8% hit rate, +17.94% ROI.
- Current selector requires at least 50 bets, so it selected 0.03. The higher-threshold rows are exploratory POLICY SELECT observations; do not relax the minimum or adopt a threshold without a pre-specified, independently untouched future validation.

POLICY CHECK (selected threshold 0.03):
- 300 matches; accuracy 54.33%; AUC 0.5637; ECE 0.0804.
- 253 matches had valid market odds.
- 169 bets; 53.8% hit rate; average odds 1.68; ROI -11.69%; net -19.76 units.
- Model Brier 0.249078 vs market Brier 0.241467; delta +0.007611 (market better).
- Model log loss 0.690823 vs market 0.675279 (market better).
- OVER: 51 bets, 70.59% hit rate, +4.43% ROI.
- UNDER: 118 bets, 46.61% hit rate, -18.66% ROI.
- Main diagnostic finding: UNDER policy is materially damaging results; the blended model also trails the market on proper scoring rules in this policy-check sample.
- POLICY CHECK has now been observed. Do not reuse it to optimize the next model or threshold. Use fresh walk-forward folds within pre-Final-Test history for development; preserve Final Test untouched.

## CI / Actions history

- Earlier diagnostics run #3 (`44338463852`, commit `d8c25f41...`) failed in pytest before the model-evaluation step. Failure: odds-bucket test incorrectly expected only one OVER row in the open-ended 2.10+ bucket, although both 2.10 and 2.12 belong there. Result: 57 passed, 1 failed. The full model command was skipped, so this run did not evaluate Final Test.
- Test expectation fixed in commit `73aab450e6ba0ff670bb18fc7f24f1caabd0322c`: 2.10+ bucket expects 2 bets and average odds 2.11.
- Diagnostics workflow changed to avoid duplicate pytest (the separate tests workflow owns pytest) and to serialize/cancel superseded diagnostics runs. Commit `b3194f1782933c78850638c6a9ed5fe1ca22b089`.
- Tests workflow changed to serialize/cancel superseded runs on the same branch. Commit `c04432ed955dfa257a537e2fe4197b0bb363ece3`.
- Latest tests workflow run #170: `44339665681`, commit `b3194f17...`; status was IN_PROGRESS at checkpoint creation. Confirm final conclusion before proceeding.
- Earlier duplicate Actions runs from intermediate commits may still appear in history. The latest diagnostics workflow run #7 is SUCCESS and used policy-only mode.

## Recent code commits

- `d8c25f410d344725847399bee7839f59c59b31c8`: add policy-only mode.
- `98dc6bfd92937efabe405bd0c849f3bca89a5cf9`: update diagnostics workflow to policy-only command.
- `0a5f4ded48783f66a6b45ee3ccf8150d37513bca`: exclude Final Test rows before feature generation in policy-only mode.
- `73aab450e6ba0ff670bb18fc7f24f1caabd0322c`: correct odds bucket boundary test.
- `b3194f1782933c78850638c6a9ed5fe1ca22b089`: serialize diagnostics workflow and remove duplicate pytest.
- `c04432ed955dfa257a537e2fe4197b0bb363ece3`: serialize tests workflow.

## Branch / merge status

- Latest verified comparison at checkpoint creation: branch is 13 commits ahead of `main`, 0 behind.
- The earlier chat checkpoint stated 109 ahead; the GitHub compare API currently reports 13. Reconcile if needed, but use GitHub's current compare as the source of truth.
- No merge to main has been performed.
- Do not merge until the user-defined acceptance conditions are met and the Final Test remains untouched.

## Next technical work

1. Confirm latest pytest run #170 conclusion.
2. Inspect the five diagnostic CSVs (artifact already generated) and preserve them with the run.
3. Develop a leakage-safe walk-forward comparison on pre-Final-Test data: market-only baseline, Poisson-only baseline, current blend, and side-specific policies. Use separate temporal folds; do not tune on the already-observed POLICY CHECK.
4. Diagnose why UNDER bets are weak and why the blend loses to the market on Brier/log loss. Do not simply suppress UNDER based on the observed POLICY CHECK; define candidate rules on training/selection folds and verify on fresh forward folds.
5. Improve calibration and probability estimation only with chronological training/calibration partitions. Keep bookmaker odds strictly as-of prediction-time features; never use closing odds if unavailable at prediction time.
6. Require adequate sample sizes and confidence intervals for accuracy/ROI. Avoid claiming a 65–70% result from tiny samples.
7. Continue to keep Final Test sealed. Only evaluate it once when the complete model and policy are frozen, and only for final reporting.

## Model artifacts

- Current run #7 generated five CSV diagnostics:
  - `policy_check_directional_roi.csv`
  - `policy_check_market_model_brier.csv`
  - `policy_check_odds_buckets.csv`
  - `policy_check_edge_buckets.csv`
  - `policy_check_reliability_bins.csv`
- Earlier user checkpoint states model artifact metadata and GitHub Actions run numbers were saved. The referenced checkpoint markdown was not present in the branch's `docs/` directory when inspected; this file now restores the current verified state.


## Latest walk-forward confidence findings (2026-11-18)

Latest research workflow: run #13 `44340587905`, commit `9e855f810823be080f6aceb0bbb95cbb74830042`.
Status: SUCCESS. Five policy-check CSVs and three walk-forward CSVs were generated and uploaded. The run log explicitly reports `POLICY-ONLY MODE: FINAL TEST SEALED — NOT ACCESSED`.

Same-calendar-day isolation was added in commit `4555482a226214fc7fda2301f3516885a80a4fde`. The date grouping now normalizes UTC dates before grouping, so different kickoff timestamps on the same calendar day cannot update each other's historical team/league xG.

Updated 10-fold pooled OOF metrics after the same-day isolation fix (2,674 valid-market matches):
- BLEND: accuracy 58.26%, Brier 0.243105, log loss 0.694352, AUC 0.593320.
- MARKET: accuracy 59.16%, Brier 0.238624, log loss 0.669980, AUC 0.613184.
- POISSON: accuracy 57.63%, Brier 0.244329, log loss 0.682144, AUC 0.588489.
- Market baseline still outperforms the blend on accuracy, Brier, log loss and AUC.

Selective accuracy (OOF, only valid-market rows):
- BLEND confidence >=0.65: 556 matches, 20.79% coverage, 67.81% accuracy. Approximate Wilson 95% interval: 63.81%–71.56%.
- BLEND confidence >=0.70: 442 matches, 16.53% coverage, 68.33% accuracy.
- BLEND confidence >=0.75: 188 matches, 7.03% coverage, 71.28% accuracy.
- MARKET confidence >=0.65: 289 matches, 10.81% coverage, 71.63% accuracy.
- POISSON confidence >=0.80: 34 matches, 1.27% coverage, 76.47% accuracy; sample is too small for a reliable claim.
- Per-fold selective accuracy varies materially. BLEND >=0.65 ranges roughly 55%–75% across folds. Do not claim a stable 65–70% production hit rate yet.
- Confidence coverage means only a fraction of all fixtures are selected. This can yield 6–7 correct per 10 selected picks, but does not mean the bot can safely make 10 picks from any arbitrary 10 fixtures.

Updated pooled flat-stake edge ROI across the 10 OOF folds:
- Edge 0.03: 1,426 bets, 55.05% hit, -7.26% ROI.
- Edge 0.05: 885 bets, 57.63% hit, -6.71% ROI.
- Edge 0.08: 457 bets, 58.21% hit, -8.83% ROI.
- Edge 0.10: 243 bets, 60.49% hit, -7.51% ROI.
- Edge 0.12: 138 bets, 63.77% hit, -5.83% ROI.
- Edge 0.15: 61 bets, 65.57% hit, -7.21% ROI.
- Edge 0.20: 33 bets, 69.70% hit, -3.55% ROI.
- UNDER-only ROI is positive at 0.12, 0.15 and 0.20 in these development folds, but sample sizes are only 62, 30 and 18 bets respectively. These are exploratory results, not a validated betting policy.

## New walk-forward implementation

- `scripts/walk_forward_over_under.py` creates 10 expanding chronological folds strictly within the first 8,500 TRAIN rows.
- Each fold uses earlier training rows, a separate 500-row calibration period, and a 300-row forward validation period.
- It compares BLEND, MARKET and POISSON probabilities on identical valid-market rows.
- It reports edge ROI by OVER/UNDER/ALL at thresholds 0.03 through 0.20 and selective accuracy/coverage at confidence cutoffs 0.55 through 0.80.
- It does not access the policy select/check or Final Test rows for feature generation or scoring.
- Latest report files: `walk_forward_metrics.csv`, `walk_forward_policy_roi.csv`, `walk_forward_confidence.csv`.

## Latest pending verification

- Tests workflow run #181: `44340587891`, commit `9e855f81...`; status IN_PROGRESS when this checkpoint was updated. It includes the walk-forward unit tests and same-calendar-day isolation test. Verify final conclusion.
- Diagnostics workflow run #13: `44340587905`; SUCCESS.
- Main remains unchanged; no merge has occurred.


## Odds provenance and production-feature restriction

- Current `xgo.csv` contains non-C fields such as `odds_Avg>2.5` / `odds_B365>2.5` and separate C-suffixed variants. `market_data()` reads only the non-C fields; it has regression tests proving it never falls back to AvgC/B365C closing prices.
- Official Football-Data documentation says that since 2019/20 it collects an early set after market opening and a second closing set whose column names include `C`. This is the naming convention, but it does not prove that this particular `xgo.csv` odds data came from Football-Data.
- Earlier project records mention Maçkolik as an odds/results source; the exact original odds CSV filename, source lineage and collection timestamp for the odds columns in `xgo.csv` remain unverified.
- Therefore non-C odds are still treated as research/benchmark data only. Do not claim their ROI or availability as production-valid until the exact xgo odds provenance and prediction-time collection schedule are confirmed.
- Primary bot model has now been changed to `PoissonOnlyModel`, which predicts exclusively from leakage-safe historical `p_poisson`. It does not read bookmaker odds. `BlendModel` remains only as a research comparator; `HistoricalXGModel` remains an experimental historical-only logistic candidate.
- A fixed 0.75 confidence policy report was added for POLICY SELECT and POLICY CHECK. This cutoff was motivated by the TRAIN-only walk-forward OOF results; do not retune it against POLICY CHECK. Final Test remains sealed.

## Latest primary-model experiment

- `PoissonOnlyModel` became the main BOT model in commit `ee8f9318369ecefe2796a22ce6b3060669b979fc`.
- The expanded historical-xG logistic candidate uses cumulative and last-five-match xG-for/xG-against features, league scoring baseline, home advantage and history counts. All features are updated only after the full UTC calendar day has been predicted.
- On the 10-fold TRAIN-only OOF comparison, raw Poisson remained stronger than the fitted historical-xG logistic candidate:
  - POISSON all-match accuracy: 57.63% (3,000 fixtures), Brier 0.244465.
  - HISTORICAL_XG all-match accuracy: 56.33% (3,000 fixtures), Brier 0.247172.
  - POISSON confidence >=0.75: 123 picks, 71.54% accuracy, 4.10% coverage.
  - POISSON confidence >=0.70: 374 picks, 68.18% accuracy, 12.47% coverage.
  - These are development OOF results, not final production claims; confidence intervals are wide and fold-level variance remains.
- Latest completed diagnostics before the fixed-confidence-report change: run #26 `44342029767`, SUCCESS. The primary model's 300-match POLICY CHECK accuracy at the ordinary 0.50 cutoff was 50.67%; this is not the confidence-gated result.
- Current fixed-confidence workflow run #27: `44342342641`, status IN_PROGRESS when this checkpoint was updated. Verify the printed 0.75 POLICY SELECT / POLICY CHECK counts and hit rates.
- Latest tests run #197: `44342342643`, status PENDING when this checkpoint was updated.


## Session checkpoint — 2026-11-18 (upcoming-prediction workflow and selective top-10 metrics)

### Work added on `diagnostics/over-under-market-analysis`

- Added `scripts/predict_upcoming.py`, a command-line prediction path for future fixtures. Usage:
  `python scripts/predict_upcoming.py xgo.csv upcoming.csv --output predictions.csv --confidence 0.70 --top 10`.
- Upcoming input requires `Date,League,HomeTeam,AwayTeam`. It builds leakage-safe historical features, uses `PoissonOnlyModel` and does not use bookmaker odds.
- The predictor applies a confidence gate, selects at most the top N eligible fixtures, and emits `PICK` / `NO_PICK` rows. It does not force picks if fewer than N fixtures qualify.
- Same-UTC-calendar-day results are excluded from historical features before future fixture predictions. Added `tests/test_predict_upcoming.py` for top-N selection and same-day exclusion.
- Wired the CLI help check and new test file into `.github/workflows/diagnostics-report.yml`.
- Fixed result ordering so selected picks appear before `NO_PICK` fixtures.
- Added top-10 highest-confidence hit-rate fields to `confidence_policy_analysis()` in `BOT.py` and to the walk-forward selective-confidence report in `scripts/walk_forward_over_under.py`; updated associated tests.
- Added 0.70 and 0.75 fixed-cutoff policy reports for both POLICY SELECT and POLICY CHECK. These are reporting cutoffs; do not optimize them against POLICY CHECK.

### Verified diagnostic evidence

- Diagnostics workflow run #35, ID `44343017252`, commit `ae3fb2c7a6f22107a784951b9b24d24e1bb407c3`: **SUCCESS**.
- Its log explicitly reports POLICY CHECK (300 fixtures):
  - At fixed confidence >=0.70: 27 picks (9.0% coverage), 22 correct, 81.48% accuracy.
  - Among the 10 highest-confidence picks within those 27: 8 correct, 80.00% accuracy.
  - At fixed confidence >=0.75: 7 picks (2.3% coverage), 5 correct, 71.43% accuracy.
  - Ordinary all-fixture POLICY CHECK accuracy at the 0.50 decision threshold: 50.67%.
- POLICY SELECT (300 fixtures), >=0.70: 33 picks, 22 correct, 66.67%; top 10: 8/10. At >=0.75: 8 picks, 7 correct (87.50%), but this is the selection segment, not the independent check.
- Interpretation: the POLICY CHECK top-10 result is an encouraging small-sample selective result, not proof of a stable 80% hit rate. Only 27/300 fixtures passed the >=0.70 gate; do not promise 6–7 correct out of every arbitrary 10 matches. Confidence intervals / further chronological validation are still required. This does not establish positive ROI or a production-ready betting strategy.
- The top-10 metric added to the walk-forward CSV is now part of the diagnostics output; inspect the generated artifact and fold-level counts before drawing conclusions.

### CI state at checkpoint time

- Diagnostics #35 (`44343017252`) is completed SUCCESS.
- Tests workflow #210 (`44343017268`), same commit, was still **IN_PROGRESS** at last check. Its pytest job ID was `134940189725`; completion and result have not been verified. Earlier test runs from intermediate commits were cancelled and are not evidence of a passing test suite.
- Do not state that the full test suite passed until run #210 (or a later run on the final branch HEAD) is completed successfully.
- Latest observed branch commit from diagnostics run: `ae3fb2c7a6f22107a784951b9b24d24e1bb407c3`. Confirm the current branch HEAD before the next change.
- Main remains separate; no merge is authorized or performed. Final Test remains sealed.

### Next actions

1. Check tests run #210 (`44343017268`) and inspect pytest logs if it fails; fix issues on the development branch only.
2. Confirm the latest branch HEAD and rerun all relevant tests after fixes.
3. Inspect `walk_forward_confidence.csv` top-10 metrics fold by fold, including sample counts and stability; do not rely only on pooled accuracy.
4. Validate the upcoming-fixture CLI with a realistic upcoming CSV and check date parsing, timezone semantics, missing history, duplicate fixtures, and no-result leakage.
5. Keep odds provenance unresolved until the exact original odds source/file lineage and timestamp are documented; primary predictions remain odds-independent.
6. Continue improving the model and evaluate on chronological data without tuning on POLICY CHECK. Do not open or use Final Test until the model/policy is frozen and the agreed merge gate is met.


### Follow-up: tests #210 completed after the checkpoint append

- Tests workflow #210 (`44343017268`) has now completed **FAILURE** on commit `ae3fb2c7a6f22107a784951b9b24d24e1bb407c3`.
- Result: **69 passed, 1 failed** (291.66 seconds).
- Sole failure: `tests/test_walk_forward_over_under.py::test_walk_forward_folds_are_expanding_and_chronological`.
- Root cause in the test assertion, not yet fixed: it calls `len(train)` where `train` is a Python `slice`; Python slices do not implement `len()`. The assertion should check `train.stop - train.start >= 5000` (or equivalent) while preserving the intended chronological fold assertions.
- Seven warnings were emitted (including pandas concat FutureWarnings and scikit-learn penalty deprecation); they did not fail the run.
- Next step: correct the slice-length assertion on the development branch, rerun pytest, then rerun diagnostics if any model/report code changes. Do not mark tests green until a new run succeeds.
- Checkpoint update commit: `54b929a25669861cf748a5e6a39bc5628c17be3b` (this is the current branch HEAD after saving this checkpoint). Main has not been merged or modified by this checkpoint operation.
