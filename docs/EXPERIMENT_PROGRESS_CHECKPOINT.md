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
