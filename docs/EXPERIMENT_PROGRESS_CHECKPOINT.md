# T-OPTA experiment progress checkpoint

Checkpoint date: 2026-11-18 (UTC)
Repository: `xsinanxaykacx-debug/t-opta`
Working branch: `experiments/three-model-comparison`
Default branch: `main`

## Safety boundary

- All experiment work is isolated on `experiments/three-model-comparison`.
- The branch comparison currently reports 108 commits ahead and 0 behind `main`.
- No merge into `main` has been performed.
- Do not merge or deploy until the latest tests pass and the evaluation evidence is reviewed.

## Current project state

- Three-model experiment code exists: market baseline, historical Poisson baseline, and odds-aware model.
- Chronological protocol and leakage-audit modules exist.
- A prediction CLI / inference entry point exists in `predict.py`.
- Production artifact build workflow exists in `.github/workflows/build-prediction-bot.yml`.
- Experiment and test workflows exist under `.github/workflows/`.
- Frozen experiment summary and ROI report are stored under `results/comparison/`.

## Evaluation evidence currently recorded

- Recorded Final Test selective accuracy: 297 / 450 = 66.00%.
- Recorded Final Test ROI per unit: approximately -0.05898 (-5.90%).
- The 65% accuracy target was exceeded on that recorded sample, but profitability was not.
- 450 selected observations are not enough to establish a guaranteed future 65% hit rate. Treat this as historical validation evidence, not a promise.
- The recorded report and the later production build log report different effective training/calibration row counts: 8,495 and 887. The proposed contract sizes (8,500 and 900) therefore must not be described as the actual manifest counts until reconciled.
- Intended five partitions remain train, calibration, policy_select, policy_check, final_test. Their exact boundaries must be loaded from the canonical pipeline manifest, not reconstructed by hand.

## GitHub Actions evidence

- Production build run `44337080532` completed successfully on commit `f1e3e6cdd9a93b6d5181bd2b495af270f637ee9f`.
- It trained using 8,495 train rows and 887 calibration rows, with threshold 0.735.
- It uploaded artifact `t-opta-odds-aware-prediction-model`, artifact ID `14441857067`, SHA-256 `255f3e1b463c0ba1911adf1af1e1018a3f10646d68d5aa9796e705e857e7c949`.
- That artifact belongs to commit `f1e3e6c`; later changes were made after it. Rebuild and retest the current branch head before treating an artifact as current.
- Latest visible pytest workflow run: `44337113685`, head `11bb64c0`; at checkpoint time it was still reported in progress. Do not assume tests passed.
- Earlier run `44335317864` failed. Its failure must be examined and resolved; subsequent runs were cancelled/retriggered while code was changing.

## Critical next actions

1. Read this checkpoint and inspect the current branch HEAD before editing.
2. Retrieve the full pytest run result/logs for run `44337113685`; resolve every failing test.
3. Run the full test suite on the latest commit and verify the result is green.
4. Reconcile the canonical split manifest against the proposed 8,500 / 900 / 300 / 300 / 5,734 counts. Do not silently alter boundaries.
5. Audit feature provenance, especially odds availability timestamp, closing-odds exclusion, post-match exclusion, and prior-only historical xG calculations.
6. Re-run the full three-model evaluation using identical chronological partitions. Keep FINAL_TEST sealed during feature engineering, fitting, calibration, threshold selection, and policy tuning.
7. Rebuild the prediction artifact from the exact validated commit and run inference smoke tests on upcoming fixtures.
8. Only consider merging after tests, reproducibility, leakage audit, and deployment readiness are all verified.

## Completion criterion

Do not announce “bot is finished and ready for predictions” unless the current code is tested, the prediction artifact is built from that exact code, the data contract and leakage audit are verified, and the held-out evaluation supports the stated accuracy claim. Clearly report ROI separately. A 65% target is not a guarantee.
