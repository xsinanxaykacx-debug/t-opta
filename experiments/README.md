# Three-model comparison

This package is an isolated research pipeline. It does not alter the legacy
`BOT.py` prediction path.

## Target

Binary Over/Under 2.5:
- OVER: `FTHG + FTAG > 2.5`
- UNDER: `FTHG + FTAG <= 2.5`

The operational objective is at least 65% hit rate on selected fixtures, with
a minimum of 100 selected Final Test fixtures. This is a pass/fail research
target, not a promised outcome.

## Models

- `market_baseline.py`: de-vigged pre-match `odds_Avg` prices, falling back
  to the paired `odds_B365` prices.
- `poisson_baseline.py`: historical xG and team/league rates, computed using
  strictly prior calendar days.
- `odds_aware.py`: logistic blend of the market probability, historical
  Poisson probability, and market-availability flag. Fit on TRAIN and
  calibrated on CALIBRATION only.
- `comparison.py`: identical raw chronological partitions, policy selection,
  policy check, final evaluation, directional ROI, and six odds buckets.

## Frozen raw split contract

| Partition | Rows |
|---|---:|
| TRAIN | 8,500 |
| CALIBRATION | 900 |
| POLICY_SELECT | 300 |
| POLICY_CHECK | 300 |
| FINAL_TEST | 5,734 |
| **Total** | **15,734** |

The source snapshot is `xgo.csv` (Git blob SHA
`1c18ccd4c4190cccc5014d186d6d34305708225c`). Rows are stable-sorted by
`Date, League, HomeTeam, AwayTeam`.

The exact raw row counts remain fixed. If a boundary splits fixtures on the
same calendar day, those boundary-day rows are purged from the preceding
operational stage before fitting, calibration, policy selection, or checking.
Raw counts and effective counts are both recorded in the split manifest.

## Leakage rules

- Closing prices, including `AvgC`, `B365C`, `MaxC`, `PC` and prefixed
  variants such as `odds_AvgC>2.5`, are audit-only.
- Match outcomes and same-match / in-play statistics are targets or historical
  source data only; they are never same-match features.
- Historical xG for all fixtures on the target calendar day is excluded.
- The model's only inputs are a pre-match de-vigged market probability, a
  strictly historical Poisson probability, and market availability.
- The approved raw odds allowlist is explicit; arbitrary `odds_*` columns are
  rejected by the protocol.

## Selection and final-test discipline

1. Fit model parameters on TRAIN.
2. Fit probability calibration on CALIBRATION.
3. Select the 65% confidence policy and candidate model on POLICY_SELECT.
4. Validate the frozen candidate and threshold on POLICY_CHECK.
5. If POLICY_CHECK fails, FINAL_TEST is not opened.
6. If POLICY_CHECK passes, evaluate all three frozen models once on FINAL_TEST.
   FINAL_TEST never selects a model, threshold, feature, or calibration method.

The experiment writes `model_metrics.csv`, `directional_roi.csv`,
`odds_buckets.csv`, `final_test_predictions.csv`, `split_manifest.json`,
and `experiment_summary.json` under `results/comparison/`.

Run:

```bash
python -m pytest
python -m experiments.comparison --data xgo.csv --output results/comparison
```
