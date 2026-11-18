# Production prediction bot

## Frozen experiment evidence

The preselected Odds-Aware model passed the one-time chronological Final Test:

| Stage | Selected | Hit rate |
|---|---:|---:|
| Policy Select | 12 | 75.00% |
| Policy Check | 23 | 78.26% |
| Final Test | 450 | 66.00% (297/450) |

- Frozen confidence threshold: **0.735**
- Final Test was evaluated once; this evidence must not be used to retune the model or threshold.
- The market and Poisson baselines remain in the experiment report for comparison.

## Profitability warning

The same 450 Final Test selections had average odds **1.4495** and unit-stake ROI **-5.90%** (net **-26.54 units**). The model met the accuracy target, but this is not a profitable betting-system validation. See final_test_roi.json and the Actions artifact reports.

## Data freshness

- Odds/target dataset xgo.csv: last target match **2026-03-14**.
- Historical xG source xg18.csv: latest history **2026-11-14**.
- The model was trained/calibrated only on the frozen TRAIN/CALIBRATION partitions. The newer xg18 rows are used only as strictly prior historical features for upcoming fixtures.

## Build and run

Build the reproducible model artifact from TRAIN + CALIBRATION only:

```powershell
python -m experiments.train_production
```

Prepare an upcoming fixtures CSV with these columns:

- Date
- League
- HomeTeam
- AwayTeam
- odds_Avg>2.5
- odds_Avg<2.5
- odds_B365>2.5
- odds_B365<2.5

Then run:

```powershell
python predict.py upcoming_fixtures.csv --output predictions.csv
```

The predictor outputs all fixtures, probabilities, predicted side, confidence, selected flag, chosen odds, and expected value per unit. It does not use closing odds. Team and league names must match the canonical names in xg18.csv.

selected=true means the market odds are valid and confidence is at least 0.735. It is a model selection flag, not a guarantee or an instruction to stake money.