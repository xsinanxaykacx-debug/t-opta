# T-OPTA Checkpoint — 2026-11-18

## Current state
- Repository: https://github.com/xsinanxaykacx-debug/t-opta
- Main branch includes UTC-day leakage fixes, reliability diagnostics, SciPy dependency, selective-accuracy reporting, and a validation gate that keeps the final holdout sealed.
- Main BOT.py blob: `311819ca30c65694a968df41481024278835ecec`
- Latest merged implementation commit: `1e7316c517452abfdd4b65ba8d30fb4ee72a4752`

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

## Safety gate
- Final holdout remains sealed unless Policy Check has at least 50 bets, hit rate >= 65%, and ROI > 0%.
- Do not tune thresholds or features using final holdout outcomes.
- If validation does not pass, do not run or inspect the final holdout.

## Code / tests
- `requirements.txt` now includes `scipy>=1.13,<2.0`.
- `tests/test_bot.py` covers same-day leakage, current-match xG exclusion, Poisson expected goals, UTC-day splits, complete-pair de-vigging, reliability bins, and policy gate behavior.
- GitHub Actions workflow added at `.github/workflows/ci.yml`; the connected status API has not shown a Python Actions run. The only reported status is the unrelated Vercel deployment rate-limit failure.
- Equivalent Python logic checks run in this session passed; the exact repository pytest suite has not yet been confirmed green.

## Next steps
1. Resolve why GitHub Actions is not reporting a run and obtain a real `pytest -q` result.
2. Improve the model on Train/Validation only; current hybrid does not beat market baseline.
3. Re-run the frozen Policy Select / Policy Check protocol after each candidate is locked.
4. Open the 5,746-match final holdout only after the validation gate passes.
