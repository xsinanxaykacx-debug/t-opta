# Three-model comparison

This package hosts isolated experiments. It must not alter the existing production prediction path.

## Models

- `market_baseline.py`: de-vigged pre-match opening/current market probabilities only.
- `poisson_baseline.py`: historical xG, team form, and prior-performance features only.
- `odds_aware.py`: permitted pre-match market probabilities plus leakage-safe historical features.
- `comparison.py`: evaluates all models on identical chronological partitions and reports comparable metrics.

Closing-price fields (`AvgC`, `B365C`, `MaxC`, `PC`) are diagnostics-only and must never enter model features, fitting, calibration, policy selection, or prediction.

Implementation is intentionally deferred until the shared data contract and sealed split manifest have been verified against the repository data.
