# T-OPTA v13 architecture

Match URL -> provider adapter -> canonical match/team IDs -> immutable historical store -> as-of feature builder -> trained model -> calibration -> confidence/EV policy -> BET or NO_BET -> prediction log -> later result reconciliation.

Safeguards:
- Chronological holdout and walk-forward evaluation only.
- Features must be available at prediction time. Post-match xG, final scores, and future/closing odds are forbidden as pre-match inputs unless timestamp evidence proves availability.
- Fit preprocessing, feature selection and calibration on training/validation periods only, never the final holdout.
- Probability, calibration, confidence and expected value are separate concepts.
- Insufficient evidence or edge returns NO_BET; no hit-rate guarantee.
- MOCK is SIMULATION_ONLY; STAGING/PRODUCTION cannot fall back to mock.
- Predictions and actual results use separate tables.
- Existing xg18.csv is source data and is not modified by bootstrap.

All provider, resolver, ingestion, feature, prediction, calibration, confidence, reporting and backtest modules will be independently implemented. Initial integrations use only user-owned datasets.
