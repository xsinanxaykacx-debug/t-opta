"""Historical-xG Poisson baseline with calendar-day leakage protection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from BOT import add_leakage_safe_features


class PoissonBaseline:
    """Generate Poisson probabilities from strictly prior calendar days.

    The legacy BOT feature routine groups on its Date values. This adapter
    normalises Date to calendar day before calling it, so no match result or
    xG from any fixture on the target calendar day can enter another fixture's
    features, even when kickoff timestamps differ.
    """

    name = "poisson"

    @staticmethod
    def add_historical_features(frame: pd.DataFrame) -> pd.DataFrame:
        if "Date" not in frame.columns:
            raise KeyError("Date column is required.")
        data = frame.copy(deep=True)
        original_dates = pd.to_datetime(data["Date"], errors="coerce", utc=True)
        if original_dates.isna().any():
            raise ValueError("Date contains invalid or missing values.")
        data["Date"] = original_dates.dt.normalize()
        featured = add_leakage_safe_features(data)
        featured["Date"] = original_dates.to_numpy()
        return featured

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if "p_poisson" not in frame.columns:
            raise KeyError("p_poisson is missing; call add_historical_features first.")
        probabilities = frame["p_poisson"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Poisson probabilities contain non-finite values.")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError("Poisson probabilities must be in [0, 1].")
        return probabilities

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= 0.5).astype(int)
