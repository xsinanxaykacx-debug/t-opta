"""Odds-aware logistic model combining pre-match market and past-xG signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from BOT import BlendModel


class OddsAwareModel:
    """Fit on TRAIN and calibrate only on CALIBRATION."""

    name = "odds_aware"
    feature_names = (
        "pre_match_market_probability",
        "hist_poisson_probability",
        "pre_match_market_available",
    )

    def __init__(self) -> None:
        self.model = BlendModel()
        self.is_fitted = False
        self.is_calibrated = False

    def fit(self, train: pd.DataFrame) -> "OddsAwareModel":
        self.model.fit(train)
        self.is_fitted = True
        return self

    def calibrate(self, calibration: pd.DataFrame) -> "OddsAwareModel":
        if not self.is_fitted:
            raise RuntimeError("Fit the model on TRAIN before calibration.")
        self.model.calibrate(calibration)
        self.is_calibrated = True
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        probabilities = np.asarray(self.model.predict_proba(frame), dtype=float)
        if probabilities.ndim != 1 or len(probabilities) != len(frame):
            raise ValueError("Model returned an invalid probability vector.")
        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned non-finite probabilities.")
        return np.clip(probabilities, 1e-6, 1.0 - 1e-6)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= 0.5).astype(int)
