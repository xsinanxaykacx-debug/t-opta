"""Odds-aware logistic model with audited historical xg18 features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from BOT import market_data


HISTORICAL_COLUMNS = (
    "p_poisson",
    "hist_lambda_home",
    "hist_lambda_away",
    "hist_home_xg_for_avg_5",
    "hist_home_xg_against_avg_5",
    "hist_home_xg_for_avg_10",
    "hist_home_xg_against_avg_10",
    "hist_away_xg_for_avg_5",
    "hist_away_xg_against_avg_5",
    "hist_away_xg_for_avg_10",
    "hist_away_xg_against_avg_10",
    "hist_home_xg_for_ewma_5",
    "hist_home_xg_against_ewma_5",
    "hist_away_xg_for_ewma_5",
    "hist_away_xg_against_ewma_5",
    "hist_home_points_avg_5",
    "hist_away_points_avg_5",
    "hist_home_rest_days",
    "hist_away_rest_days",
    "hist_h2h_matches_available",
    "hist_h2h_home_goals_avg",
    "hist_h2h_away_goals_avg",
)

FEATURE_NAMES = (
    "pre_match_market_probability",
    "pre_match_market_available",
    "hist_poisson_probability",
    *tuple(column for column in HISTORICAL_COLUMNS if column != "p_poisson"),
)


class OddsAwareModel:
    """Fit on TRAIN, calibrate on CALIBRATION, and predict from pre-match data."""

    name = "odds_aware"
    feature_names = FEATURE_NAMES
    min_calibration_rows = 100

    def __init__(self) -> None:
        self.pipeline = make_pipeline(
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                max_iter=3000,
                solver="lbfgs",
                random_state=42,
            ),
        )
        self.calibrator: IsotonicRegression | None = None
        self.is_fitted = False
        self.is_calibrated = False

    @staticmethod
    def _matrix(frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in HISTORICAL_COLUMNS if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing approved historical model features: {missing}")

        p_market, _, _, market_valid = market_data(frame)
        columns = [
            np.asarray(p_market, dtype=float),
            np.asarray(market_valid, dtype=float),
        ]
        for column in HISTORICAL_COLUMNS:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            columns.append(values)
        matrix = np.column_stack(columns)
        if matrix.ndim != 2 or matrix.shape[0] != len(frame):
            raise ValueError("Invalid odds-aware feature matrix.")
        return matrix

    @staticmethod
    def _target(frame: pd.DataFrame) -> np.ndarray:
        required = {"FTHG", "FTAG"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Missing training target columns: {missing}")
        home_goals = pd.to_numeric(frame["FTHG"], errors="coerce").to_numpy(dtype=float)
        away_goals = pd.to_numeric(frame["FTAG"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(home_goals).all() or not np.isfinite(away_goals).all():
            raise ValueError("Training targets contain missing or non-finite scores.")
        return ((home_goals + away_goals) > 2.5).astype(int)

    def fit(self, train: pd.DataFrame) -> "OddsAwareModel":
        if len(train) < 100:
            raise ValueError("TRAIN partition is too small.")
        X = self._matrix(train)
        y = self._target(train)
        if np.unique(y).size != 2:
            raise ValueError("TRAIN must contain both OVER and UNDER outcomes.")
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def calibrate(self, calibration: pd.DataFrame) -> "OddsAwareModel":
        if not self.is_fitted:
            raise RuntimeError("Fit the model on TRAIN before calibration.")
        if len(calibration) < self.min_calibration_rows:
            raise ValueError("CALIBRATION partition is too small.")
        X = self._matrix(calibration)
        y = self._target(calibration)
        if np.unique(y).size != 2:
            raise ValueError("CALIBRATION must contain both outcome classes.")
        raw = self.pipeline.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw, y)
        self.is_calibrated = True
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        raw = self.pipeline.predict_proba(self._matrix(frame))[:, 1]
        if self.is_calibrated and self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        probabilities = np.asarray(raw, dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned non-finite probabilities.")
        return np.clip(probabilities, 1e-6, 1.0 - 1e-6)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= 0.5).astype(int)
