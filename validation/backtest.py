"""Chronological 1X2 backtesting with an isolated calibration period."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from validation.metrics import EvaluationMetrics


class HistoricalBacktester:
    """Evaluate an adapter on disjoint, ordered train/calibration/test periods.

    The feature engine builds each row by MatchID from its historical source;
    its calendar-day exclusion policy prevents same-day fixtures leaking into
    one another. The adapter owns train-fitted preprocessing and model fitting.
    """

    def __init__(
        self,
        feature_engine: Any,
        model_adapter: Any,
        confidence_engine: Any = None,
        decision_fn: Callable | None = None,
    ):
        self.feature_engine = feature_engine
        self.model_adapter = model_adapter
        self.confidence_engine = confidence_engine
        self.decision_fn = decision_fn

    @staticmethod
    def _prepare_partition(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        required = {"MatchID", "Date", "FTHG", "FTAG"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
        data = frame.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True)
        if data["Date"].isna().any():
            raise ValueError(f"{name} contains invalid or missing dates.")
        if data["MatchID"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate MatchID values.")
        if data[["FTHG", "FTAG"]].isna().any().any():
            raise ValueError(f"{name} contains incomplete full-time scores.")
        if (data[["FTHG", "FTAG"]] < 0).any().any():
            raise ValueError(f"{name} contains negative full-time scores.")
        return data.sort_values(["Date", "MatchID"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _labels(frame: pd.DataFrame) -> np.ndarray:
        home = frame["FTHG"].to_numpy(dtype=float)
        away = frame["FTAG"].to_numpy(dtype=float)
        return np.where(home > away, 0, np.where(home == away, 1, 2)).astype(int)

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engine.build_dataset(frame["MatchID"].astype(str).tolist())

    def _decisions(self, probabilities: np.ndarray, features: pd.DataFrame) -> list[str]:
        decisions = []
        for index, row in enumerate(probabilities):
            payload = {
                "prediction": {
                    "probabilities": {
                        "1": float(row[0]), "X": float(row[1]), "2": float(row[2])
                    }
                }
            }
            if self.decision_fn is not None:
                decision = self.decision_fn(payload, features.iloc[index])
            elif self.confidence_engine is not None and hasattr(self.confidence_engine, "evaluate"):
                result = self.confidence_engine.evaluate(payload)
                decision = result.get("confidence_evaluation", {}).get(
                    "decision", result.get("decision", "NO BET")
                )
            else:
                decision = "NO BET"
            normalized = str(decision).strip().upper().replace("_", " ")
            decisions.append("BET" if normalized == "BET" else "NO BET")
        return decisions

    def run_backtest(
        self,
        train_df: pd.DataFrame,
        calib_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
    ) -> dict:
        train = self._prepare_partition(train_df, "train_df")
        test = self._prepare_partition(test_df, "test_df")
        calibration = None
        if calib_df is not None and not calib_df.empty:
            calibration = self._prepare_partition(calib_df, "calib_df")

        if train["Date"].max().normalize() >= test["Date"].min().normalize():
            raise ValueError("Train and test must be separated by calendar day.")
        if calibration is not None:
            if train["Date"].max().normalize() >= calibration["Date"].min().normalize():
                raise ValueError("Train must end before calibration begins (calendar-day strict).")
            if calibration["Date"].max().normalize() >= test["Date"].min().normalize():
                raise ValueError("Calibration must end before test begins (calendar-day strict).")

        partitions = [train]
        if calibration is not None:
            partitions.append(calibration)
        partitions.append(test)
        all_ids = pd.concat([part["MatchID"].astype(str) for part in partitions], ignore_index=True)
        if all_ids.duplicated().any():
            raise ValueError("MatchID overlap detected between backtest partitions.")

        x_train, y_train = self._features(train), self._labels(train)
        self.model_adapter.fit(x_train, y_train)

        calibrated = False
        if calibration is not None and hasattr(self.model_adapter, "calibrate"):
            x_calib, y_calib = self._features(calibration), self._labels(calibration)
            self.model_adapter.calibrate(x_calib, y_calib)
            calibrated = True

        x_test, y_test = self._features(test), self._labels(test)
        probabilities = np.asarray(self.model_adapter.predict_proba(x_test), dtype=float)
        if probabilities.shape != (len(test), 3):
            raise ValueError("predict_proba must return an (n_test, 3) matrix in [home, draw, away] order.")
        # Reuse the metrics validator before applying any decision policy.
        EvaluationMetrics._validate(y_test, probabilities)
        decisions = self._decisions(probabilities, x_test)
        report = EvaluationMetrics.evaluate_predictions(y_test, probabilities, decisions)
        report.update({
            "status": "EVALUATED",
            "target": "1X2",
            "class_order": ["HOME", "DRAW", "AWAY"],
            "train_matches": int(len(train)),
            "calibration_matches": int(len(calibration)) if calibration is not None else 0,
            "test_matches": int(len(test)),
            "calibration_applied": calibrated,
            "model_version": str(getattr(self.model_adapter, "version", "unknown")),
            "roi": None,
            "roi_note": "Not calculated: historical dataset contains no odds columns.",
            "decision_note": (
                "No confidence policy supplied; all matches default to NO BET."
                if self.confidence_engine is None and self.decision_fn is None
                else "BET coverage is policy-selected; ROI is unavailable without odds."
            ),
        })
        return report
