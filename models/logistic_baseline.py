"""Leakage-safe multinomial Logistic Regression baseline for 1X2."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LogisticBaseline:
    """Train-fitted preprocessing, chronological C selection, and temperature scaling."""

    version = "logistic-regression-baseline-v1"
    C_GRID = (0.01, 0.1, 1.0, 10.0)
    CLASS_ORDER = (0, 1, 2)

    def __init__(self, c_grid=None, max_iter=2000, validation_fraction=0.2):
        self.c_grid = tuple(float(c) for c in (c_grid or self.C_GRID))
        if not self.c_grid or any(not math.isfinite(c) or c <= 0 for c in self.c_grid):
            raise ValueError("c_grid must contain positive finite values.")
        if isinstance(max_iter, bool) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not 0.05 <= float(validation_fraction) <= 0.4:
            raise ValueError("validation_fraction must be between 0.05 and 0.4.")
        self.max_iter = int(max_iter)
        self.validation_fraction = float(validation_fraction)
        self.pipeline_ = None
        self.selected_c_ = None
        self.temperature_ = 1.0
        self.calibration_applied_ = False
        self.feature_columns_ = None
        self.numeric_columns_ = None
        self.categorical_columns_ = None
        self.validation_scores_ = {}

    @staticmethod
    def _prepare_features(features):
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("features must be a non-empty DataFrame.")
        forbidden = {"FTHG", "FTAG", "Home_xG", "Away_xG", "MatchID", "match_date"}
        leakage = forbidden.intersection(features.columns)
        if leakage:
            raise ValueError(f"Target/identity columns are not model features: {sorted(leakage)}")
        frame = features.copy()
        frame.columns = [str(column) for column in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError("Feature column names must be unique.")
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        categorical = [c for c in frame.columns if c not in numeric]
        if not numeric and not categorical:
            raise ValueError("No usable feature columns.")
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
        for column in categorical:
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
        return frame, numeric, categorical

    @staticmethod
    def _validate_labels(labels):
        y = np.asarray(labels, dtype=int)
        if y.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1, 2]).all():
            raise ValueError("labels must be a non-empty 1D array containing only 0, 1, 2.")
        if set(np.unique(y)) != {0, 1, 2}:
            raise ValueError("Training partition must contain all three 1X2 classes.")
        return y

    @staticmethod
    def _make_pipeline(numeric, categorical, c_value, max_iter):
        transformers = []
        if numeric:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("numeric", numeric_pipeline, numeric))
        if categorical:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])
            transformers.append(("categorical", categorical_pipeline, categorical))
        preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
        classifier = LogisticRegression(
            C=float(c_value), solver="lbfgs", penalty="l2", max_iter=max_iter,
        )
        return Pipeline([("preprocess", preprocess), ("classifier", classifier)])

    @staticmethod
    def _ordered_probabilities(pipeline, features):
        raw = pipeline.predict_proba(features)
        classes = list(pipeline.named_steps["classifier"].classes_)
        if set(classes) != {0, 1, 2}:
            raise ValueError("Fitted classifier does not contain all 1X2 classes.")
        return np.asarray(raw[:, [classes.index(c) for c in (0, 1, 2)]], dtype=float)

    @staticmethod
    def _chronological_indices(features, fraction):
        if "match_date" not in features.columns:
            raise ValueError("match_date is required for chronological C selection.")
        dates = pd.to_datetime(features["match_date"], errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError("match_date contains invalid or missing values.")
        days = np.array(sorted(dates.dt.normalize().unique()))
        if len(days) < 2:
            raise ValueError("At least two distinct calendar days are required for C selection.")
        split_at = max(1, min(len(days) - 1, int(np.floor(len(days) * (1.0 - fraction)))))
        validation_start = pd.Timestamp(days[split_at])
        train_mask = (dates.dt.normalize() < validation_start).to_numpy()
        valid_mask = ~train_mask
        if not train_mask.any() or not valid_mask.any():
            raise ValueError("Chronological internal validation produced an empty partition.")
        return np.flatnonzero(train_mask), np.flatnonzero(valid_mask)

    def fit(self, features, labels):
        frame, numeric, categorical = self._prepare_features(features)
        y = self._validate_labels(labels)
        if len(frame) != len(y):
            raise ValueError("features and labels length mismatch.")
        train_idx, valid_idx = self._chronological_indices(frame, self.validation_fraction)
        if set(np.unique(y[train_idx])) != {0, 1, 2}:
            raise ValueError("Internal chronological training fold must contain all 1X2 classes.")
        if len(np.unique(y[valid_idx])) < 2:
            raise ValueError("Internal chronological validation fold must contain at least two classes.")

        scores = {}
        for c_value in self.c_grid:
            candidate = self._make_pipeline(numeric, categorical, c_value, self.max_iter)
            candidate.fit(frame.iloc[train_idx], y[train_idx])
            probabilities = self._ordered_probabilities(candidate, frame.iloc[valid_idx])
            actual = y[valid_idx]
            loss = -np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-15, 1.0)).mean()
            scores[c_value] = float(loss)

        self.selected_c_ = min(self.c_grid, key=lambda value: (scores[value], value))
        self.validation_scores_ = scores
        self.pipeline_ = self._make_pipeline(numeric, categorical, self.selected_c_, self.max_iter)
        self.pipeline_.fit(frame, y)
        self.feature_columns_ = list(frame.columns)
        self.numeric_columns_ = list(numeric)
        self.categorical_columns_ = list(categorical)
        self.temperature_ = 1.0
        self.calibration_applied_ = False
        return self

    @staticmethod
    def _temperature_loss(log_temperature, probabilities, labels):
        temperature = float(np.exp(log_temperature))
        logits = np.log(np.clip(probabilities, 1e-15, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        scaled = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(scaled[np.arange(len(labels)), labels], 1e-15, 1.0)).mean())

    def calibrate(self, features, labels):
        if self.pipeline_ is None:
            raise RuntimeError("fit must be called before calibrate.")
        frame, _, _ = self._prepare_features(features)
        if list(frame.columns) != self.feature_columns_:
            raise ValueError("Calibration feature schema differs from training schema.")
        y = np.asarray(labels, dtype=int)
        if len(frame) != len(y) or len(y) == 0 or not np.isin(y, [0, 1, 2]).all():
            raise ValueError("Invalid calibration labels or feature count.")
        probabilities = self._ordered_probabilities(self.pipeline_, frame)
        # Deterministic bounded golden-section search over T in [0.05, 20].
        left, right = math.log(0.05), math.log(20.0)
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = right - ratio * (right - left)
        x2 = left + ratio * (right - left)
        f1 = self._temperature_loss(x1, probabilities, y)
        f2 = self._temperature_loss(x2, probabilities, y)
        for _ in range(100):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - ratio * (right - left)
                f1 = self._temperature_loss(x1, probabilities, y)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + ratio * (right - left)
                f2 = self._temperature_loss(x2, probabilities, y)
        self.temperature_ = float(np.exp((left + right) / 2.0))
        self.calibration_applied_ = True
        return self

    def predict_proba(self, features):
        if self.pipeline_ is None:
            raise RuntimeError("Model has not been fitted.")
        frame, _, _ = self._prepare_features(features)
        if list(frame.columns) != self.feature_columns_:
            raise ValueError("Prediction feature schema differs from training schema.")
        probabilities = self._ordered_probabilities(self.pipeline_, frame)
        if self.calibration_applied_:
            logits = np.log(np.clip(probabilities, 1e-15, 1.0)) / self.temperature_
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probabilities
