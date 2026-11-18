"""Expanding-window Over/Under 2.5 research using pre-Final-Test history only.

The input CSV is read and cleaned to establish chronological row order, then
capped at BOT.N_TRAIN before feature generation, model fitting or scoring.
Policy validation partitions and the sealed Final Test are not evaluated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BOT import (  # noqa: E402
    N_TRAIN,
    BlendModel,
    HistoricalXGModel,
    add_leakage_safe_features,
    market_data,
    require_columns,
)


DEFAULT_INITIAL_TRAIN = 5000
DEFAULT_CALIBRATION_SIZE = 500
DEFAULT_VALIDATION_SIZE = 300
DEFAULT_FOLDS = 10
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "walk_forward"



class HistGradientBoostingCandidate:
    """Nonlinear historical-xG research candidate; never uses bookmaker odds."""

    def __init__(self):
        self.clf = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=50,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=42,
        )
        self.calibrator = None

    @staticmethod
    def _X(df):
        return HistoricalXGModel._X(df)

    @staticmethod
    def _y(df):
        return BlendModel._y(df)

    def fit(self, df):
        self.clf.fit(self._X(df), self._y(df))
        return self

    def raw_predict(self, df):
        return self.clf.predict_proba(self._X(df))[:, 1]

    def calibrate(self, df):
        y = self._y(df)
        if np.unique(y).size != 2:
            raise ValueError("HGB calibration requires both outcome classes.")
        self.calibrator = IsotonicRegression(
            y_min=0.001,
            y_max=0.999,
            out_of_bounds="clip",
        )
        self.calibrator.fit(self.raw_predict(df), y)
        return self

    def predict_proba(self, df):
        probabilities = self.raw_predict(df)
        if self.calibrator is not None:
            probabilities = self.calibrator.predict(probabilities)
        return np.clip(probabilities, 1e-6, 1.0 - 1e-6)



class LogisticOverUnderCandidate:
    """Chronologically tuned binary logistic model on historical xG features."""

    C_GRID = (0.01, 0.1, 1.0, 10.0)

    def __init__(self):
        self.pipeline = None
        self.calibrator = None
        self.best_c = None
        self.validation_scores = {}

    @staticmethod
    def _X(frame: pd.DataFrame) -> np.ndarray:
        return HistoricalXGModel._X(frame)

    @staticmethod
    def _y(frame: pd.DataFrame) -> np.ndarray:
        return BlendModel._y(frame)

    @staticmethod
    def _make_pipeline(c_value: float) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=float(c_value),
                solver="lbfgs",
                max_iter=1500,
                random_state=42,
            )),
        ])

    def fit(self, train: pd.DataFrame):
        dates = pd.to_datetime(train["Date"], utc=True, errors="coerce")
        if dates.isna().any():
            raise ValueError("Logistic candidate requires valid chronological training dates.")
        days = pd.Series(dates.dt.normalize().unique()).sort_values().reset_index(drop=True)
        if len(days) < 2:
            raise ValueError("Logistic candidate needs at least two distinct training days.")
        split_at = max(1, min(len(days) - 1, int(np.floor(len(days) * 0.8))))
        validation_start = pd.Timestamp(days.iloc[split_at])
        normalized_days = dates.dt.normalize()
        fit_mask = (normalized_days < validation_start).to_numpy()
        validation_mask = ~fit_mask
        y = self._y(train)
        if np.unique(y[fit_mask]).size != 2 or np.unique(y[validation_mask]).size != 2:
            raise ValueError("Logistic inner chronological split must contain both outcome classes.")

        X = self._X(train)
        scores = {}
        for c_value in self.C_GRID:
            candidate = self._make_pipeline(c_value)
            candidate.fit(X[fit_mask], y[fit_mask])
            probabilities = candidate.predict_proba(X[validation_mask])[:, 1]
            scores[c_value] = float(log_loss(y[validation_mask], probabilities, labels=[0, 1]))

        self.best_c = min(self.C_GRID, key=lambda value: (scores[value], value))
        self.validation_scores = scores
        self.pipeline = self._make_pipeline(self.best_c)
        self.pipeline.fit(X, y)
        self.calibrator = None
        return self

    def raw_predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Logistic candidate has not been fitted.")
        return self.pipeline.predict_proba(self._X(frame))[:, 1]

    def calibrate(self, calibration: pd.DataFrame):
        probabilities = self.raw_predict(calibration)
        labels = self._y(calibration)
        if np.unique(labels).size != 2:
            raise ValueError("Logistic calibration requires both outcome classes.")
        self.calibrator = IsotonicRegression(
            y_min=0.001,
            y_max=0.999,
            out_of_bounds="clip",
        )
        self.calibrator.fit(probabilities, labels)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.raw_predict(frame)
        if self.calibrator is not None:
            probabilities = self.calibrator.predict(probabilities)
        return np.clip(probabilities, 1e-6, 1.0 - 1e-6)

class NegativeBinomialCandidate:
    """Train-only overdispersed total-goals model using historical xG rates."""

    def __init__(self):
        self.alpha = 0.0

    @staticmethod
    def _mu(frame: pd.DataFrame) -> np.ndarray:
        required = ("lambda_home", "lambda_away")
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError("Negative-binomial features missing: " + ", ".join(missing))
        return np.clip(
            frame["lambda_home"].to_numpy(dtype=float)
            + frame["lambda_away"].to_numpy(dtype=float),
            1e-4,
            10.0,
        )

    @staticmethod
    def _goals(frame: pd.DataFrame) -> np.ndarray:
        return (
            frame["FTHG"].to_numpy(dtype=float)
            + frame["FTAG"].to_numpy(dtype=float)
        )

    def fit(self, train: pd.DataFrame):
        mu = self._mu(train)
        goals = self._goals(train)
        valid = np.isfinite(mu) & np.isfinite(goals)
        if valid.sum() < 100:
            raise ValueError("Negative-binomial dispersion needs at least 100 valid training matches.")

        mu = mu[valid]
        goals = goals[valid]
        denominator = float(np.square(mu).sum())
        numerator = float((np.square(goals - mu) - mu).sum())
        estimate = numerator / denominator if denominator > 0 else 0.0
        self.alpha = float(np.clip(estimate, 0.0, 2.0))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        mu = self._mu(frame)
        if self.alpha <= 1e-8:
            probabilities = poisson.sf(2, mu)
        else:
            shape = 1.0 / self.alpha
            probability = shape / (shape + mu)
            probabilities = nbinom.sf(2, shape, probability)
        return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)


def build_fold_slices(
    n_rows: int,
    initial_train: int = DEFAULT_INITIAL_TRAIN,
    calibration_size: int = DEFAULT_CALIBRATION_SIZE,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    folds: int = DEFAULT_FOLDS,
) -> list[tuple[slice, slice, slice]]:
    """Create strictly ordered expanding train/calibration/validation folds."""
    values = (n_rows, initial_train, calibration_size, validation_size, folds)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError("Fold sizes must be integers.")
    if initial_train < 1000 or calibration_size < 100 or validation_size < 50 or folds < 1:
        raise ValueError("Fold sizes are below safe minimums.")

    windows = []
    for fold_index in range(folds):
        train_end = initial_train + fold_index * validation_size
        calibration_end = train_end + calibration_size
        validation_end = calibration_end + validation_size
        if validation_end > n_rows:
            break
        windows.append((
            slice(0, train_end),
            slice(train_end, calibration_end),
            slice(calibration_end, validation_end),
        ))

    if not windows:
        raise ValueError("Not enough development rows for one complete fold.")
    return windows


def _score_row(
    y: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    fold: int,
    scope: str = "VALID_MARKET",
) -> dict:
    selected_y = y[mask]
    selected_p = np.clip(probabilities[mask], 1e-6, 1.0 - 1e-6)
    if len(selected_y) == 0:
        raise ValueError(f"No valid observations for {model_name} in fold {fold}.")

    return {
        "fold": fold,
        "model": model_name,
        "scope": scope,
        "matches": int(len(selected_y)),
        "accuracy": float(((selected_p >= 0.5).astype(int) == selected_y).mean()),
        "brier": float(brier_score_loss(selected_y, selected_p)),
        "log_loss": float(log_loss(selected_y, selected_p, labels=[0, 1])),
        "auc": (
            float(roc_auc_score(selected_y, selected_p))
            if np.unique(selected_y).size == 2
            else np.nan
        ),
    }



def platt_scale_probabilities(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    target_probabilities: np.ndarray,
) -> np.ndarray:
    """Fit one-dimensional Platt scaling on a separate chronological calibration set."""
    calibration_probabilities = np.clip(
        np.asarray(calibration_probabilities, dtype=float), 1e-6, 1.0 - 1e-6
    )
    target_probabilities = np.clip(
        np.asarray(target_probabilities, dtype=float), 1e-6, 1.0 - 1e-6
    )
    labels = np.asarray(calibration_labels, dtype=int)
    if len(calibration_probabilities) != len(labels):
        raise ValueError("Calibration probabilities and labels must have equal length.")
    if np.unique(labels).size != 2:
        raise ValueError("Platt calibration requires both outcome classes.")

    calibration_logits = np.log(
        calibration_probabilities / (1.0 - calibration_probabilities)
    ).reshape(-1, 1)
    target_logits = np.log(
        target_probabilities / (1.0 - target_probabilities)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    calibrator.fit(calibration_logits, labels)
    return np.clip(
        calibrator.predict_proba(target_logits)[:, 1],
        1e-6,
        1.0 - 1e-6,
    )


def build_odds_provenance_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize source-file and match-status metadata without using odds as features."""
    required = ("odds_source_file", "odds_match_status", "League", "Date")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return pd.DataFrame([{
            "odds_source_file": "MISSING_METADATA",
            "matches": int(len(frame)),
            "matched_rows": 0,
            "match_rate": np.nan,
            "leagues": "",
            "first_date": "",
            "last_date": "",
            "avg_ou_pair_rows": 0,
            "b365_ou_pair_rows": 0,
        }])

    work = frame.loc[:, list(required)].copy()
    work["odds_source_file"] = work["odds_source_file"].fillna("MISSING").astype(str)
    work["odds_match_status"] = work["odds_match_status"].fillna("MISSING").astype(str)
    work["Date"] = pd.to_datetime(work["Date"], utc=True, errors="coerce")

    avg_pair = (
        pd.to_numeric(frame.get("odds_Avg>2.5", np.nan), errors="coerce").notna()
        & pd.to_numeric(frame.get("odds_Avg<2.5", np.nan), errors="coerce").notna()
    ) if "odds_Avg>2.5" in frame and "odds_Avg<2.5" in frame else np.zeros(len(frame), dtype=bool)
    b365_pair = (
        pd.to_numeric(frame.get("odds_B365>2.5", np.nan), errors="coerce").notna()
        & pd.to_numeric(frame.get("odds_B365<2.5", np.nan), errors="coerce").notna()
    ) if "odds_B365>2.5" in frame and "odds_B365<2.5" in frame else np.zeros(len(frame), dtype=bool)
    work["avg_ou_pair"] = np.asarray(avg_pair, dtype=bool)
    work["b365_ou_pair"] = np.asarray(b365_pair, dtype=bool)

    rows = []
    for source_file, group in work.groupby("odds_source_file", sort=True):
        matched = group["odds_match_status"].str.lower().eq("matched")
        valid_dates = group["Date"].dropna()
        rows.append({
            "odds_source_file": source_file,
            "matches": int(len(group)),
            "matched_rows": int(matched.sum()),
            "match_rate": float(matched.mean()) if len(group) else np.nan,
            "leagues": ";".join(sorted(group["League"].dropna().astype(str).unique())),
            "first_date": valid_dates.min().isoformat() if len(valid_dates) else "",
            "last_date": valid_dates.max().isoformat() if len(valid_dates) else "",
            "avg_ou_pair_rows": int(group["avg_ou_pair"].sum()),
            "b365_ou_pair_rows": int(group["b365_ou_pair"].sum()),
        })
    return pd.DataFrame(rows)


def calculate_policy_rows(
    y: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    over_odds: np.ndarray,
    under_odds: np.ndarray,
    market_valid: np.ndarray,
    threshold: float,
    fold: int,
    model_name: str = "BLEND",
) -> list[dict]:
    """Calculate OVER, UNDER and combined flat-stake ROI for a fixed edge."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite.")

    edge = p_model - p_market
    masks = {
        "OVER": market_valid & (edge > threshold) & np.isfinite(over_odds) & (over_odds > 1),
        "UNDER": market_valid & (edge < -threshold) & np.isfinite(under_odds) & (under_odds > 1),
    }

    rows = []
    all_profits = []
    all_wins = []
    all_odds = []

    for side, mask in masks.items():
        odds = over_odds[mask] if side == "OVER" else under_odds[mask]
        wins = (y[mask] == (1 if side == "OVER" else 0)).astype(int)
        profits = np.where(wins == 1, odds - 1.0, -1.0)
        all_profits.extend(profits.tolist())
        all_wins.extend(wins.tolist())
        all_odds.extend(odds.tolist())

        rows.append({
            "fold": fold,
            "model": model_name,
            "threshold": float(threshold),
            "side": side,
            "bets": int(len(odds)),
            "wins": int(wins.sum()),
            "hit_rate": float(wins.mean()) if len(wins) else np.nan,
            "avg_odds": float(odds.mean()) if len(odds) else np.nan,
            "roi_pct": float(profits.sum() / len(profits) * 100.0) if len(profits) else np.nan,
            "net_profit_units": float(profits.sum()) if len(profits) else 0.0,
        })

    rows.append({
        "fold": fold,
        "model": model_name,
        "threshold": float(threshold),
        "side": "ALL",
        "bets": len(all_profits),
        "wins": int(sum(all_wins)),
        "hit_rate": float(np.mean(all_wins)) if all_wins else np.nan,
        "avg_odds": float(np.mean(all_odds)) if all_odds else np.nan,
        "roi_pct": float(np.sum(all_profits) / len(all_profits) * 100.0) if all_profits else np.nan,
        "net_profit_units": float(np.sum(all_profits)) if all_profits else 0.0,
    })
    return rows


def calculate_confidence_rows(
    y: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    fold: int,
    cutoffs: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
    scope: str = "VALID_MARKET",
) -> list[dict]:
    """Measure selective accuracy when the system abstains on low confidence."""
    rows = []
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predicted = (probabilities >= 0.5).astype(int)
    for cutoff in cutoffs:
        selected = confidence >= cutoff
        count = int(selected.sum())
        selected_indices = np.flatnonzero(selected)
        selected_order = np.argsort(confidence[selected_indices])[::-1]
        top_indices = selected_indices[selected_order[:10]]
        top_n = int(len(top_indices))
        top_n_correct = int((predicted[top_indices] == y[top_indices]).sum())
        over_selected = selected & (predicted == 1)
        under_selected = selected & (predicted == 0)
        top_over = top_indices[predicted[top_indices] == 1]
        top_under = top_indices[predicted[top_indices] == 0]
        rows.append({
            "fold": fold,
            "model": model_name,
            "scope": scope,
            "confidence_cutoff": float(cutoff),
            "matches": count,
            "coverage": float(count / len(y)) if len(y) else 0.0,
            "accuracy": float((predicted[selected] == y[selected]).mean()) if count else np.nan,
            "mean_confidence": float(confidence[selected].mean()) if count else np.nan,
            "top_n_matches": top_n,
            "top_n_correct": top_n_correct,
            "top_n_accuracy": float(top_n_correct / top_n) if top_n else np.nan,
            "over_matches": int(over_selected.sum()),
            "over_correct": int((predicted[over_selected] == y[over_selected]).sum()),
            "over_accuracy": float((predicted[over_selected] == y[over_selected]).mean()) if over_selected.any() else np.nan,
            "under_matches": int(under_selected.sum()),
            "under_correct": int((predicted[under_selected] == y[under_selected]).sum()),
            "under_accuracy": float((predicted[under_selected] == y[under_selected]).mean()) if under_selected.any() else np.nan,
            "top_n_over": int(len(top_over)),
            "top_n_over_correct": int((predicted[top_over] == y[top_over]).sum()),
            "top_n_under": int(len(top_under)),
            "top_n_under_correct": int((predicted[top_under] == y[top_under]).sum()),
        })
    return rows



def build_prediction_diagnostic_rows(
    validation: pd.DataFrame,
    y: np.ndarray,
    model_probabilities: dict[str, np.ndarray],
    fold: int,
) -> list[dict]:
    """Keep row-level, strictly out-of-fold predictions for error analysis."""
    required = (
        "Date", "League", "HomeTeam", "AwayTeam",
        "lambda_home", "lambda_away",
        "home_history_count", "away_history_count",
        "home_for_mean", "away_for_mean",
        "home_recent_for", "away_recent_for",
        "home_recent_for_10", "away_recent_for_10",
        "home_recent_for_15", "away_recent_for_15",
    )
    missing = [column for column in required if column not in validation.columns]
    if missing:
        raise ValueError("Prediction diagnostics missing: " + ", ".join(missing))

    frame = validation.loc[:, required].reset_index(drop=True)
    actual = np.asarray(y, dtype=int)
    if len(frame) != len(actual):
        raise ValueError("Prediction diagnostics require aligned validation rows and labels.")

    total_lambda = (
        frame["lambda_home"].to_numpy(dtype=float)
        + frame["lambda_away"].to_numpy(dtype=float)
    )
    history_depth = np.minimum(
        frame["home_history_count"].to_numpy(dtype=float),
        frame["away_history_count"].to_numpy(dtype=float),
    )
    recent_delta = (
        frame["home_recent_for"].to_numpy(dtype=float)
        + frame["away_recent_for"].to_numpy(dtype=float)
        - frame["home_recent_for_15"].to_numpy(dtype=float)
        - frame["away_recent_for_15"].to_numpy(dtype=float)
    )
    recent_form = np.select(
        [recent_delta <= -0.15, recent_delta >= 0.15],
        ["FALLING", "RISING"],
        default="STABLE",
    )

    rows = []
    for model_name, raw_probabilities in model_probabilities.items():
        probabilities = np.asarray(raw_probabilities, dtype=float)
        if len(probabilities) != len(actual):
            raise ValueError(f"{model_name} predictions are not aligned with validation labels.")
        valid = np.isfinite(probabilities)
        if not valid.any():
            continue

        confidence = np.maximum(probabilities, 1.0 - probabilities)
        predicted = (probabilities >= 0.5).astype(int)
        confidence_band = pd.cut(
            confidence,
            bins=[0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.000001],
            labels=["<55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80%+"],
            include_lowest=True,
        ).astype(str)
        lambda_band = pd.cut(
            total_lambda,
            bins=[0.0, 2.0, 2.3, 2.6, 2.9, 3.2, 3.6, 4.2, np.inf],
            labels=["<2.0", "2.0-2.3", "2.3-2.6", "2.6-2.9", "2.9-3.2", "3.2-3.6", "3.6-4.2", "4.2+"],
            include_lowest=True,
        ).astype(str)
        history_band = pd.cut(
            history_depth,
            bins=[-0.001, 2, 5, 8, 12, np.inf],
            labels=["0-2", "3-5", "6-8", "9-12", "13+"],
        ).astype(str)

        for index in np.flatnonzero(valid):
            rows.append({
                "fold": int(fold),
                "model": model_name,
                "date": pd.Timestamp(frame.iloc[index]["Date"]).isoformat(),
                "league": str(frame.iloc[index]["League"]),
                "home_team": str(frame.iloc[index]["HomeTeam"]),
                "away_team": str(frame.iloc[index]["AwayTeam"]),
                "actual_over": int(actual[index]),
                "predicted_over": int(predicted[index]),
                "correct": int(predicted[index] == actual[index]),
                "probability_over": float(probabilities[index]),
                "confidence": float(confidence[index]),
                "predicted_side": "OVER" if predicted[index] else "UNDER",
                "confidence_band": str(confidence_band[index]),
                "total_lambda": float(total_lambda[index]),
                "total_lambda_band": str(lambda_band[index]),
                "history_depth": float(history_depth[index]),
                "history_band": str(history_band[index]),
                "recent_xg_5": float(frame.iloc[index]["home_recent_for"] + frame.iloc[index]["away_recent_for"]),
                "recent_xg_10": float(frame.iloc[index]["home_recent_for_10"] + frame.iloc[index]["away_recent_for_10"]),
                "recent_xg_15": float(frame.iloc[index]["home_recent_for_15"] + frame.iloc[index]["away_recent_for_15"]),
                "recent_form_delta": float(recent_delta[index]),
                "recent_form": str(recent_form[index]),
            })
    return rows


def aggregate_prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize out-of-fold errors by league, side, confidence and team context."""
    if predictions.empty:
        return pd.DataFrame(columns=[
            "fold", "model", "dimension", "bucket", "matches",
            "correct", "accuracy", "error_rate", "mean_confidence",
            "mean_total_lambda",
        ])

    dimensions = {
        "LEAGUE": "league",
        "SIDE": "predicted_side",
        "CONFIDENCE": "confidence_band",
        "TOTAL_LAMBDA": "total_lambda_band",
        "HISTORY_DEPTH": "history_band",
        "RECENT_FORM": "recent_form",
    }
    frames = []
    for dimension, column in dimensions.items():
        grouped = predictions.groupby(
            ["fold", "model", column],
            dropna=False,
            observed=True,
        ).agg(
            matches=("correct", "size"),
            correct=("correct", "sum"),
            mean_confidence=("confidence", "mean"),
            mean_total_lambda=("total_lambda", "mean"),
        ).reset_index().rename(columns={column: "bucket"})
        grouped["dimension"] = dimension
        grouped["accuracy"] = grouped["correct"] / grouped["matches"]
        grouped["error_rate"] = 1.0 - grouped["accuracy"]
        frames.append(grouped[[
            "fold", "model", "dimension", "bucket", "matches",
            "correct", "accuracy", "error_rate", "mean_confidence",
            "mean_total_lambda",
        ]])

    detail = pd.concat(frames, ignore_index=True)
    detail["confidence_sum"] = detail["mean_confidence"] * detail["matches"]
    detail["total_lambda_sum"] = detail["mean_total_lambda"] * detail["matches"]
    pooled = detail.loc[detail["fold"] != 0].groupby(
        ["model", "dimension", "bucket"],
        dropna=False,
        observed=True,
    ).agg(
        matches=("matches", "sum"),
        correct=("correct", "sum"),
        confidence_sum=("confidence_sum", "sum"),
        total_lambda_sum=("total_lambda_sum", "sum"),
    ).reset_index()
    pooled["mean_confidence"] = pooled["confidence_sum"] / pooled["matches"]
    pooled["mean_total_lambda"] = pooled["total_lambda_sum"] / pooled["matches"]
    pooled["fold"] = 0
    pooled["accuracy"] = pooled["correct"] / pooled["matches"]
    pooled["error_rate"] = 1.0 - pooled["accuracy"]
    pooled = pooled[detail.columns]
    detail = detail.drop(columns=["confidence_sum", "total_lambda_sum"])
    return pd.concat([detail, pooled], ignore_index=True)

def load_development_data(path: str | Path) -> pd.DataFrame:
    """Load and clean history, then cap it before feature generation."""
    frame = pd.read_csv(path, encoding="latin-1", low_memory=False)
    require_columns(frame)
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True, errors="coerce")

    for column in ("Home_xG", "Away_xG", "FTHG", "FTAG"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=[
        "Date", "League", "HomeTeam", "AwayTeam",
        "Home_xG", "Away_xG", "FTHG", "FTAG",
    ]).copy()

    frame = frame.sort_values(
        ["Date", "League", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(frame) < N_TRAIN:
        raise ValueError(
            f"Need at least {N_TRAIN} cleaned rows before the sealed holdout; found {len(frame)}."
        )

    # The historical development boundary is fixed at the original TRAIN size.
    # Do not build features or labels for calibration, policy-check or Final Test rows.
    return frame.iloc[:N_TRAIN].copy()


def run_walk_forward(
    csv_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    thresholds: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_development_data(csv_path)
    odds_provenance = build_odds_provenance_report(frame)
    frame = add_leakage_safe_features(frame)
    windows = build_fold_slices(len(frame))

    metric_rows = []
    policy_rows = []
    confidence_rows = []
    prediction_rows = []
    oof = {
        "y": [], "blend": [], "historical_xg": [], "market": [],
        "poisson": [], "poisson_cal": [], "poisson_platt": [], "goal_poisson": [], "xg_goal_blend": [], "negative_binomial": [], "hgb": [], "logistic": [],
    }
    oof_all = {
        "y": [], "historical_xg": [], "poisson": [], "poisson_cal": [], "poisson_platt": [], "goal_poisson": [], "xg_goal_blend": [], "negative_binomial": [], "hgb": [], "logistic": [],
    }

    for fold_number, (train_slice, calibration_slice, validation_slice) in enumerate(windows, 1):
        train = frame.iloc[train_slice].copy()
        calibration = frame.iloc[calibration_slice].copy()
        validation = frame.iloc[validation_slice].copy()

        blend_model = BlendModel()
        blend_model.fit(train)
        blend_model.calibrate(calibration)

        historical_model = HistoricalXGModel()
        historical_model.fit(train)
        historical_model.calibrate(calibration)

        hgb_model = HistGradientBoostingCandidate()
        hgb_model.fit(train)
        hgb_model.calibrate(calibration)

        logistic_model = LogisticOverUnderCandidate().fit(train)
        logistic_model.calibrate(calibration)

        negative_binomial_model = NegativeBinomialCandidate().fit(train)

        y = historical_model._y(validation)
        y_calibration = historical_model._y(calibration)
        p_blend = blend_model.predict_proba(validation)
        p_historical = historical_model.predict_proba(validation)
        p_hgb = hgb_model.predict_proba(validation)
        p_logistic = logistic_model.predict_proba(validation)
        p_poisson = validation["p_poisson"].to_numpy(dtype=float)
        p_goal_poisson = validation["p_poisson_goals"].to_numpy(dtype=float)
        p_xg_goal_blend = np.clip(0.5 * p_poisson + 0.5 * p_goal_poisson, 1e-6, 1.0 - 1e-6)
        p_negative_binomial = negative_binomial_model.predict_proba(validation)
        p_market, over_odds, under_odds, market_valid = market_data(validation)


        # Calibrated Poisson-only candidate: no bookmaker odds in its features.
        poisson_calibrator = IsotonicRegression(
            y_min=0.001,
            y_max=0.999,
            out_of_bounds="clip",
        )
        poisson_calibrator.fit(
            calibration["p_poisson"].to_numpy(dtype=float),
            y_calibration,
        )
        p_poisson_cal = poisson_calibrator.predict(p_poisson)
        p_poisson_platt = platt_scale_probabilities(
            calibration["p_poisson"].to_numpy(dtype=float),
            y_calibration,
            p_poisson,
        )

        prediction_rows.extend(build_prediction_diagnostic_rows(
            validation,
            y,
            {
                "BLEND_RESEARCH": p_blend,
                "HISTORICAL_XG": p_historical,
                "POISSON": p_poisson,
                "POISSON_CAL": p_poisson_cal,
                "POISSON_PLATT": p_poisson_platt,
                "GOAL_POISSON": p_goal_poisson,
                "XG_GOAL_BLEND": p_xg_goal_blend,
                "NEGATIVE_BINOMIAL": p_negative_binomial,
                "HIST_GB": p_hgb,
                "LOGISTIC": p_logistic,
            },
            fold_number,
        ))

        # Compare the market-dependent research blend with the production
        # historical-only model on identical rows where market prices exist.
        metric_rows.extend([
            _score_row(y, p_blend, market_valid, "BLEND_RESEARCH", fold_number),
            _score_row(y, p_historical, market_valid, "HISTORICAL_XG", fold_number),
            _score_row(y, p_market, market_valid, "MARKET", fold_number),
            _score_row(y, p_poisson, market_valid, "POISSON", fold_number),
            _score_row(y, p_poisson_cal, market_valid, "POISSON_CAL", fold_number),
            _score_row(y, p_poisson_platt, market_valid, "POISSON_PLATT", fold_number),
            _score_row(y, p_goal_poisson, market_valid, "GOAL_POISSON", fold_number),
            _score_row(y, p_xg_goal_blend, market_valid, "XG_GOAL_BLEND", fold_number),
            _score_row(y, p_negative_binomial, market_valid, "NEGATIVE_BINOMIAL", fold_number),
            _score_row(y, p_hgb, market_valid, "HIST_GB", fold_number),
            _score_row(y, p_logistic, market_valid, "LOGISTIC", fold_number),
            _score_row(y, p_historical, np.ones(len(y), dtype=bool), "HISTORICAL_XG", fold_number, "ALL_MATCHES"),
            _score_row(y, p_poisson, np.ones(len(y), dtype=bool), "POISSON", fold_number, "ALL_MATCHES"),
            _score_row(y, p_poisson_cal, np.ones(len(y), dtype=bool), "POISSON_CAL", fold_number, "ALL_MATCHES"),
            _score_row(y, p_poisson_platt, np.ones(len(y), dtype=bool), "POISSON_PLATT", fold_number, "ALL_MATCHES"),
            _score_row(y, p_goal_poisson, np.ones(len(y), dtype=bool), "GOAL_POISSON", fold_number, "ALL_MATCHES"),
            _score_row(y, p_xg_goal_blend, np.ones(len(y), dtype=bool), "XG_GOAL_BLEND", fold_number, "ALL_MATCHES"),
            _score_row(y, p_negative_binomial, np.ones(len(y), dtype=bool), "NEGATIVE_BINOMIAL", fold_number, "ALL_MATCHES"),
            _score_row(y, p_hgb, np.ones(len(y), dtype=bool), "HIST_GB", fold_number, "ALL_MATCHES"),
            _score_row(y, p_logistic, np.ones(len(y), dtype=bool), "LOGISTIC", fold_number, "ALL_MATCHES"),
        ])

        for model_name, probabilities in (
            ("BLEND_RESEARCH", p_blend),
            ("HISTORICAL_XG", p_historical),
            ("MARKET", p_market),
            ("POISSON", p_poisson),
            ("POISSON_CAL", p_poisson_cal),
            ("POISSON_PLATT", p_poisson_platt),
            ("GOAL_POISSON", p_goal_poisson),
            ("XG_GOAL_BLEND", p_xg_goal_blend),
            ("NEGATIVE_BINOMIAL", p_negative_binomial),
            ("HIST_GB", p_hgb),
            ("LOGISTIC", p_logistic),
        ):
            confidence_rows.extend(calculate_confidence_rows(
                y[market_valid],
                probabilities[market_valid],
                model_name,
                fold_number,
            ))

        for model_name, probabilities in (
            ("HISTORICAL_XG", p_historical),
            ("POISSON", p_poisson),
            ("POISSON_CAL", p_poisson_cal),
            ("POISSON_PLATT", p_poisson_platt),
            ("GOAL_POISSON", p_goal_poisson),
            ("XG_GOAL_BLEND", p_xg_goal_blend),
            ("NEGATIVE_BINOMIAL", p_negative_binomial),
            ("HIST_GB", p_hgb),
        ):
            confidence_rows.extend(calculate_confidence_rows(
                y,
                probabilities,
                model_name,
                fold_number,
                scope="ALL_MATCHES",
            ))

        for model_name, probabilities in (
            ("BLEND_RESEARCH", p_blend),
            ("HISTORICAL_XG", p_historical),
            ("POISSON_CAL", p_poisson_cal),
            ("POISSON_PLATT", p_poisson_platt),
            ("GOAL_POISSON", p_goal_poisson),
            ("XG_GOAL_BLEND", p_xg_goal_blend),
            ("NEGATIVE_BINOMIAL", p_negative_binomial),
            ("HIST_GB", p_hgb),
        ):
            for threshold in thresholds:
                policy_rows.extend(calculate_policy_rows(
                    y=y,
                    p_model=probabilities,
                    p_market=p_market,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    market_valid=market_valid,
                    threshold=float(threshold),
                    fold=fold_number,
                    model_name=model_name,
                ))

        valid_indices = np.flatnonzero(market_valid)
        oof["y"].extend(y[valid_indices].tolist())
        oof["blend"].extend(p_blend[valid_indices].tolist())
        oof["historical_xg"].extend(p_historical[valid_indices].tolist())
        oof["market"].extend(p_market[valid_indices].tolist())
        oof["poisson"].extend(p_poisson[valid_indices].tolist())
        oof["poisson_cal"].extend(p_poisson_cal[valid_indices].tolist())
        oof["poisson_platt"].extend(p_poisson_platt[valid_indices].tolist())
        oof["goal_poisson"].extend(p_goal_poisson[valid_indices].tolist())
        oof["xg_goal_blend"].extend(p_xg_goal_blend[valid_indices].tolist())
        oof["negative_binomial"].extend(p_negative_binomial[valid_indices].tolist())
        oof["hgb"].extend(p_hgb[valid_indices].tolist())
        oof["logistic"].extend(p_logistic[valid_indices].tolist())
        oof_all["y"].extend(y.tolist())
        oof_all["historical_xg"].extend(p_historical.tolist())
        oof_all["poisson"].extend(p_poisson.tolist())
        oof_all["poisson_cal"].extend(p_poisson_cal.tolist())
        oof_all["poisson_platt"].extend(p_poisson_platt.tolist())
        oof_all["goal_poisson"].extend(p_goal_poisson.tolist())
        oof_all["xg_goal_blend"].extend(p_xg_goal_blend.tolist())
        oof_all["negative_binomial"].extend(p_negative_binomial.tolist())
        oof_all["hgb"].extend(p_hgb.tolist())
        oof_all["logistic"].extend(p_logistic.tolist())

        print(
            f"Fold {fold_number:02d}: train={len(train)} "
            f"calibration={len(calibration)} validation={len(validation)} "
            f"valid_market={int(market_valid.sum())}"
        )

    y_all = np.asarray(oof["y"], dtype=int)
    all_mask = np.ones(len(y_all), dtype=bool)
    metric_rows.extend([
        _score_row(y_all, np.asarray(oof["blend"]), all_mask, "BLEND_RESEARCH", 0),
        _score_row(y_all, np.asarray(oof["historical_xg"]), all_mask, "HISTORICAL_XG", 0),
        _score_row(y_all, np.asarray(oof["market"]), all_mask, "MARKET", 0),
        _score_row(y_all, np.asarray(oof["poisson"]), all_mask, "POISSON", 0),
        _score_row(y_all, np.asarray(oof["poisson_cal"]), all_mask, "POISSON_CAL", 0),
        _score_row(y_all, np.asarray(oof["poisson_platt"]), all_mask, "POISSON_PLATT", 0),
        _score_row(y_all, np.asarray(oof["goal_poisson"]), all_mask, "GOAL_POISSON", 0),
        _score_row(y_all, np.asarray(oof["xg_goal_blend"]), all_mask, "XG_GOAL_BLEND", 0),
        _score_row(y_all, np.asarray(oof["negative_binomial"]), all_mask, "NEGATIVE_BINOMIAL", 0),
        _score_row(y_all, np.asarray(oof["hgb"]), all_mask, "HIST_GB", 0),
        _score_row(y_all, np.asarray(oof["logistic"]), all_mask, "LOGISTIC", 0),
    ])

    y_all_matches = np.asarray(oof_all["y"], dtype=int)
    all_match_mask = np.ones(len(y_all_matches), dtype=bool)
    metric_rows.extend([
        _score_row(y_all_matches, np.asarray(oof_all["historical_xg"]), all_match_mask, "HISTORICAL_XG", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["poisson"]), all_match_mask, "POISSON", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["poisson_cal"]), all_match_mask, "POISSON_CAL", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["poisson_platt"]), all_match_mask, "POISSON_PLATT", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["goal_poisson"]), all_match_mask, "GOAL_POISSON", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["xg_goal_blend"]), all_match_mask, "XG_GOAL_BLEND", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["negative_binomial"]), all_match_mask, "NEGATIVE_BINOMIAL", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["hgb"]), all_match_mask, "HIST_GB", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["logistic"]), all_match_mask, "LOGISTIC", 0, "ALL_MATCHES"),
    ])

    metrics = pd.DataFrame(metric_rows)
    policies = pd.DataFrame(policy_rows)

    for model_name, probabilities in (
        ("BLEND_RESEARCH", np.asarray(oof["blend"], dtype=float)),
        ("HISTORICAL_XG", np.asarray(oof["historical_xg"], dtype=float)),
        ("MARKET", np.asarray(oof["market"], dtype=float)),
        ("POISSON", np.asarray(oof["poisson"], dtype=float)),
        ("POISSON_CAL", np.asarray(oof["poisson_cal"], dtype=float)),
        ("POISSON_PLATT", np.asarray(oof["poisson_platt"], dtype=float)),
        ("GOAL_POISSON", np.asarray(oof["goal_poisson"], dtype=float)),
        ("XG_GOAL_BLEND", np.asarray(oof["xg_goal_blend"], dtype=float)),
        ("NEGATIVE_BINOMIAL", np.asarray(oof["negative_binomial"], dtype=float)),
        ("HIST_GB", np.asarray(oof["hgb"], dtype=float)),
        ("LOGISTIC", np.asarray(oof["logistic"], dtype=float)),
    ):
        confidence_rows.extend(calculate_confidence_rows(
            y_all,
            probabilities,
            model_name,
            0,
        ))

    for model_name, probabilities in (
        ("HISTORICAL_XG", np.asarray(oof_all["historical_xg"], dtype=float)),
        ("POISSON", np.asarray(oof_all["poisson"], dtype=float)),
        ("POISSON_CAL", np.asarray(oof_all["poisson_cal"], dtype=float)),
        ("POISSON_PLATT", np.asarray(oof_all["poisson_platt"], dtype=float)),
        ("GOAL_POISSON", np.asarray(oof_all["goal_poisson"], dtype=float)),
        ("XG_GOAL_BLEND", np.asarray(oof_all["xg_goal_blend"], dtype=float)),
        ("NEGATIVE_BINOMIAL", np.asarray(oof_all["negative_binomial"], dtype=float)),
        ("HIST_GB", np.asarray(oof_all["hgb"], dtype=float)),
        ("LOGISTIC", np.asarray(oof_all["logistic"], dtype=float)),
    ):
        confidence_rows.extend(calculate_confidence_rows(
            y_all_matches,
            probabilities,
            model_name,
            0,
            scope="ALL_MATCHES",
        ))

    confidence = pd.DataFrame(confidence_rows)
    prediction_diagnostics = pd.DataFrame(prediction_rows)
    error_analysis = aggregate_prediction_diagnostics(prediction_diagnostics)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
    policies.to_csv(output_dir / "walk_forward_policy_roi.csv", index=False, encoding="utf-8-sig")
    confidence.to_csv(output_dir / "walk_forward_confidence.csv", index=False, encoding="utf-8-sig")
    prediction_diagnostics.to_csv(output_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    error_analysis.to_csv(output_dir / "walk_forward_error_analysis.csv", index=False, encoding="utf-8-sig")
    odds_provenance.to_csv(output_dir / "walk_forward_odds_provenance.csv", index=False, encoding="utf-8-sig")

    print("\nOOF metrics (fold=0 is pooled out-of-fold):")
    print(metrics.loc[metrics["fold"] == 0].to_string(index=False))
    print("\nOOF selective accuracy (fold=0 is pooled out-of-fold):")
    print(confidence.loc[confidence["fold"] == 0].to_string(index=False))
    print("\\nPooled out-of-fold error analysis (fold=0):")
    print(error_analysis.loc[error_analysis["fold"] == 0].sort_values(["model", "dimension", "matches"], ascending=[True, True, False]).to_string(index=False))
    print("\\nOdds provenance audit (TRAIN rows only):")
    print(odds_provenance.to_string(index=False))
    print(f"\nReports saved to: {output_dir.resolve()}")
    return metrics, policies, confidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="xgo.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run_walk_forward(args.csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
