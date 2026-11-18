"""Create odds-independent Over/Under 2.5 picks for future fixtures.

Usage:
    python scripts/predict_upcoming.py xgo.csv upcoming.csv --output predictions.csv

The predictor calibrates a blend of long-term and recent opponent-adjusted
Poisson probabilities on the latest 500 historical matches. It does not read
bookmaker odds and never uses same-calendar-day match results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BOT import (  # noqa: E402
    PoissonOnlyModel,
    add_leakage_safe_features,
    require_columns,
)


UPCOMING_REQUIRED = ("Date", "League", "HomeTeam", "AwayTeam")
RESULT_COLUMNS = ("Home_xG", "Away_xG", "FTHG", "FTAG")


def build_prediction_table(
    upcoming_features: pd.DataFrame,
    confidence_cutoff: float = 0.70,
    top_n: int = 10,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Rank future fixtures and mark only the top-N above the fixed confidence gate."""
    if not 0.5 < confidence_cutoff < 1.0:
        raise ValueError("confidence_cutoff must be greater than 0.5 and less than 1.0.")
    if isinstance(top_n, bool) or int(top_n) != top_n or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    missing = [column for column in UPCOMING_REQUIRED if column not in upcoming_features.columns]
    if missing:
        raise ValueError("Upcoming fixtures missing: " + ", ".join(missing))
    if "p_poisson" not in upcoming_features.columns:
        raise ValueError("Upcoming features do not contain p_poisson.")

    if probabilities is None:
        probabilities = PoissonOnlyModel.predict_proba(upcoming_features)
    else:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 1 or len(probabilities) != len(upcoming_features):
            raise ValueError("Custom probabilities must be a 1D array aligned with upcoming fixtures.")
        if not np.isfinite(probabilities).all() or ((probabilities <= 0.0) | (probabilities >= 1.0)).any():
            raise ValueError("Custom probabilities must be finite and strictly between 0 and 1.")
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    side = np.where(probabilities >= 0.5, "OVER", "UNDER")
    side_probability = np.maximum(probabilities, 1.0 - probabilities)

    result = upcoming_features.loc[:, list(UPCOMING_REQUIRED)].copy()
    result["lambda_home"] = upcoming_features["lambda_home"].to_numpy(dtype=float)
    result["lambda_away"] = upcoming_features["lambda_away"].to_numpy(dtype=float)
    result["p_over"] = probabilities
    result["p_under"] = 1.0 - probabilities
    result["predicted_side"] = side
    result["confidence"] = confidence
    result["side_probability"] = side_probability
    result["confidence_rank"] = result["confidence"].rank(
        method="first",
        ascending=False,
    ).astype(int)

    eligible = result.loc[result["confidence"] >= confidence_cutoff].sort_values(
        ["confidence", "Date"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected_indices = eligible.head(int(top_n)).index
    result["status"] = np.where(result.index.isin(selected_indices), "PICK", "NO_PICK")
    result["confidence_cutoff"] = float(confidence_cutoff)
    result["top_n_limit"] = int(top_n)

    result["_pick_order"] = np.where(result["status"] == "PICK", 0, 1)
    return result.sort_values(
        ["_pick_order", "confidence_rank"],
        ascending=[True, True],
        kind="mergesort",
    ).drop(columns=["_pick_order"]).reset_index(drop=True)


def _prepare_upcoming_feature_sets(
    history: pd.DataFrame,
    upcoming: pd.DataFrame,
    calibration_size: int | None = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build future and calibration rows from one strictly chronological feature pass."""
    history = history.copy()
    upcoming = upcoming.copy()

    require_columns(history)
    missing = [column for column in UPCOMING_REQUIRED if column not in upcoming.columns]
    if missing:
        raise ValueError("Upcoming fixtures missing: " + ", ".join(missing))

    history["Date"] = pd.to_datetime(history["Date"], utc=True, errors="coerce")
    upcoming["Date"] = pd.to_datetime(upcoming["Date"], utc=True, errors="coerce")

    if history["Date"].isna().any() or upcoming["Date"].isna().any():
        raise ValueError("Invalid Date value in history or upcoming fixtures.")
    if upcoming.empty:
        raise ValueError("Upcoming fixture file is empty.")
    if calibration_size is not None and (
        isinstance(calibration_size, bool) or int(calibration_size) < 100
    ):
        raise ValueError("calibration_size must be an integer of at least 100.")

    for column in ("Home_xG", "Away_xG", "FTHG", "FTAG"):
        history[column] = pd.to_numeric(history[column], errors="coerce")

    history = history.dropna(subset=[
        "Date", "League", "HomeTeam", "AwayTeam",
        "Home_xG", "Away_xG", "FTHG", "FTAG",
    ]).copy()

    first_upcoming_day = upcoming["Date"].dt.normalize().min()
    history = history.loc[
        history["Date"].dt.normalize() < first_upcoming_day
    ].copy()

    if calibration_size is not None and len(history) < int(calibration_size):
        raise ValueError(
            f"Need at least {calibration_size} historical matches for recent-model calibration; "
            f"found {len(history)}."
        )

    for column in RESULT_COLUMNS:
        upcoming[column] = np.nan

    history["_is_upcoming"] = False
    upcoming["_is_upcoming"] = True

    combined = pd.concat(
        [history, upcoming],
        ignore_index=True,
        sort=False,
    ).sort_values(
        ["Date", "League", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    ).reset_index(drop=True)

    featured = add_leakage_safe_features(combined)
    future = featured.loc[featured["_is_upcoming"]].copy()
    historical_features = featured.loc[~featured["_is_upcoming"]].copy()
    calibration = (
        historical_features.tail(int(calibration_size)).copy()
        if calibration_size is not None
        else historical_features.iloc[0:0].copy()
    )

    if len(future) != len(upcoming):
        raise RuntimeError("Not all upcoming fixtures received prediction features.")
    if calibration_size is not None and len(calibration) != int(calibration_size):
        raise RuntimeError("Recent-model calibration partition has an unexpected size.")

    return future, calibration


def prepare_upcoming_features(
    history: pd.DataFrame,
    upcoming: pd.DataFrame,
) -> pd.DataFrame:
    """Build future fixture features using only history before upcoming calendar days."""
    future, _ = _prepare_upcoming_feature_sets(history, upcoming, calibration_size=None)
    return future


def predict_upcoming(
    history_path: str | Path,
    upcoming_path: str | Path,
    output_path: str | Path = "predictions.csv",
    confidence_cutoff: float = 0.70,
    top_n: int = 10,
) -> pd.DataFrame:
    history = pd.read_csv(history_path, encoding="latin-1", low_memory=False)
    upcoming = pd.read_csv(upcoming_path, encoding="utf-8-sig", low_memory=False)

    future_features, calibration_features = _prepare_upcoming_feature_sets(
        history,
        upcoming,
        calibration_size=500,
    )
    recent_model = RecentPoissonBlendCandidate().calibrate(calibration_features)
    future_probabilities = recent_model.predict_proba(future_features)
    predictions = build_prediction_table(
        future_features,
        confidence_cutoff=confidence_cutoff,
        top_n=top_n,
        probabilities=future_probabilities,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")

    picks = predictions.loc[predictions["status"] == "PICK"]
    print(f"Fixtures scanned : {len(predictions)}")
    print(f"Selected picks   : {len(picks)} / {top_n} maximum")
    print(f"Confidence gate  : {confidence_cutoff:.0%}")
    if picks.empty:
        print("No fixture passed the confidence gate. No forced picks were added.")
    else:
        print(picks[[
            "Date", "League", "HomeTeam", "AwayTeam",
            "predicted_side", "side_probability", "confidence",
        ]].to_string(index=False))
    print(f"Predictions saved: {output_path.resolve()}")
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", help="Historical xG dataset, e.g. xgo.csv")
    parser.add_argument("upcoming", help="CSV with Date, League, HomeTeam, AwayTeam")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    predict_upcoming(
        args.history,
        args.upcoming,
        args.output,
        confidence_cutoff=args.confidence,
        top_n=args.top,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
