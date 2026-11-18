"""Predict O/U 2.5 probabilities for upcoming fixtures.

The selected policy is confidence >= the frozen threshold and valid
pre-match market odds. This is an accuracy-oriented research model, not a
claim of profitable betting performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from BOT import market_data
from experiments.history_features import build_upcoming_feature_frame


def predict_fixtures(
    fixtures: pd.DataFrame,
    model: Any,
    metadata: dict[str, Any],
    *,
    history_path: str | Path = "xg18.csv",
) -> pd.DataFrame:
    """Return probabilities, confidence selection, odds, and diagnostics."""
    threshold = float(metadata["confidence_threshold"])
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.50 and 1.00.")

    featured, history_metadata = build_upcoming_feature_frame(
        fixtures,
        history_path=history_path,
    )
    probability = np.asarray(model.predict_proba(featured), dtype=float)
    if probability.shape != (len(featured),):
        raise ValueError("Model returned an invalid probability vector.")
    if not np.isfinite(probability).all():
        raise ValueError("Model returned non-finite probabilities.")

    p_market, over_odds, under_odds, market_valid = market_data(featured)
    p_market = np.asarray(p_market, dtype=float)
    over_odds = np.asarray(over_odds, dtype=float)
    under_odds = np.asarray(under_odds, dtype=float)
    market_valid = np.asarray(market_valid, dtype=bool)

    confidence = np.maximum(probability, 1.0 - probability)
    predicts_over = probability >= 0.5
    selected = market_valid & (confidence >= threshold)
    chosen_odds = np.where(predicts_over, over_odds, under_odds)
    side_probability = np.where(predicts_over, probability, 1.0 - probability)
    expected_value = side_probability * chosen_odds - 1.0

    out = featured.copy()
    out["p_market_over"] = p_market
    out["p_poisson_over"] = pd.to_numeric(out["p_poisson"], errors="coerce")
    out["p_odds_aware_over"] = probability
    out["predicted_side"] = np.where(predicts_over, "OVER 2.5", "UNDER 2.5")
    out["side_probability"] = side_probability
    out["confidence"] = confidence
    out["market_valid"] = market_valid
    out["selected"] = selected
    out["chosen_odds"] = chosen_odds
    out["expected_value_per_unit"] = expected_value
    out["selection_threshold"] = threshold
    out["home_history_matches"] = pd.to_numeric(
        out["hist_home_matches_available"], errors="coerce"
    ).fillna(0).astype(int)
    out["away_history_matches"] = pd.to_numeric(
        out["hist_away_matches_available"], errors="coerce"
    ).fillna(0).astype(int)
    out["history_date_max"] = history_metadata["history_date_max"]
    out["model_name"] = str(metadata.get("model_name", "odds_aware"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe O/U 2.5 predictions for upcoming fixtures."
    )
    parser.add_argument("fixtures_csv", help="CSV of upcoming fixtures and pre-match odds.")
    parser.add_argument(
        "--history", default="xg18.csv", help="Pinned xg18 historical source CSV."
    )
    parser.add_argument(
        "--model",
        default="artifacts/odds_aware_model.joblib",
        help="Trained Odds-Aware model artifact.",
    )
    parser.add_argument(
        "--metadata",
        default="artifacts/odds_aware_model.json",
        help="Model metadata JSON.",
    )
    parser.add_argument(
        "--output", default="predictions.csv", help="Output predictions CSV."
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    metadata_path = Path(args.metadata)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run the production-model build workflow first."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    fixtures = pd.read_csv(args.fixtures_csv, encoding="utf-8-sig", low_memory=False)
    model = joblib.load(model_path)
    predictions = predict_fixtures(
        fixtures,
        model,
        metadata,
        history_path=args.history,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")

    selected = predictions.loc[predictions["selected"]]
    print(f"Fixtures: {len(predictions)}")
    print(f"Market-valid: {int(predictions['market_valid'].sum())}")
    print(f"Selected at confidence >= {metadata['confidence_threshold']:.3f}: {len(selected)}")
    print(f"Predictions written: {output_path}")
    print(
        "Warning: historical Final Test accuracy was 66.00%, but selective ROI "
        "was -5.91%. This is not evidence of profitable betting."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
