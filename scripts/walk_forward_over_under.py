"""Expanding-window Over/Under 2.5 research using pre-Final-Test history only.

This script deliberately caps development data at BOT.N_TRAIN rows. It never
loads or evaluates the policy validation partitions or the sealed Final Test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

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
        })
    return rows


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
    frame = add_leakage_safe_features(frame)
    windows = build_fold_slices(len(frame))

    metric_rows = []
    policy_rows = []
    confidence_rows = []
    oof = {
        "y": [], "blend": [], "historical_xg": [], "market": [],
        "poisson": [], "poisson_cal": [],
    }
    oof_all = {
        "y": [], "historical_xg": [], "poisson": [], "poisson_cal": [],
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

        y = historical_model._y(validation)
        y_calibration = historical_model._y(calibration)
        p_blend = blend_model.predict_proba(validation)
        p_historical = historical_model.predict_proba(validation)
        p_poisson = validation["p_poisson"].to_numpy(dtype=float)
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

        # Compare the market-dependent research blend with the production
        # historical-only model on identical rows where market prices exist.
        metric_rows.extend([
            _score_row(y, p_blend, market_valid, "BLEND_RESEARCH", fold_number),
            _score_row(y, p_historical, market_valid, "HISTORICAL_XG", fold_number),
            _score_row(y, p_market, market_valid, "MARKET", fold_number),
            _score_row(y, p_poisson, market_valid, "POISSON", fold_number),
            _score_row(y, p_poisson_cal, market_valid, "POISSON_CAL", fold_number),
            _score_row(y, p_historical, np.ones(len(y), dtype=bool), "HISTORICAL_XG", fold_number, "ALL_MATCHES"),
            _score_row(y, p_poisson, np.ones(len(y), dtype=bool), "POISSON", fold_number, "ALL_MATCHES"),
            _score_row(y, p_poisson_cal, np.ones(len(y), dtype=bool), "POISSON_CAL", fold_number, "ALL_MATCHES"),
        ])

        for model_name, probabilities in (
            ("BLEND_RESEARCH", p_blend),
            ("HISTORICAL_XG", p_historical),
            ("MARKET", p_market),
            ("POISSON", p_poisson),
            ("POISSON_CAL", p_poisson_cal),
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
        oof_all["y"].extend(y.tolist())
        oof_all["historical_xg"].extend(p_historical.tolist())
        oof_all["poisson"].extend(p_poisson.tolist())
        oof_all["poisson_cal"].extend(p_poisson_cal.tolist())

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
    ])

    y_all_matches = np.asarray(oof_all["y"], dtype=int)
    all_match_mask = np.ones(len(y_all_matches), dtype=bool)
    metric_rows.extend([
        _score_row(y_all_matches, np.asarray(oof_all["historical_xg"]), all_match_mask, "HISTORICAL_XG", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["poisson"]), all_match_mask, "POISSON", 0, "ALL_MATCHES"),
        _score_row(y_all_matches, np.asarray(oof_all["poisson_cal"]), all_match_mask, "POISSON_CAL", 0, "ALL_MATCHES"),
    ])

    metrics = pd.DataFrame(metric_rows)
    policies = pd.DataFrame(policy_rows)

    for model_name, probabilities in (
        ("BLEND_RESEARCH", np.asarray(oof["blend"], dtype=float)),
        ("HISTORICAL_XG", np.asarray(oof["historical_xg"], dtype=float)),
        ("MARKET", np.asarray(oof["market"], dtype=float)),
        ("POISSON", np.asarray(oof["poisson"], dtype=float)),
        ("POISSON_CAL", np.asarray(oof["poisson_cal"], dtype=float)),
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
    ):
        confidence_rows.extend(calculate_confidence_rows(
            y_all_matches,
            probabilities,
            model_name,
            0,
            scope="ALL_MATCHES",
        ))

    confidence = pd.DataFrame(confidence_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "walk_forward_metrics.csv", index=False, encoding="utf-8-sig")
    policies.to_csv(output_dir / "walk_forward_policy_roi.csv", index=False, encoding="utf-8-sig")
    confidence.to_csv(output_dir / "walk_forward_confidence.csv", index=False, encoding="utf-8-sig")

    print("\nOOF metrics (fold=0 is pooled out-of-fold):")
    print(metrics.loc[metrics["fold"] == 0].to_string(index=False))
    print("\nOOF selective accuracy (fold=0 is pooled out-of-fold):")
    print(confidence.loc[confidence["fold"] == 0].to_string(index=False))
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
