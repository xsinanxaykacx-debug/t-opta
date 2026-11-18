"""
Independent diagnostics layer for BOT.py.

This module observes an already-fitted model. It does not fit,
calibrate, tune, or mutate the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss, log_loss


DEFAULT_OUTPUT_DIR = Path("reports/diagnostics")


def _bet_rows(y, over_odds, under_odds, market_valid, edge, threshold):
    over_mask = (
        market_valid
        & (edge > threshold)
        & np.isfinite(over_odds)
        & (over_odds > 1.0)
    )
    under_mask = (
        market_valid
        & (edge < -threshold)
        & np.isfinite(under_odds)
        & (under_odds > 1.0)
    )

    over = pd.DataFrame({
        "side": "OVER",
        "odds": over_odds[over_mask],
        "edge": edge[over_mask],
        "won": (y[over_mask] == 1).astype(int),
    })
    under = pd.DataFrame({
        "side": "UNDER",
        "odds": under_odds[under_mask],
        "edge": np.abs(edge[under_mask]),
        "won": (y[under_mask] == 0).astype(int),
    })

    return pd.concat([over, under], ignore_index=True)


def directional_roi_report(y, p_model, bets):
    rows = []

    for side in ("OVER", "UNDER"):
        if side == "OVER":
            prediction_mask = p_model >= 0.5
        else:
            prediction_mask = p_model < 0.5

        side_bets = bets.loc[bets["side"] == side].copy()
        n_bets = len(side_bets)

        if n_bets:
            profits = np.where(
                side_bets["won"].to_numpy(dtype=int) == 1,
                side_bets["odds"].to_numpy(dtype=float) - 1.0,
                -1.0,
            )
            net_profit = float(profits.sum())
            hit_rate = float(side_bets["won"].mean())
            avg_odds = float(side_bets["odds"].mean())
            avg_edge = float(side_bets["edge"].mean())
            roi_pct = net_profit / n_bets * 100.0
        else:
            net_profit = 0.0
            hit_rate = np.nan
            avg_odds = np.nan
            avg_edge = np.nan
            roi_pct = np.nan

        if prediction_mask.any():
            classification_accuracy = float(
                (y[prediction_mask] == (1 if side == "OVER" else 0)).mean()
            )
        else:
            classification_accuracy = np.nan

        rows.append({
            "side": side,
            "predicted_matches": int(prediction_mask.sum()),
            "classification_accuracy": classification_accuracy,
            "bets": n_bets,
            "bet_hit_rate": hit_rate,
            "avg_odds": avg_odds,
            "avg_abs_edge": avg_edge,
            "roi_pct": roi_pct,
            "net_profit_units": net_profit,
        })

    return pd.DataFrame(rows)


def market_model_brier_report(y, p_model, p_market, market_valid):
    # Compare both forecasts on exactly the same valid-market observations.
    mask = (
        market_valid
        & np.isfinite(p_market)
        & np.isfinite(p_model)
    )
    n = int(mask.sum())

    if n == 0:
        return pd.DataFrame([{
            "matches": 0,
            "model_brier": np.nan,
            "market_brier": np.nan,
            "delta_brier_model_minus_market": np.nan,
            "model_logloss": np.nan,
            "market_logloss": np.nan,
        }])

    y_eval = y[mask]
    model_p = np.clip(p_model[mask], 1e-6, 1.0 - 1e-6)
    market_p = np.clip(p_market[mask], 1e-6, 1.0 - 1e-6)

    model_brier = float(brier_score_loss(y_eval, model_p))
    market_brier = float(brier_score_loss(y_eval, market_p))

    return pd.DataFrame([{
        "matches": n,
        "model_brier": model_brier,
        "market_brier": market_brier,
        "delta_brier_model_minus_market": model_brier - market_brier,
        "model_logloss": float(log_loss(y_eval, model_p, labels=[0, 1])),
        "market_logloss": float(log_loss(y_eval, market_p, labels=[0, 1])),
    }])


def _segment_report(bets, value_column, edges, labels):
    columns = [
        "side", "bucket", "bets", "hit_rate", "avg_odds",
        "avg_edge", "roi_pct", "net_profit_units",
    ]

    if bets.empty:
        return pd.DataFrame(columns=columns)

    work = bets.copy()
    work["bucket"] = pd.cut(
        work[value_column],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    work = work.dropna(subset=["bucket"])

    if work.empty:
        return pd.DataFrame(columns=columns)

    work["profit"] = np.where(
        work["won"].to_numpy(dtype=int) == 1,
        work["odds"].to_numpy(dtype=float) - 1.0,
        -1.0,
    )

    result = work.groupby(
        ["side", "bucket"],
        observed=True,
        sort=True,
    ).agg(
        bets=("won", "size"),
        hit_rate=("won", "mean"),
        avg_odds=("odds", "mean"),
        avg_edge=("edge", "mean"),
        net_profit_units=("profit", "sum"),
    ).reset_index()

    result["roi_pct"] = (
        result["net_profit_units"] / result["bets"] * 100.0
    )
    return result[columns]


def odds_bucket_report(bets):
    return _segment_report(
        bets=bets,
        value_column="odds",
        edges=[1.0, 1.30, 1.50, 1.70, 2.00, 2.50, np.inf],
        labels=[
            "1.01-1.29",
            "1.30-1.49",
            "1.50-1.69",
            "1.70-1.99",
            "2.00-2.49",
            "2.50+",
        ],
    )


def edge_bucket_report(bets, threshold):
    # Keep the requested standard boundaries while supporting a higher
    # selected threshold without creating invalid or duplicate intervals.
    boundaries = sorted({
        float(threshold),
        *(value for value in (0.05, 0.08, 0.10) if value > threshold),
    })
    edges = boundaries + [np.inf]
    labels = [
        f"{edges[i]:.2f}-{edges[i + 1]:.2f}"
        if np.isfinite(edges[i + 1])
        else f"{edges[i]:.2f}+"
        for i in range(len(edges) - 1)
    ]

    return _segment_report(
        bets=bets,
        value_column="edge",
        edges=edges,
        labels=labels,
    )


def reliability_report(y, p_model, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.minimum(
        np.digitize(p_model, edges[1:-1], right=False),
        bins - 1,
    )

    frame = pd.DataFrame({
        "bin": bin_index,
        "predicted_over": p_model,
        "observed_over": y,
    })

    result = frame.groupby("bin", sort=True).agg(
        matches=("observed_over", "size"),
        mean_predicted_over=("predicted_over", "mean"),
        observed_over_rate=("observed_over", "mean"),
    ).reset_index()

    result["bin_lower"] = result["bin"] / bins
    result["bin_upper"] = (result["bin"] + 1) / bins
    result["calibration_gap"] = (
        result["observed_over_rate"] - result["mean_predicted_over"]
    )
    result["absolute_gap"] = result["calibration_gap"].abs()
    result["ece_contribution"] = (
        result["matches"] / len(y) * result["absolute_gap"]
    )
    result["overall_ece"] = float(result["ece_contribution"].sum())

    return result


def run_diagnostics(
    df,
    model,
    label,
    threshold,
    market_data_fn,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Generate diagnostic CSVs from an already fitted/calibrated BOT model.
    market_data_fn must be BOT.py's existing market_data function.
    """

    if threshold is None or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Diagnostics requires a positive finite edge threshold.")

    p_model = np.asarray(model.predict_proba(df), dtype=float)
    y = np.asarray(model._y(df), dtype=int)

    p_market, over_odds, under_odds, market_valid = market_data_fn(df)

    p_market = np.asarray(p_market, dtype=float)
    over_odds = np.asarray(over_odds, dtype=float)
    under_odds = np.asarray(under_odds, dtype=float)
    market_valid = np.asarray(market_valid, dtype=bool)

    lengths = {
        len(df), len(y), len(p_model), len(p_market),
        len(over_odds), len(under_odds), len(market_valid),
    }
    if len(lengths) != 1:
        raise ValueError("Diagnostics arrays do not match dataframe length.")

    if not np.isfinite(p_model).all():
        raise ValueError("Model predictions contain NaN or infinite values.")

    p_model = np.clip(p_model, 1e-6, 1.0 - 1e-6)
    edge = p_model - p_market

    bets = _bet_rows(
        y=y,
        over_odds=over_odds,
        under_odds=under_odds,
        market_valid=market_valid,
        edge=edge,
        threshold=float(threshold),
    )

    reports = {
        "directional_roi": directional_roi_report(y, p_model, bets),
        "market_model_brier": market_model_brier_report(
            y, p_model, p_market, market_valid
        ),
        "odds_buckets": odds_bucket_report(bets),
        "edge_buckets": edge_bucket_report(bets, float(threshold)),
        "reliability_bins": reliability_report(y, p_model, bins=10),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_label = (
        str(label).strip().lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    for report_name, report in reports.items():
        report.to_csv(
            output_dir / f"{safe_label}_{report_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    brier_delta = reports["market_model_brier"].iloc[0][
        "delta_brier_model_minus_market"
    ]

    print(f"\n[{label}] diagnostics saved to: {output_dir.resolve()}")
    if np.isfinite(brier_delta):
        print(
            "Brier delta (model - market): "
            f"{brier_delta:.6f} "
            "(- = model better; + = market better)"
        )
    else:
        print("Brier comparison unavailable: no valid market odds.")

    return reports
