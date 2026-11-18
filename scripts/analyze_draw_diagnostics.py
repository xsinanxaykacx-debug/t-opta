"""Diagnose draw probabilities, Logistic Regression weights, and draw thresholds.

The 2024/25 calibration season is split chronologically:
60% temperature-fit, 20% threshold-selection, 20% threshold-confirmation.
The 2025/26+ test partition is never used by this diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.feature_engine import FeatureEngine
from modules.historical_engine import HistoricalEngine
from models.logistic_baseline import LogisticBaseline


def season_start_year(value):
    import re
    match = re.search(r"(20\d{2})\s*[/\-]\s*(?:20)?\d{2}", str(value).strip())
    return int(match.group(1)) if match else None


def labels_for(frame):
    home = frame["FTHG"].to_numpy(dtype=float)
    away = frame["FTAG"].to_numpy(dtype=float)
    return np.where(home > away, 0, np.where(home == away, 1, 2)).astype(int)


def split_calibration(frame):
    ordered = frame.sort_values(["Date", "MatchID"], kind="stable").reset_index(drop=True)
    days = np.array(sorted(pd.to_datetime(ordered["Date"], utc=True).dt.normalize().unique()))
    if len(days) < 10:
        raise ValueError("Calibration partition is too small for nested chronological diagnostics.")
    fit_cut = days[max(1, int(len(days) * 0.60))]
    select_cut = days[max(2, int(len(days) * 0.80))]
    dates = pd.to_datetime(ordered["Date"], utc=True).dt.normalize()
    fit = ordered.loc[dates < fit_cut].copy()
    select = ordered.loc[(dates >= fit_cut) & (dates < select_cut)].copy()
    confirm = ordered.loc[dates >= select_cut].copy()
    if min(len(fit), len(select), len(confirm)) == 0:
        raise ValueError("Nested calibration split produced an empty partition.")
    return fit, select, confirm


def summarize_probabilities(labels, probabilities):
    names = ("HOME", "DRAW", "AWAY")
    predicted = probabilities.argmax(axis=1)
    result = {
        "matches": int(len(labels)),
        "actual_class_counts": {name: int(np.sum(labels == i)) for i, name in enumerate(names)},
        "predicted_argmax_counts": {name: int(np.sum(predicted == i)) for i, name in enumerate(names)},
        "probability_distribution": {},
    }
    for i, name in enumerate(names):
        values = probabilities[:, i]
        result["probability_distribution"][name] = {
            "mean": round(float(values.mean()), 6),
            "median": round(float(np.median(values)), 6),
            "p10": round(float(np.quantile(values, 0.10)), 6),
            "p25": round(float(np.quantile(values, 0.25)), 6),
            "p75": round(float(np.quantile(values, 0.75)), 6),
            "p90": round(float(np.quantile(values, 0.90)), 6),
        }
    draw_bins = []
    edges = np.linspace(0.0, 1.0, 11)
    for i in range(10):
        mask = ((probabilities[:, 1] >= edges[i]) &
                (probabilities[:, 1] < edges[i + 1] if i < 9 else probabilities[:, 1] <= edges[i + 1]))
        if mask.any():
            draw_bins.append({
                "lower": round(float(edges[i]), 2),
                "upper": round(float(edges[i + 1]), 2),
                "matches": int(mask.sum()),
                "mean_draw_probability": round(float(probabilities[mask, 1].mean()), 6),
                "observed_draw_rate": round(float(np.mean(labels[mask] == 1)), 6),
            })
    result["draw_probability_bins"] = draw_bins
    return result


def draw_threshold_curve(labels, probabilities, policy):
    """Evaluate a draw decision policy without changing model probabilities."""
    if policy not in ("argmax_and_threshold", "probability_threshold_only"):
        raise ValueError(f"Unknown draw decision policy: {policy}")
    rows = []
    predicted = probabilities.argmax(axis=1)
    total_draws = max(1, int(np.sum(labels == 1)))
    for threshold in np.round(np.arange(0.10, 0.501, 0.025), 3):
        above_threshold = probabilities[:, 1] >= threshold
        if policy == "argmax_and_threshold":
            selected = (predicted == 1) & above_threshold
        else:
            selected = above_threshold
        count = int(selected.sum())
        tp = int(np.sum(selected & (labels == 1)))
        precision = tp / count if count else 0.0
        recall = tp / total_draws
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": float(threshold),
            "selected_matches": count,
            "coverage": round(count / len(labels), 6),
            "draw_precision": round(precision, 6),
            "draw_recall": round(recall, 6),
            "draw_f1": round(f1, 6),
        })
    return rows


def select_threshold(curve):
    """Choose on selection data only; deterministic tie-breaks favor coverage then lower threshold."""
    return max(curve, key=lambda row: (row["draw_f1"], row["coverage"], -row["threshold"]))


def feature_weight_report(model):
    preprocess = model.pipeline_.named_steps["preprocess"]
    classifier = model.pipeline_.named_steps["classifier"]
    names = list(preprocess.get_feature_names_out())
    class_ids = list(classifier.classes_)
    draw_row = classifier.coef_[class_ids.index(1)]
    numeric = list(model.numeric_columns_ or [])
    categorical = list(model.categorical_columns_ or [])
    rows = []
    for feature_name, coefficient in zip(names, draw_row):
        raw = feature_name.split("__", 1)[-1]
        source = next((c for c in sorted(numeric + categorical, key=len, reverse=True)
                       if raw == c or raw.startswith(c + "_")), raw)
        rows.append({
            "encoded_feature": feature_name,
            "source_feature": source,
            "draw_coefficient": float(coefficient),
            "abs_draw_coefficient": abs(float(coefficient)),
        })
    rows.sort(key=lambda item: item["abs_draw_coefficient"], reverse=True)
    grouped = {}
    for row in rows:
        bucket = grouped.setdefault(row["source_feature"], {"sum_abs_coefficient": 0.0, "encoded_terms": 0})
        bucket["sum_abs_coefficient"] += row["abs_draw_coefficient"]
        bucket["encoded_terms"] += 1
    group_rows = [
        {"source_feature": key, "sum_abs_coefficient": round(value["sum_abs_coefficient"], 6),
         "encoded_terms": value["encoded_terms"]}
        for key, value in grouped.items()
    ]
    group_rows.sort(key=lambda item: item["sum_abs_coefficient"], reverse=True)
    return {
        "interpretation_note": (
            "Coefficients are conditional model weights, not causal effects. Numeric inputs "
            "are standardized; categorical one-hot terms are shown individually. Grouped "
            "absolute sums are descriptive and can favor fields with many categories."
        ),
        "top_positive_draw_terms": [
            {**row, "draw_coefficient": round(row["draw_coefficient"], 6)}
            for row in sorted(rows, key=lambda item: item["draw_coefficient"], reverse=True)[:20]
        ],
        "top_negative_draw_terms": [
            {**row, "draw_coefficient": round(row["draw_coefficient"], 6)}
            for row in sorted(rows, key=lambda item: item["draw_coefficient"])[:20]
        ],
        "grouped_feature_weights": group_rows,
    }


def main():
    history = HistoricalEngine(ROOT / "xg18.csv")
    data = history.data.loc[
        history.data["Date"].notna()
        & history.data["FTHG"].notna()
        & history.data["FTAG"].notna()
    ].copy()
    data["_season_start"] = data["Season"].map(season_start_year)
    if data["_season_start"].isna().any():
        unknown = sorted(data.loc[data["_season_start"].isna(), "Season"].astype(str).unique())
        raise ValueError(f"Unrecognized season labels: {unknown}")

    train = data.loc[data["_season_start"].isin([2022, 2023])].drop(columns="_season_start")
    calibration = data.loc[data["_season_start"].eq(2024)].drop(columns="_season_start")
    test = data.loc[data["_season_start"].isin([2025, 2026])].drop(columns="_season_start")
    if train.empty or calibration.empty or test.empty:
        raise ValueError("Expected train 2022/23-2023/24, calibration 2024/25, test 2025/26-2026/27.")

    calibration_fit, threshold_select, threshold_confirm = split_calibration(calibration)
    engine = FeatureEngine(history)

    x_train = engine.build_dataset(train["MatchID"].tolist())
    y_train = labels_for(train)
    model = LogisticBaseline().fit(x_train, y_train)

    x_cal_fit = engine.build_dataset(calibration_fit["MatchID"].tolist())
    y_cal_fit = labels_for(calibration_fit)
    model.calibrate(x_cal_fit, y_cal_fit)

    x_select = engine.build_dataset(threshold_select["MatchID"].tolist())
    y_select = labels_for(threshold_select)
    p_select = model.predict_proba(x_select)

    x_confirm = engine.build_dataset(threshold_confirm["MatchID"].tolist())
    y_confirm = labels_for(threshold_confirm)
    p_confirm = model.predict_proba(x_confirm)

    policy_results = {}
    for policy in ("argmax_and_threshold", "probability_threshold_only"):
        selection_curve = draw_threshold_curve(y_select, p_select, policy)
        best = select_threshold(selection_curve)
        chosen_threshold = best["threshold"]
        confirmation_curve = draw_threshold_curve(y_confirm, p_confirm, policy)
        confirmation_row = next(
            row for row in confirmation_curve if row["threshold"] == chosen_threshold
        )
        policy_results[policy] = {
            "selection_objective": "maximize draw-class F1 on the chronological threshold-selection slice",
            "selection_tie_break": "higher coverage, then lower threshold",
            "grid": selection_curve,
            "selected_threshold": chosen_threshold,
            "selection_slice_result": best,
            "confirmation_slice_result": confirmation_row,
        }

    # Descriptive test probabilities are reported without using test labels to select thresholds.
    x_test = engine.build_dataset(test["MatchID"].tolist())
    y_test = labels_for(test)
    p_test = model.predict_proba(x_test)

    report = {
        "status": "DIAGNOSTIC_EVALUATED",
        "model_version": model.version,
        "selected_C": model.selected_c_,
        "temperature": model.temperature_,
        "class_order": ["HOME", "DRAW", "AWAY"],
        "calibration_subpartitions": {
            "temperature_fit": {"matches": len(calibration_fit), "start": str(calibration_fit["Date"].min().date()), "end": str(calibration_fit["Date"].max().date())},
            "threshold_selection": {"matches": len(threshold_select), "start": str(threshold_select["Date"].min().date()), "end": str(threshold_select["Date"].max().date())},
            "threshold_confirmation": {"matches": len(threshold_confirm), "start": str(threshold_confirm["Date"].min().date()), "end": str(threshold_confirm["Date"].max().date())},
        },
        "draw_feature_weights": feature_weight_report(model),
        "decision_policy_comparison": {
            "policies": {
                "argmax_and_threshold": "Select DRAW only when argmax is DRAW and P(DRAW) >= threshold.",
                "probability_threshold_only": "Select DRAW whenever P(DRAW) >= threshold, regardless of whether DRAW is argmax.",
            },
            "probability_model_unchanged": True,
            "test_data_used_for_policy_selection": False,
            "results": policy_results,
        },
        "probability_diagnostics": {
            "threshold_selection": summarize_probabilities(y_select, p_select),
            "threshold_confirmation": summarize_probabilities(y_confirm, p_confirm),
            "test_descriptive_only": summarize_probabilities(y_test, p_test),
        },
        "test_policy_warning": (
            "Test outcomes were not used for threshold selection. Test probability summaries "
            "are descriptive only; no test threshold optimization is performed."
        ),
        "decision_policy_warning": (
            "This is a classification-threshold study, not a profitable betting strategy. "
            "No odds are present, so EV and ROI cannot be evaluated."
        ),
    }

    output = ROOT / "reports" / "draw_diagnostics_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved: {output}")


if __name__ == "__main__":
    main()
