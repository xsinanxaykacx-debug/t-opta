"""Build the frozen Odds-Aware production model from TRAIN + CALIBRATION only.

This workflow does not evaluate, tune on, or read target labels from
POLICY_SELECT, POLICY_CHECK, or FINAL_TEST.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from experiments.comparison import (
    load_dataset,
    make_partitions,
    purge_same_calendar_day_boundaries,
)
from experiments.history_features import add_xg18_historical_features
from experiments.odds_aware import OddsAwareModel


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "results" / "comparison" / "experiment_summary.json"
ROI_PATH = ROOT / "results" / "comparison" / "final_test_roi.json"
DATA_PATH = ROOT / "xgo.csv"
HISTORY_PATH = ROOT / "xg18.csv"
OUTPUT_DIR = ROOT / "artifacts"


def _load_and_verify_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    if not EVIDENCE_PATH.is_file() or not ROI_PATH.is_file():
        raise FileNotFoundError(
            "Frozen experiment evidence is missing. Refusing to build a model "
            "without the recorded Final Test gate."
        )
    with EVIDENCE_PATH.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    with ROI_PATH.open("r", encoding="utf-8") as stream:
        roi = json.load(stream)

    if summary.get("ready") is not True:
        raise RuntimeError("The recorded experiment did not pass its Final Test gate.")
    if summary.get("selected_model_on_policy_select") != "odds_aware":
        raise RuntimeError("The frozen selected model is not Odds-Aware.")

    selected = summary["policy_selection"]["odds_aware"]
    check = summary["policy_check"]["odds_aware"]
    final = summary["final_test"]["odds_aware"]["selective"]

    threshold = float(selected["threshold"])
    if not np.isclose(threshold, 0.735, atol=1e-12):
        raise RuntimeError(f"Unexpected frozen confidence threshold: {threshold}")
    if not check.get("target_met") or not final.get("target_met"):
        raise RuntimeError("Policy Check or Final Test evidence does not meet the gate.")
    if int(final["bets"]) < 100 or float(final["hit_rate"]) < 0.65:
        raise RuntimeError("Final Test evidence is below the approved accuracy gate.")

    if roi.get("selected_model") != "odds_aware":
        raise RuntimeError("ROI evidence refers to a different selected model.")
    if not np.isclose(float(roi.get("threshold", -1)), threshold, atol=1e-12):
        raise RuntimeError("ROI evidence threshold does not match the frozen threshold.")
    return summary, roi


def train_production_model(
    *,
    data_path: Path = DATA_PATH,
    history_path: Path = HISTORY_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    summary, roi = _load_and_verify_evidence()

    frame = load_dataset(data_path)
    frame["_source_row"] = np.arange(len(frame), dtype=int)

    raw_partitions = make_partitions(frame)
    effective, purged = purge_same_calendar_day_boundaries(raw_partitions)
    train = effective["train"].copy()
    calibration = effective["calibration"].copy()

    if train.empty or calibration.empty:
        raise RuntimeError("TRAIN or CALIBRATION became empty after boundary purging.")

    # Only TRAIN and CALIBRATION are passed into the feature/target pipeline.
    # Policy Select, Policy Check, and Final Test target columns are not read.
    pre_final = pd.concat([train, calibration], ignore_index=True)
    pre_final = pre_final.sort_values("_source_row", kind="mergesort").reset_index(drop=True)
    featured, history_metadata = add_xg18_historical_features(
        pre_final,
        history_path,
        target_validation_limit=len(pre_final),
    )

    train_source_rows = set(train["_source_row"].astype(int).tolist())
    calibration_source_rows = set(calibration["_source_row"].astype(int).tolist())
    train_features = featured.loc[
        featured["_source_row"].isin(train_source_rows)
    ].copy()
    calibration_features = featured.loc[
        featured["_source_row"].isin(calibration_source_rows)
    ].copy()

    if len(train_features) != len(train) or len(calibration_features) != len(calibration):
        raise RuntimeError("Feature rows do not match the frozen TRAIN/CALIBRATION partitions.")

    model = OddsAwareModel().fit(train_features).calibrate(calibration_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "odds_aware_model.joblib"
    metadata_path = output_dir / "odds_aware_model.json"
    joblib.dump(model, model_path, compress=3)

    selected = summary["policy_selection"]["odds_aware"]
    policy_check = summary["policy_check"]["odds_aware"]
    final = summary["final_test"]["odds_aware"]["selective"]

    metadata = {
        "model_name": "odds_aware",
        "model_version": "t-opta-odds-aware-v1",
        "target": "Over/Under 2.5 total goals",
        "confidence_threshold": float(selected["threshold"]),
        "selection_rule": (
            "market_valid and max(p_over, 1-p_over) >= confidence_threshold"
        ),
        "feature_policy": {
            "closing_odds": "forbidden",
            "post_match_and_live_data": "forbidden",
            "historical_xg": "same league; strictly earlier calendar days",
            "odds": ["odds_Avg>2.5", "odds_Avg<2.5", "odds_B365>2.5", "odds_B365<2.5"],
        },
        "source_data": {
            "xgo_path": str(data_path.name),
            "xgo_git_blob_sha": summary["historical_source"].get(
                "xgo_git_blob_sha", None
            ),
            "xg18_path": str(history_path.name),
            "xg18_git_blob_sha": history_metadata["history_git_blob_sha"],
            "xg18_history_date_max": history_metadata["history_date_max"],
            "training_target_date_max": str(train["Date"].max()),
        },
        "effective_partition_rows": {
            "train": int(len(train_features)),
            "calibration": int(len(calibration_features)),
            "purged_same_calendar_day": purged,
        },
        "partition_date_ranges": {
            "train": {
                "first": train["Date"].min().isoformat(),
                "last": train["Date"].max().isoformat(),
            },
            "calibration": {
                "first": calibration["Date"].min().isoformat(),
                "last": calibration["Date"].max().isoformat(),
            },
        },
        "frozen_validation_evidence": {
            "policy_select": {
                "bets": int(selected["bets"]),
                "hit_rate": float(selected["hit_rate"]),
            },
            "policy_check": {
                "bets": int(policy_check["bets"]),
                "hit_rate": float(policy_check["hit_rate"]),
            },
            "final_test": {
                "bets": int(final["bets"]),
                "hits": int(round(final["bets"] * final["hit_rate"])),
                "hit_rate": float(final["hit_rate"]),
                "roi_per_unit": float(roi["roi_per_unit"]),
                "average_odds": float(roi["average_odds"]),
                "net_units": float(roi["net_units"]),
            },
        },
        "warning": (
            "Historical accuracy passed 65%, but Final Test selective ROI was "
            "negative. This artifact is an accuracy-oriented predictor, not a "
            "validated profitable betting system."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "train_rows": len(train_features),
        "calibration_rows": len(calibration_features),
        "threshold": metadata["confidence_threshold"],
        "final_test_hit_rate": metadata["frozen_validation_evidence"]["final_test"]["hit_rate"],
        "final_test_roi_per_unit": metadata["frozen_validation_evidence"]["final_test"]["roi_per_unit"],
    }


if __name__ == "__main__":
    result = train_production_model()
    print(json.dumps(result, ensure_ascii=False, indent=2))
