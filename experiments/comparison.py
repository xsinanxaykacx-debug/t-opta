"""Reproducible three-model Over/Under 2.5 comparison on xgo.csv.

The script keeps the five approved row partitions immutable, purges same-day
boundary rows from the preceding operational stage, and never tunes on the
sealed final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from experiments.market_baseline import MarketBaseline
from experiments.odds_aware import OddsAwareModel
from experiments.history_features import add_xg18_historical_features
from experiments.poisson_baseline import PoissonBaseline
from validation.experiment_protocol import (
    EXPECTED_SPLIT_SIZES,
    EXPECTED_TOTAL_ROWS,
    PARTITION_ORDER,
    assert_final_test_sealed,
    validate_chronological_order,
    validate_feature_columns,
    validate_partition_dates,
    validate_split_indices,
)


DATASET_GIT_BLOB_SHA = "1c18ccd4c4190cccc5014d186d6d34305708225c"
TARGET_HIT_RATE = 0.65
MIN_SELECT_BETS = 10
MIN_CHECK_BETS = 10
MIN_FINAL_TEST_BETS = 100
EDGE_THRESHOLDS = (0.03, 0.05, 0.08, 0.10)
ODDS_BUCKET_EDGES = np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10])


def git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    actual_sha = git_blob_sha(path)
    if actual_sha != DATASET_GIT_BLOB_SHA:
        raise ValueError(
            "xgo.csv snapshot SHA mismatch. "
            f"Expected {DATASET_GIT_BLOB_SHA}, received {actual_sha}."
        )

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = [
        "MatchID", "Date", "League", "HomeTeam", "AwayTeam",
        "Home_xG", "Away_xG", "FTHG", "FTAG",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Raw dataset has {len(frame)} rows; expected {EXPECTED_TOTAL_ROWS}."
        )

    # Validate only pre-match identity/history inputs here. Target score
    # values are parsed and checked partition-by-partition when permitted.
    frame = frame.copy()
    frame["MatchID"] = frame["MatchID"].astype("string").str.strip()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
    for column in ("Home_xG", "Away_xG"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    feature_required = [
        "MatchID", "Date", "League", "HomeTeam", "AwayTeam", "Home_xG", "Away_xG"
    ]
    invalid = frame[feature_required].isna().any()
    if invalid.any():
        raise ValueError(
            "Missing pre-match identity/history values: "
            f"{invalid[invalid].index.tolist()}"
        )
    if not np.isfinite(frame[["Home_xG", "Away_xG"]].to_numpy(dtype=float)).all():
        raise ValueError("Historical xG columns contain non-finite values.")

    frame = frame.sort_values(
        ["Date", "League", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    ).reset_index(drop=True)

    if frame.duplicated(["Date", "League", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate fixture key detected.")
    validate_chronological_order(frame["Date"].tolist(), column_name="Date")

    validate_feature_columns(
        [
            "odds_Avg>2.5",
            "odds_Avg<2.5",
            "odds_B365>2.5",
            "odds_B365<2.5",
        ]
    )
    validate_feature_columns(OddsAwareModel.feature_names)
    return frame

def make_partitions(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cuts = (8500, 9400, 9700, 10000, EXPECTED_TOTAL_ROWS)
    split_indices: dict[str, list[int]] = {}
    partitions: dict[str, pd.DataFrame] = {}
    start = 0

    for name, end in zip(PARTITION_ORDER, cuts):
        split_indices[name] = list(range(start, end))
        partitions[name] = frame.iloc[start:end].copy().reset_index(drop=True)
        start = end

    validate_split_indices(split_indices)
    validate_partition_dates(
        {name: part["Date"].tolist() for name, part in partitions.items()}
    )
    return partitions


def purge_same_calendar_day_boundaries(
    partitions: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Remove boundary-day rows from the preceding operational stage.

    Fixed raw split sizes remain intact in the manifest. The rows removed here
    are not used to fit/calibrate/select/check a policy because their outcomes
    may not have been available before fixtures on the next partition's first
    calendar day. They remain in the original source history for later dates.
    """
    effective = {name: part.copy() for name, part in partitions.items()}
    purged: dict[str, int] = {name: 0 for name in PARTITION_ORDER}

    for index in range(1, len(PARTITION_ORDER)):
        previous_name = PARTITION_ORDER[index - 1]
        current_name = PARTITION_ORDER[index]
        previous = effective[previous_name]
        current = effective[current_name]
        cutoff_day = current["Date"].min().normalize()
        keep = previous["Date"].dt.normalize() < cutoff_day
        purged[previous_name] += int((~keep).sum())
        effective[previous_name] = previous.loc[keep].reset_index(drop=True)

    for previous_name, current_name in zip(PARTITION_ORDER, PARTITION_ORDER[1:]):
        previous = effective[previous_name]
        current = effective[current_name]
        if previous.empty or current.empty:
            raise ValueError(
                f"Boundary purge emptied {previous_name} or {current_name}."
            )
        if previous["Date"].dt.normalize().max() >= current["Date"].dt.normalize().min():
            raise RuntimeError(
                f"Calendar-day leakage remains between {previous_name} and {current_name}."
            )
    return effective, purged


def labels(frame: pd.DataFrame) -> np.ndarray:
    home_goals = pd.to_numeric(frame["FTHG"], errors="coerce").to_numpy(dtype=float)
    away_goals = pd.to_numeric(frame["FTAG"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(home_goals).all() or not np.isfinite(away_goals).all():
        raise ValueError("The currently permitted partition has missing/invalid target scores.")
    return ((home_goals + away_goals) > 2.5).astype(int)


def market_components(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return MarketBaseline().predict_components(frame)


def safe_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    result: dict[str, Any] = {
        "matches": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "over_rate": float(np.mean(y_true)),
    }
    result["auc"] = (
        float(roc_auc_score(y_true, probability))
        if np.unique(y_true).size == 2
        else None
    )
    return result


def choose_confidence_policy(
    y_true: np.ndarray,
    probability: np.ndarray,
    market_valid: np.ndarray,
) -> dict[str, Any] | None:
    confidence = np.maximum(probability, 1.0 - probability)
    candidates: list[dict[str, Any]] = []

    for threshold in np.round(np.arange(0.50, 0.951, 0.005), 3):
        selected = market_valid & (confidence >= threshold)
        count = int(selected.sum())
        if count < MIN_SELECT_BETS:
            continue
        hit_rate = float(np.mean((probability[selected] >= 0.5) == y_true[selected]))
        if hit_rate >= TARGET_HIT_RATE:
            candidates.append(
                {
                    "threshold": float(round(threshold, 3)),
                    "bets": count,
                    "hit_rate": hit_rate,
                    "coverage": float(count / max(int(market_valid.sum()), 1)),
                }
            )

    if not candidates:
        return None
    # Maximise observed hit rate on POLICY_SELECT, then sample size.
    return max(candidates, key=lambda item: (item["hit_rate"], item["bets"]))


def selective_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    market_valid: np.ndarray,
    threshold: float | None,
    minimum_bets: int,
) -> dict[str, Any]:
    if threshold is None:
        return {
            "threshold": None,
            "bets": 0,
            "hit_rate": None,
            "coverage": 0.0,
            "minimum_bets_met": False,
            "target_met": False,
        }

    confidence = np.maximum(probability, 1.0 - probability)
    selected = market_valid & (confidence >= threshold)
    count = int(selected.sum())
    hit_rate = (
        float(np.mean((probability[selected] >= 0.5) == y_true[selected]))
        if count
        else None
    )
    return {
        "threshold": float(threshold),
        "bets": count,
        "hit_rate": hit_rate,
        "coverage": float(count / max(int(market_valid.sum()), 1)),
        "minimum_bets_met": bool(count >= minimum_bets),
        "target_met": bool(
            count >= minimum_bets
            and hit_rate is not None
            and hit_rate >= TARGET_HIT_RATE
        ),
    }


def _returns(
    y_true: np.ndarray,
    predicted_over: np.ndarray,
    over_odds: np.ndarray,
    under_odds: np.ndarray,
    selected: np.ndarray,
) -> tuple[int, float | None, float | None, float | None]:
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        return 0, None, None, None

    direction = np.asarray(predicted_over, dtype=bool)[indices]
    chosen_odds = np.where(direction, over_odds[indices], under_odds[indices])
    won = np.where(direction, y_true[indices] == 1, y_true[indices] == 0)
    valid = np.isfinite(chosen_odds) & (chosen_odds > 1.0)
    if not valid.any():
        return 0, None, None, None

    chosen_odds = chosen_odds[valid]
    won = won[valid]
    profit = np.where(won, chosen_odds - 1.0, -1.0)
    roi = float(profit.sum() / len(profit))
    hit_rate = float(won.mean())
    average_odds = float(chosen_odds.mean())
    return int(len(profit)), hit_rate, roi, average_odds

def build_directional_roi(
    model_probabilities: dict[str, np.ndarray],
    partitions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition_name in PARTITION_ORDER:
        frame = partitions[partition_name]
        source_rows = frame["_source_row"].to_numpy(dtype=int)
        y = labels(frame)
        market = market_components(frame)
        p_market = market["p_over"]
        valid = market["market_valid"]
        for model_name, full_probability in model_probabilities.items():
            p_model = full_probability[source_rows]
            if model_name == "market":
                predicted_over = p_market >= 0.5
                count, hit, roi, average_odds = _returns(
                    y, predicted_over, market["over_odds"], market["under_odds"], valid
                )
                rows.append(
                    {
                        "model": model_name,
                        "partition": partition_name,
                        "edge_threshold": None,
                        "bets": count,
                        "hit_rate": hit,
                        "roi_per_unit": roi,
                        "average_odds": average_odds,
                    }
                )
                continue

            edge = p_model - p_market
            for threshold in EDGE_THRESHOLDS:
                selected = valid & (np.abs(edge) >= threshold)
                predicted_over = edge >= 0.0
                count, hit, roi, average_odds = _returns(
                    y,
                    predicted_over,
                    market["over_odds"],
                    market["under_odds"],
                    selected,
                )
                rows.append(
                    {
                        "model": model_name,
                        "partition": partition_name,
                        "edge_threshold": threshold,
                        "bets": count,
                        "hit_rate": hit,
                        "roi_per_unit": roi,
                        "average_odds": average_odds,
                    }
                )
    return pd.DataFrame(rows)

def build_odds_buckets(
    model_probabilities: dict[str, np.ndarray],
    policy_thresholds: dict[str, float | None],
    partitions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition_name in PARTITION_ORDER:
        frame = partitions[partition_name]
        source_rows = frame["_source_row"].to_numpy(dtype=int)
        y = labels(frame)
        market = market_components(frame)
        for model_name, full_probability in model_probabilities.items():
            threshold = policy_thresholds[model_name]
            if threshold is None:
                continue
            probability = full_probability[source_rows]
            confidence = np.maximum(probability, 1.0 - probability)
            selected = market["market_valid"] & (confidence >= threshold)
            predicted_over = probability >= 0.5
            chosen_odds = np.where(
                predicted_over, market["over_odds"], market["under_odds"]
            )
            hit = predicted_over == y
            for bucket_index in range(6):
                low = ODDS_BUCKET_EDGES[bucket_index]
                high = ODDS_BUCKET_EDGES[bucket_index + 1]
                in_bucket = selected & (chosen_odds >= low)
                if bucket_index < 5:
                    in_bucket &= chosen_odds < high
                else:
                    in_bucket &= chosen_odds <= high
                count = int(in_bucket.sum())
                if count:
                    profit = np.where(
                        hit[in_bucket],
                        chosen_odds[in_bucket] - 1.0,
                        -1.0,
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "partition": partition_name,
                            "odds_bucket": f"{low:.2f}-{high:.2f}",
                            "bets": count,
                            "hit_rate": float(hit[in_bucket].mean()),
                            "roi_per_unit": float(profit.sum() / count),
                            "average_odds": float(chosen_odds[in_bucket].mean()),
                        }
                    )
                else:
                    rows.append(
                        {
                            "model": model_name,
                            "partition": partition_name,
                            "odds_bucket": f"{low:.2f}-{high:.2f}",
                            "bets": 0,
                            "hit_rate": None,
                            "roi_per_unit": None,
                            "average_odds": None,
                        }
                    )
    return pd.DataFrame(rows)

def run_experiment(data_path: Path, output_dir: Path) -> dict[str, Any]:
    frame = load_dataset(data_path)
    frame["_source_row"] = np.arange(len(frame), dtype=int)
    featured, history_metadata = add_xg18_historical_features(
        frame,
        Path("xg18.csv"),
        target_validation_limit=10_000,
    )
    raw_partitions = make_partitions(featured)
    effective, purged = purge_same_calendar_day_boundaries(raw_partitions)

    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = {
        name: (
            int(raw_partitions[name]["_source_row"].min()),
            int(raw_partitions[name]["_source_row"].max()) + 1,
        )
        for name in PARTITION_ORDER
    }
    manifest = {
        "dataset": str(data_path.name),
        "dataset_git_blob_sha": DATASET_GIT_BLOB_SHA,
        "historical_source": history_metadata,
        "total_rows": EXPECTED_TOTAL_ROWS,
        "split_sizes_raw": EXPECTED_SPLIT_SIZES,
        "split_index_ranges": {
            name: [offsets[name][0], offsets[name][1]]
            for name in PARTITION_ORDER
        },
        "split_dates_raw": {
            name: {
                "first": raw_partitions[name]["Date"].min().isoformat(),
                "last": raw_partitions[name]["Date"].max().isoformat(),
            }
            for name in PARTITION_ORDER
        },
        "purged_same_calendar_day_rows": purged,
        "effective_rows": {
            name: int(len(effective[name]))
            for name in PARTITION_ORDER
        },
        "feature_history_policy": "prior calendar days only; all same-day xG excluded",
        "closing_odds_policy": "audit-only; never model features",
        "final_test_policy": "candidate and thresholds frozen before one evaluation",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train = effective["train"]
    calibration = effective["calibration"]
    policy_select = effective["policy_select"]
    policy_check = effective["policy_check"]
    final_test = effective["final_test"]

    assert_final_test_sealed(["train"])
    assert_final_test_sealed(["calibration"])
    assert_final_test_sealed(["policy_select"])
    assert_final_test_sealed(["policy_check"])

    odds_model = OddsAwareModel()
    odds_model.fit(train)
    odds_model.calibrate(calibration)

    model_objects = {
        "market": MarketBaseline(),
        "poisson": PoissonBaseline(),
        "odds_aware": odds_model,
    }
    model_probabilities = {
        name: np.full(len(featured), np.nan, dtype=float)
        for name in model_objects
    }

    # Generate predictions only for partitions preceding FINAL_TEST.
    pre_final_parts = [
        effective[name] for name in PARTITION_ORDER[:-1]
    ]
    pre_final_frame = pd.concat(pre_final_parts, ignore_index=True)
    pre_final_frame = pre_final_frame.sort_values(
        "_source_row", kind="stable"
    ).reset_index(drop=True)
    pre_final_rows = pre_final_frame["_source_row"].to_numpy(dtype=int)
    for model_name, model in model_objects.items():
        model_probabilities[model_name][pre_final_rows] = model.predict_proba(
            pre_final_frame
        )

    # POLICY_SELECT alone chooses each threshold and the candidate model.
    select_rows = policy_select["_source_row"].to_numpy(dtype=int)
    select_y = labels(policy_select)
    select_market_valid = market_components(policy_select)["market_valid"]
    policy_thresholds: dict[str, float | None] = {}
    policy_selection: dict[str, Any] = {}

    for model_name in model_objects:
        selection = choose_confidence_policy(
            select_y,
            model_probabilities[model_name][select_rows],
            select_market_valid,
        )
        policy_selection[model_name] = selection
        policy_thresholds[model_name] = (
            selection["threshold"] if selection is not None else None
        )

    eligible_candidates = [
        (name, result)
        for name, result in policy_selection.items()
        if result is not None
    ]
    selected_model = (
        max(
            eligible_candidates,
            key=lambda item: (item[1]["hit_rate"], item[1]["bets"]),
        )[0]
        if eligible_candidates
        else None
    )

    # POLICY_CHECK independently validates the thresholds chosen above.
    check_rows = policy_check["_source_row"].to_numpy(dtype=int)
    check_y = labels(policy_check)
    check_market_valid = market_components(policy_check)["market_valid"]
    policy_check_results: dict[str, Any] = {}
    for model_name in model_objects:
        policy_check_results[model_name] = selective_metrics(
            check_y,
            model_probabilities[model_name][check_rows],
            check_market_valid,
            policy_thresholds[model_name],
            MIN_CHECK_BETS,
        )

    selected_check = (
        policy_check_results[selected_model]
        if selected_model is not None
        else None
    )
    policy_gate_passed = bool(
        selected_model is not None
        and selected_check is not None
        and selected_check["target_met"]
    )

    policy_rows = []
    for model_name in model_objects:
        selection = policy_selection[model_name]
        check = policy_check_results[model_name]
        policy_rows.append(
            {
                "model": model_name,
                "selected_on_policy_select": model_name == selected_model,
                "confidence_threshold": policy_thresholds[model_name],
                "select_bets": selection["bets"] if selection else 0,
                "select_hit_rate": selection["hit_rate"] if selection else None,
                "check_bets": check["bets"],
                "check_hit_rate": check["hit_rate"],
                "check_target_met": check["target_met"],
            }
        )
    pd.DataFrame(policy_rows).to_csv(
        output_dir / "policy_results.csv", index=False
    )

    selection_report = {
        "selected_model_on_policy_select": selected_model,
        "policy_selection": policy_selection,
        "policy_check": policy_check_results,
        "policy_gate_passed": policy_gate_passed,
    }

    # If the candidate fails, FINAL_TEST predictions and labels remain untouched.
    if not policy_gate_passed:
        summary = {
            "target_hit_rate": TARGET_HIT_RATE,
            "selected_model": selected_model,
            "historical_source": history_metadata,
            **selection_report,
            "final_test_opened": False,
            "ready": False,
            "interpretation": (
                "NOT READY: no preselected model passed POLICY_CHECK. "
                "FINAL_TEST remains sealed and was not predicted or evaluated."
            ),
        }
        (output_dir / "experiment_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    # Candidate and all thresholds are frozen. Only now generate FINAL_TEST
    # predictions and read its labels for the single confirmatory evaluation.
    assert_final_test_sealed(
        ["train", "calibration", "policy_select", "policy_check"]
    )
    final_rows = final_test["_source_row"].to_numpy(dtype=int)
    for model_name, model in model_objects.items():
        model_probabilities[model_name][final_rows] = model.predict_proba(
            final_test
        )

    final_y = labels(final_test)
    final_market = market_components(final_test)
    final_market_valid = final_market["market_valid"]

    final_results: dict[str, Any] = {}
    model_metrics_rows: list[dict[str, Any]] = []
    for model_name in model_objects:
        final_probability = model_probabilities[model_name][final_rows]
        metrics = safe_metrics(
            final_y[final_market_valid],
            final_probability[final_market_valid],
        )
        selective = selective_metrics(
            final_y,
            final_probability,
            final_market_valid,
            policy_thresholds[model_name],
            MIN_FINAL_TEST_BETS,
        )
        final_results[model_name] = {
            **metrics,
            "selective": selective,
            "preselected_candidate": model_name == selected_model,
        }
        model_metrics_rows.append(
            {
                "model": model_name,
                "partition": "final_test",
                **metrics,
                "common_market_eligible": int(final_market_valid.sum()),
                "raw_partition_rows": int(len(final_test)),
                "policy_threshold_frozen": policy_thresholds[model_name],
                "selective_bets": selective["bets"],
                "selective_hit_rate": selective["hit_rate"],
                "selective_target_met": selective["target_met"],
            }
        )

    # Descriptive metrics for the earlier partitions; never used to tune
    # thresholds after the candidate has been selected.
    for partition_name in PARTITION_ORDER[:-1]:
        part = effective[partition_name]
        source_rows = part["_source_row"].to_numpy(dtype=int)
        y_part = labels(part)
        valid = market_components(part)["market_valid"]
        for model_name in model_objects:
            metrics = safe_metrics(
                y_part[valid],
                model_probabilities[model_name][source_rows][valid],
            )
            model_metrics_rows.append(
                {
                    "model": model_name,
                    "partition": partition_name,
                    **metrics,
                    "common_market_eligible": int(valid.sum()),
                    "raw_partition_rows": int(
                        len(raw_partitions[partition_name])
                    ),
                    "effective_partition_rows": int(len(part)),
                }
            )

    pd.DataFrame(model_metrics_rows).to_csv(
        output_dir / "model_metrics.csv", index=False
    )
    build_directional_roi(model_probabilities, effective).to_csv(
        output_dir / "directional_roi.csv", index=False
    )
    build_odds_buckets(
        model_probabilities, policy_thresholds, effective
    ).to_csv(output_dir / "odds_buckets.csv", index=False)

    final_predictions = pd.DataFrame(
        {
            "MatchID": final_test["MatchID"].to_numpy(),
            "Date": final_test["Date"].astype(str).to_numpy(),
            "League": final_test["League"].to_numpy(),
            "HomeTeam": final_test["HomeTeam"].to_numpy(),
            "AwayTeam": final_test["AwayTeam"].to_numpy(),
            "actual_over_2_5": final_y,
            "p_market_over": model_probabilities["market"][final_rows],
            "p_poisson_over": model_probabilities["poisson"][final_rows],
            "p_odds_aware_over": model_probabilities["odds_aware"][final_rows],
            "market_valid": final_market_valid,
            "over_odds": final_market["over_odds"],
            "under_odds": final_market["under_odds"],
        }
    )
    final_predictions.to_csv(
        output_dir / "final_test_predictions.csv", index=False
    )

    candidate_final = final_results[selected_model]["selective"]
    ready = bool(candidate_final["target_met"])
    summary = {
        "target_hit_rate": TARGET_HIT_RATE,
        "minimum_final_test_bets": MIN_FINAL_TEST_BETS,
        "historical_source": history_metadata,
        **selection_report,
        "final_test": final_results,
        "final_test_opened": True,
        "final_test_evaluated_once": True,
        "ready": ready,
        "interpretation": (
            "READY: the preselected model passed the sealed Final Test gate."
            if ready
            else "NOT READY: the preselected model did not demonstrate the target on Final Test."
        ),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("xgo.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()
    run_experiment(args.data, args.output)


if __name__ == "__main__":
    main()
