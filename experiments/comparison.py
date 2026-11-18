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
MIN_SELECT_BETS = 50
MIN_CHECK_BETS = 50
MIN_FINAL_TEST_BETS = 100
EDGE_THRESHOLDS = (0.03, 0.05, 0.08, 0.10)
ODDS_BUCKET_EDGES = np.array([1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10])


def git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\\0".encode("ascii")
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

    frame = pd.read_csv(path, encoding="latin-1", low_memory=False)
    required = [
        "Date", "League", "HomeTeam", "AwayTeam",
        "Home_xG", "Away_xG", "FTHG", "FTAG",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
    for column in ("Home_xG", "Away_xG", "FTHG", "FTAG"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=required).copy()
    frame = frame.sort_values(
        ["Date", "League", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Clean dataset has {len(frame)} rows; expected {EXPECTED_TOTAL_ROWS}."
        )
    if frame.duplicated(["Date", "League", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate fixture key detected.")
    validate_chronological_order(frame["Date"].tolist(), column_name="Date")

    # Validate the only raw odds columns consumed by BOT.market_data.
    validate_feature_columns(
        [
            "odds_Avg>2.5",
            "odds_Avg<2.5",
            "odds_B365>2.5",
            "odds_B365<2.5",
        ]
    )
    # Internal model inputs are explicitly named as pre-match / historical.
    validate_feature_columns(
        [
            "pre_match_market_probability",
            "hist_poisson_probability",
            "pre_match_market_available",
        ]
    )
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
    return (
        frame["FTHG"].to_numpy(dtype=float)
        + frame["FTAG"].to_numpy(dtype=float)
        > 2.5
    ).astype(int)


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

    for threshold in np.arange(0.50, 0.951, 0.005):
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
    # Maximise selection coverage on POLICY_SELECT, then hit rate.
    return max(candidates, key=lambda item: (item["bets"], item["hit_rate"]))


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
    p_over: np.ndarray,
    over_odds: np.ndarray,
    under_odds: np.ndarray,
    selected: np.ndarray,
) -> tuple[int, float | None, float | None, float | None]:
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        return 0, None, None, None

    predicted_over = p_over[indices] >= 0.5
    chosen_odds = np.where(predicted_over, over_odds[indices], under_odds[indices])
    won = np.where(predicted_over, y_true[indices] == 1, y_true[indices] == 0)
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
        y = labels(frame)
        market = market_components(frame)
        p_market = market["p_over"]
        valid = market["market_valid"]
        for model_name, p_model in model_probabilities.items():
            if model_name == "market":
                selected = valid
                count, hit, roi, average_odds = _returns(
                    y, p_market, market["over_odds"], market["under_odds"], selected
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
                direction_probability = np.where(edge >= 0, p_model, p_model)
                count, hit, roi, average_odds = _returns(
                    y,
                    direction_probability,
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
        y = labels(frame)
        market = market_components(frame)
        for model_name, probability in model_probabilities.items():
            threshold = policy_thresholds[model_name]
            if threshold is None:
                continue
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
    raw_partitions = make_partitions(frame)

    # Historical xG is made available only after the entire target calendar
    # day has elapsed. The raw source labels remain available only as targets
    # and as past-history inputs for later calendar days.
    featured = PoissonBaseline.add_historical_features(frame)
    partitions = {
        name: featured.iloc[
            sum(EXPECTED_SPLIT_SIZES[p] for p in PARTITION_ORDER[:index]):
            sum(EXPECTED_SPLIT_SIZES[p] for p in PARTITION_ORDER[: index + 1])
        ].copy().reset_index(drop=True)
        for index, name in enumerate(PARTITION_ORDER)
    }

    effective, purged = purge_same_calendar_day_boundaries(partitions)
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

    model_probabilities: dict[str, np.ndarray] = {
        "market": MarketBaseline().predict_proba(featured),
        "poisson": PoissonBaseline().predict_proba(featured),
        "odds_aware": odds_model.predict_proba(featured),
    }

    market = market_components(featured)
    y_all = labels(featured)
    common_valid = market["market_valid"]

    raw_rows: list[dict[str, Any]] = []
    policy_thresholds: dict[str, float | None] = {}
    policy_selection: dict[str, Any] = {}
    policy_check_results: dict[str, Any] = {}
    final_results: dict[str, Any] = {}

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in PARTITION_ORDER:
        end = cursor + EXPECTED_SPLIT_SIZES[name]
        offsets[name] = (cursor, end)
        cursor = end

    for model_name, probability in model_probabilities.items():
        for partition_name in PARTITION_ORDER:
            start, end = offsets[partition_name]
            part_y = y_all[start:end]
            part_probability = probability[start:end]
            part_valid = common_valid[start:end]
            eligible = part_valid
            metrics = safe_metrics(part_y[eligible], part_probability[eligible])
            raw_rows.append(
                {
                    "model": model_name,
                    "partition": partition_name,
                    **metrics,
                    "common_market_eligible": int(eligible.sum()),
                    "raw_partition_rows": int(end - start),
                }
            )

        select_start, select_end = offsets["policy_select"]
        selection = choose_confidence_policy(
            y_all[select_start:select_end],
            probability[select_start:select_end],
            common_valid[select_start:select_end],
        )
        threshold = selection["threshold"] if selection else None
        policy_thresholds[model_name] = threshold
        policy_selection[model_name] = selection

        check_start, check_end = offsets["policy_check"]
        check = selective_metrics(
            y_all[check_start:check_end],
            probability[check_start:check_end],
            common_valid[check_start:check_end],
            threshold,
            MIN_CHECK_BETS,
        )
        policy_check_results[model_name] = check

        test_start, test_end = offsets["final_test"]
        final = selective_metrics(
            y_all[test_start:test_end],
            probability[test_start:test_end],
            common_valid[test_start:test_end],
            threshold,
            MIN_FINAL_TEST_BETS,
        )
        final_results[model_name] = {
            **safe_metrics(
                y_all[test_start:test_end][common_valid[test_start:test_end]],
                probability[test_start:test_end][common_valid[test_start:test_end]],
            ),
            "selective": final,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(raw_rows)
    metrics_frame.to_csv(output_dir / "model_metrics.csv", index=False)

    directional = build_directional_roi(model_probabilities, effective)
    directional.to_csv(output_dir / "directional_roi.csv", index=False)

    buckets = build_odds_buckets(model_probabilities, policy_thresholds, effective)
    buckets.to_csv(output_dir / "odds_buckets.csv", index=False)

    start, end = offsets["final_test"]
    final_frame = featured.iloc[start:end].copy()
    final_predictions = pd.DataFrame(
        {
            "MatchID": final_frame["MatchID"] if "MatchID" in final_frame else np.arange(len(final_frame)),
            "Date": final_frame["Date"].astype(str),
            "League": final_frame["League"],
            "HomeTeam": final_frame["HomeTeam"],
            "AwayTeam": final_frame["AwayTeam"],
            "actual_over_2_5": y_all[start:end],
            "p_market_over": model_probabilities["market"][start:end],
            "p_poisson_over": model_probabilities["poisson"][start:end],
            "p_odds_aware_over": model_probabilities["odds_aware"][start:end],
            "market_valid": common_valid[start:end],
            "over_odds": market["over_odds"][start:end],
            "under_odds": market["under_odds"][start:end],
        }
    )
    final_predictions.to_csv(output_dir / "final_test_predictions.csv", index=False)

    manifest = {
        "dataset": str(data_path.name),
        "dataset_git_blob_sha": DATASET_GIT_BLOB_SHA,
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
        "final_test_policy": "sealed; evaluated once; no tuning",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "target_hit_rate": TARGET_HIT_RATE,
        "minimum_final_test_bets": MIN_FINAL_TEST_BETS,
        "policy_selection": policy_selection,
        "policy_check": policy_check_results,
        "final_test": final_results,
        "all_models_pass_policy_check": all(
            result["target_met"] for result in policy_check_results.values()
        ),
        "any_model_passes_final_test": any(
            result["selective"]["target_met"] for result in final_results.values()
        ),
        "ready": any(
            result["selective"]["target_met"] for result in final_results.values()
        ),
        "final_test_evaluated_once": True,
        "interpretation": (
            "READY: at least one model reached the target on the sealed final test."
            if any(result["selective"]["target_met"] for result in final_results.values())
            else "NOT READY: no model has yet demonstrated the required hit rate and sample size."
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
