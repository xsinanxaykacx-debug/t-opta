"""Leakage-conscious 1X2 bookmaker-market diagnostics.

Uses only a complete pre-match 1X2 odds triplet as market input.
FTHG/FTAG are used solely to derive the final-result label.
The source archive has no odds capture timestamps, so kickoff availability
cannot be independently proven; reports carry this limitation explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ODDS_ALIASES = {
    "home": ("odds_AvgH", "AvgH", "odds_BbAvH", "BbAvH", "odds_B365H", "B365H"),
    "draw": ("odds_AvgD", "AvgD", "odds_BbAvD", "BbAvD", "odds_B365D", "B365D"),
    "away": ("odds_AvgA", "AvgA", "odds_BbAvA", "BbAvA", "odds_B365A", "B365A"),
}
SOURCE_GROUPS = (
    ("Avg", ("odds_AvgH", "AvgH"), ("odds_AvgD", "AvgD"), ("odds_AvgA", "AvgA")),
    ("BbAv", ("odds_BbAvH", "BbAvH"), ("odds_BbAvD", "BbAvD"), ("odds_BbAvA", "BbAvA")),
    ("B365", ("odds_B365H", "B365H"), ("odds_B365D", "B365D"), ("odds_B365A", "B365A")),
)
PROBABILITY_BINS = (0.60, 0.71, 0.81, 0.91, 1.0000001)
PROBABILITY_BAND_LABELS = ('60.00%-70.99%', '71.00%-80.99%', '81.00%-90.99%', '91.00%-100.00%')


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().casefold(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    return None


def _valid_odds(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values[~np.isfinite(values) | (values <= 1.0)] = np.nan
    return values


def market_probabilities(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select one complete H/D/A triplet per row; never mix bookmakers."""
    n = len(frame)
    probs = np.full((n, 3), np.nan, dtype=float)
    odds_selected = np.full((n, 3), np.nan, dtype=float)
    margin = np.full(n, np.nan, dtype=float)
    source_selected = np.full(n, "", dtype=object)
    columns_by_source = {}
    for source, home_aliases, draw_aliases, away_aliases in SOURCE_GROUPS:
        columns = (
            _find_column(frame, home_aliases),
            _find_column(frame, draw_aliases),
            _find_column(frame, away_aliases),
        )
        if not all(columns):
            continue
        raw = [_valid_odds(frame[column]) for column in columns]
        valid = np.logical_and.reduce([np.isfinite(values) for values in raw])
        use = valid & (source_selected == "")
        if not use.any():
            continue
        inverse = np.column_stack([1.0 / values for values in raw])
        inverse_sum = inverse.sum(axis=1)
        # Raw implied probability = 1 / decimal odds; deliberately no de-vigging.
        probs[use] = inverse[use]
        odds_selected[use] = np.column_stack(raw)[use]
        margin[use] = inverse_sum[use] - 1.0
        source_selected[use] = source
        columns_by_source[source] = list(columns)

    valid_all = source_selected != ""
    if not valid_all.any():
        available = [str(c) for c in frame.columns if any(
            token in str(c).casefold() for token in
            ("avgh", "avgd", "avga", "bbavh", "bbavd", "bbava", "b365h", "b365d", "b365a")
        )]
        raise ValueError(
            "Complete 1X2 odds triplet not found. Recognized aliases: "
            + json.dumps(ODDS_ALIASES, ensure_ascii=False)
            + "; candidate columns present: " + repr(available)
        )

    result = pd.DataFrame(probs, columns=["p_home", "p_draw", "p_away"], index=frame.index)
    result["market_margin"] = margin
    result["odds_home"] = odds_selected[:, 0]
    result["odds_draw"] = odds_selected[:, 1]
    result["odds_away"] = odds_selected[:, 2]
    result["odds_source"] = source_selected
    result["odds_valid"] = valid_all
    source_counts = {source[0]: int(np.sum(source_selected == source[0])) for source in SOURCE_GROUPS}
    return result, {"source_counts": source_counts, "valid_matches": int(valid_all.sum()),
                    "source_columns": columns_by_source}


def _season_label(dates: pd.Series) -> pd.Series:
    years = dates.dt.year.where(dates.dt.month >= 7, dates.dt.year - 1)
    return years.astype("Int64").astype(str) + "/" + (years + 1).astype("Int64").astype(str)


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 chronological split on whole UTC dates; holdout stays unread."""
    day = frame["Date"].dt.floor("D")
    counts = day.value_counts().sort_index()
    if len(counts) < 10:
        raise ValueError("At least 10 distinct kickoff dates are required.")
    cumulative = counts.cumsum().to_numpy()
    train_cut = int(np.argmin(np.abs(cumulative - len(frame) * 0.70)))
    val_target = len(frame) * 0.85
    candidates = np.arange(train_cut + 1, len(counts) - 1)
    if not len(candidates):
        raise ValueError("Insufficient dates for chronological 70/15/15 split.")
    val_cut = int(candidates[np.argmin(np.abs(cumulative[candidates] - val_target))])
    days = counts.index
    train_end, val_end = days[train_cut], days[val_cut]
    discovery = frame.loc[day <= train_end].copy()
    validation = frame.loc[(day > train_end) & (day <= val_end)].copy()
    sealed_holdout = frame.loc[day > val_end].copy()
    if min(map(len, (discovery, validation, sealed_holdout))) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return discovery, validation, sealed_holdout


def _summary(group: pd.DataFrame) -> dict:
    n = len(group)
    if not n:
        return {"matches": 0, "accuracy": np.nan, "mean_confidence": np.nan,
                "mean_margin": np.nan, "raw_multiclass_squared_error": np.nan}
    p = group[["p_home", "p_draw", "p_away"]].to_numpy(float)
    actual = group["actual_code"].to_numpy(int)
    predicted = np.argmax(p, axis=1)
    onehot = np.eye(3)[actual]
    return {
        "matches": int(n),
        "accuracy": float(np.mean(predicted == actual)),
        "mean_confidence": float(np.mean(np.max(p, axis=1))),
        "mean_margin": float(group["market_margin"].mean()),
        "raw_multiclass_squared_error": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
    }


def _report_rows(data: pd.DataFrame, scope: str, keys: list[str]) -> list[dict]:
    rows = []
    grouped = [((), data)] if not keys else data.groupby(keys, dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {"scope": scope}
        row.update(dict(zip(keys, key)))
        row.update(_summary(group))
        p = group[["p_home", "p_draw", "p_away"]].to_numpy(float)
        actual = group["actual_code"].to_numpy(int)
        pred = np.argmax(p, axis=1)
        for code, name in enumerate(("home", "draw", "away")):
            selected = pred == code
            row[f"{name}_predicted_count"] = int(selected.sum())
            precision = float(np.mean(actual[selected] == code)) if selected.any() else np.nan
            actual_count = int(np.sum(actual == code))
            recall = float(np.sum((pred == code) & (actual == code)) / actual_count) if actual_count else np.nan
            row[f"{name}_predicted_accuracy"] = precision
            row[f"{name}_actual_count"] = actual_count
            row[f"{name}_actual_recall"] = recall
            row[f"{name}_f1"] = (
                2 * precision * recall / (precision + recall)
                if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
                else np.nan
            )
        rows.append(row)
    return rows


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for an observed binary frequency."""
    if total <= 0:
        return np.nan, np.nan
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def _calibration_summary_rows(calibration_rows: list[dict]) -> list[dict]:
    """Sample-weighted calibration diagnostics, with sparse-bin guardrails."""
    rows = []
    calibration = pd.DataFrame(calibration_rows)
    for (partition, outcome), group in calibration.groupby(["partition", "outcome"], sort=True):
        weights = group["matches"].to_numpy(float)
        errors = group["calibration_error"].to_numpy(float)
        reliable_bins = group.loc[group["matches"] >= 30, "calibration_error"].abs()
        rows.append({
            "partition": partition,
            "outcome": outcome,
            "matches": int(weights.sum()),
            "weighted_absolute_calibration_error": float(np.average(np.abs(errors), weights=weights)),
            "weighted_signed_calibration_error": float(np.average(errors, weights=weights)),
            "bins_with_at_least_30_matches": int((group["matches"] >= 30).sum()),
            "max_absolute_bin_error_n_ge_30": float(reliable_bins.max()) if len(reliable_bins) else np.nan,
        })
    return rows


def _outcome_calibration_rows(data: pd.DataFrame, partition: str) -> list[dict]:
    """Reliability table for each H/D/A probability, independent of favorite selection."""
    rows = []
    edges = np.linspace(0.0, 1.0, 11)
    probability_columns = ("p_home", "p_draw", "p_away")
    outcome_names = ("home", "draw", "away")
    actual = data["actual_code"].to_numpy(int)
    for code, (name, column) in enumerate(zip(outcome_names, probability_columns)):
        probabilities = data[column].to_numpy(float)
        observed = (actual == code).astype(float)
        bin_ids = np.minimum(np.digitize(probabilities, edges, right=False) - 1, 9)
        for bin_index in range(10):
            mask = bin_ids == bin_index
            count = int(mask.sum())
            if not count:
                continue
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(observed[mask].mean())
            successes = int(observed[mask].sum())
            ci_lower, ci_upper = _wilson_interval(successes, count)
            rows.append({
                "partition": partition,
                "outcome": name,
                "probability_band": f"{edges[bin_index]:.0%}-{edges[bin_index + 1]:.0%}",
                "matches": count,
                "mean_predicted_probability": mean_probability,
                "observed_frequency": observed_rate,
                "observed_frequency_ci95_lower": ci_lower,
                "observed_frequency_ci95_upper": ci_upper,
                "calibration_error": observed_rate - mean_probability,
                "brier_score_one_vs_rest": float(np.mean((probabilities[mask] - observed[mask]) ** 2)),
            })
    return rows

def _select_candidate_bands(discovery: pd.DataFrame, minimum_matches: int = 50,
                            minimum_hit_rate: float = 0.75) -> list[dict]:
    """Choose reliable-looking bands on discovery only; validation remains untouched."""
    selected = []
    for label, lower, upper in zip(PROBABILITY_BAND_LABELS, PROBABILITY_BINS[:-1], PROBABILITY_BINS[1:]):
        subset = discovery.loc[
            (discovery["favorite_probability"] >= lower)
            & (discovery["favorite_probability"] < upper)
        ]
        count = len(subset)
        hit_rate = float(subset["favorite_correct"].mean()) if count else float("nan")
        if count >= minimum_matches and hit_rate >= minimum_hit_rate:
            selected.append({
                "band": label, "lower": float(lower), "upper": float(upper),
                "discovery_matches": int(count), "discovery_hit_rate": hit_rate,
            })
    return selected


def _coupon_simulations(validation: pd.DataFrame, output_dir: Path, selected_bands: list[dict], simulations: int = 50,
                        group_size: int = 10, legs_per_coupon: int = 3,
                        stake_per_coupon: float = 100.0, seed: int = 20261119) -> dict:
    """Evaluate only bands that passed the discovery selection gate."""
    selected_mask = np.zeros(len(validation), dtype=bool)
    for band in selected_bands:
        selected_mask |= (
            (validation["favorite_probability"].to_numpy() >= band["lower"])
            & (validation["favorite_probability"].to_numpy() < band["upper"])
        )
    candidates = validation.loc[selected_mask].copy()
    candidates = candidates.sort_values("Date").reset_index(drop=True)
    columns = ["simulation", "candidate_pool", "correct_of_10", "group_hit_rate",
               "coupon1_won", "coupon2_won", "coupon3_won", "winning_coupons",
               "total_return", "total_stake", "net_result", "any_coupon_won"]
    if len(candidates) < group_size:
        pd.DataFrame(columns=columns).to_csv(output_dir / "market_1x2_coupon_simulations.csv", index=False)
        status = "no_discovery_band_passed" if not selected_bands else "insufficient_validation_matches"
        return {"status": status, "candidate_matches": int(len(candidates)),
                "minimum_required": group_size, "simulations_requested": simulations,
                "selected_bands": selected_bands}
    rng = np.random.default_rng(seed)
    odds_columns = ("odds_home", "odds_draw", "odds_away")
    rows = []
    for simulation in range(1, simulations + 1):
        indices = rng.choice(len(candidates), size=group_size, replace=False)
        group = candidates.iloc[indices].reset_index(drop=True)
        coupons = []
        for coupon_index in range(3):
            coupon = group.iloc[coupon_index * legs_per_coupon:(coupon_index + 1) * legs_per_coupon]
            leg_odds, leg_hits = [], []
            for _, match in coupon.iterrows():
                favorite = int(match["market_favorite"])
                leg_odds.append(float(match[odds_columns[favorite]]))
                leg_hits.append(bool(match["favorite_correct"]))
            won = all(leg_hits)
            coupons.append((won, float(np.prod(leg_odds) * stake_per_coupon) if won else 0.0,
                            float(np.prod(leg_odds))))
        total_return = sum(item[1] for item in coupons)
        total_stake = 3.0 * stake_per_coupon
        correct = int(group["favorite_correct"].sum())
        rows.append({
            "simulation": simulation, "candidate_pool": len(candidates),
            "correct_of_10": correct, "group_hit_rate": correct / group_size,
            "coupon1_won": coupons[0][0], "coupon2_won": coupons[1][0],
            "coupon3_won": coupons[2][0],
            "coupon1_odds": coupons[0][2], "coupon2_odds": coupons[1][2],
            "coupon3_odds": coupons[2][2],
            "winning_coupons": sum(item[0] for item in coupons),
            "total_return": total_return, "total_stake": total_stake,
            "net_result": total_return - total_stake,
            "any_coupon_won": any(item[0] for item in coupons),
        })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "market_1x2_coupon_simulations.csv", index=False)
    return {
        "status": "completed", "simulation_count": simulations,
        "candidate_matches": int(len(candidates)), "selected_bands": selected_bands, "group_size": group_size,
        "coupons_per_group": 3, "legs_per_coupon": legs_per_coupon,
        "unused_match_per_group": group_size - 3 * legs_per_coupon,
        "stake_per_coupon": stake_per_coupon, "stake_per_group": total_stake,
        "groups_with_7_or_more_correct": int((result["correct_of_10"] >= 7).sum()),
        "groups_with_8_or_more_correct": int((result["correct_of_10"] >= 8).sum()),
        "groups_with_any_coupon_won": int(result["any_coupon_won"].sum()),
        "coupon_wins_total": int(result["winning_coupons"].sum()),
        "mean_correct_of_10": float(result["correct_of_10"].mean()),
        "mean_net_result": float(result["net_result"].mean()),
        "profitable_groups": int((result["net_result"] > 0).sum()),
        "break_even_or_better_groups": int((result["net_result"] >= 0).sum()),
        "candidate_band_policy": "Discovery-only gate: include bands with at least 50 discovery matches and at least 75% favorite hit rate; evaluate those selected bands on validation.",
        "simulation_note": "50 independently drawn groups sampled without replacement within each group from validation favorites in the discovery-selected probability bands; groups may overlap between draws. Three 3-leg coupons per group, 100 TL stake each; the 10th match is unused. Actual odds of the selected favorite are used. Historical out-of-sample simulation only, not a forecast.",
    }


def build_reports(frame: pd.DataFrame, output_dir: Path) -> dict:
    required = {"Date", "FTHG", "FTAG"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True)
    data["FTHG"] = pd.to_numeric(data["FTHG"], errors="coerce")
    data["FTAG"] = pd.to_numeric(data["FTAG"], errors="coerce")
    data = data.loc[data["Date"].notna() & data["FTHG"].notna() & data["FTAG"].notna()].copy()
    data["actual_code"] = np.select(
        [data["FTHG"] > data["FTAG"], data["FTHG"] == data["FTAG"]],
        [0, 1], default=2
    ).astype(int)
    market, source_info = market_probabilities(data)
    data = pd.concat([data, market], axis=1)
    data = data.loc[data["odds_valid"]].copy()
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} complete 1X2 odds rows; at least 100 required.")
    data["market_favorite"] = np.argmax(data[["p_home", "p_draw", "p_away"]].to_numpy(), axis=1)
    data["favorite_probability"] = data[["p_home", "p_draw", "p_away"]].max(axis=1)
    data["favorite_correct"] = data["market_favorite"].to_numpy() == data["actual_code"].to_numpy()
    data["season"] = _season_label(data["Date"])

    discovery, validation, sealed = chronological_split(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    scope_rows = []
    for partition_name, part in (("discovery", discovery), ("validation", validation)):
        scope_rows.extend(_report_rows(part, f"{partition_name}_overall", []))
        scope_rows.extend(_report_rows(part, f"{partition_name}_league", ["League"]) if "League" in part else [])
        scope_rows.extend(_report_rows(part, f"{partition_name}_season", ["season"]))
        scope_rows.extend(_report_rows(part, f"{partition_name}_league_season", ["League", "season"]) if "League" in part else [])
    pd.DataFrame(scope_rows).to_csv(output_dir / "market_1x2_segments.csv", index=False)

    bin_rows = []
    favorite_type_rows = []
    for partition_name, part in (("discovery", discovery), ("validation", validation)):
        for band_index, (lower, upper) in enumerate(zip(PROBABILITY_BINS[:-1], PROBABILITY_BINS[1:])):
            chosen = part.loc[(part["favorite_probability"] >= lower) & (part["favorite_probability"] < upper)]
            stats = _summary(chosen)
            favorite_count = int(len(chosen))
            favorite_wins = int(chosen["favorite_correct"].sum()) if favorite_count else 0
            favorite_rate = favorite_wins / favorite_count if favorite_count else np.nan
            favorite_ci_lower, favorite_ci_upper = _wilson_interval(favorite_wins, favorite_count)
            stats.update({
                "partition": partition_name,
                "favorite_probability_band": PROBABILITY_BAND_LABELS[band_index],
                "favorite_hit_rate": favorite_rate,
                "favorite_hit_rate_ci95_lower": favorite_ci_lower,
                "favorite_hit_rate_ci95_upper": favorite_ci_upper,
                "favorite_calibration_gap": favorite_rate - stats["mean_confidence"] if favorite_count else np.nan,
                "favorite_calibration_gap_ci95_lower": favorite_ci_lower - stats["mean_confidence"] if favorite_count else np.nan,
                "favorite_calibration_gap_ci95_upper": favorite_ci_upper - stats["mean_confidence"] if favorite_count else np.nan,
                "favorite_wins": favorite_wins,
                "favorite_count": favorite_count,
            })
            bin_rows.append(stats)
            for code, outcome_name in enumerate(("MS1", "MSX", "MS2")):
                subset = chosen.loc[chosen["market_favorite"] == code]
                n_type = len(subset)
                wins_type = int(subset["favorite_correct"].sum()) if n_type else 0
                ci_low, ci_high = _wilson_interval(wins_type, n_type)
                favorite_type_rows.append({
                    "partition": partition_name,
                    "favorite_probability_band": PROBABILITY_BAND_LABELS[band_index],
                    "favorite_type": outcome_name,
                    "matches": int(n_type), "wins": wins_type,
                    "hit_rate": wins_type / n_type if n_type else np.nan,
                    "hit_rate_ci95_lower": ci_low, "hit_rate_ci95_upper": ci_high,
                })
    pd.DataFrame(bin_rows).to_csv(output_dir / "market_1x2_probability_bands.csv", index=False)
    pd.DataFrame(favorite_type_rows).to_csv(output_dir / "market_1x2_favorite_types.csv", index=False)

    calibration_rows = []
    for partition_name, part in (("discovery", discovery), ("validation", validation)):
        calibration_rows.extend(_outcome_calibration_rows(part, partition_name))
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(
        output_dir / "market_1x2_outcome_calibration.csv", index=False
    )
    pd.DataFrame(_calibration_summary_rows(calibration_rows)).to_csv(
        output_dir / "market_1x2_calibration_summary.csv", index=False
    )

    selected_candidate_bands = _select_candidate_bands(discovery)
    coupon_simulation = _coupon_simulations(validation, output_dir, selected_candidate_bands)
    overall = {
        "method": "raw implied probabilities: 1 / decimal odds per outcome; no normalization and no margin removal",
        "selected_odds_source": source_info,
        "odds_timestamp_limitation": "The archive does not provide odds capture timestamps; pre-kickoff availability is not independently verified.",
        "result_label": "FTHG/FTAG used only to derive H/D/A outcome label; never used as a predictor.",
        "discovery": {"rows": len(discovery), "start": str(discovery["Date"].min()), "end": str(discovery["Date"].max()), **_summary(discovery)},
        "validation": {"rows": len(validation), "start": str(validation["Date"].min()), "end": str(validation["Date"].max()), **_summary(validation)},
        "sealed_holdout": {"rows": len(sealed), "start": str(sealed["Date"].min()), "end": str(sealed["Date"].max()), "metrics_not_computed": True},
        "favorite_hit_rate_discovery": float(discovery["favorite_correct"].mean()),
        "favorite_hit_rate_validation": float(validation["favorite_correct"].mean()),
        "favorite_class_counts_validation": {name: int((validation["market_favorite"] == code).sum()) for code, name in enumerate(("home", "draw", "away"))},
        "actual_class_counts_validation": {name: int((validation["actual_code"] == code).sum()) for code, name in enumerate(("home", "draw", "away"))},
        "raw_probability_definition": "Each outcome probability equals 1 / its decimal odds. Values are intentionally not normalized and can sum above 100% because bookmaker margin is retained.",
        "favorite_probability_bands": list(PROBABILITY_BAND_LABELS),
        "selected_candidate_bands_from_discovery": selected_candidate_bands,
        "coupon_simulation": coupon_simulation,
    }
    (output_dir / "market_1x2_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results_market_1x2"))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, encoding="latin-1", low_memory=False)
    report = build_reports(frame, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
