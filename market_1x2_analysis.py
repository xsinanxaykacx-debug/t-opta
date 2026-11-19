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
PROBABILITY_BINS = (0.50, 0.60, 0.70, 0.80, 1.0000001)


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
        probs[use] = inverse[use] / inverse_sum[use, None]
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
                "mean_margin": np.nan, "brier_multiclass": np.nan}
    p = group[["p_home", "p_draw", "p_away"]].to_numpy(float)
    actual = group["actual_code"].to_numpy(int)
    predicted = np.argmax(p, axis=1)
    onehot = np.eye(3)[actual]
    return {
        "matches": int(n),
        "accuracy": float(np.mean(predicted == actual)),
        "mean_confidence": float(np.mean(np.max(p, axis=1))),
        "mean_margin": float(group["market_margin"].mean()),
        "brier_multiclass": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
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
            row[f"{name}_predicted_accuracy"] = float(np.mean(actual[selected] == code)) if selected.any() else np.nan
            row[f"{name}_actual_count"] = int(np.sum(actual == code))
        rows.append(row)
    return rows


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
    for partition_name, part in (("discovery", discovery), ("validation", validation)):
        for lower, upper in zip(PROBABILITY_BINS[:-1], PROBABILITY_BINS[1:]):
            chosen = part.loc[
                (part["favorite_probability"] >= lower)
                & (part["favorite_probability"] < upper)
            ]
            stats = _summary(chosen)
            stats.update({
                "partition": partition_name,
                "favorite_probability_band": f"{lower:.0%}-{min(upper, 1.0):.0%}",
                "favorite_hit_rate": float(chosen["favorite_correct"].mean()) if len(chosen) else np.nan,
                "favorite_count": int(len(chosen)),
            })
            bin_rows.append(stats)
    pd.DataFrame(bin_rows).to_csv(output_dir / "market_1x2_probability_bands.csv", index=False)

    overall = {
        "method": "de-vigged implied probabilities from one complete bookmaker H/D/A triplet",
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
