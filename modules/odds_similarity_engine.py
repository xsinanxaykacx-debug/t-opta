"""Find historically similar 1X2 markets without crossing the 2020 data-era boundary.

Similarity is measured on de-vigged H/D/A probabilities (root-mean-square
absolute difference, in percentage points), not on post-match outcomes.
Only completed matches from strictly earlier calendar days are eligible.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from market_1x2_analysis import market_probabilities


_RESULT_NAMES = ("MS1", "MSX", "MS2")
_REQUIRED_HISTORY_COLUMNS = ("Date", "FTHG", "FTAG")


def _target_odds_frame(target_odds: Mapping[str, Any] | Sequence[Any]) -> pd.DataFrame:
    if isinstance(target_odds, Mapping):
        values = [target_odds.get(key) for key in ("home", "draw", "away")]
    else:
        values = list(target_odds)
    if len(values) != 3:
        raise ValueError("target_odds must contain home, draw and away prices.")

    prices = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 1.0).any():
        raise ValueError("All target 1X2 odds must be finite decimal prices greater than 1.")
    return pd.DataFrame([{"AvgH": prices[0], "AvgD": prices[1], "AvgA": prices[2]}])


def find_similar_matches(
    history: pd.DataFrame,
    target_odds: Mapping[str, Any] | Sequence[Any],
    target_date: Any,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    """Return nearest earlier matches from the same pre-2020 / 2020+ era.

    The source archive has no odds-capture timestamps. This function therefore
    reports historical similarity only; it does not claim that archived prices
    were captured at a verified pre-kickoff instant.
    """
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    missing = [column for column in _REQUIRED_HISTORY_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"History is missing required columns: {missing}")

    cutoff = pd.to_datetime(target_date, errors="coerce", utc=True)
    if pd.isna(cutoff):
        raise ValueError(f"Invalid target_date: {target_date!r}")

    target_market, _ = market_probabilities(_target_odds_frame(target_odds))
    target_probability = target_market.loc[0, ["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

    data = history.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True)
    data["FTHG"] = pd.to_numeric(data["FTHG"], errors="coerce")
    data["FTAG"] = pd.to_numeric(data["FTAG"], errors="coerce")

    target_era = "pre_2020" if cutoff.year < 2020 else "2020_plus"
    data_era = np.where(data["Date"].dt.year < 2020, "pre_2020", "2020_plus")
    eligible = (
        data["Date"].notna()
        & (data["Date"].dt.normalize() < cutoff.normalize())
        & (data_era == target_era)
        & data["FTHG"].notna()
        & data["FTAG"].notna()
    )
    data = data.loc[eligible].copy()
    if data.empty:
        return pd.DataFrame(columns=[
            "Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
            "actual_result", "odds_home", "odds_draw", "odds_away",
            "p_home", "p_draw", "p_away", "probability_distance_pp",
        ])

    historical_market, _ = market_probabilities(data)
    data = data.join(historical_market)
    data = data.loc[data["odds_valid"]].copy()
    if data.empty:
        return pd.DataFrame(columns=[
            "Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
            "actual_result", "odds_home", "odds_draw", "odds_away",
            "p_home", "p_draw", "p_away", "probability_distance_pp",
        ])

    probabilities = data[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    distances = np.sqrt(np.mean(np.square(probabilities - target_probability), axis=1)) * 100.0
    data["probability_distance_pp"] = distances
    data["actual_result"] = np.select(
        [data["FTHG"] > data["FTAG"], data["FTHG"] == data["FTAG"]],
        _RESULT_NAMES[:2],
        default="MS2",
    )

    output_columns = [
        column for column in (
            "Date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
            "actual_result", "odds_home", "odds_draw", "odds_away",
            "p_home", "p_draw", "p_away", "probability_distance_pp",
            "odds_source", "MatchID",
        ) if column in data.columns
    ]
    return data.sort_values(
        ["probability_distance_pp", "Date"],
        ascending=[True, False],
        kind="stable",
    ).loc[:, output_columns].head(limit).reset_index(drop=True)
