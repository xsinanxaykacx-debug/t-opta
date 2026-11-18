import numpy as np
import pandas as pd

from BOT import add_leakage_safe_features


def test_matches_on_same_calendar_day_do_not_update_each_others_history():
    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-12-31T12:00:00Z",
            "2025-01-01T12:00:00Z",
            "2025-01-01T18:00:00Z",
        ], utc=True),
        "League": ["L1", "L1", "L1"],
        "HomeTeam": ["X", "A", "C"],
        "AwayTeam": ["Y", "B", "D"],
        "Home_xG": [4.0, 0.5, 0.5],
        "Away_xG": [4.0, 0.5, 0.5],
        "FTHG": [4, 0, 0],
        "FTAG": [4, 0, 0],
    })

    result = add_leakage_safe_features(frame)

    # The previous day's 4+4 xG is known to both fixtures.
    # The first January 1 fixture must not change the second's prior-day features.
    for column in (
        "p_poisson",
        "lambda_home",
        "lambda_away",
        "league_average",
        "home_advantage",
        "home_recent_for_10",
        "home_recent_against_10",
        "away_recent_for_10",
        "away_recent_against_10",
        "home_recent_for_15",
        "home_recent_against_15",
        "away_recent_for_15",
        "away_recent_against_15",
    ):
        assert np.isclose(
            result.loc[1, column],
            result.loc[2, column],
        )



def test_goal_based_poisson_features_exclude_same_day_results():
    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-12-31T12:00:00Z",
            "2025-01-01T12:00:00Z",
            "2025-01-01T18:00:00Z",
        ], utc=True),
        "League": ["L1", "L1", "L1"],
        "HomeTeam": ["X", "A", "C"],
        "AwayTeam": ["Y", "B", "D"],
        "Home_xG": [4.0, 0.5, 0.5],
        "Away_xG": [4.0, 0.5, 0.5],
        "FTHG": [4, 0, 0],
        "FTAG": [4, 0, 0],
    })

    result = add_leakage_safe_features(frame)

    for column in (
        "p_poisson_goals",
        "lambda_home_goals",
        "lambda_away_goals",
    ):
        assert np.isfinite(result.loc[1, column])
        assert np.isclose(result.loc[1, column], result.loc[2, column])


def test_recent_xg_windows_use_only_prior_matches_and_keep_window_lengths():
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=17, freq="7D", tz="UTC"),
        "League": ["L1"] * 17,
        "HomeTeam": ["A"] * 17,
        "AwayTeam": [f"OPP_{index}" for index in range(17)],
        "Home_xG": [float(index) for index in range(1, 18)],
        "Away_xG": [0.5] * 17,
        "FTHG": [1] * 17,
        "FTAG": [0] * 17,
    })

    result = add_leakage_safe_features(frame)
    target = result.iloc[-1]

    # The final fixture sees only the first 16 matches, never its own xG.
    assert np.isclose(target["home_recent_for"], np.mean([12, 13, 14, 15, 16]))
    assert np.isclose(target["home_recent_for_10"], np.mean(range(7, 17)))
    assert np.isclose(target["home_recent_for_15"], np.mean(range(2, 17)))
    assert np.isclose(target["home_recent_for_10"], 11.5)
    assert np.isclose(target["home_recent_for_15"], 9.0)
