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
