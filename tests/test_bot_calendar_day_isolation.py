import numpy as np
import pandas as pd

from BOT import add_leakage_safe_features


def test_matches_on_same_calendar_day_do_not_update_each_others_history():
    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2025-01-01T12:00:00Z",
            "2025-01-01T18:00:00Z",
        ], utc=True),
        "League": ["L1", "L1"],
        "HomeTeam": ["A", "C"],
        "AwayTeam": ["B", "D"],
        "Home_xG": [4.0, 0.5],
        "Away_xG": [4.0, 0.5],
        "FTHG": [4, 0],
        "FTAG": [4, 0],
    })

    result = add_leakage_safe_features(frame)

    assert np.isclose(
        result.loc[0, "p_poisson"],
        result.loc[1, "p_poisson"],
    )
    assert np.isclose(
        result.loc[0, "lambda_home"],
        result.loc[1, "lambda_home"],
    )
    assert np.isclose(
        result.loc[0, "lambda_away"],
        result.loc[1, "lambda_away"],
    )
