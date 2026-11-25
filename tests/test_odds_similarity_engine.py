import pandas as pd

from modules.odds_similarity_engine import find_similar_matches


def _history():
    return pd.DataFrame([
        {
            "Date": "2019-05-01", "League": "L1", "HomeTeam": "Old Home",
            "AwayTeam": "Old Away", "FTHG": 2, "FTAG": 0,
            "AvgH": 1.80, "AvgD": 3.50, "AvgA": 4.50,
        },
        {
            "Date": "2020-05-01", "League": "L2", "HomeTeam": "Prior Home",
            "AwayTeam": "Prior Away", "FTHG": 1, "FTAG": 1,
            "AvgH": 1.80, "AvgD": 3.50, "AvgA": 4.50,
        },
        {
            "Date": "2021-01-10", "League": "L3", "HomeTeam": "Same Day",
            "AwayTeam": "Same Day Away", "FTHG": 0, "FTAG": 1,
            "AvgH": 1.80, "AvgD": 3.50, "AvgA": 4.50,
        },
        {
            "Date": "2021-01-09", "League": "L4", "HomeTeam": "Near Home",
            "AwayTeam": "Near Away", "FTHG": 0, "FTAG": 2,
            "AvgH": 2.10, "AvgD": 3.20, "AvgA": 3.80,
        },
    ])


def test_similarity_uses_only_strictly_earlier_matches_in_same_era():
    result = find_similar_matches(
        _history(),
        {"home": 1.80, "draw": 3.50, "away": 4.50},
        "2021-01-10",
        limit=10,
    )

    assert result["HomeTeam"].tolist() == ["Prior Home", "Near Home"]
    assert "Old Home" not in result["HomeTeam"].tolist()
    assert "Same Day" not in result["HomeTeam"].tolist()
    assert result.iloc[0]["actual_result"] == "MSX"
    assert result.iloc[0]["probability_distance_pp"] == 0.0


def test_pre_2020_target_never_uses_2020_plus_rows():
    result = find_similar_matches(
        _history(),
        [1.80, 3.50, 4.50],
        "2019-06-01",
        limit=10,
    )

    assert result["HomeTeam"].tolist() == ["Old Home"]


def test_invalid_target_odds_are_rejected():
    try:
        find_similar_matches(_history(), [1.8, 1.0, 4.5], "2021-01-10")
    except ValueError as exc:
        assert "greater than 1" in str(exc)
    else:
        raise AssertionError("Invalid decimal odds should be rejected.")
