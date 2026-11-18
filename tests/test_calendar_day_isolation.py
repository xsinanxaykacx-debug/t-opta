import pandas as pd
import pytest

from modules.historical_engine import HistoricalEngine, filter_strict_past
from modules.feature_engine import FeatureEngine
from validation.leakage_audit import audit_match_features


def test_filter_strict_past_excludes_entire_target_calendar_day():
    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-01-01T09:00:00Z", "2024-01-02T08:00:00Z",
            "2024-01-02T20:00:00Z", "2024-01-03T00:00:00Z",
        ], utc=True),
        "MatchID": ["old", "same-early", "same-late", "future"],
    })
    result = filter_strict_past(frame, "2024-01-02T19:00:00Z")
    assert result["MatchID"].tolist() == ["old"]


def test_team_history_excludes_same_day_even_with_earlier_timestamp(tmp_path):
    frame = pd.DataFrame([
        ["1", "L", "S", "2024-01-02T08:00:00Z", "A", "B", 1, 0, 1.0, 0.2],
        ["2", "L", "S", "2024-01-02T20:00:00Z", "C", "A", 0, 0, 0.4, 0.4],
        ["3", "L", "S", "2024-01-03T12:00:00Z", "A", "C", 2, 0, 1.5, 0.3],
    ], columns=[
        "MatchID", "League", "Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "Home_xG", "Away_xG",
    ])
    path = tmp_path / "matches.csv"
    frame.to_csv(path, index=False)
    engine = HistoricalEngine(path)
    history = engine.get_team_history("A", "2024-01-02T20:00:00Z")
    assert history.empty


def test_leakage_audit_reports_only_prior_calendar_days(tmp_path):
    frame = pd.DataFrame([
        ["1", "L", "S", "2024-01-01", "A", "B", 1, 0, 1.0, 0.2],
        ["2", "L", "S", "2024-01-02", "C", "A", 0, 0, 0.4, 0.4],
        ["3", "L", "S", "2024-01-02", "A", "C", 2, 0, 1.5, 0.3],
    ], columns=[
        "MatchID", "League", "Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "Home_xG", "Away_xG",
    ])
    path = tmp_path / "matches.csv"
    frame.to_csv(path, index=False)
    engine = FeatureEngine(HistoricalEngine(path), windows=(2,))
    audit = audit_match_features(engine, "3")
    assert audit["passed"]
    assert audit["source_match_ids"] == ["1"]
