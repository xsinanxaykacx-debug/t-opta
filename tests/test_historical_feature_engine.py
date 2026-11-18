import pandas as pd
import pytest

from modules.historical_engine import HistoricalEngine
from modules.feature_engine import FeatureEngine


@pytest.fixture
def engine(tmp_path):
    rows = [
        ["1","L","2024/2025","2024-01-01T12:00:00Z","A","B","R",2,0,1,0,1.5,.4,None,None],
        ["2","L","2024/2025","2024-01-08T12:00:00Z","C","A","R",1,1,0,0,.7,1.1,None,None],
        ["3","L","2024/2025","2024-01-15T12:00:00Z","A","C","R",3,1,1,0,2.2,.8,None,None],
        ["4","L","2024/2025","2024-01-22T12:00:00Z","B","A","R",0,1,0,0,.5,1.3,None,None],
        ["5","L","2024/2025","2024-01-29T12:00:00Z","A","B","R",99,99,99,99,99,99,None,None],
    ]
    frame = pd.DataFrame(rows, columns=[
        "MatchID","League","Season","Date","HomeTeam","AwayTeam","Referee",
        "FTHG","FTAG","HT_Home_Goals","HT_Away_Goals","Home_xG","Away_xG",
        "HT_Home_xG","HT_Away_xG"
    ])
    path = tmp_path / "xg18.csv"
    frame.to_csv(path, index=False)
    return HistoricalEngine(path)


def test_schema_loads_and_date_is_utc(engine):
    assert len(engine.data) == 5
    assert str(engine.data["Date"].dtype) == "datetime64[ns, UTC]"


def test_team_history_is_strictly_before_cutoff(engine):
    history = engine.get_team_history("A", "2024-01-29T12:00:00Z")
    assert set(history["MatchID"]) == {"1", "2", "3", "4"}
    assert (history["Date"] < pd.Timestamp("2024-01-29T12:00:00Z")).all()
    assert set(history["TeamSide"]) == {"home", "away"}


def test_team_history_normalizes_perspective(engine):
    history = engine.get_team_history("A", "2024-01-10T00:00:00Z")
    row = history.iloc[0]
    assert row["TeamSide"] == "away"
    assert row["TeamGoals"] == 1
    assert row["TeamXG"] == 1.1


def test_invalid_date_rows_are_audited_but_not_returned(tmp_path):
    frame = pd.DataFrame([["1","L","S","","A","B",0,0,1,1]], columns=[
        "MatchID","League","Season","Date","HomeTeam","AwayTeam",
        "FTHG","FTAG","Home_xG","Away_xG"
    ])
    path = tmp_path / "xg.csv"
    frame.to_csv(path, index=False)
    hist = HistoricalEngine(path)
    assert hist.schema_report["undated_rows"] == 1
    assert hist.get_team_history("A", "2025-01-01").empty


def test_feature_engine_excludes_target_match_values(engine):
    feature = FeatureEngine(engine, windows=(2,)).build_features("5")
    assert feature["home_matches_available"] == 2
    assert feature["home_goals_for_avg_2"] == pytest.approx(2.5)
    assert feature["home_xg_for_avg_2"] == pytest.approx(1.85)
    assert feature["h2h_matches_available"] == 2
    assert feature["home_goals_for_avg_2"] != 99


def test_unknown_or_undated_match_is_rejected(engine):
    with pytest.raises(KeyError):
        FeatureEngine(engine).build_features("missing")
