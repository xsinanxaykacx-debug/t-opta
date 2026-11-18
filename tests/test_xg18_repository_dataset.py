from modules.historical_engine import HistoricalEngine
from modules.feature_engine import FeatureEngine


def test_repository_xg18_snapshot_schema_and_features():
    history = HistoricalEngine("xg18.csv")
    report = history.schema_report
    assert report["rows"] == 18509
    assert report["columns"] == 39
    assert report["dated_rows"] == 18505
    assert report["undated_rows"] == 4
    assert len(report["leagues"]) == 15

    # Pick a fixture late enough to have prior club history.
    dated = history.data.loc[history.data["Date"].notna()]
    target = dated.iloc[len(dated) // 2]
    features = FeatureEngine(history).build_features(str(target["MatchID"]))
    assert features["MatchID"] == str(target["MatchID"])
    assert features["match_date"] == target["Date"]
    assert "FTHG" not in features
    assert "FTAG" not in features
    assert "Home_xG" not in features
    assert "Away_xG" not in features
