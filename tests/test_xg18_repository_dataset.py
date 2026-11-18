from modules.historical_engine import HistoricalEngine
from modules.feature_engine import FeatureEngine
from validation.leakage_audit import audit_feature_dataset


def test_repository_xg18_snapshot_schema_and_features():
    history = HistoricalEngine("xg18.csv")
    report = history.schema_report
    assert report["rows"] == 18509
    assert report["columns"] == 39
    assert report["dated_rows"] == 18505
    assert report["undated_rows"] == 4
    assert len(report["leagues"]) == 15

    engine = FeatureEngine(history)
    dated = history.data.loc[history.data["Date"].notna()]
    target = dated.iloc[len(dated) // 2]
    features = engine.build_features(str(target["MatchID"]))
    assert features["MatchID"] == str(target["MatchID"])
    assert features["match_date"] == target["Date"]
    assert "FTHG" not in features
    assert "FTAG" not in features
    assert "Home_xG" not in features
    assert "Away_xG" not in features

    # Full-dataset temporal/leakage audit, not a sampled smoke test.
    audit = audit_feature_dataset(engine, batch_size=250)
    assert audit["passed"] is True
    assert audit["total_source_rows"] == 18509
    assert audit["eligible_matches"] == 18505
    assert audit["audited_matches"] == 18505
    assert audit["failed_count"] == 0
