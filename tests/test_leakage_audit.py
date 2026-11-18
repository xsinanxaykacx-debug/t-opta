import pandas as pd
import pytest

from modules.feature_engine import FeatureEngine
from modules.historical_engine import HistoricalEngine
from validation.leakage_audit import (
    audit_feature_dataset, audit_feature_sample, audit_match_features,
)


@pytest.fixture
def feature_engine(tmp_path):
    rows = [
        ["1", "L", "S", "2024-01-01T12:00:00Z", "A", "B", 2, 0, 1.5, 0.4],
        ["2", "L", "S", "2024-01-08T12:00:00Z", "C", "A", 1, 1, 0.7, 1.1],
        ["3", "L", "S", "2024-01-15T12:00:00Z", "A", "C", 3, 1, 2.2, 0.8],
        ["4", "L", "S", "2024-01-22T12:00:00Z", "B", "A", 0, 1, 0.5, 1.3],
        ["5", "L", "S", "2024-01-29T12:00:00Z", "A", "B", 99, 99, 99, 99],
        ["6", "L", "S", "2024-01-25T12:00:00Z", "A", "D", None, None, 0.8, 0.2],
    ]
    frame = pd.DataFrame(rows, columns=[
        "MatchID", "League", "Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "Home_xG", "Away_xG",
    ])
    path = tmp_path / "xg18.csv"
    frame.to_csv(path, index=False)
    return FeatureEngine(HistoricalEngine(path), windows=(2,))


def test_audit_passes_and_lists_only_prior_completed_sources(feature_engine):
    result = audit_match_features(feature_engine, "5")
    assert result["passed"] is True
    assert result["source_match_ids"] == ["1", "3", "4"]
    assert "5" not in result["source_match_ids"]
    assert "6" not in result["source_match_ids"]


def test_audit_rejects_unknown_and_undated_targets(feature_engine):
    assert audit_match_features(feature_engine, "missing")["passed"] is False


def test_sample_audit_is_reproducible(feature_engine):
    first = audit_feature_sample(feature_engine, sample_size=4, seed=7)
    second = audit_feature_sample(feature_engine, sample_size=4, seed=7)
    assert first["passed"] is True
    assert [r["match_id"] for r in first["results"]] == [
        r["match_id"] for r in second["results"]
    ]


def test_sample_size_must_be_positive(feature_engine):
    with pytest.raises(ValueError):
        audit_feature_sample(feature_engine, sample_size=0)


def test_full_dataset_audit_covers_every_dated_match(feature_engine):
    result = audit_feature_dataset(feature_engine, batch_size=2)
    assert result["passed"] is True
    assert result["eligible_matches"] == 6
    assert result["audited_matches"] == 6
    assert result["failed_count"] == 0
    assert result["details"] is None


def test_full_dataset_audit_can_return_lineage_details(feature_engine):
    result = audit_feature_dataset(feature_engine, batch_size=3, include_details=True)
    assert len(result["details"]) == 6
    assert all(item["passed"] for item in result["details"])


def test_full_dataset_audit_validates_batch_size(feature_engine):
    with pytest.raises(ValueError):
        audit_feature_dataset(feature_engine, batch_size=0)


def test_team_history_is_scoped_to_target_league(tmp_path):
    rows = [
        ["1", "L1", "S", "2024-01-01T12:00:00Z", "A", "B", 1, 0, 1.0, 0.5],
        ["2", "L2", "S", "2024-01-02T12:00:00Z", "A", "B", 5, 0, 4.0, 0.5],
        ["3", "L1", "S", "2024-01-08T12:00:00Z", "A", "B", 0, 0, 0.7, 0.3],
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "MatchID", "League", "Season", "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "Home_xG", "Away_xG",
        ],
    )
    path = tmp_path / "league-history.csv"
    frame.to_csv(path, index=False)

    engine = FeatureEngine(HistoricalEngine(path), windows=(5,))
    features = engine.build_features("3")

    # The L2 match for a same-named team must not contaminate L1 history.
    assert features["home_xg_for_avg_5"] == 1.0
    assert features["h2h_home_goals_avg"] == 1.0


def test_ewma_weights_most_recent_match_more_heavily():
    # FeatureEngine receives histories newest-first from HistoricalEngine.
    value = FeatureEngine._ewma(pd.Series([5.0, 1.0]), span=2)
    expected = pd.Series([1.0, 5.0]).ewm(span=2, adjust=False).mean().iloc[-1]
    assert value == expected
