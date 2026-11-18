import numpy as np
import pandas as pd
import pytest

import predict
from experiments.history_features import build_upcoming_feature_frame


class DummyModel:
    def predict_proba(self, frame):
        return np.array([0.80, 0.20, 0.55], dtype=float)


def _fixtures():
    return pd.DataFrame(
        [
            {
                "Date": "2026-11-20T18:00:00Z",
                "League": "L",
                "HomeTeam": "A",
                "AwayTeam": "B",
                "odds_Avg>2.5": 1.80,
                "odds_Avg<2.5": 2.00,
                "odds_B365>2.5": 1.82,
                "odds_B365<2.5": 1.98,
            },
            {
                "Date": "2026-11-20T20:00:00Z",
                "League": "L",
                "HomeTeam": "C",
                "AwayTeam": "D",
                "odds_Avg>2.5": 2.20,
                "odds_Avg<2.5": 1.70,
                "odds_B365>2.5": 2.15,
                "odds_B365<2.5": 1.72,
            },
            {
                "Date": "2026-11-21T18:00:00Z",
                "League": "L",
                "HomeTeam": "E",
                "AwayTeam": "F",
                "odds_Avg>2.5": np.nan,
                "odds_Avg<2.5": np.nan,
                "odds_B365>2.5": np.nan,
                "odds_B365<2.5": np.nan,
            },
        ]
    )


def test_predict_fixtures_applies_frozen_confidence_and_market_gates(monkeypatch):
    def fake_features(fixtures, history_path):
        out = fixtures.copy()
        out["p_poisson"] = 0.55
        out["hist_home_matches_available"] = 8
        out["hist_away_matches_available"] = 9
        return out, {
            "history_git_blob_sha": "history-sha",
            "history_date_max": "2026-11-14T20:00:00+00:00",
        }

    monkeypatch.setattr(predict, "build_upcoming_feature_frame", fake_features)
    metadata = {
        "model_name": "odds_aware",
        "confidence_threshold": 0.735,
        "source_data": {"xg18_git_blob_sha": "history-sha"},
    }

    result = predict.predict_fixtures(
        _fixtures(),
        DummyModel(),
        metadata,
        history_path="xg18.csv",
    )

    assert result["selected"].tolist() == [True, True, False]
    assert result["predicted_side"].tolist() == [
        "OVER 2.5",
        "UNDER 2.5",
        "OVER 2.5",
    ]
    assert result.loc[0, "chosen_odds"] == pytest.approx(1.80)
    assert result.loc[1, "chosen_odds"] == pytest.approx(1.70)
    assert result.loc[0, "expected_value_per_unit"] == pytest.approx(0.44)
    assert result.loc[1, "expected_value_per_unit"] == pytest.approx(0.36)
    assert result.loc[2, "market_valid"] == False


def test_predict_fixtures_rejects_history_snapshot_mismatch(monkeypatch):
    def fake_features(fixtures, history_path):
        out = fixtures.copy()
        out["p_poisson"] = 0.55
        out["hist_home_matches_available"] = 8
        out["hist_away_matches_available"] = 9
        return out, {
            "history_git_blob_sha": "different-history-sha",
            "history_date_max": "2026-11-14T20:00:00+00:00",
        }

    monkeypatch.setattr(predict, "build_upcoming_feature_frame", fake_features)
    metadata = {
        "confidence_threshold": 0.735,
        "source_data": {"xg18_git_blob_sha": "history-sha"},
    }

    with pytest.raises(ValueError, match="different xg18.csv snapshot"):
        predict.predict_fixtures(
            _fixtures(),
            DummyModel(),
            metadata,
            history_path="xg18.csv",
        )


def test_upcoming_feature_frame_uses_past_xg18_history_only():
    history = pd.read_csv("xg18.csv", encoding="utf-8-sig", low_memory=False)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce", utc=True)
    last = history.loc[history["Date"].notna()].sort_values("Date").iloc[-1]

    fixture = pd.DataFrame(
        [
            {
                "Date": (last["Date"] + pd.Timedelta(days=7)).isoformat(),
                "League": str(last["League"]),
                "HomeTeam": str(last["HomeTeam"]),
                "AwayTeam": str(last["AwayTeam"]),
            }
        ]
    )

    featured, metadata = build_upcoming_feature_frame(fixture, "xg18.csv")

    assert len(featured) == 1
    assert metadata["history_git_blob_sha"]
    assert featured.loc[0, "hist_home_matches_available"] >= 1
    assert featured.loc[0, "hist_away_matches_available"] >= 1
    assert 0.0 < featured.loc[0, "p_poisson"] < 1.0
