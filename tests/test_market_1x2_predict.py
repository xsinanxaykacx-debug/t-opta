import json

import pandas as pd

from market_1x2_predict import predict_upcoming


def _upcoming(n=10):
    return pd.DataFrame({
        "Date": pd.date_range("2026-11-20", periods=n, freq="D"),
        "HomeTeam": [f"Home {i}" for i in range(n)],
        "AwayTeam": [f"Away {i}" for i in range(n)],
        "AvgH": [1.3] * n,
        "AvgD": [4.0] * n,
        "AvgA": [5.0] * n,
    })


def test_upcoming_picker_uses_raw_odds_and_builds_three_coupons(tmp_path):
    selected_bands = [{
        "band": "71.00%-80.99%", "lower": 0.71, "upper": 0.81,
        "discovery_matches": 800, "discovery_hit_rate": 0.77,
    }]
    result = predict_upcoming(_upcoming(), selected_bands, tmp_path)
    assert result["status"] == "ready"
    assert result["fixtures_received"] == 10
    assert result["eligible_fixtures"] == 10
    assert result["picks_in_group"] == 10
    assert result["coupon_count"] == 3
    assert result["coupon_stake_total"] == 300
    picks = pd.read_csv(tmp_path / "upcoming_market_picks.csv")
    assert picks.iloc[0]["market_favorite"] == "MS1"
    assert picks.iloc[0]["favorite_probability_pct"] == 100 / 1.3
    assert picks.iloc[0]["favorite_odds"] == 1.3
    coupons = pd.read_csv(tmp_path / "upcoming_coupons.csv")
    assert len(coupons) == 9
    assert coupons.groupby("coupon").size().to_dict() == {
        "Kupon 1": 3, "Kupon 2": 3, "Kupon 3": 3
    }
    summary = json.loads((tmp_path / "upcoming_prediction_summary.json").read_text())
    assert summary["unused_tenth_pick"]["pick_rank"] == 10
    assert "FTHG" not in picks.columns
    assert "FTAG" not in picks.columns


def test_upcoming_picker_refuses_to_make_incomplete_ten_match_coupons(tmp_path):
    selected_bands = [{
        "band": "71.00%-80.99%", "lower": 0.71, "upper": 0.81,
        "discovery_matches": 800, "discovery_hit_rate": 0.77,
    }]
    result = predict_upcoming(_upcoming(7), selected_bands, tmp_path)
    assert result["status"] == "fewer_than_10_eligible_fixtures"
    assert result["coupon_count"] == 0
    assert result["eligible_fixtures"] == 7
    coupons = pd.read_csv(tmp_path / "upcoming_coupons.csv")
    assert coupons.empty
