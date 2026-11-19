import json

import pandas as pd

from market_1x2_predict import predict_upcoming


def _upcoming(n=10, same_day=False):
    dates = ([pd.Timestamp("2026-11-20")] * n if same_day else
             pd.date_range("2026-11-20", periods=n, freq="D"))
    return pd.DataFrame({
        "Date": dates,
        "HomeTeam": [f"Home {i}" for i in range(n)],
        "AwayTeam": [f"Away {i}" for i in range(n)],
        "AvgH": [1.3] * n,
        "AvgD": [4.0] * n,
        "AvgA": [5.0] * n,
    })


def test_upcoming_picker_uses_raw_odds_and_builds_three_coupons(tmp_path):
    selected_bands = [{
        "band": "71.00%-80.99%", "lower": 0.71, "upper": 0.81,
        "favorite_type": "MS1", "discovery_matches": 800, "discovery_hit_rate": 0.77,
    }]
    result = predict_upcoming(_upcoming(same_day=True), selected_bands, tmp_path)
    assert result["status"] == "ready"
    assert result["fixtures_received"] == 10
    assert result["eligible_fixtures"] == 10
    assert result["picks_in_group"] == 10
    assert result["picks_exported"] == 10
    assert result["coupon_count"] == 3
    assert result["coupon_stake_total"] == 300
    assert result["eligible_dates_with_10plus"] == 1
    assert result["selected_match_date"] == "2026-11-20"
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
        "favorite_type": "MS1", "discovery_matches": 800, "discovery_hit_rate": 0.77,
    }]
    result = predict_upcoming(_upcoming(7, same_day=True), selected_bands, tmp_path)
    assert result["status"] == "fewer_than_10_eligible_fixtures"
    assert result["coupon_count"] == 0
    assert result["picks_exported"] == 9
    assert result["picks_in_group"] == 0
    assert result["eligible_fixtures"] == 7
    assert result["picks_exported"] == 7
    assert result["picks_in_group"] == 0
    picks = pd.read_csv(tmp_path / "upcoming_market_picks.csv")
    assert len(picks) == 7
    coupons = pd.read_csv(tmp_path / "upcoming_coupons.csv")
    assert coupons.empty


def test_upcoming_picker_never_mixes_fixtures_from_different_dates(tmp_path):
    selected_bands = [{
        "band": "71.00%-80.99%", "lower": 0.71, "upper": 0.81,
        "favorite_type": "MS1", "discovery_matches": 800, "discovery_hit_rate": 0.77,
    }]
    result = predict_upcoming(_upcoming(10, same_day=False), selected_bands, tmp_path)
    assert result["status"] == "no_same_day_group_of_10"
    assert result["eligible_fixtures"] == 10
    assert result["eligible_dates_with_10plus"] == 0
    assert result["coupon_count"] == 0
    assert result["picks_exported"] == 10
    assert result["picks_in_group"] == 0
    picks = pd.read_csv(tmp_path / "upcoming_market_picks.csv")
    assert len(picks) == 10


def test_picker_does_not_borrow_other_outcome_types_to_fill_same_day_group(tmp_path):
    selected_bands = [{
        "band": "71.00%-80.99%", "lower": 0.71, "upper": 0.81,
        "favorite_type": "MS1", "discovery_matches": 800, "discovery_hit_rate": 0.77,
    }]
    frame = _upcoming(same_day=True)
    frame.loc[0, ["AvgH", "AvgD", "AvgA"]] = [4.0, 1.3, 5.0]
    result = predict_upcoming(frame, selected_bands, tmp_path)
    assert result["eligible_fixtures"] == 9
    assert result["status"] == "fewer_than_10_eligible_fixtures"
    assert result["coupon_count"] == 0
