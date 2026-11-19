import numpy as np
import pandas as pd
import pytest

import market_1x2_analysis as market


def _frame(n=120):
    dates = pd.date_range("2025-01-01", periods=(n + 9) // 10, freq="D", tz="UTC").repeat(10)[:n]
    return pd.DataFrame({
        "Date": dates,
        "League": ["L1" if i % 2 else "L2" for i in range(n)],
        "FTHG": [2] * n,
        "FTAG": [0] * n,
        "AvgH": [1.3] * n,
        "AvgD": [4.0] * n,
        "AvgA": [5.0] * n,
        "B365H": [1.25] * n,
        "B365D": [4.2] * n,
        "B365A": [5.2] * n,
    })


def test_market_probabilities_keep_raw_implied_probability_and_use_complete_source():
    frame = pd.DataFrame({
        "AvgH": [2.0, np.nan],
        "AvgD": [3.0, np.nan],
        "AvgA": [4.0, np.nan],
        "B365H": [1.8, 1.8],
        "B365D": [3.2, 3.2],
        "B365A": [4.5, 4.5],
    })
    probabilities, info = market.market_probabilities(frame)
    assert info["source_counts"] == {"Avg": 1, "BbAv": 0, "B365": 1}
    assert probabilities.loc[0, "p_home"] == pytest.approx(1 / 2)
    assert probabilities.loc[0, "p_draw"] == pytest.approx(1 / 3)
    assert probabilities.loc[0, "p_away"] == pytest.approx(1 / 4)
    assert probabilities.loc[1, "p_home"] == pytest.approx(1 / 1.8)
    assert probabilities.loc[1, "p_home"] + probabilities.loc[1, "p_draw"] + probabilities.loc[1, "p_away"] > 1.0
    assert probabilities.loc[1, "odds_source"] == "B365"


def test_market_probabilities_never_mix_bookmakers():
    frame = pd.DataFrame({
        "AvgH": [2.0], "AvgD": [3.0], "AvgA": [np.nan],
        "B365H": [np.nan], "B365D": [3.2], "B365A": [4.5],
    })
    with pytest.raises(ValueError, match="Complete 1X2 odds triplet"):
        market.market_probabilities(frame)


def test_chronological_split_uses_whole_dates_and_seals_last_partition():
    frame = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")})
    discovery, validation, holdout = market.chronological_split(frame)
    assert discovery["Date"].max() < validation["Date"].min()
    assert validation["Date"].max() < holdout["Date"].min()
    assert len(discovery) > len(validation) > 0
    assert len(holdout) > 0


def test_build_reports_creates_market_only_reports_and_does_not_score_holdout(tmp_path):
    report = market.build_reports(_frame(), tmp_path)
    assert report["discovery"]["rows"] > report["validation"]["rows"]
    assert report["selected_odds_source"]["source_counts"]["Avg"] == 120
    assert report["method"].startswith("raw implied probabilities")
    assert report["coupon_simulation"]["status"] == "completed"
    assert report["coupon_simulation"]["simulation_count"] == 50
    assert report["coupon_simulation"]["unique_fixture_groups"] <= 50
    assert report["coupon_simulation"]["duplicate_fixture_group_draws"] == (
        50 - report["coupon_simulation"]["unique_fixture_groups"]
    )
    assert report["coupon_simulation"]["eligible_match_dates"] >= 1
    assert report["selected_candidate_bands_from_discovery"]
    assert all(item["discovery_matches"] >= 50 and item["discovery_hit_rate"] >= 0.75
               for item in report["selected_candidate_bands_from_discovery"])
    assert {item["favorite_type"] for item in report["selected_candidate_bands_from_discovery"]} == {"MS1"}
    assert report["sealed_holdout"]["metrics_not_computed"] is True
    assert report["odds_timestamp_limitation"]
    segments = pd.read_csv(tmp_path / "market_1x2_segments.csv")
    assert {"discovery_overall", "validation_overall", "validation_league_season"} <= set(segments["scope"])
    assert "draw_actual_recall" in segments.columns
    assert "away_f1" in segments.columns
    bands = pd.read_csv(tmp_path / "market_1x2_probability_bands.csv")
    assert set(bands["partition"]) == {"discovery", "validation"}
    assert {
        "favorite_hit_rate_ci95_lower",
        "favorite_hit_rate_ci95_upper",
        "favorite_calibration_gap",
        "favorite_calibration_gap_ci95_lower",
        "favorite_calibration_gap_ci95_upper",
        "favorite_wins",
    } <= set(bands.columns)
    nonempty = bands["favorite_count"] > 0
    assert (bands.loc[nonempty, "favorite_hit_rate_ci95_lower"] <=
            bands.loc[nonempty, "favorite_hit_rate"]).all()
    assert (bands.loc[nonempty, "favorite_hit_rate"] <=
            bands.loc[nonempty, "favorite_hit_rate_ci95_upper"]).all()
    assert np.allclose(
        bands.loc[nonempty, "favorite_calibration_gap"],
        bands.loc[nonempty, "favorite_hit_rate"] - bands.loc[nonempty, "mean_confidence"],
    )
    assert (tmp_path / "market_1x2_summary.json").exists()
    favorite_types = pd.read_csv(tmp_path / "market_1x2_favorite_types.csv")
    assert set(favorite_types["favorite_type"]) == {"MS1", "MSX", "MS2"}
    assert set(favorite_types["favorite_probability_band"]) == {
        "60.00%-70.99%", "71.00%-80.99%", "81.00%-90.99%", "91.00%-100.00%"
    }
    simulations = pd.read_csv(tmp_path / "market_1x2_coupon_simulations.csv")
    assert len(simulations) == 50
    assert simulations["correct_of_10"].between(0, 10).all()
    assert simulations["match_date"].nunique() <= report["coupon_simulation"]["eligible_match_dates"]
    assert (simulations["candidate_pool"] >= 10).all()
    assert (simulations["total_stake"] == 300).all()
    assert report["coupon_simulation"]["groups_with_7_or_more_correct"] == int(
        (simulations["correct_of_10"] >= 7).sum()
    )


def test_result_label_is_derived_from_final_goals_not_features(tmp_path):
    frame = _frame()
    frame["Home_xG"] = np.arange(len(frame)) * 100
    frame["Away_xG"] = -np.arange(len(frame)) * 100
    report = market.build_reports(frame, tmp_path)
    assert report["result_label"].startswith("FTHG/FTAG used only")

def test_outcome_calibration_reports_each_class_probability_separately(tmp_path):
    report = market.build_reports(_frame(), tmp_path)
    calibration = pd.read_csv(tmp_path / "market_1x2_outcome_calibration.csv")
    assert set(calibration["partition"]) == {"discovery", "validation"}
    assert set(calibration["outcome"]) == {"home", "draw", "away"}
    assert {"mean_predicted_probability", "observed_frequency", "calibration_error",
            "observed_frequency_ci95_lower", "observed_frequency_ci95_upper"} <= set(calibration.columns)
    assert calibration["matches"].sum() == 3 * (report["discovery"]["rows"] + report["validation"]["rows"])
    assert calibration["matches"].gt(0).all()
    assert (calibration["observed_frequency_ci95_lower"] <= calibration["observed_frequency"]).all()
    assert (calibration["observed_frequency"] <= calibration["observed_frequency_ci95_upper"]).all()
    summary = pd.read_csv(tmp_path / "market_1x2_calibration_summary.csv")
    assert len(summary) == 6
    assert set(summary["outcome"]) == {"home", "draw", "away"}
    assert summary["matches"].sum() == 3 * (report["discovery"]["rows"] + report["validation"]["rows"])
    assert summary["weighted_absolute_calibration_error"].between(0, 1).all()


def test_candidate_gate_is_independent_for_ms1_msx_and_ms2():
    discovery = pd.DataFrame({
        "favorite_probability": [0.75] * 160,
        "market_favorite": [0] * 80 + [1] * 80,
        "favorite_correct": [True] * 70 + [False] * 10 + [True] * 40 + [False] * 40,
    })
    selected = market._select_candidate_bands(discovery, minimum_matches=50, minimum_hit_rate=0.75)
    assert len(selected) == 1
    assert selected[0]["favorite_type"] == "MS1"
    assert selected[0]["discovery_matches"] == 80
    assert selected[0]["discovery_hit_rate"] == pytest.approx(0.875)
