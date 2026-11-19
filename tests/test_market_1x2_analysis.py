import numpy as np
import pandas as pd
import pytest

import market_1x2_analysis as market


def _frame(n=120):
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "Date": dates,
        "League": ["L1" if i % 2 else "L2" for i in range(n)],
        "FTHG": [2 if i % 3 == 0 else (0 if i % 3 == 1 else 1) for i in range(n)],
        "FTAG": [0 if i % 3 == 0 else (0 if i % 3 == 1 else 2) for i in range(n)],
        "AvgH": [1.8] * n,
        "AvgD": [3.6] * n,
        "AvgA": [4.5] * n,
        "B365H": [1.7] * n,
        "B365D": [3.8] * n,
        "B365A": [5.0] * n,
    })


def test_market_probabilities_remove_overround_and_use_complete_source():
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
    assert probabilities.loc[0, "p_home"] == pytest.approx((1 / 2) / (1 / 2 + 1 / 3 + 1 / 4))
    assert probabilities.loc[1, "p_home"] == pytest.approx((1 / 1.8) / (1 / 1.8 + 1 / 3.2 + 1 / 4.5))
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
    assert report["sealed_holdout"]["metrics_not_computed"] is True
    assert report["odds_timestamp_limitation"]
    segments = pd.read_csv(tmp_path / "market_1x2_segments.csv")
    assert {"discovery_overall", "validation_overall", "validation_league_season"} <= set(segments["scope"])
    assert "draw_actual_recall" in segments.columns
    assert "away_f1" in segments.columns
    bands = pd.read_csv(tmp_path / "market_1x2_probability_bands.csv")
    assert set(bands["partition"]) == {"discovery", "validation"}
    assert (tmp_path / "market_1x2_summary.json").exists()


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
