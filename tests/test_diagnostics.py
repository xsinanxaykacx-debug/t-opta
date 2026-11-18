import numpy as np
import pandas as pd

from diagnostics import (
    directional_roi_report,
    market_model_brier_report,
    reliability_report,
    run_diagnostics,
    _bet_rows,
)


def test_market_and_model_brier_use_same_valid_market_rows():
    y = np.array([1, 0, 1], dtype=int)
    p_model = np.array([0.8, 0.2, 0.7])
    p_market = np.array([0.6, 0.4, 0.5])
    market_valid = np.array([True, True, False])

    report = market_model_brier_report(
        y, p_model, p_market, market_valid
    ).iloc[0]

    assert report["matches"] == 2
    assert np.isclose(report["model_brier"], 0.04)
    assert np.isclose(report["market_brier"], 0.16)
    assert np.isclose(
        report["delta_brier_model_minus_market"],
        -0.12,
    )


def test_directional_roi_keeps_over_and_under_separate():
    y = np.array([1, 0, 1, 0], dtype=int)
    p_model = np.array([0.70, 0.30, 0.80, 0.20])
    over_odds = np.array([2.0, 2.0, 1.5, 1.5])
    under_odds = np.array([2.2, 2.2, 1.8, 1.8])
    market_valid = np.ones(4, dtype=bool)
    edge = np.array([0.10, -0.10, 0.05, -0.05])

    bets = _bet_rows(
        y,
        over_odds,
        under_odds,
        market_valid,
        edge,
        threshold=0.04,
    )
    report = directional_roi_report(y, p_model, bets)
    over = report.loc[report["side"] == "OVER"].iloc[0]
    under = report.loc[report["side"] == "UNDER"].iloc[0]

    assert over["bets"] == 2
    assert under["bets"] == 2
    assert np.isclose(over["bet_hit_rate"], 1.0)
    assert np.isclose(under["bet_hit_rate"], 1.0)
    assert np.isclose(over["roi_pct"], 75.0)
    assert np.isclose(under["roi_pct"], 100.0)
    assert np.isclose(over["classification_accuracy"], 1.0)
    assert np.isclose(under["classification_accuracy"], 1.0)


def test_reliability_ece_contributions_sum_to_overall_ece():
    y = np.array([0, 0, 1, 1], dtype=int)
    p_model = np.array([0.1, 0.2, 0.8, 0.9])

    report = reliability_report(y, p_model, bins=10)

    assert np.isclose(
        report["ece_contribution"].sum(),
        report["overall_ece"].iloc[0],
    )
    assert report["matches"].sum() == len(y)


def test_run_diagnostics_writes_five_csv_reports(tmp_path):
    class DummyModel:
        @staticmethod
        def predict_proba(frame):
            return np.array([0.8, 0.2, 0.7, 0.3])

        @staticmethod
        def _y(frame):
            return np.array([1, 0, 1, 0])

    frame = pd.DataFrame({"fixture": [1, 2, 3, 4]})

    def fake_market_data(_frame):
        return (
            np.array([0.6, 0.4, 0.6, 0.4]),
            np.array([2.0, 2.0, 1.8, 2.0]),
            np.array([2.0, 2.0, 2.0, 1.8]),
            np.array([True, True, True, True]),
        )

    reports = run_diagnostics(
        df=frame,
        model=DummyModel(),
        label="TEST",
        threshold=0.05,
        market_data_fn=fake_market_data,
        output_dir=tmp_path,
    )

    expected = {
        "directional_roi",
        "market_model_brier",
        "odds_buckets",
        "edge_buckets",
        "reliability_bins",
    }

    assert set(reports) == expected
    assert len(list(tmp_path.glob("test_*.csv"))) == 5
