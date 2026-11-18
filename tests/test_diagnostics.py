import numpy as np
import pandas as pd

from BOT import HistoricalXGModel, PoissonOnlyModel, confidence_policy_analysis, market_data

from diagnostics import (
    directional_roi_report,
    market_model_brier_report,
    reliability_report,
    run_diagnostics,
    odds_bucket_report,
    _bet_rows,
)

def test_odds_buckets_use_actual_decimal_odds_and_keep_sides_separate():
    bets = pd.DataFrame({
        "side": [
            "OVER", "OVER", "OVER", "OVER", "OVER",
            "OVER", "OVER", "OVER", "OVER", "UNDER",
            "UNDER", "UNDER",
        ],
        "odds": [
            1.44, 1.58, 1.60, 1.72, 1.86,
            1.98, 2.05, 2.10, 2.12, 1.44,
            1.86, 1.98,
        ],
        "edge": [0.06] * 12,
        "won": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    })

    report = odds_bucket_report(bets)

    over = report.loc[report["side"] == "OVER"].set_index("bucket")
    under = report.loc[report["side"] == "UNDER"].set_index("bucket")

    assert over.loc["1.30-1.49", "bets"] == 1
    assert over.loc["1.50-<1.60", "bets"] == 1
    assert over.loc["1.60-<1.70", "bets"] == 1
    assert over.loc["1.70-<1.80", "bets"] == 1
    assert over.loc["1.80-<1.90", "bets"] == 1
    assert over.loc["1.90-<2.00", "bets"] == 1
    assert over.loc["2.00-<2.10", "bets"] == 1
    # The 2.10 boundary and all higher odds belong to the open-ended bucket.
    assert over.loc["2.10+", "bets"] == 2

    assert under.loc["1.30-1.49", "bets"] == 1
    assert under.loc["1.80-<1.90", "bets"] == 1
    assert under.loc["1.90-<2.00", "bets"] == 1

    # Exact boundary values are assigned once, to the bucket starting there.
    assert over.loc["1.60-<1.70", "avg_odds"] == 1.60
    assert over.loc["2.00-<2.10", "avg_odds"] == 2.05
    assert np.isclose(over.loc["2.10+", "avg_odds"], 2.11)



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


def test_market_data_uses_only_non_closing_odds_columns():
    frame = pd.DataFrame({
        "odds_Avg>2.5": [2.00],
        "odds_Avg<2.5": [2.00],
        "odds_B365>2.5": [2.20],
        "odds_B365<2.5": [1.80],
        "odds_AvgC>2.5": [1.10],
        "odds_AvgC<2.5": [8.00],
        "odds_B365C>2.5": [1.05],
        "odds_B365C<2.5": [9.00],
    })

    p_market, over, under, valid = market_data(frame)

    assert valid.tolist() == [True]
    assert np.isclose(over[0], 2.00)
    assert np.isclose(under[0], 2.00)
    assert np.isclose(p_market[0], 0.50)


def test_market_data_never_falls_back_to_closing_odds():
    frame = pd.DataFrame({
        "odds_AvgC>2.5": [1.10],
        "odds_AvgC<2.5": [8.00],
        "odds_B365C>2.5": [1.05],
        "odds_B365C<2.5": [9.00],
    })

    p_market, over, under, valid = market_data(frame)

    assert valid.tolist() == [False]
    assert np.isclose(p_market[0], 0.50)
    assert np.isnan(over[0])
    assert np.isnan(under[0])


def test_primary_historical_model_features_do_not_depend_on_any_odds():
    historical = pd.DataFrame({
        "lambda_home": [1.4, 1.8],
        "lambda_away": [0.9, 1.2],
        "p_poisson": [0.55, 0.72],
        "home_for_mean": [1.5, 1.7],
        "home_against_mean": [1.0, 1.1],
        "away_for_mean": [1.1, 1.3],
        "away_against_mean": [1.4, 1.2],
        "home_recent_for": [1.8, 2.0],
        "home_recent_against": [0.9, 1.0],
        "away_recent_for": [1.0, 1.2],
        "away_recent_against": [1.6, 1.4],
        "home_recent_for_10": [1.7, 1.9],
        "home_recent_against_10": [1.0, 1.1],
        "home_recent_for_15": [1.6, 1.8],
        "home_recent_against_15": [1.1, 1.2],
        "away_recent_for_10": [1.7, 1.9],
        "away_recent_against_10": [1.0, 1.1],
        "away_recent_for_15": [1.6, 1.8],
        "away_recent_against_15": [1.1, 1.2],
        "lambda_home_recent_5": [1.5, 1.7],
        "lambda_away_recent_5": [1.0, 1.1],
        "lambda_total_recent_5": [2.5, 2.8],
        "p_poisson_recent_5": [0.55, 0.65],
        "lambda_home_recent_10": [1.5, 1.7],
        "lambda_away_recent_10": [1.0, 1.1],
        "lambda_total_recent_10": [2.5, 2.8],
        "p_poisson_recent_10": [0.55, 0.65],
        "lambda_home_recent_15": [1.5, 1.7],
        "lambda_away_recent_15": [1.0, 1.1],
        "lambda_total_recent_15": [2.5, 2.8],
        "p_poisson_recent_15": [0.55, 0.65],
        "league_average": [2.5, 2.6],
        "home_advantage": [1.1, 1.1],
        "home_history_count": [6.0, 7.0],
        "away_history_count": [6.0, 7.0],
    })
    with_odds = historical.assign(
        **{
            "odds_Avg>2.5": [1.20, 8.00],
            "odds_Avg<2.5": [9.00, 1.10],
            "odds_AvgC>2.5": [1.01, 99.0],
            "odds_AvgC<2.5": [99.0, 1.01],
        }
    )

    expected = HistoricalXGModel._X(historical)
    actual = HistoricalXGModel._X(with_odds)

    assert expected.shape == (2, 35)
    assert np.array_equal(actual, expected)


def test_primary_poisson_model_is_odds_independent_and_preserves_raw_probabilities():
    model = PoissonOnlyModel()
    frame = pd.DataFrame({
        "p_poisson": [0.72, 0.31],
        "odds_Avg>2.5": [1.01, 99.0],
        "odds_Avg<2.5": [99.0, 1.01],
        "odds_AvgC>2.5": [1.001, 999.0],
        "odds_AvgC<2.5": [999.0, 1.001],
    })

    assert model.fit(frame) is model
    assert model.calibrate(frame) is model
    assert np.allclose(model.predict_proba(frame), [0.72, 0.31])


def test_fixed_confidence_policy_reports_only_selected_high_confidence_picks():
    class DummyModel:
        @staticmethod
        def _y(frame):
            return frame["y"].to_numpy(dtype=int)

        @staticmethod
        def predict_proba(frame):
            return frame["p"].to_numpy(dtype=float)

    frame = pd.DataFrame({
        "p": [0.80, 0.20, 0.60, 0.40],
        "y": [1, 0, 1, 1],
    })

    report = confidence_policy_analysis(
        frame,
        DummyModel(),
        "TEST",
        cutoff=0.75,
    )

    assert report["matches"] == 2
    assert report["correct"] == 2
    assert np.isclose(report["accuracy"], 1.0)
    assert report["top_n"] == 2
    assert report["top_n_correct"] == 2
    assert np.isclose(report["top_n_accuracy"], 1.0)
    assert np.isclose(report["coverage"], 0.5)
