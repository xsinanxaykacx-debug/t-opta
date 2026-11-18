import numpy as np
import pandas as pd
import pytest

from modules.feature_engine import FeatureEngine
from modules.historical_engine import HistoricalEngine
from validation.backtest import HistoricalBacktester


class TinyAdapter:
    version = "test-baseline"

    def __init__(self):
        self.fit_ids = None
        self.calibration_size = 0

    def fit(self, features, labels):
        self.fit_ids = set(features["MatchID"].astype(str))
        assert len(features) == len(labels)

    def calibrate(self, features, labels):
        self.calibration_size = len(labels)

    def predict_proba(self, features):
        return np.tile([0.7, 0.2, 0.1], (len(features), 1))


@pytest.fixture
def backtest_inputs(tmp_path):
    rows = [
        ["1", "L", "S", "2024-01-01", "A", "B", 1, 0, 1.0, 0.2],
        ["2", "L", "S", "2024-01-08", "C", "A", 0, 0, 0.4, 0.4],
        ["3", "L", "S", "2024-01-15", "A", "C", 2, 0, 1.5, 0.3],
        ["4", "L", "S", "2024-01-22", "B", "A", 0, 1, 0.2, 0.8],
        ["5", "L", "S", "2024-01-29", "A", "B", 2, 1, 1.4, 0.7],
    ]
    columns = [
        "MatchID", "League", "Season", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "Home_xG", "Away_xG",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    path = tmp_path / "matches.csv"
    frame.to_csv(path, index=False)
    engine = FeatureEngine(HistoricalEngine(path), windows=(2,))
    train = frame.iloc[:2].copy()
    calibration = frame.iloc[2:3].copy()
    test = frame.iloc[3:].copy()
    return engine, train, calibration, test


def test_backtest_fits_train_calibrates_separately_and_defaults_no_bet(backtest_inputs):
    engine, train, calibration, test = backtest_inputs
    adapter = TinyAdapter()
    report = HistoricalBacktester(engine, adapter).run_backtest(
        train, calibration, test
    )
    assert adapter.fit_ids == {"1", "2"}
    assert adapter.calibration_size == 1
    assert report["train_matches"] == 2
    assert report["calibration_matches"] == 1
    assert report["test_matches"] == 2
    assert report["calibration_applied"] is True
    assert report["bet_count"] == 0
    assert report["no_bet_count"] == 2
    assert report["roi"] is None


def test_backtest_applies_explicit_decision_function(backtest_inputs):
    engine, train, calibration, test = backtest_inputs
    report = HistoricalBacktester(
        engine, TinyAdapter(), decision_fn=lambda payload, features: "BET"
    ).run_backtest(train, calibration, test)
    assert report["bet_count"] == 2
    assert report["bet_coverage_ratio"] == 1.0


def test_backtest_rejects_calendar_day_overlap(backtest_inputs):
    engine, train, calibration, test = backtest_inputs
    calibration.loc[calibration.index[0], "Date"] = "2024-01-08T20:00:00Z"
    with pytest.raises(ValueError, match="Train must end before calibration"):
        HistoricalBacktester(engine, TinyAdapter()).run_backtest(
            train, calibration, test
        )


def test_backtest_rejects_partition_match_id_overlap(backtest_inputs):
    engine, train, calibration, test = backtest_inputs
    test.loc[test.index[0], "MatchID"] = "2"
    with pytest.raises(ValueError):
        HistoricalBacktester(engine, TinyAdapter()).run_backtest(
            train, calibration, test
        )
