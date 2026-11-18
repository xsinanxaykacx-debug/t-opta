import numpy as np
import pandas as pd

import odds_only_bot as bot


def _frame(over, under, goals=(3, 1, 0, 4)):
    n = len(over)
    return pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
        "FTHG": list(goals)[:n],
        "FTAG": [0] * n,
        "odds_Avg>2.5": over,
        "odds_Avg<2.5": under,
        "odds_B365>2.5": [np.nan] * n,
        "odds_B365<2.5": [np.nan] * n,
    })


def test_model_predictors_are_only_odds_derived():
    frame = _frame([1.8, 2.1], [2.1, 1.8])
    frame["Home_xG"] = [1000, -1000]
    frame["Home_ShotsOnTarget"] = [999, 0]
    x_before = bot.odds_features(frame)
    frame["Home_xG"] = [-5000, 5000]
    frame["Home_ShotsOnTarget"] = [0, 9999]
    x_after = bot.odds_features(frame)
    assert np.allclose(x_before, x_after, equal_nan=True)
    assert x_before.shape[1] == 4


def test_market_data_uses_complete_avg_pair_then_complete_b365_pair():
    frame = pd.DataFrame({
        "odds_Avg>2.5": [2.0, np.nan],
        "odds_Avg<2.5": [2.0, np.nan],
        "odds_B365>2.5": [1.8, 1.8],
        "odds_B365<2.5": [2.2, 2.2],
    })
    p, over, under, margin, valid = bot.market_data(frame)
    assert valid.tolist() == [True, True]
    assert np.allclose(over, [2.0, 1.8])
    assert np.allclose(under, [2.0, 2.2])
    assert np.isclose(p[0], 0.5)
    assert np.isclose(p[1], (1 / 1.8) / ((1 / 1.8) + (1 / 2.2)))
    assert np.all(np.isfinite(margin))


def test_chronological_split_keeps_utc_days_whole():
    dates = []
    for day in range(1, 21):
        dates += [f"2025-01-{day:02d}T08:00:00Z", f"2025-01-{day:02d}T20:00:00Z"]
    frame = pd.DataFrame({"Date": pd.to_datetime(dates, utc=True)})
    train, val, holdout = bot.chronological_split(frame, 0.6, 0.2)
    day_sets = [set(part.Date.dt.floor("D")) for part in (train, val, holdout)]
    assert day_sets[0].isdisjoint(day_sets[1])
    assert day_sets[0].isdisjoint(day_sets[2])
    assert day_sets[1].isdisjoint(day_sets[2])
    assert train.Date.max() < val.Date.min() < holdout.Date.min()


def test_roi_is_calculated_per_bet_not_mean_odds_times_hit_rate():
    odds = np.array([1.6, 2.5, 1.8])
    won = np.array([True, False, True])
    result = bot._roi_summary(odds, won)
    exact_profit = (1.6 - 1) - 1 + (1.8 - 1)
    assert result["bets"] == 3
    assert result["wins"] == 2
    assert np.isclose(result["profit_units"], exact_profit)
    assert np.isclose(result["roi_pct"], exact_profit / 3 * 100)


def test_high_odds_filter_includes_only_selected_side_between_160_and_250():
    frame = _frame([1.59, 1.8, 2.6, 1.9], [2.4, 1.7, 1.8, 2.2])
    p_model = np.array([0.7, 0.7, 0.2, 0.2])
    sides = bot._side_columns(frame, p_model)
    assert sides["eligible"].tolist() == [False, True, True, True]
