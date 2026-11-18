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

def test_league_season_diagnostics_are_descriptive_and_flag_small_samples():
    predictions = pd.DataFrame({
        "Date": pd.to_datetime([
            "2025-01-10T12:00:00Z",
            "2025-08-10T12:00:00Z",
            "2025-08-11T12:00:00Z",
        ], utc=True),
        "League": ["L1", "L1", "L2"],
        "actual_over25": [1, 0, 1],
        "p_market_over": [0.55, 0.45, 0.60],
        "predicted_side": ["OVER", "UNDER", "OVER"],
        "high_odds_eligible": [True, True, False],
        "selected_side_odds": [1.8, 2.0, 1.7],
        "selected_side_won": [True, False, True],
        "market_high_odds_eligible": [True, True, False],
        "market_selected_odds": [1.8, 2.0, 1.7],
        "market_side_won": [True, False, True],
    })
    report = bot.segment_diagnostics(predictions)
    league_seasons = report[report["scope"] == "league_season"]
    assert set(league_seasons["season"]) == {"2024/2025", "2025/2026"}
    assert not league_seasons["sample_at_least_50_bets"].any()
    l1_2425 = league_seasons[
        (league_seasons["league"] == "L1") &
        (league_seasons["season"] == "2024/2025")
    ].iloc[0]
    assert l1_2425["bets_160_250"] == 1
    assert np.isclose(l1_2425["roi_pct"], 80.0)
    assert l1_2425["market_bets_160_250"] == 1
    assert np.isclose(l1_2425["market_roi_pct"], 80.0)

