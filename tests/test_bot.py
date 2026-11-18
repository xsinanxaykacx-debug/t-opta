import numpy as np
import pandas as pd

import BOT


def _matches(dates, home_xg=None, away_xg=None):
    n = len(dates)
    return pd.DataFrame({
        "Date": pd.to_datetime(dates, utc=True),
        "League": ["Test League"] * n,
        "HomeTeam": [f"Home {i}" for i in range(n)],
        "AwayTeam": [f"Away {i}" for i in range(n)],
        "Home_xG": home_xg if home_xg is not None else [1.2] * n,
        "Away_xG": away_xg if away_xg is not None else [0.9] * n,
        "FTHG": [1] * n,
        "FTAG": [0] * n,
    })


def test_same_utc_day_results_do_not_change_later_fixture_features():
    frame = _matches(
        ["2025-01-01T10:00:00Z", "2025-01-01T19:00:00Z"],
        home_xg=[9.0, 0.2],
        away_xg=[0.1, 0.2],
    )
    featured = BOT.add_leakage_safe_features(frame)
    assert np.isclose(featured.loc[0, "p_poisson"], featured.loc[1, "p_poisson"])
    assert np.isclose(featured.loc[0, "lambda_home"], featured.loc[1, "lambda_home"])
    assert np.isclose(featured.loc[0, "lambda_away"], featured.loc[1, "lambda_away"])


def test_current_match_xg_is_not_used_for_its_own_poisson_prediction():
    base = _matches(
        ["2025-01-01T12:00:00Z", "2025-01-03T12:00:00Z"],
        home_xg=[1.4, 1.0],
        away_xg=[0.8, 1.0],
    )
    changed = base.copy()
    changed.loc[1, "Home_xG"] = 9.9
    changed.loc[1, "Away_xG"] = 0.1

    p_base = BOT.add_leakage_safe_features(base).loc[1, "p_poisson"]
    p_changed = BOT.add_leakage_safe_features(changed).loc[1, "p_poisson"]
    assert np.isclose(p_base, p_changed)


def test_poisson_expected_goals_are_exposed_and_consistent():
    frame = _matches(["2025-01-01T12:00:00Z"])
    featured = BOT.add_leakage_safe_features(frame)
    assert np.isclose(
        featured.loc[0, "expected_total_goals"],
        featured.loc[0, "lambda_home"] + featured.loc[0, "lambda_away"],
    )


def test_chronological_splits_never_cut_a_utc_day():
    dates = []
    for day in range(1, 13):
        dates.extend([f"2025-01-{day:02d}T10:00:00Z", f"2025-01-{day:02d}T19:00:00Z"])
    frame = _matches(dates)
    train, validation, test = BOT.chronological_split(frame, train_target=5, validation_target=4)

    train_days = set(train.Date.dt.floor("D"))
    validation_days = set(validation.Date.dt.floor("D"))
    test_days = set(test.Date.dt.floor("D"))
    assert train_days.isdisjoint(validation_days)
    assert train_days.isdisjoint(test_days)
    assert validation_days.isdisjoint(test_days)
    assert train.Date.max() < validation.Date.min()
    assert validation.Date.max() < test.Date.min()


def test_validation_subsplits_never_cut_a_utc_day():
    dates = []
    for day in range(1, 10):
        dates.extend([f"2025-02-{day:02d}T10:00:00Z", f"2025-02-{day:02d}T19:00:00Z"])
    frame = _matches(dates)
    calibration, selection, check = BOT.split_validation_by_day(frame)

    sets = [
        set(calibration.Date.dt.floor("D")),
        set(selection.Date.dt.floor("D")),
        set(check.Date.dt.floor("D")),
    ]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_market_probability_devigs_complete_same_source_pairs():
    frame = pd.DataFrame({
        "odds_Avg>2.5": [2.0, np.nan],
        "odds_Avg<2.5": [2.0, np.nan],
        "odds_B365>2.5": [np.nan, 1.8],
        "odds_B365<2.5": [np.nan, 2.2],
    })
    p_market, over, under, valid = BOT.market_data(frame)
    assert valid.tolist() == [True, True]
    assert np.isclose(p_market[0], 0.5)
    assert np.isclose(p_market[1], (1 / 1.8) / ((1 / 1.8) + (1 / 2.2)))
    assert np.allclose(over, [2.0, 1.8])
    assert np.allclose(under, [2.0, 2.2])


def test_reliability_report_reports_market_bias_and_model_edge():
    y = np.array([1, 0, 1, 0])
    p_model = np.array([0.85, 0.70, 0.80, 0.40])
    p_market = np.array([0.75, 0.78, 0.74, 0.42])
    report = BOT.reliability_report(y, p_model, p_market, n_bins=10)

    high_bin = report.loc[report["bin_range"] == "0.70-0.80"].iloc[0]
    assert high_bin["count"] == 3
    assert np.isclose(high_bin["actual_freq"], 2 / 3)
    assert np.isclose(high_bin["market_bias"], high_bin["market_prob"] - 2 / 3)
    assert np.isclose(high_bin["edge_opportunity"], high_bin["model_prob"] - high_bin["market_prob"])
