import numpy as np

from scripts.walk_forward_over_under import (
    HistGradientBoostingCandidate,
    LogisticOverUnderCandidate,
    NegativeBinomialCandidate,
    build_odds_provenance_report,
    platt_scale_probabilities,
    build_fold_slices,
    calculate_policy_rows,
)


def test_walk_forward_folds_are_expanding_and_chronological():
    folds = build_fold_slices(
        n_rows=8500,
        initial_train=5000,
        calibration_size=500,
        validation_size=300,
        folds=10,
    )

    assert len(folds) == 10
    for train, calibration, validation in folds:
        assert train.start == 0
        assert train.stop == calibration.start
        assert calibration.stop == validation.start
        assert train.stop - train.start >= 5000
        assert calibration.stop - calibration.start == 500
        assert validation.stop - validation.start == 300
    assert folds[-1][2].stop == 8500


def test_walk_forward_policy_reports_over_under_and_combined_roi():
    y = np.array([1, 0, 1, 0], dtype=int)
    p_model = np.array([0.70, 0.30, 0.80, 0.20])
    p_market = np.array([0.60, 0.40, 0.60, 0.40])
    over_odds = np.array([2.0, 2.0, 1.5, 1.5])
    under_odds = np.array([2.0, 2.0, 2.0, 2.0])
    market_valid = np.ones(4, dtype=bool)

    rows = calculate_policy_rows(
        y=y,
        p_model=p_model,
        p_market=p_market,
        over_odds=over_odds,
        under_odds=under_odds,
        market_valid=market_valid,
        threshold=0.05,
        fold=1,
    )

    by_side = {row["side"]: row for row in rows}
    assert by_side["OVER"]["bets"] == 2
    assert by_side["UNDER"]["bets"] == 2
    assert by_side["ALL"]["bets"] == 4
    assert np.isclose(by_side["OVER"]["hit_rate"], 1.0)
    assert np.isclose(by_side["UNDER"]["hit_rate"], 1.0)
    assert np.isclose(by_side["ALL"]["roi_pct"], 87.5)


def test_confidence_report_measures_selective_accuracy_and_coverage():
    from scripts.walk_forward_over_under import calculate_confidence_rows

    y = np.array([1, 0, 1, 0], dtype=int)
    probabilities = np.array([0.90, 0.10, 0.55, 0.45])

    rows = calculate_confidence_rows(
        y=y,
        probabilities=probabilities,
        model_name="BLEND",
        fold=1,
        cutoffs=(0.60, 0.80),
    )

    by_cutoff = {row["confidence_cutoff"]: row for row in rows}
    assert by_cutoff[0.60]["matches"] == 2
    assert np.isclose(by_cutoff[0.60]["coverage"], 0.5)
    assert np.isclose(by_cutoff[0.60]["accuracy"], 1.0)
    assert by_cutoff[0.60]["top_n_matches"] == 2
    assert by_cutoff[0.60]["top_n_correct"] == 2
    assert np.isclose(by_cutoff[0.60]["top_n_accuracy"], 1.0)
    assert by_cutoff[0.60]["over_matches"] == 1
    assert by_cutoff[0.60]["under_matches"] == 1
    assert by_cutoff[0.60]["top_n_over_correct"] == 1
    assert by_cutoff[0.60]["top_n_under_correct"] == 1
    assert by_cutoff[0.80]["matches"] == 2
    assert np.isclose(by_cutoff[0.80]["mean_confidence"], 0.90)


def test_walk_forward_fold_builder_rejects_insufficient_history():
    try:
        build_fold_slices(
            n_rows=500,
            initial_train=5000,
            calibration_size=500,
            validation_size=300,
            folds=2,
        )
    except ValueError as exc:
        assert "Not enough development rows" in str(exc)
    else:
        raise AssertionError("Expected insufficient history to be rejected.")



def test_hist_gradient_boosting_candidate_is_odds_independent_and_calibrated():
    import pandas as pd

    rng = np.random.default_rng(42)
    feature_names = (
        "lambda_home", "lambda_away", "p_poisson",
        "home_for_mean", "home_against_mean",
        "away_for_mean", "away_against_mean",
        "home_recent_for", "home_recent_against",
        "away_recent_for", "away_recent_against",
        "home_recent_for_10", "home_recent_against_10",
        "away_recent_for_10", "away_recent_against_10",
        "home_recent_for_15", "home_recent_against_15",
        "away_recent_for_15", "away_recent_against_15",
        "lambda_home_recent_5", "lambda_away_recent_5",
        "lambda_total_recent_5", "p_poisson_recent_5",
        "lambda_home_recent_10", "lambda_away_recent_10",
        "lambda_total_recent_10", "p_poisson_recent_10",
        "lambda_home_recent_15", "lambda_away_recent_15",
        "lambda_total_recent_15", "p_poisson_recent_15",
        "league_average", "home_advantage",
        "home_history_count", "away_history_count",
    )

    def make_frame(size, offset):
        signal = np.linspace(0.0, 1.0, size, endpoint=False) + offset
        labels = (np.arange(size) % 2).astype(int)
        values = {
            name: (0.5 + signal if "lambda" in name or "mean" in name else signal)
            for name in feature_names
        }
        values["FTHG"] = np.where(labels == 1, 2, 1)
        values["FTAG"] = np.where(labels == 1, 1, 0)
        return pd.DataFrame(values)

    train = make_frame(240, 0.0)
    calibration = make_frame(120, 0.03)
    validation = make_frame(40, 0.07)

    model = HistGradientBoostingCandidate().fit(train).calibrate(calibration)
    probabilities = model.predict_proba(validation)

    assert len(probabilities) == len(validation)
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()
    assert not any(column.startswith("odds_") for column in train.columns)



def test_platt_calibration_is_bounded_and_monotonic():
    calibration_probabilities = np.linspace(0.05, 0.95, 200)
    calibration_labels = (calibration_probabilities >= 0.52).astype(int)
    target_probabilities = np.array([0.20, 0.50, 0.80])

    calibrated = platt_scale_probabilities(
        calibration_probabilities,
        calibration_labels,
        target_probabilities,
    )

    assert np.isfinite(calibrated).all()
    assert ((calibrated > 0.0) & (calibrated < 1.0)).all()
    assert np.all(np.diff(calibrated) > 0)


def test_odds_provenance_report_groups_source_files_and_match_status():
    import pandas as pd

    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-01-01T12:00:00Z",
            "2024-01-02T12:00:00Z",
            "2024-01-03T12:00:00Z",
        ], utc=True),
        "League": ["L1", "L1", "L2"],
        "odds_source_file": ["D2 (4).csv", "D2 (4).csv", "E0.csv"],
        "odds_match_status": ["matched", "unmatched", "matched"],
        "odds_Avg>2.5": [1.90, np.nan, 2.00],
        "odds_Avg<2.5": [1.90, np.nan, 1.80],
        "odds_B365>2.5": [1.95, np.nan, 2.05],
        "odds_B365<2.5": [1.85, np.nan, 1.75],
    })

    report = build_odds_provenance_report(frame)
    by_source = report.set_index("odds_source_file")

    assert by_source.loc["D2 (4).csv", "matches"] == 2
    assert by_source.loc["D2 (4).csv", "matched_rows"] == 1
    assert np.isclose(by_source.loc["D2 (4).csv", "match_rate"], 0.5)
    assert by_source.loc["D2 (4).csv", "avg_ou_pair_rows"] == 1
    assert by_source.loc["E0.csv", "b365_ou_pair_rows"] == 1



def test_negative_binomial_candidate_learns_dispersion_from_training_only():
    import pandas as pd

    goals = np.tile(np.array([0, 0, 1, 1, 5, 6, 0, 5, 1, 6]), 20)
    train = pd.DataFrame({
        "lambda_home": np.full(len(goals), 1.25),
        "lambda_away": np.full(len(goals), 1.25),
        "FTHG": goals // 2,
        "FTAG": goals - (goals // 2),
    })
    validation = pd.DataFrame({
        "lambda_home": [1.25, 1.0, 1.5],
        "lambda_away": [1.25, 1.0, 1.5],
    })

    model = NegativeBinomialCandidate().fit(train)
    probabilities = model.predict_proba(validation)

    assert model.alpha > 0.0
    assert model.alpha <= 2.0
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()
    assert not any(column.startswith("odds_") for column in train.columns)


def test_negative_binomial_candidate_reduces_to_poisson_without_overdispersion():
    import pandas as pd
    from scipy.stats import poisson

    goals = np.tile(np.array([1, 2, 2, 3]), 30)
    train = pd.DataFrame({
        "lambda_home": np.full(len(goals), 1.0),
        "lambda_away": np.full(len(goals), 1.0),
        "FTHG": goals // 2,
        "FTAG": goals - (goals // 2),
    })
    validation = pd.DataFrame({
        "lambda_home": [1.0, 1.25],
        "lambda_away": [1.0, 1.25],
    })

    model = NegativeBinomialCandidate().fit(train)
    probabilities = model.predict_proba(validation)

    assert model.alpha == 0.0
    assert np.allclose(
        probabilities,
        poisson.sf(2, np.array([2.0, 2.5])),
    )


def test_prediction_diagnostics_keep_out_of_fold_rows_and_group_error_sources():
    import pandas as pd
    from scripts.walk_forward_over_under import (
        aggregate_prediction_diagnostics,
        build_prediction_diagnostic_rows,
    )

    validation = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "League": ["L1", "L1", "L2"],
        "HomeTeam": ["A", "B", "C"],
        "AwayTeam": ["D", "E", "F"],
        "lambda_home": [1.4, 1.0, 1.8],
        "lambda_away": [1.2, 1.1, 1.5],
        "home_history_count": [10, 4, 15],
        "away_history_count": [12, 3, 16],
        "home_for_mean": [1.4, 1.0, 1.8],
        "away_for_mean": [1.2, 1.1, 1.5],
        "home_recent_for": [1.6, 0.7, 1.9],
        "away_recent_for": [1.3, 0.9, 1.6],
        "home_recent_for_10": [1.5, 0.8, 1.8],
        "away_recent_for_10": [1.2, 1.0, 1.5],
        "home_recent_for_15": [1.4, 0.9, 1.7],
        "away_recent_for_15": [1.1, 1.0, 1.4],
    })
    labels = np.array([1, 0, 1], dtype=int)
    probabilities = np.array([0.82, 0.25, 0.61])

    rows = build_prediction_diagnostic_rows(
        validation,
        labels,
        {"TEST_MODEL": probabilities},
        fold=2,
    )
    raw = pd.DataFrame(rows)
    report = aggregate_prediction_diagnostics(raw)

    assert len(raw) == 3
    assert set(raw["fold"]) == {2}
    assert raw["correct"].tolist() == [1, 1, 1]
    assert {"LEAGUE", "SIDE", "CONFIDENCE", "TOTAL_LAMBDA", "HISTORY_DEPTH", "RECENT_FORM"} == set(report["dimension"])
    pooled_league = report.loc[
        (report["fold"] == 0)
        & (report["dimension"] == "LEAGUE")
    ]
    assert pooled_league["matches"].sum() == 3
    assert np.isclose(pooled_league["accuracy"].mean(), 1.0)


def test_prediction_diagnostics_reject_misaligned_labels():
    import pandas as pd
    from scripts.walk_forward_over_under import build_prediction_diagnostic_rows

    validation = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01"]),
        "League": ["L1"],
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "lambda_home": [1.2],
        "lambda_away": [1.1],
        "home_history_count": [5],
        "away_history_count": [5],
        "home_for_mean": [1.2],
        "away_for_mean": [1.1],
        "home_recent_for": [1.2],
        "away_recent_for": [1.1],
        "home_recent_for_10": [1.2],
        "away_recent_for_10": [1.1],
        "home_recent_for_15": [1.2],
        "away_recent_for_15": [1.1],
    })

    try:
        build_prediction_diagnostic_rows(
            validation,
            np.array([1, 0]),
            {"TEST_MODEL": np.array([0.7, 0.2])},
            fold=1,
        )
    except ValueError as exc:
        assert "aligned validation rows" in str(exc)
    else:
        raise AssertionError("Expected misaligned labels to be rejected.")


def test_logistic_over_under_candidate_uses_train_only_selection_and_calibration():
    import pandas as pd

    rng = np.random.default_rng(17)
    feature_names = (
        "lambda_home", "lambda_away", "p_poisson",
        "home_for_mean", "home_against_mean",
        "away_for_mean", "away_against_mean",
        "home_recent_for", "home_recent_against",
        "away_recent_for", "away_recent_against",
        "home_recent_for_10", "home_recent_against_10",
        "away_recent_for_10", "away_recent_against_10",
        "home_recent_for_15", "home_recent_against_15",
        "away_recent_for_15", "away_recent_against_15",
        "league_average", "home_advantage",
        "home_history_count", "away_history_count",
    )

    def make_frame(size, start):
        labels = (np.arange(size) % 2).astype(int)
        values = {
            name: rng.normal(1.5, 0.3, size)
            for name in feature_names
        }
        values["Date"] = pd.date_range(start, periods=size, freq="D", tz="UTC")
        values["FTHG"] = np.where(labels == 1, 2, 1)
        values["FTAG"] = np.where(labels == 1, 1, 0)
        return pd.DataFrame(values)

    train = make_frame(240, "2022-01-01")
    calibration = make_frame(100, "2023-01-01")
    validation = make_frame(30, "2023-05-01")

    model = LogisticOverUnderCandidate().fit(train).calibrate(calibration)
    probabilities = model.predict_proba(validation)

    assert model.best_c in model.C_GRID
    assert set(model.validation_scores) == set(model.C_GRID)
    assert len(probabilities) == len(validation)
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()
