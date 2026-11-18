import numpy as np

from scripts.walk_forward_over_under import (
    HistGradientBoostingCandidate,
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
