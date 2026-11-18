import numpy as np

from scripts.walk_forward_over_under import (
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
        assert len(train) >= 5000
        assert len(calibration) == 500
        assert len(validation) == 300
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
    probabilities = np.array([0.90, 0.80, 0.55, 0.45])

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
    assert by_cutoff[0.80]["matches"] == 2
    assert np.isclose(by_cutoff[0.80]["mean_confidence"], 0.85)


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
