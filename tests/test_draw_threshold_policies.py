import numpy as np

from scripts.analyze_draw_diagnostics import draw_threshold_curve, select_threshold


def test_probability_only_policy_can_select_draw_when_draw_is_not_argmax():
    labels = np.array([1, 0, 1, 2])
    probabilities = np.array([
        [0.40, 0.35, 0.25],
        [0.45, 0.35, 0.20],
        [0.30, 0.40, 0.30],
        [0.40, 0.35, 0.25],
    ])

    strict = draw_threshold_curve(labels, probabilities, "argmax_and_threshold")
    probability_only = draw_threshold_curve(labels, probabilities, "probability_threshold_only")

    strict_at_035 = next(row for row in strict if row["threshold"] == 0.35)
    probability_at_035 = next(row for row in probability_only if row["threshold"] == 0.35)

    assert strict_at_035["selected_matches"] == 1
    assert probability_at_035["selected_matches"] == 4
    assert strict_at_035["draw_recall"] == 0.5
    assert probability_at_035["draw_recall"] == 1.0


def test_probability_only_policy_reports_precision_recall_and_coverage():
    labels = np.array([1, 0, 1, 2])
    probabilities = np.array([
        [0.40, 0.35, 0.25],
        [0.45, 0.35, 0.20],
        [0.30, 0.40, 0.30],
        [0.40, 0.35, 0.25],
    ])

    row = next(
        row for row in draw_threshold_curve(labels, probabilities, "probability_threshold_only")
        if row["threshold"] == 0.35
    )

    assert row["selected_matches"] == 4
    assert row["coverage"] == 1.0
    assert row["draw_precision"] == 0.5
    assert row["draw_recall"] == 1.0
    assert row["draw_f1"] == round(2 * 0.5 * 1.0 / 1.5, 6)


def test_threshold_selection_has_deterministic_tie_break():
    curve = [
        {"threshold": 0.2, "draw_f1": 0.5, "coverage": 0.1},
        {"threshold": 0.3, "draw_f1": 0.5, "coverage": 0.2},
        {"threshold": 0.4, "draw_f1": 0.5, "coverage": 0.2},
    ]

    assert select_threshold(curve)["threshold"] == 0.3


def test_unknown_policy_is_rejected():
    with np.testing.assert_raises(ValueError):
        draw_threshold_curve(np.array([1]), np.array([[0.2, 0.5, 0.3]]), "unknown")
