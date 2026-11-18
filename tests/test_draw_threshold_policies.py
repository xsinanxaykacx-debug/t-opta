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


def test_coverage_curve_uses_selection_cutoff_and_reports_metrics():
    from scripts.analyze_draw_diagnostics import coverage_precision_recall_curve

    labels = np.array([1, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    draw_p = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05])
    probabilities = np.column_stack([1 - draw_p, draw_p, np.zeros_like(draw_p)])

    row = coverage_precision_recall_curve(labels, probabilities, [0.20])[0]

    assert row["threshold"] == 0.8
    assert row["selected_matches"] == 2
    assert row["coverage"] == 0.2
    assert row["true_positives"] == 1
    assert row["false_positives"] == 1
    assert row["draw_precision"] == 0.5
    assert row["draw_recall"] == 0.25


def test_coverage_curve_includes_all_matches_tied_at_cutoff():
    from scripts.analyze_draw_diagnostics import coverage_precision_recall_curve

    labels = np.array([1, 0, 1, 2])
    draw_p = np.array([0.8, 0.8, 0.4, 0.1])
    probabilities = np.column_stack([1 - draw_p, draw_p, np.zeros_like(draw_p)])

    row = coverage_precision_recall_curve(labels, probabilities, [0.25])[0]

    assert row["threshold"] == 0.8
    assert row["selected_matches"] == 2
    assert row["coverage"] == 0.5
    assert row["target_coverage"] == 0.25


def test_coverage_curve_rejects_invalid_targets():
    from scripts.analyze_draw_diagnostics import coverage_precision_recall_curve

    labels = np.array([1])
    probabilities = np.array([[0.2, 0.5, 0.3]])

    with np.testing.assert_raises(ValueError):
        coverage_precision_recall_curve(labels, probabilities, [0.0])



def test_league_diagnostics_uses_one_fixed_global_threshold_and_reports_base_rate():
    import pandas as pd
    from scripts.analyze_draw_diagnostics import league_draw_diagnostics

    frame = pd.DataFrame({"League": ["A", "A", "B", "B"]})
    labels = np.array([1, 0, 0, 1])
    probabilities = np.array([
        [0.30, 0.50, 0.20],
        [0.40, 0.40, 0.20],
        [0.50, 0.30, 0.20],
        [0.20, 0.60, 0.20],
    ])

    rows = league_draw_diagnostics(frame, labels, probabilities, 0.45)
    by_league = {row["league"]: row for row in rows}

    assert by_league["A"]["matches"] == 2
    assert by_league["A"]["actual_draw_rate"] == 0.5
    assert by_league["A"]["mean_p_draw"] == 0.45
    assert by_league["A"]["selected_matches"] == 1
    assert by_league["A"]["draw_precision"] == 1.0
    assert by_league["B"]["matches"] == 2
    assert by_league["B"]["selected_matches"] == 1
    assert by_league["A"]["sample_reliability"] == "small_sample_caution"
    assert by_league["B"]["fixed_global_threshold"] == 0.45


def test_league_diagnostics_rejects_misaligned_inputs():
    import pandas as pd
    from scripts.analyze_draw_diagnostics import league_draw_diagnostics

    with np.testing.assert_raises(ValueError):
        league_draw_diagnostics(
            pd.DataFrame({"League": ["A"]}),
            np.array([1, 0]),
            np.array([[0.2, 0.5, 0.3], [0.3, 0.4, 0.3]]),
            0.4,
        )
