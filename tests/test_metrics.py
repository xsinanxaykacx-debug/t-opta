import numpy as np
import pytest

from validation.metrics import EvaluationMetrics


def test_metrics_report_overall_and_selected_bets():
    y_true = np.array([0, 1, 2, 0])
    y_prob = np.array([
        [0.80, 0.10, 0.10],
        [0.20, 0.60, 0.20],
        [0.10, 0.20, 0.70],
        [0.40, 0.30, 0.30],
    ])
    report = EvaluationMetrics.evaluate_predictions(
        y_true, y_prob, ["BET", "NO BET", "BET", "NO BET"]
    )
    assert report["total_matches"] == 4
    assert report["bet_count"] == 2
    assert report["no_bet_count"] == 2
    assert report["bet_coverage_ratio"] == 0.5
    assert report["overall"]["accuracy"] == 1.0
    assert report["selected_bets_only"]["accuracy"] == 1.0


def test_metrics_return_null_selected_scores_when_no_bets():
    report = EvaluationMetrics.evaluate_predictions(
        [0, 1], [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], ["NO BET", "NO BET"]
    )
    assert report["bet_count"] == 0
    assert report["selected_bets_only"]["accuracy"] is None


@pytest.mark.parametrize("probabilities", [
    [[0.2, 0.2, 0.2]],
    [[-0.1, 0.5, 0.6]],
    [[float("nan"), 0.5, 0.5]],
])
def test_metrics_reject_invalid_probability_rows(probabilities):
    with pytest.raises(ValueError):
        EvaluationMetrics.evaluate_predictions([0], probabilities, ["NO BET"])


def test_metrics_reject_invalid_decision_labels():
    with pytest.raises(ValueError, match="decisions"):
        EvaluationMetrics.evaluate_predictions(
            [0], [[0.7, 0.2, 0.1]], ["SIMULATION_ONLY"]
        )
