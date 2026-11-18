"""Dependency-light evaluation metrics for 1X2 probability forecasts."""
from __future__ import annotations

import numpy as np


class EvaluationMetrics:
    """Calculate proper scoring rules and selective-prediction coverage.

    Class order is fixed: 0=home win, 1=draw, 2=away win.
    """

    @staticmethod
    def _validate(y_true, y_prob):
        labels = np.asarray(y_true)
        probabilities = np.asarray(y_prob, dtype=float)
        if labels.ndim != 1:
            raise ValueError("y_true must be a one-dimensional array.")
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError("y_prob must have shape (n_samples, 3) in 1X2 order.")
        if len(labels) != len(probabilities) or len(labels) == 0:
            raise ValueError("y_true and y_prob must have the same non-zero length.")
        if not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("y_true labels must be 0 (home), 1 (draw), or 2 (away).")
        if not np.isfinite(probabilities).all():
            raise ValueError("y_prob must contain only finite values.")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Probabilities must be between 0 and 1.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each probability row must sum to 1.")
        return labels.astype(int), probabilities

    @staticmethod
    def multiclass_brier_score(y_true, y_prob) -> float:
        labels = np.asarray(y_true, dtype=int)
        probabilities = np.asarray(y_prob, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError("y_prob must have shape (n_samples, 3).")
        if labels.ndim != 1 or len(labels) != len(probabilities) or len(labels) == 0:
            raise ValueError("y_true and y_prob must have the same non-zero length.")
        if not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("y_true labels must be 0, 1, or 2.")
        one_hot = np.eye(3, dtype=float)[labels]
        return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    @classmethod
    def evaluate_predictions(cls, y_true, y_prob, decisions) -> dict:
        labels, probabilities = cls._validate(y_true, y_prob)
        decision_array = np.asarray(decisions, dtype=str)
        if decision_array.ndim != 1 or len(decision_array) != len(labels):
            raise ValueError("decisions must contain one decision per prediction.")
        if not np.isin(decision_array, ["BET", "NO BET"]).all():
            raise ValueError("decisions must be exactly 'BET' or 'NO BET'.")

        predicted = probabilities.argmax(axis=1)
        row_index = np.arange(len(labels))
        clipped = np.clip(probabilities[row_index, labels], 1e-15, 1.0)
        log_loss = float(-np.mean(np.log(clipped)))
        brier = cls.multiclass_brier_score(labels, probabilities)
        accuracy = float(np.mean(predicted == labels))

        bet_mask = decision_array == "BET"
        bet_count = int(bet_mask.sum())
        total = int(len(labels))
        no_bet_count = total - bet_count
        if bet_count:
            selected_labels = labels[bet_mask]
            selected_probabilities = probabilities[bet_mask]
            selected_predicted = predicted[bet_mask]
            selected_rows = np.arange(bet_count)
            selected_log_loss = float(-np.mean(np.log(np.clip(
                selected_probabilities[selected_rows, selected_labels], 1e-15, 1.0
            ))))
            selected_brier = cls.multiclass_brier_score(selected_labels, selected_probabilities)
            selected_accuracy = float(np.mean(selected_predicted == selected_labels))
        else:
            selected_log_loss = selected_brier = selected_accuracy = None

        return {
            "total_matches": total,
            "bet_count": bet_count,
            "no_bet_count": no_bet_count,
            "bet_coverage_ratio": round(bet_count / total, 4),
            "no_bet_ratio": round(no_bet_count / total, 4),
            "overall": {
                "log_loss": round(log_loss, 6),
                "brier_score": round(brier, 6),
                "accuracy": round(accuracy, 6),
            },
            "selected_bets_only": {
                "log_loss": round(selected_log_loss, 6) if selected_log_loss is not None else None,
                "brier_score": round(selected_brier, 6) if selected_brier is not None else None,
                "accuracy": round(selected_accuracy, 6) if selected_accuracy is not None else None,
            },
        }
