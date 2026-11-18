"""Dependency-light evaluation metrics for 1X2 probability forecasts."""
from __future__ import annotations

import numpy as np


class EvaluationMetrics:
    """Calculate proper scoring rules, class metrics, calibration, and coverage.

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

    @staticmethod
    def _classification_report(labels, predicted):
        names = ("HOME", "DRAW", "AWAY")
        per_class = {}
        precisions, recalls, f1_scores = [], [], []
        for class_id, name in enumerate(names):
            tp = int(np.sum((labels == class_id) & (predicted == class_id)))
            fp = int(np.sum((labels != class_id) & (predicted == class_id)))
            fn = int(np.sum((labels == class_id) & (predicted != class_id)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[name] = {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": int(np.sum(labels == class_id)),
            }
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        return {
            "per_class": per_class,
            "macro_precision": round(float(np.mean(precisions)), 6),
            "macro_recall": round(float(np.mean(recalls)), 6),
            "macro_f1": round(float(np.mean(f1_scores)), 6),
        }

    @staticmethod
    def _calibration_report(labels, probabilities, bins=10):
        confidence = probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)
        correct = (predicted == labels).astype(float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        max_error = 0.0
        bin_rows = []
        for index in range(bins):
            lower, upper = edges[index], edges[index + 1]
            mask = ((confidence >= lower) & (confidence < upper)
                    if index < bins - 1 else
                    (confidence >= lower) & (confidence <= upper))
            count = int(mask.sum())
            if not count:
                continue
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            gap = abs(mean_confidence - accuracy)
            ece += (count / len(labels)) * gap
            max_error = max(max_error, gap)
            bin_rows.append({
                "lower": round(float(lower), 2),
                "upper": round(float(upper), 2),
                "count": count,
                "mean_confidence": round(mean_confidence, 6),
                "accuracy": round(accuracy, 6),
            })
        return {
            "expected_calibration_error": round(float(ece), 6),
            "max_calibration_error": round(float(max_error), 6),
            "bins": bin_rows,
            "method": "top-label, equal-width bins",
        }

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
            "classification": cls._classification_report(labels, predicted),
            "calibration": cls._calibration_report(labels, probabilities),
            "selected_bets_only": {
                "log_loss": round(selected_log_loss, 6) if selected_log_loss is not None else None,
                "brier_score": round(selected_brier, 6) if selected_brier is not None else None,
                "accuracy": round(selected_accuracy, 6) if selected_accuracy is not None else None,
            },
        }
