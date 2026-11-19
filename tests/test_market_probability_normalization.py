import pytest

from mackolik_web import _normalize_1x2_probabilities


def test_normalized_market_probabilities_sum_to_one_and_remove_overround():
    raw = {"1": 1 / 1.11, "X": 1 / 7.0, "2": 1 / 15.0}
    normalized = _normalize_1x2_probabilities(raw)

    assert sum(normalized.values()) == pytest.approx(1.0)
    assert normalized["1"] * 100 == pytest.approx(81.13, abs=0.01)
    assert normalized["X"] * 100 == pytest.approx(12.86, abs=0.01)
    assert normalized["2"] * 100 == pytest.approx(6.00, abs=0.01)


def test_normalization_preserves_favorite_order():
    raw = {"1": 1 / 1.80, "X": 1 / 3.50, "2": 1 / 4.50}
    normalized = _normalize_1x2_probabilities(raw)

    assert max(raw, key=raw.get) == max(normalized, key=normalized.get)


def test_normalization_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        _normalize_1x2_probabilities({"1": 0.0, "X": 0.2, "2": 0.3})
