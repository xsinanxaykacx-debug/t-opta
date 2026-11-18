import numpy as np
import pandas as pd
import pytest

from models.logistic_baseline import LogisticBaseline


def make_training_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-08-01", periods=180, freq="D", tz="UTC")
    labels = np.tile([0, 1, 2], 60)
    home = np.where(labels == 0, 1.5, np.where(labels == 1, 0.0, -1.0))
    away = np.where(labels == 2, 1.5, np.where(labels == 1, 0.0, -1.0))
    frame = pd.DataFrame({
        "match_date": dates,
        "home_form": home + rng.normal(0, 0.4, len(labels)),
        "away_form": away + rng.normal(0, 0.4, len(labels)),
        "League": np.where(np.arange(len(labels)) % 2, "A", "B"),
        "sparse_value": np.where(np.arange(len(labels)) % 7, rng.normal(size=len(labels)), np.nan),
    })
    return frame, labels


def test_logistic_baseline_fits_train_pipeline_and_outputs_ordered_probabilities():
    features, labels = make_training_data()
    model = LogisticBaseline(c_grid=(0.1, 1.0))
    model.fit(features, labels)
    probabilities = model.predict_proba(features.iloc[-12:])
    assert model.selected_c_ in (0.1, 1.0)
    assert probabilities.shape == (12, 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_temperature_scaling_is_fitted_only_when_calibration_is_called():
    features, labels = make_training_data()
    model = LogisticBaseline(c_grid=(1.0,)).fit(features, labels)
    assert model.calibration_applied_ is False
    model.calibrate(features.iloc[-30:], labels[-30:])
    probabilities = model.predict_proba(features.iloc[-5:])
    assert model.calibration_applied_ is True
    assert 0.05 <= model.temperature_ <= 20.0
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_rejects_identity_and_target_leakage_columns():
    features, labels = make_training_data()
    features["FTHG"] = 1
    with pytest.raises(ValueError, match="not model features"):
        LogisticBaseline().fit(features, labels)


def test_model_rejects_non_chronological_training_schema():
    features, labels = make_training_data().copy()
    features = features.drop(columns="match_date")
    with pytest.raises(ValueError, match="match_date"):
        LogisticBaseline().fit(features, labels)
