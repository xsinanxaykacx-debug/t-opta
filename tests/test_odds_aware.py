import numpy as np
import pandas as pd

from experiments.odds_aware import HISTORICAL_COLUMNS, OddsAwareModel


def _frame(n: int = 360) -> pd.DataFrame:
    index = np.arange(n)
    p_poisson = 0.25 + 0.5 * ((index % 17) / 16)
    over = np.where(index % 2 == 0, 1.75, 2.15)
    under = np.where(index % 2 == 0, 2.15, 1.75)
    target = ((p_poisson + (1 / over) / ((1 / over) + (1 / under))) / 2) >= 0.5
    data = {
        "p_poisson": p_poisson,
        "odds_Avg>2.5": over,
        "odds_Avg<2.5": under,
        "odds_B365>2.5": over + 0.05,
        "odds_B365<2.5": under + 0.05,
        "FTHG": np.where(target, 2, 0),
        "FTAG": np.where(target, 1, 0),
    }
    for offset, column in enumerate(HISTORICAL_COLUMNS):
        if column != "p_poisson":
            data[column] = 0.5 + ((index + offset) % 11) / 10
    return pd.DataFrame(data)


def test_odds_aware_model_fits_calibrates_and_predicts_probabilities():
    frame = _frame()
    model = OddsAwareModel().fit(frame.iloc[:160])
    model.calibrate(frame.iloc[160:260])
    probabilities = model.predict_proba(frame.iloc[260:])

    assert model.is_fitted is True
    assert model.is_calibrated is True
    assert probabilities.shape == (100,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()

    prediction_frame = frame.iloc[260:].copy()
    before = model.predict_proba(prediction_frame)
    prediction_frame["odds_AvgC>2.5"] = 1.01
    prediction_frame["odds_B365C<2.5"] = 99.0
    prediction_frame["FTHG"] = 99
    prediction_frame["FTAG"] = 99
    after = model.predict_proba(prediction_frame)
    assert np.allclose(before, after)
