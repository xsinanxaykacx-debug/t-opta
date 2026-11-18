import numpy as np
import pandas as pd

from experiments.odds_aware import OddsAwareModel


def _frame(n: int = 240) -> pd.DataFrame:
    index = np.arange(n)
    p_poisson = 0.25 + 0.5 * ((index % 17) / 16)
    over = np.where(index % 2 == 0, 1.75, 2.15)
    under = np.where(index % 2 == 0, 2.15, 1.75)
    target = ((p_poisson + (1 / over) / ((1 / over) + (1 / under))) / 2) >= 0.5
    return pd.DataFrame(
        {
            "p_poisson": p_poisson,
            "odds_Avg>2.5": over,
            "odds_Avg<2.5": under,
            "odds_B365>2.5": over + 0.05,
            "odds_B365<2.5": under + 0.05,
            "FTHG": np.where(target, 2, 0),
            "FTAG": np.where(target, 1, 0),
        }
    )


def test_odds_aware_model_fits_calibrates_and_predicts_probabilities():
    frame = _frame()
    model = OddsAwareModel().fit(frame.iloc[:160])
    model.calibrate(frame.iloc[160:210])
    probabilities = model.predict_proba(frame.iloc[210:])

    assert model.is_fitted is True
    assert model.is_calibrated is True
    assert probabilities.shape == (30,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()

    prediction_frame = frame.iloc[210:].copy()
    before = model.predict_proba(prediction_frame)
    prediction_frame["odds_AvgC>2.5"] = 1.01
    prediction_frame["odds_B365C<2.5"] = 99.0
    prediction_frame["FTHG"] = 99
    prediction_frame["FTAG"] = 99
    after = model.predict_proba(prediction_frame)
    assert np.allclose(before, after)
