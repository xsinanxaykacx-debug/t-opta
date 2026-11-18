import numpy as np
import pandas as pd

from experiments.market_baseline import MarketBaseline


def test_market_baseline_uses_de_vigged_prematch_prices():
    frame = pd.DataFrame(
        {
            "odds_Avg>2.5": [2.0, np.nan],
            "odds_Avg<2.5": [1.8, np.nan],
            "odds_B365>2.5": [2.1, 1.9],
            "odds_B365<2.5": [1.7, 1.9],
            "odds_AvgC>2.5": [1.01, 1.01],
            "odds_B365C<2.5": [1.01, 1.01],
        }
    )
    result = MarketBaseline().predict_components(frame)
    expected_first = (1 / 2.0) / ((1 / 2.0) + (1 / 1.8))
    expected_second = (1 / 1.9) / ((1 / 1.9) + (1 / 1.9))

    assert np.isclose(result["p_over"][0], expected_first)
    assert np.isclose(result["p_over"][1], expected_second)
    assert result["market_valid"].tolist() == [True, True]
    assert np.all(result["over_odds"] > 1.0)
