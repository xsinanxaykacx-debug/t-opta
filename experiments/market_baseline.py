"""De-vigged pre-match market baseline for Over/Under 2.5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from BOT import market_data


class MarketBaseline:
    """Use only approved pre-match Over/Under 2.5 odds.

    market_data prefers the paired odds_Avg quotes and falls back to the
    paired odds_B365 quotes. It never reads any closing-price column.
    """

    name = "market"

    def predict_components(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        p_over, over_odds, under_odds, valid = market_data(frame)
        return {
            "p_over": np.asarray(p_over, dtype=float),
            "over_odds": np.asarray(over_odds, dtype=float),
            "under_odds": np.asarray(under_odds, dtype=float),
            "market_valid": np.asarray(valid, dtype=bool),
        }

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_components(frame)["p_over"]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= 0.5).astype(int)
