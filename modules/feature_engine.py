"""Leakage-safe pre-match feature generation from prior xG match history."""
from __future__ import annotations

import math
import pandas as pd


class FeatureEngine:
    def __init__(self, historical_engine, windows=(5, 10)):
        self.history = historical_engine
        self.windows = tuple(windows)
        if not self.windows or any(
            isinstance(w, bool) or not isinstance(w, int) or w <= 0
            for w in self.windows
        ):
            raise ValueError("windows must contain positive integers.")

    @staticmethod
    def _mean(series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.mean()) if len(values) else math.nan

    @staticmethod
    def _ewma(series, span=5):
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.ewm(span=span, adjust=False).mean().iloc[-1]) if len(values) else math.nan

    def _team_features(self, history: pd.DataFrame, before_date, prefix: str, venue: str) -> dict:
        if history.empty:
            return {
                f"{prefix}_matches_available": 0,
                f"{prefix}_goals_for_avg_5": np.nan,
                f"{prefix}_goals_against_avg_5": np.nan,
                f"{prefix}_xg_for_avg_5": np.nan,
                f"{prefix}_xg_against_avg_5": np.nan,
                f"{prefix}_points_avg_5": np.nan,
                f"{prefix}_goals_for_avg_10": np.nan,
                f"{prefix}_goals_against_avg_10": np.nan,
                f"{prefix}_xg_for_avg_10": np.nan,
                f"{prefix}_xg_against_avg_10": np.nan,
                f"{prefix}_points_avg_10": np.nan,
                f"{prefix}_xg_for_ewma_5": np.nan,
                f"{prefix}_xg_against_ewma_5": np.nan,
                f"{prefix}_rest_days": np.nan,
            }

        venue_history = history.loc[history["TeamSide"].eq(venue)].copy()
        latest = venue_history.head(10)
        recent5 = latest.head(5)
        latest_match_date = history.iloc[0]["Date"]
        rest_days = (
            pd.Timestamp(before_date).normalize()
            - pd.Timestamp(latest_match_date).normalize()
        ).days

        return {
            f"{prefix}_matches_available": int(len(venue_history)),
            f"{prefix}_goals_for_avg_5": self._mean(recent5["TeamGoals"]),
            f"{prefix}_goals_against_avg_5": self._mean(recent5["OpponentGoals"]),
            f"{prefix}_xg_for_avg_5": self._mean(recent5["Team_xG"]),
            f"{prefix}_xg_against_avg_5": self._mean(recent5["Opponent_xG"]),
            f"{prefix}_points_avg_5": self._mean(recent5["Points"]),
            f"{prefix}_goals_for_avg_10": self._mean(latest["TeamGoals"]),
            f"{prefix}_goals_against_avg_10": self._mean(latest["OpponentGoals"]),
            f"{prefix}_xg_for_avg_10": self._mean(latest["Team_xG"]),
            f"{prefix}_xg_against_avg_10": self._mean(latest["Opponent_xG"]),
            f"{prefix}_points_avg_10": self._mean(latest["Points"]),
            f"{prefix}_xg_for_ewma_5": self._ewma(recent5["Team_xG"]),
            f"{prefix}_xg_against_ewma_5": self._ewma(recent5["Opponent_xG"]),
            f"{prefix}_rest_days": rest_days,
        }

    def build_features(self, match_id: str) -> dict:
        """Build one pre-match row; target match outcomes/statistics are never features."""
        try:
            match = self.history._match_index.loc[str(match_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown MatchID={match_id!r}.") from exc
        if pd.isna(match["Date"]):
            raise ValueError(f"MatchID={match_id!r} has no valid kickoff date.")
        cutoff = match["Date"]
        home, away = str(match["HomeTeam"]), str(match["AwayTeam"])
        features = {
            "MatchID": str(match_id),
            "League": str(match["League"]),
            "Season": str(match["Season"]),
            "match_date": cutoff,
        }
        league = str(match["League"])
        home_history = self.history.get_team_history(home, cutoff, league=league)
        away_history = self.history.get_team_history(away, cutoff, league=league)
        features.update(self._team_features(home_history, cutoff, "home", "home"))
        features.update(self._team_features(away_history, cutoff, "away", "away"))
        h2h = home_history.loc[
            home_history["Opponent"].astype(str).str.casefold().eq(away.casefold())
        ].copy()
        features["h2h_matches_available"] = int(len(h2h))
        features["h2h_home_goals_avg"] = self._mean(h2h["TeamGoals"])
        features["h2h_away_goals_avg"] = self._mean(h2h["OpponentGoals"])
        return features

    def build_dataset(self, match_ids=None) -> pd.DataFrame:
        """Build features for selected IDs; pass IDs explicitly for controlled batches."""
        if match_ids is None:
            rows = self.history.data.loc[
                self.history.data["Date"].notna(), "MatchID"
            ].tolist()
        else:
            rows = [str(value) for value in match_ids]
        return pd.DataFrame([self.build_features(match_id) for match_id in rows])
