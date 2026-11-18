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

    def _team_features(self, team: str, before_date, prefix: str, venue: str) -> dict:
        history = self.history.get_team_history(team, before_date)
        if venue in {"home", "away"}:
            history = history.loc[history["TeamSide"].eq(venue)]
        history = history.sort_values("Date", ascending=False, kind="stable")
        features = {f"{prefix}_matches_available": int(len(history))}
        for window in self.windows:
            recent = history.head(window)
            points = (
                (recent["TeamGoals"] > recent["OpponentGoals"]).astype(float) * 3
                + (recent["TeamGoals"] == recent["OpponentGoals"]).astype(float)
            )
            features.update({
                f"{prefix}_goals_for_avg_{window}": self._mean(recent["TeamGoals"]),
                f"{prefix}_goals_against_avg_{window}": self._mean(recent["OpponentGoals"]),
                f"{prefix}_xg_for_avg_{window}": self._mean(recent["TeamXG"]),
                f"{prefix}_xg_against_avg_{window}": self._mean(recent["OpponentXG"]),
                f"{prefix}_points_avg_{window}": self._mean(points),
            })
        chronological = history.sort_values("Date", kind="stable")
        features[f"{prefix}_xg_for_ewma_5"] = self._ewma(chronological["TeamXG"], 5)
        features[f"{prefix}_xg_against_ewma_5"] = self._ewma(chronological["OpponentXG"], 5)
        if len(history):
            latest = history["Date"].max()
            target = self.history._utc_timestamp(before_date)
            features[f"{prefix}_rest_days"] = float(
                (target - latest).total_seconds() / 86400
            )
        else:
            features[f"{prefix}_rest_days"] = math.nan
        return features

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
        features.update(self._team_features(home, cutoff, "home", "home"))
        features.update(self._team_features(away, cutoff, "away", "away"))
        h2h = self.history.get_match_history(home, away, cutoff)
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
