"""Leakage-safe pre-match feature generation from prior xG match history."""
from __future__ import annotations

import math
import numpy as np
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
        # Team histories are newest-first; reverse so EWMA gives greater
        # weight to the most recent completed match.
        values = values.iloc[::-1]
        return float(values.ewm(span=span, adjust=False).mean().iloc[-1]) if len(values) else math.nan

    def _team_features(
        self,
        history: pd.DataFrame,
        before_date,
        prefix: str,
        venue: str,
    ) -> dict:
        features = {f"{prefix}_matches_available": 0}
        for window in self.windows:
            for metric in (
                "goals_for_avg",
                "goals_against_avg",
                "xg_for_avg",
                "xg_against_avg",
                "points_avg",
            ):
                features[f"{prefix}_{metric}_{window}"] = math.nan
        features[f"{prefix}_xg_for_ewma_5"] = math.nan
        features[f"{prefix}_xg_against_ewma_5"] = math.nan
        features[f"{prefix}_rest_days"] = math.nan

        if history.empty:
            return features

        venue_history = history.loc[history["TeamSide"].eq(venue)].copy()
        team_goals = pd.to_numeric(venue_history["TeamGoals"], errors="coerce")
        opponent_goals = pd.to_numeric(venue_history["OpponentGoals"], errors="coerce")
        venue_history["Points"] = np.select(
            [team_goals > opponent_goals, team_goals == opponent_goals],
            [3, 1],
            default=0,
        )
        features[f"{prefix}_matches_available"] = int(len(venue_history))

        latest_match_date = history.iloc[0]["Date"]
        features[f"{prefix}_rest_days"] = (
            pd.Timestamp(before_date).normalize()
            - pd.Timestamp(latest_match_date).normalize()
        ).days

        for window in self.windows:
            recent = venue_history.head(window)
            features[f"{prefix}_goals_for_avg_{window}"] = self._mean(
                recent["TeamGoals"]
            )
            features[f"{prefix}_goals_against_avg_{window}"] = self._mean(
                recent["OpponentGoals"]
            )
            features[f"{prefix}_xg_for_avg_{window}"] = self._mean(
                recent["TeamXG"]
            )
            features[f"{prefix}_xg_against_avg_{window}"] = self._mean(
                recent["OpponentXG"]
            )
            features[f"{prefix}_points_avg_{window}"] = self._mean(
                recent["Points"]
            )

        recent5 = venue_history.head(5)
        features[f"{prefix}_xg_for_ewma_5"] = self._ewma(recent5["TeamXG"])
        features[f"{prefix}_xg_against_ewma_5"] = self._ewma(
            recent5["OpponentXG"]
        )
        return features

    def build_features_for_fixture(
        self,
        league: str,
        match_date,
        home_team: str,
        away_team: str,
    ) -> dict:
        """Build leakage-safe features for a fixture not yet present in history."""
        for name, value in (
            ("league", league),
            ("home_team", home_team),
            ("away_team", away_team),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

        cutoff = pd.to_datetime(match_date, errors="coerce", utc=True)
        if pd.isna(cutoff):
            raise ValueError("match_date must be a valid timestamp.")

        league = league.strip()
        home = home_team.strip()
        away = away_team.strip()
        features = {
            "League": league,
            "match_date": cutoff,
        }

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

    def build_features(self, match_id: str) -> dict:
        """Build one historical pre-match row; target outcomes are never features."""
        try:
            match = self.history._match_index.loc[str(match_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown MatchID={match_id!r}.") from exc
        if pd.isna(match["Date"]):
            raise ValueError(f"MatchID={match_id!r} has no valid kickoff date.")

        features = self.build_features_for_fixture(
            league=str(match["League"]),
            match_date=match["Date"],
            home_team=str(match["HomeTeam"]),
            away_team=str(match["AwayTeam"]),
        )
        features["MatchID"] = str(match_id)
        features["Season"] = str(match["Season"])
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
