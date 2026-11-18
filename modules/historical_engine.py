"""Read-only historical access for the xg18.csv dataset."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = (
    "MatchID", "League", "Season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "Home_xG", "Away_xG",
)
NUMERIC_COLUMNS = (
    "FTHG", "FTAG", "HT_Home_Goals", "HT_Away_Goals", "Home_xG",
    "Away_xG", "HT_Home_xG", "HT_Away_xG", "Home_Shots", "Away_Shots",
    "Home_ShotsInBox", "Away_ShotsInBox", "Home_ShotsSetPiece",
    "Away_ShotsSetPiece", "Home_ShotsOnTarget", "Away_ShotsOnTarget",
    "Home_BigChance", "Away_BigChance", "Home_Corners", "Away_Corners",
    "Home_Poss", "Away_Poss", "Home_RedCards", "Away_RedCards",
    "Penalty_Awarded", "Home_Squad_Value_M", "Away_Squad_Value_M",
    "Home_Injured_Key_Players", "Away_Injured_Key_Players",
    "Weather_Temp", "Weather_Rain", "Attendance",
)


class HistoricalEngine:
    """Loads the repository dataset and exposes strictly pre-match history.

    The source CSV is never modified. Rows with missing/unparseable dates remain
    in data for audit, but cannot be returned by time-bounded history queries.
    """

    def __init__(self, data_path: str | Path = "data/raw/xg18.csv"):
        requested = Path(data_path)
        candidates = [requested]
        if requested == Path("data/raw/xg18.csv"):
            candidates.extend([Path("xg18.csv"), Path(__file__).resolve().parents[1] / "xg18.csv"])
        self.data_path = next((p for p in candidates if p.is_file()), None)
        if self.data_path is None:
            raise FileNotFoundError(
                f"xG dataset not found. Checked: {', '.join(map(str, candidates))}"
            )
        self.data = self._load_data()
        self.df = self.data  # Backwards-compatible alias; treat as read-only.
        self._team_index = self._build_team_index()
        self._match_index = self.data.set_index("MatchID", drop=False)

    def _load_data(self) -> pd.DataFrame:
        frame = pd.read_csv(self.data_path, encoding="utf-8-sig")
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"xG dataset missing required columns: {missing}")
        frame = frame.copy()
        frame["MatchID"] = frame["MatchID"].astype("string").str.strip()
        for column in ("League", "Season", "HomeTeam", "AwayTeam", "Referee"):
            if column in frame:
                frame[column] = frame[column].astype("string").str.strip()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
        for column in NUMERIC_COLUMNS:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame["MatchID"].isna().any() or frame["MatchID"].duplicated().any():
            raise ValueError("MatchID must be populated and unique.")
        if frame[["HomeTeam", "AwayTeam"]].isna().any().any():
            raise ValueError("HomeTeam/AwayTeam cannot be empty.")
        return frame.sort_values(
            ["Date", "MatchID"], na_position="last", kind="stable"
        ).reset_index(drop=True)

    @property
    def schema_report(self) -> dict:
        return {
            "path": str(self.data_path),
            "rows": int(len(self.data)),
            "columns": int(len(self.data.columns)),
            "dated_rows": int(self.data["Date"].notna().sum()),
            "undated_rows": int(self.data["Date"].isna().sum()),
            "duplicate_match_ids": int(self.data["MatchID"].duplicated().sum()),
            "leagues": sorted(self.data["League"].dropna().unique().tolist()),
            "seasons": sorted(self.data["Season"].dropna().unique().tolist()),
            "missing_by_column": self.data.isna().sum().astype(int).to_dict(),
        }

    @staticmethod
    def _utc_timestamp(value) -> pd.Timestamp:
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(timestamp):
            raise ValueError(f"Invalid before_date: {value!r}")
        return timestamp

    def _build_team_index(self) -> dict[str, pd.DataFrame]:
        dated = self.data.loc[self.data["Date"].notna()].copy()
        home = dated.copy()
        home["Team"] = home["HomeTeam"]
        home["Opponent"] = home["AwayTeam"]
        home["TeamSide"] = "home"
        home["TeamGoals"] = home["FTHG"]
        home["OpponentGoals"] = home["FTAG"]
        home["TeamXG"] = home["Home_xG"]
        home["OpponentXG"] = home["Away_xG"]

        away = dated.copy()
        away["Team"] = away["AwayTeam"]
        away["Opponent"] = away["HomeTeam"]
        away["TeamSide"] = "away"
        away["TeamGoals"] = away["FTAG"]
        away["OpponentGoals"] = away["FTHG"]
        away["TeamXG"] = away["Away_xG"]
        away["OpponentXG"] = away["Home_xG"]

        combined = pd.concat([home, away], ignore_index=True)
        index = {}
        for team, group in combined.groupby(combined["Team"].str.casefold(), sort=False):
            index[team] = group.sort_values(
                ["Date", "MatchID"], kind="stable"
            ).reset_index(drop=True)
        return index

    def get_team_history(
        self, team_name: str, before_date, league: str | None = None
    ) -> pd.DataFrame:
        """Return prior team matches, newest first; exact-cutoff matches are excluded."""
        if not isinstance(team_name, str) or not team_name.strip():
            raise ValueError("team_name must be a non-empty string.")
        cutoff = self._utc_timestamp(before_date)
        source = self._team_index.get(team_name.strip().casefold())
        if source is None:
            return pd.DataFrame(columns=list(self.data.columns) + [
                "Team", "Opponent", "TeamSide", "TeamGoals", "OpponentGoals",
                "TeamXG", "OpponentXG"
            ])
        mask = source["Date"] < cutoff
        if league is not None:
            mask &= source["League"].str.casefold() == league.strip().casefold()
        return source.loc[mask].sort_values(
            ["Date", "MatchID"], ascending=[False, False], kind="stable"
        ).copy().reset_index(drop=True)

    def get_match_history(self, home_team: str, away_team: str, before_date) -> pd.DataFrame:
        """Return prior head-to-head matches, excluding the target timestamp."""
        if not isinstance(away_team, str) or not away_team.strip():
            raise ValueError("away_team must be a non-empty string.")
        history = self.get_team_history(home_team, before_date)
        mask = history["Opponent"].str.casefold() == away_team.strip().casefold()
        return history.loc[mask].copy().reset_index(drop=True)
