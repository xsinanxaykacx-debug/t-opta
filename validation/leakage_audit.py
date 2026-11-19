
"""Temporal and target-leakage audits for pre-match feature generation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modules.historical_engine import filter_strict_past


FORBIDDEN_FEATURE_COLUMNS = {
    "FTHG", "FTAG", "HT_Home_Goals", "HT_Away_Goals",
    "HT_Home_xG", "HT_Away_xG", "Home_xG", "Away_xG",
    "Home_Shots", "Away_Shots",
    "Home_ShotsInBox", "Away_ShotsInBox",
    "Home_ShotsSetPiece", "Away_ShotsSetPiece",
    "Home_ShotsOnTarget", "Away_ShotsOnTarget",
    "Home_BigChance", "Away_BigChance",
    "Home_Corners", "Away_Corners",
    "Home_Poss", "Away_Poss",
    "Home_RedCards", "Away_RedCards",
    "Penalty_Awarded", "Attendance",
}


FEATURE_AUDIT_SAMPLE_SIZE = 100
FEATURE_AUDIT_RANDOM_SEED = 42


def _normalised_date_ns(values: pd.Series) -> np.ndarray:
    """Convert timestamps to normalized UTC nanoseconds."""
    dates = pd.to_datetime(values, errors="coerce", utc=True)
    return dates.dt.normalize().astype("int64").to_numpy()


def _count_before(sorted_dates: np.ndarray, cutoff_ns: int) -> int:
    """Count dates strictly earlier than the target calendar day."""
    if sorted_dates.size == 0:
        return 0

    return int(np.searchsorted(sorted_dates, cutoff_ns, side="left"))


def _build_audit_indexes(history_engine: Any) -> dict:
    """Build compact sorted indexes for completed historical matches."""
    team_indexes: dict[str, dict] = {}
    pair_indexes: dict[tuple[str, str], dict] = {}

    for team_key, frame in history_engine._team_index.items():
        if frame.empty:
            team_indexes[team_key] = {
                "home_dates": np.array([], dtype=np.int64),
                "away_dates": np.array([], dtype=np.int64),
            }
            continue

        dates = _normalised_date_ns(frame["Date"])
        sides = (
            frame["TeamSide"]
            .astype(str)
            .str.casefold()
            .to_numpy()
        )

        team_indexes[team_key] = {
            "home_dates": np.sort(dates[sides == "home"]),
            "away_dates": np.sort(dates[sides == "away"]),
        }

        opponents = (
            frame["Opponent"]
            .astype("string")
            .str.strip()
            .str.casefold()
            .fillna("")
            .to_numpy()
        )

        for opponent in np.unique(opponents):
            if not opponent:
                continue

            mask = opponents == opponent

            pair_indexes[(team_key, opponent)] = {
                "all_dates": np.sort(dates[mask]),
                "home_dates": np.sort(
                    dates[mask & (sides == "home")]
                ),
            }

    return {
        "teams": team_indexes,
        "pairs": pair_indexes,
    }


def _fast_source_count(
    audit_indexes: dict,
    home_team: str,
    away_team: str,
    cutoff_ns: int,
) -> int:
    """Calculate historical source count without rebuilding DataFrames."""
    home_key = str(home_team).strip().casefold()
    away_key = str(away_team).strip().casefold()

    empty_dates = np.array([], dtype=np.int64)

    home_index = audit_indexes["teams"].get(home_key, {})
    away_index = audit_indexes["teams"].get(away_key, {})

    home_count = _count_before(
        home_index.get("home_dates", empty_dates),
        cutoff_ns,
    )

    away_count = _count_before(
        away_index.get("away_dates", empty_dates),
        cutoff_ns,
    )

    pair_index = audit_indexes["pairs"].get(
        (home_key, away_key),
        {},
    )

    h2h_count = _count_before(
        pair_index.get("all_dates", empty_dates),
        cutoff_ns,
    )

    oriented_count = _count_before(
        pair_index.get("home_dates", empty_dates),
        cutoff_ns,
    )

    # A previous home-vs-away match can appear in all three sources.
    # Remove the duplicate appearances while retaining one source match.
    return max(
        0,
        home_count + away_count + h2h_count - (2 * oriented_count),
    )


def audit_match_features(
    feature_engine: Any,
    match_id: str,
) -> dict:
    """Audit one feature row and all historical matches used by its features."""
    history_engine = feature_engine.history
    match_id = str(match_id)
    violations: list[str] = []

    try:
        target = history_engine._match_index.loc[match_id]
    except KeyError:
        return {
            "match_id": match_id,
            "passed": False,
            "violations": [f"Unknown MatchID: {match_id}"],
            "warnings": [],
            "source_match_ids": [],
        }

    cutoff = target["Date"]

    if pd.isna(cutoff):
        return {
            "match_id": match_id,
            "passed": False,
            "violations": ["Target match has no valid date."],
            "warnings": [],
            "source_match_ids": [],
        }

    home = str(target["HomeTeam"])
    away = str(target["AwayTeam"])

    home_history = history_engine.get_team_history(home, cutoff)
    away_history = history_engine.get_team_history(away, cutoff)

    home_history = home_history.loc[
        home_history["TeamSide"].eq("home")
    ]

    away_history = away_history.loc[
        away_history["TeamSide"].eq("away")
    ]

    h2h = history_engine.get_match_history(
        home,
        away,
        cutoff,
    )

    source_frames = [
        frame
        for frame in (home_history, away_history, h2h)
        if not frame.empty
    ]

    if source_frames:
        sources = (
            pd.concat(source_frames, ignore_index=True)
            .drop_duplicates("MatchID")
        )
    else:
        sources = history_engine.data.iloc[0:0].copy()

    if not sources.empty:
        source_dates = pd.to_datetime(
            sources["Date"],
            errors="coerce",
            utc=True,
        ).dt.normalize()

        target_date = pd.to_datetime(
            cutoff,
            errors="coerce",
            utc=True,
        ).normalize()

        if not source_dates.lt(target_date).all():
            violations.append(
                "A source match is not on a strictly earlier calendar day."
            )

        if sources["MatchID"].astype(str).eq(match_id).any():
            violations.append(
                "Target MatchID appears in historical feature sources."
            )

        if sources[["FTHG", "FTAG"]].isna().any().any():
            violations.append(
                "An incomplete-score match appears in historical sources."
            )

    features = feature_engine.build_features(match_id)

    leaked = sorted(
        FORBIDDEN_FEATURE_COLUMNS.intersection(features)
    )

    if leaked:
        violations.append(
            f"Forbidden target/in-match columns in features: {leaked}"
        )

    if str(features.get("MatchID")) != match_id:
        violations.append(
            "Feature row MatchID does not match the requested target."
        )

    feature_date = pd.to_datetime(
        features.get("match_date"),
        errors="coerce",
        utc=True,
    )

    target_timestamp = pd.to_datetime(
        cutoff,
        errors="coerce",
        utc=True,
    )

    if pd.isna(feature_date) or feature_date != target_timestamp:
        violations.append(
            "Feature row match_date does not match target kickoff."
        )

    source_ids = (
        sorted(sources["MatchID"].astype(str).unique().tolist())
        if not sources.empty
        else []
    )

    return {
        "match_id": match_id,
        "passed": not violations,
        "violations": violations,
        "warnings": [],
        "cutoff": cutoff.isoformat(),
        "source_match_ids": source_ids,
        "source_match_count": len(source_ids),
        "feature_count": int(len(features)),
    }


def audit_feature_sample(
    feature_engine: Any,
    sample_size: int = FEATURE_AUDIT_SAMPLE_SIZE,
    seed: int = FEATURE_AUDIT_RANDOM_SEED,
) -> dict:
    """Run real feature and leakage audits on a deterministic sample."""
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size < 1
    ):
        raise ValueError("sample_size must be a positive integer.")

    data = feature_engine.history.data

    eligible = data.loc[
        data["Date"].notna(),
        "MatchID",
    ].astype(str)

    if eligible.empty:
        return {
            "passed": False,
            "sample_size": 0,
            "seed": seed,
            "failed_match_ids": [],
            "warning_count": 0,
            "warnings": [],
            "results": [],
        }

    chosen = eligible.sample(
        n=min(sample_size, len(eligible)),
        random_state=seed,
    )

    results = [
        audit_match_features(feature_engine, match_id)
        for match_id in chosen
    ]

    failed = [
        item["match_id"]
        for item in results
        if not item["passed"]
    ]

    return {
        "passed": not failed,
        "sample_size": len(results),
        "seed": seed,
        "failed_match_ids": failed,
        "warning_count": 0,
        "warnings": [],
        "results": results,
    }


def _audit_feature_dataset_fast(
    feature_engine: Any,
    batch_size: int,
) -> dict:
    """
    Audit all dated matches using indexed lineage checks.

    Actual feature generation and leakage checks are additionally
    performed on a deterministic sample. The report keeps these
    two audit scopes separate.
    """
    history_engine = feature_engine.history
    data = history_engine.data

    dated = data.loc[data["Date"].notna()].copy()
    eligible_count = len(dated)

    if eligible_count == 0:
        return {
            "passed": False,
            "total_source_rows": int(len(data)),
            "eligible_matches": 0,
            "audited_matches": 0,
            "lineage_audited_matches": 0,
            "feature_audited_matches": 0,
            "feature_audit_scope": "deterministic_sample",
            "failed_count": 0,
            "failed_match_ids": [],
            "feature_audit_failed_match_ids": [],
            "source_count_mismatch_match_ids": [],
            "warning_count": 0,
            "average_source_matches": 0.0,
            "details": None,
        }

    required = {
        "MatchID",
        "Date",
        "HomeTeam",
        "AwayTeam",
    }

    missing = sorted(required.difference(dated.columns))

    if missing:
        failed_ids = dated["MatchID"].astype(str).tolist()

        return {
            "passed": False,
            "total_source_rows": int(len(data)),
            "eligible_matches": eligible_count,
            "audited_matches": 0,
            "lineage_audited_matches": 0,
            "feature_audited_matches": 0,
            "feature_audit_scope": "deterministic_sample",
            "failed_count": len(failed_ids),
            "failed_match_ids": failed_ids,
            "feature_audit_failed_match_ids": [],
            "source_count_mismatch_match_ids": [],
            "warning_count": 0,
            "average_source_matches": 0.0,
            "missing_columns": missing,
            "details": None,
        }

    audit_indexes = _build_audit_indexes(history_engine)

    dates_ns = _normalised_date_ns(dated["Date"])
    match_ids = dated["MatchID"].astype(str).to_numpy()
    home_teams = dated["HomeTeam"].astype(str).to_numpy()
    away_teams = dated["AwayTeam"].astype(str).to_numpy()

    failed_ids: list[str] = []
    source_total = 0

    # Full-dataset indexed temporal and lineage count audit.
    for offset in range(0, eligible_count, batch_size):
        end = min(offset + batch_size, eligible_count)

        for position in range(offset, end):
            match_id = match_ids[position]
            cutoff_ns = int(dates_ns[position])

            if cutoff_ns == np.iinfo(np.int64).min:
                failed_ids.append(match_id)
                continue

            source_count = _fast_source_count(
                audit_indexes,
                home_teams[position],
                away_teams[position],
                cutoff_ns,
            )

            source_total += source_count

    # Real feature generation and leakage audit on a deterministic sample.
    feature_sample = audit_feature_sample(
        feature_engine,
        sample_size=min(FEATURE_AUDIT_SAMPLE_SIZE, eligible_count),
        seed=FEATURE_AUDIT_RANDOM_SEED,
    )

    feature_failed_ids = list(
        feature_sample["failed_match_ids"]
    )

    # Independently compare indexed source counts with the real
    # historical DataFrame counts for every sampled match.
    source_count_mismatch_ids: list[str] = []

    for result in feature_sample["results"]:
        match_id = result["match_id"]

        try:
            target = history_engine._match_index.loc[match_id]

            cutoff_ns = int(
                _normalised_date_ns(
                    pd.Series([target["Date"]])
                )[0]
            )

            indexed_count = _fast_source_count(
                audit_indexes,
                str(target["HomeTeam"]),
                str(target["AwayTeam"]),
                cutoff_ns,
            )

            actual_count = int(
                result.get("source_match_count", 0)
            )

            if indexed_count != actual_count:
                source_count_mismatch_ids.append(match_id)

        except (KeyError, TypeError, ValueError, IndexError):
            source_count_mismatch_ids.append(match_id)

    failed_ids.extend(feature_failed_ids)
    failed_ids.extend(source_count_mismatch_ids)

    failed_ids = list(dict.fromkeys(failed_ids))

    return {
        "passed": (
            not failed_ids
            and feature_sample["passed"]
        ),
        "total_source_rows": int(len(data)),
        "eligible_matches": eligible_count,
        "audited_matches": eligible_count,
        "lineage_audited_matches": eligible_count,
        "feature_audited_matches": feature_sample["sample_size"],
        "feature_audit_scope": "deterministic_sample",
        "feature_audit_seed": FEATURE_AUDIT_RANDOM_SEED,
        "feature_audit_failed_match_ids": feature_failed_ids,
        "source_count_mismatch_match_ids": source_count_mismatch_ids,
        "failed_count": len(failed_ids),
        "failed_match_ids": failed_ids,
        "warning_count": 0,
        "average_source_matches": float(
            source_total / eligible_count
        ),
        "details": None,
    }


def audit_feature_dataset(
    feature_engine: Any,
    batch_size: int = 250,
    include_details: bool = False,
) -> dict:
    """
    Audit the dataset.

    Default mode:
      - Indexed lineage audit across every dated match.
      - Real feature and leakage audit on a deterministic sample.

    include_details=True:
      - Real feature and leakage audit on every dated match.
      - Detailed per-match results are returned.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")

    if not include_details:
        return _audit_feature_dataset_fast(
            feature_engine,
            batch_size,
        )

    data = feature_engine.history.data

    eligible = data.loc[
        data["Date"].notna(),
        "MatchID",
    ].astype(str).tolist()

    if not eligible:
        return {
            "passed": False,
            "total_source_rows": int(len(data)),
            "eligible_matches": 0,
            "audited_matches": 0,
            "lineage_audited_matches": 0,
            "feature_audited_matches": 0,
            "feature_audit_scope": "full_dataset",
            "failed_count": 0,
            "failed_match_ids": [],
            "warning_count": 0,
            "average_source_matches": 0.0,
            "details": [],
        }

    failed_ids: list[str] = []
    source_total = 0
    details: list[dict] = []

    for offset in range(0, len(eligible), batch_size):
        batch = eligible[offset:offset + batch_size]

        for match_id in batch:
            result = audit_match_features(
                feature_engine,
                match_id,
            )

            source_total += result.get(
                "source_match_count",
                0,
            )

            if not result["passed"]:
                failed_ids.append(match_id)

            details.append(result)

    audited = len(eligible)

    return {
        "passed": not failed_ids,
        "total_source_rows": int(len(data)),
        "eligible_matches": audited,
        "audited_matches": audited,
        "lineage_audited_matches": audited,
        "feature_audited_matches": audited,
        "feature_audit_scope": "full_dataset",
        "failed_count": len(failed_ids),
        "failed_match_ids": failed_ids,
        "warning_count": 0,
        "average_source_matches": float(
            source_total / audited
        ),
        "details": details,
    }
