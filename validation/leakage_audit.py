"""Temporal and target-leakage audits for pre-match feature generation."""
from __future__ import annotations

from typing import Any

import pandas as pd

FORBIDDEN_FEATURE_COLUMNS = {
    "FTHG", "FTAG", "HT_Home_Goals", "HT_Away_Goals",
    "HT_Home_xG", "HT_Away_xG", "Home_xG", "Away_xG",
    "Home_Shots", "Away_Shots", "Home_ShotsInBox", "Away_ShotsInBox",
    "Home_ShotsSetPiece", "Away_ShotsSetPiece",
    "Home_ShotsOnTarget", "Away_ShotsOnTarget",
    "Home_BigChance", "Away_BigChance", "Home_Corners", "Away_Corners",
    "Home_Poss", "Away_Poss", "Home_RedCards", "Away_RedCards",
    "Penalty_Awarded", "Attendance",
}


def audit_match_features(feature_engine: Any, match_id: str) -> dict:
    """Audit one feature row and the historical match IDs that can influence it."""
    history_engine = feature_engine.history
    match_id = str(match_id)
    violations: list[str] = []
    warnings: list[str] = []
    try:
        target = history_engine._match_index.loc[match_id]
    except KeyError:
        return {"match_id": match_id, "passed": False,
                "violations": [f"Unknown MatchID: {match_id}"],
                "warnings": [], "source_match_ids": []}

    cutoff = target["Date"]
    if pd.isna(cutoff):
        return {"match_id": match_id, "passed": False,
                "violations": ["Target match has no valid date."],
                "warnings": [], "source_match_ids": []}

    home, away = str(target["HomeTeam"]), str(target["AwayTeam"])
    home_history = history_engine.get_team_history(home, cutoff)
    away_history = history_engine.get_team_history(away, cutoff)
    home_history = home_history.loc[home_history["TeamSide"].eq("home")]
    away_history = away_history.loc[away_history["TeamSide"].eq("away")]
    h2h = history_engine.get_match_history(home, away, cutoff)
    sources = pd.concat([home_history, away_history, h2h], ignore_index=True)
    sources = sources.drop_duplicates("MatchID")

    if not sources.empty:
        if not (sources["Date"] < cutoff).all():
            violations.append("A source match is not strictly earlier than target kickoff.")
        if sources["MatchID"].astype(str).eq(match_id).any():
            violations.append("Target MatchID appears in historical feature sources.")
        if sources[["FTHG", "FTAG"]].isna().any().any():
            violations.append("An incomplete-score match appears in historical sources.")

    features = feature_engine.build_features(match_id)
    leaked = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(features))
    if leaked:
        violations.append(f"Forbidden target/in-match columns in features: {leaked}")
    if str(features.get("MatchID")) != match_id:
        violations.append("Feature row MatchID does not match the requested target.")
    if pd.Timestamp(features["match_date"]) != cutoff:
        violations.append("Feature row match_date does not match target kickoff.")

    dates = history_engine.data["Date"].dropna()
    if not dates.empty and (dates == dates.dt.normalize()).any():
        same_day = sources.loc[sources["Date"].dt.date == cutoff.date()]
        if not same_day.empty:
            warnings.append(
                "Same-day source rows exist alongside date-only timestamps; "
                "confirm kickoff-time provenance before using them."
            )

    return {
        "match_id": match_id, "passed": not violations,
        "violations": violations, "warnings": warnings,
        "cutoff": cutoff.isoformat(),
        "source_match_ids": sorted(sources["MatchID"].astype(str).unique().tolist()),
        "source_match_count": int(sources["MatchID"].nunique()),
        "feature_count": int(len(features)),
    }


def audit_feature_sample(feature_engine: Any, sample_size: int = 100, seed: int = 42) -> dict:
    """Run deterministic audits over a reproducible sample of dated matches."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer.")
    data = feature_engine.history.data
    eligible = data.loc[data["Date"].notna(), "MatchID"].astype(str)
    if eligible.empty:
        return {"passed": False, "sample_size": 0, "results": [], "failed_match_ids": []}
    chosen = eligible.sample(n=min(sample_size, len(eligible)), random_state=seed)
    results = [audit_match_features(feature_engine, mid) for mid in chosen]
    failed = [item["match_id"] for item in results if not item["passed"]]
    warnings = [{"match_id": item["match_id"], "warning": warning}
                for item in results for warning in item["warnings"]]
    return {"passed": not failed, "sample_size": len(results), "seed": seed,
            "failed_match_ids": failed, "warning_count": len(warnings),
            "warnings": warnings, "results": results}


def audit_feature_dataset(
    feature_engine: Any, batch_size: int = 250, include_details: bool = False
) -> dict:
    """Audit every dated match while bounding report memory by default."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    data = feature_engine.history.data
    eligible = data.loc[data["Date"].notna(), "MatchID"].astype(str).tolist()
    if not eligible:
        return {"passed": False, "total_source_rows": int(len(data)),
                "eligible_matches": 0, "audited_matches": 0, "failed_count": 0,
                "warning_count": 0, "failed_match_ids": [],
                "details": [] if include_details else None}

    failed_ids: list[str] = []
    warning_count = 0
    source_total = 0
    details: list[dict] = []
    for offset in range(0, len(eligible), batch_size):
        for match_id in eligible[offset:offset + batch_size]:
            result = audit_match_features(feature_engine, match_id)
            source_total += result.get("source_match_count", 0)
            if not result["passed"]:
                failed_ids.append(match_id)
            warning_count += len(result.get("warnings", []))
            if include_details:
                details.append(result)

    audited = len(eligible)
    return {
        "passed": not failed_ids,
        "total_source_rows": int(len(data)),
        "eligible_matches": audited,
        "audited_matches": audited,
        "failed_count": len(failed_ids),
        "failed_match_ids": failed_ids,
        "warning_count": warning_count,
        "average_source_matches": float(source_total / audited),
        "details": details if include_details else None,
    }
