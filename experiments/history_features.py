"""Join xgo odds with the repository's audited, strict-past xg18 features."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import poisson

from modules.feature_engine import FeatureEngine
from modules.historical_engine import HistoricalEngine


XG18_GIT_BLOB_SHA = "ed2213748abb12a1815c11376931b97568c17cf9"


def git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def add_xg18_historical_features(
    frame: pd.DataFrame,
    history_path: str | Path = "xg18.csv",
    *,
    target_validation_limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build strictly prior-calendar-day xG features for every xgo fixture.

    xgo.csv is the odds-bearing modelling sample. xg18.csv is the broader
    historical source, including matches without odds. Joining by MatchID lets
    those additional matches contribute to history without becoming target
    rows in the odds comparison.
    """
    required = {"MatchID", "Date"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing xgo identity columns: {missing_columns}")

    out = frame.copy(deep=True)
    out["MatchID"] = out["MatchID"].astype("string").str.strip()
    if out["MatchID"].isna().any() or out["MatchID"].duplicated().any():
        raise ValueError("xgo MatchID values must be populated and unique.")

    history_file = Path(history_path)
    if not history_file.is_file():
        raise FileNotFoundError(f"xg18 history file not found: {history_file}")
    actual_history_sha = git_blob_sha(history_file)
    if actual_history_sha != XG18_GIT_BLOB_SHA:
        raise ValueError(
            "xg18.csv snapshot SHA mismatch. "
            f"Expected {XG18_GIT_BLOB_SHA}, received {actual_history_sha}."
        )
    history = HistoricalEngine(history_file)
    history_index = history.data.set_index("MatchID", drop=False)
    history_ids = set(history.data["MatchID"].astype(str))
    requested_ids = out["MatchID"].astype(str).tolist()
    missing_ids = [match_id for match_id in requested_ids if match_id not in history_ids]
    if missing_ids:
        preview = missing_ids[:10]
        raise ValueError(
            f"{len(missing_ids)} xgo MatchID values are absent from xg18.csv; "
            f"examples: {preview}"
        )

    matched_history = history_index.loc[requested_ids].reset_index(drop=True)

    for column in ("League", "HomeTeam", "AwayTeam"):
        source_values = out[column].astype(str).str.strip().str.casefold().reset_index(drop=True)
        history_values = matched_history[column].astype(str).str.strip().str.casefold().reset_index(drop=True)
        if not source_values.equals(history_values):
            mismatch_mask = source_values != history_values
            mismatch_count = int(mismatch_mask.sum())
            examples = list(
                zip(
                    out.loc[mismatch_mask, "MatchID"].astype(str).tolist()[:5],
                    source_values.loc[mismatch_mask].tolist()[:5],
                    history_values.loc[mismatch_mask].tolist()[:5],
                )
            )
            raise ValueError(
                f"xgo/xg18 {column} mismatch for {mismatch_count} MatchID rows; "
                f"examples (MatchID, xgo, xg18): {examples}"
            )

    if target_validation_limit is not None:
        if (
            isinstance(target_validation_limit, bool)
            or not isinstance(target_validation_limit, int)
            or not 0 <= target_validation_limit <= len(out)
        ):
            raise ValueError("target_validation_limit must be an integer row count in range.")
        validation_out = out.iloc[:target_validation_limit]
        validation_history = matched_history.iloc[:target_validation_limit]
    else:
        validation_out = out
        validation_history = matched_history

    # Outcome/xG consistency is checked only on pre-final rows. The sealed
    # Final Test target columns are not inspected before the policy gate.
    for column in ("FTHG", "FTAG", "Home_xG", "Away_xG"):
        source_values = pd.to_numeric(validation_out[column], errors="coerce").to_numpy(dtype=float)
        history_values = pd.to_numeric(validation_history[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(source_values, history_values, rtol=0.0, atol=1e-6, equal_nan=True):
            mismatch_count = int(
                (~np.isclose(source_values, history_values, rtol=0.0, atol=1e-6, equal_nan=True)).sum()
            )
            raise ValueError(
                f"Pre-final xgo/xg18 {column} mismatch for {mismatch_count} MatchID rows."
            )

    feature_engine = FeatureEngine(history)
    features = feature_engine.build_dataset(requested_ids)
    features["MatchID"] = features["MatchID"].astype("string").str.strip()
    if features["MatchID"].duplicated().any():
        raise ValueError("xg18 feature builder returned duplicate MatchID values.")

    if len(features) != len(out):
        raise ValueError(
            f"xg18 feature rows ({len(features)}) do not match xgo rows ({len(out)})."
        )

    history_date = pd.to_datetime(features["match_date"], errors="coerce", utc=True)
    source_date = pd.to_datetime(out["Date"], errors="coerce", utc=True)
    if history_date.isna().any() or source_date.isna().any():
        raise ValueError("Invalid match date encountered during xg18 join.")
    if not history_date.dt.normalize().reset_index(drop=True).equals(
        source_date.dt.normalize().reset_index(drop=True)
    ):
        raise ValueError("xgo and xg18 MatchID dates do not agree by calendar day.")

    identity_columns = {"MatchID", "League", "Season", "match_date"}
    historical_columns = [
        column for column in features.columns if column not in identity_columns
    ]
    historical = features[["MatchID", "match_date", *historical_columns]].copy()
    rename = {column: f"hist_{column}" for column in historical_columns}
    historical = historical.rename(columns=rename)
    historical = historical.rename(columns={"match_date": "hist_match_date"})

    out = out.merge(
        historical,
        on="MatchID",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if out[[f"hist_{column}" for column in historical_columns]].isna().all(axis=1).any():
        raise ValueError("One or more xgo fixtures have no joined historical features.")

    def positive_feature(name: str, fallback_name: str) -> np.ndarray:
        primary = pd.to_numeric(out[name], errors="coerce")
        fallback = pd.to_numeric(out[fallback_name], errors="coerce")
        values = primary.fillna(fallback).fillna(1.25)
        return values.clip(lower=0.05, upper=5.0).to_numpy(dtype=float)

    home_for = positive_feature(
        "hist_home_xg_for_ewma_5", "hist_home_xg_for_avg_5"
    )
    home_against = positive_feature(
        "hist_home_xg_against_ewma_5", "hist_home_xg_against_avg_5"
    )
    away_for = positive_feature(
        "hist_away_xg_for_ewma_5", "hist_away_xg_for_avg_5"
    )
    away_against = positive_feature(
        "hist_away_xg_against_ewma_5", "hist_away_xg_against_avg_5"
    )

    # Symmetric attack-vs-defence interaction with a modest, fixed home factor.
    lambda_home = np.clip(np.sqrt(home_for * away_against) * 1.05, 0.05, 5.0)
    lambda_away = np.clip(np.sqrt(away_for * home_against) / 1.05, 0.05, 5.0)
    out["hist_lambda_home"] = lambda_home
    out["hist_lambda_away"] = lambda_away
    out["p_poisson"] = np.clip(
        poisson.sf(2, lambda_home + lambda_away),
        1e-6,
        1.0 - 1e-6,
    )

    metadata = {
        "history_source": str(history.data_path),
        "history_git_blob_sha": XG18_GIT_BLOB_SHA,
        "history_rows": int(len(history.data)),
        "history_dated_rows": int(history.data["Date"].notna().sum()),
        "history_date_min": history.data["Date"].min().isoformat(),
        "history_date_max": history.data["Date"].max().isoformat(),
        "xgo_match_id_coverage": 1.0,
        "fixture_identity_verified": True,
        "pre_final_target_and_xg_verified": True,
        "pre_final_target_rows_verified": int(len(validation_out)),
        "historical_feature_count": int(len(historical_columns)),
        "history_rule": "same league; strictly earlier calendar days; same-day matches excluded",
    }
    return out, metadata
