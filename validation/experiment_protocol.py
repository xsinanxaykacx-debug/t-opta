"""Immutable chronological and leakage protocol for the three-model experiment.

The partition sizes below are the approved experiment contract. They are NOT
permission to recreate split boundaries: the canonical upstream manifest must
supply the row assignments and dates, and must match this contract exactly.

No model, threshold, calibration map, feature selector, or policy may inspect
FINAL_TEST while being fitted or selected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# ---------------------------------------------------------------------------
# Immutable chronological split contract
# ---------------------------------------------------------------------------

PARTITION_ORDER = (
    "train",
    "calibration",
    "policy_select",
    "policy_check",
    "final_test",
)

EXPECTED_SPLIT_SIZES = {
    "train": 8_500,
    "calibration": 900,
    "policy_select": 300,
    "policy_check": 300,
    "final_test": 5_734,
}

EXPECTED_TOTAL_ROWS = sum(EXPECTED_SPLIT_SIZES.values())
assert EXPECTED_TOTAL_ROWS == 15_734

SEALED_PARTITIONS = frozenset({"final_test"})


# ---------------------------------------------------------------------------
# Feature policy: denylist always wins over the allowlist
# ---------------------------------------------------------------------------

# Closing prices are post-market information and are audit-only.
CLOSING_ODDS_COLUMNS = frozenset(
    {"avgc", "b365c", "maxc", "pc"}
)

# Match outcomes, half-time outcomes, and post-match statistics are never
# model inputs. Add source-specific result columns here when the schema is
# inspected; do not weaken this guard to make a pipeline pass.
POST_MATCH_COLUMNS = frozenset(
    {
        "fthg", "ftag", "ftr",
        "hthg", "htag", "htr",
        "ht goals", "ft goals",
        "home_goals", "away_goals",
        "home_score", "away_score",
        "result", "match_result", "outcome",
        "winner", "target",
        "xg", "xg_home", "xg_away", "home_xg", "away_xg",
        "match_xg", "live_xg", "inplay_xg",
    }
)

# These prefixes are allowed only for genuinely pre-match information.
# A denylisted column is rejected even if its name starts with an allowed
# prefix. Source-specific feature names should be explicitly reviewed and
# passed through approved_features.
SAFE_FEATURE_PREFIXES = (
    "odds_",
    "hist_",
    "rolling_",
    "form_",
    "pre_match_",
)

APPROVED_OPENING_OR_CURRENT_ODDS = frozenset(
    {
        "odds_avg",
        "odds_b365",
    }
)


class ProtocolError(ValueError):
    """Raised when data or feature inputs violate the experiment contract."""


def _normalise_column(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ProtocolError(f"Invalid feature column name: {name!r}")
    return name.strip().casefold()


def _is_post_match_column(normalised_name: str) -> bool:
    if normalised_name in CLOSING_ODDS_COLUMNS:
        return True
    if normalised_name in POST_MATCH_COLUMNS:
        return True

    # Common source variants for full-time / half-time result fields.
    compact = normalised_name.replace("_", "").replace(" ", "")
    if compact in {"fthg", "ftag", "ftr", "hthg", "htag", "htr"}:
        return True

    # Explicitly reject in-play/live xG columns without rejecting historical
    # rolling xG aggregates such as hist_home_xg or rolling_xg_for.
    if any(token in normalised_name for token in ("inplay", "in_play", "live_")):
        return True

    return False


def validate_feature_columns(
    feature_columns: Sequence[str],
    *,
    approved_features: Sequence[str] = (),
) -> tuple[str, ...]:
    """Validate feature names; return the original names in input order.

    Policy:
    1. Closing odds and post-match/live data are always rejected.
    2. Remaining columns must be explicitly approved, be approved opening /
       current odds, or use a documented pre-match historical prefix.
    3. This function validates names, not provenance. The data pipeline must
       still prove that historical aggregates were computed using past rows
       only and that odds were available at prediction time.
    """
    if not feature_columns:
        raise ProtocolError("Feature list is empty.")

    approved = {_normalise_column(name) for name in approved_features}
    seen: set[str] = set()
    validated: list[str] = []

    for original_name in feature_columns:
        name = _normalise_column(original_name)

        if name in seen:
            raise ProtocolError(f"Duplicate feature column: {original_name!r}")
        seen.add(name)

        if _is_post_match_column(name):
            raise ProtocolError(
                f"Leakage guard rejected post-match/closing feature: {original_name!r}"
            )

        is_allowed = (
            name in approved
            or name in APPROVED_OPENING_OR_CURRENT_ODDS
            or name.startswith(SAFE_FEATURE_PREFIXES)
        )
        if not is_allowed:
            raise ProtocolError(
                f"Feature is not on the pre-match allowlist: {original_name!r}. "
                "Review its provenance and add it to approved_features explicitly."
            )

        validated.append(original_name)

    return tuple(validated)


# ---------------------------------------------------------------------------
# Determinism and split-integrity checks
# ---------------------------------------------------------------------------

def validate_chronological_order(
    dates: Sequence[Any],
    *,
    column_name: str = "date",
) -> None:
    """Assert that supplied row dates are present and nondecreasing.

    Values must be mutually comparable (for example datetime/date objects or
    consistently formatted ISO-8601 strings). No sorting is performed here:
    silently sorting would conceal an upstream ordering defect.
    """
    if not dates:
        raise ProtocolError(f"{column_name!r} contains no rows.")

    previous = None
    for position, current in enumerate(dates):
        if current is None:
            raise ProtocolError(
                f"Missing {column_name!r} at row position {position}."
            )
        try:
            if previous is not None and current < previous:
                raise ProtocolError(
                    f"Rows are not chronological at position {position}: "
                    f"{current!r} precedes {previous!r}."
                )
        except TypeError as exc:
            raise ProtocolError(
                f"{column_name!r} values are not mutually comparable; "
                "normalise dates before validation."
            ) from exc
        previous = current


def validate_split_indices(
    split_indices: Mapping[str, Sequence[int]],
    *,
    expected_sizes: Mapping[str, int] = EXPECTED_SPLIT_SIZES,
    require_full_coverage: bool = True,
) -> None:
    """Validate exact partition sizes, no overlap, and chronological ordering.

    Indices must refer to positions in the already chronologically ordered
    source dataset. This function never creates or reassigns a split.
    """
    expected_names = tuple(expected_sizes)
    if tuple(split_indices) != expected_names:
        raise ProtocolError(
            "Split names/order mismatch. "
            f"Expected {expected_names}, received {tuple(split_indices)}."
        )

    all_indices: list[int] = []
    previous_max = -1

    for partition in expected_names:
        indices = list(split_indices[partition])
        expected_count = expected_sizes[partition]

        if len(indices) != expected_count:
            raise ProtocolError(
                f"{partition} has {len(indices)} rows; expected {expected_count}."
            )
        if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
            raise ProtocolError(f"{partition} contains a non-integer row index.")
        if len(set(indices)) != len(indices):
            raise ProtocolError(f"{partition} contains duplicate row indices.")
        if indices != sorted(indices):
            raise ProtocolError(f"{partition} indices are not in source order.")
        if indices and indices[0] <= previous_max:
            raise ProtocolError(
                f"{partition} overlaps or precedes the previous partition."
            )

        if indices:
            previous_max = indices[-1]
        all_indices.extend(indices)

    if len(set(all_indices)) != len(all_indices):
        raise ProtocolError("Chronological partitions overlap.")

    if require_full_coverage:
        expected_total = sum(expected_sizes.values())
        if len(all_indices) != expected_total:
            raise ProtocolError(
                f"Split coverage is {len(all_indices)} rows; "
                f"expected {expected_total}."
            )
        if sorted(all_indices) != list(range(expected_total)):
            raise ProtocolError(
                "Split indices must cover every source row exactly once "
                "from 0 through N-1."
            )


def validate_partition_dates(
    partition_dates: Mapping[str, Sequence[Any]],
) -> None:
    """Validate row order inside each partition and strict date progression
    between adjacent partitions. Equal dates across a boundary are rejected
    to prevent same-timestamp fixtures from being split across partitions.
    """
    if tuple(partition_dates) != PARTITION_ORDER:
        raise ProtocolError(
            f"Partition date order must be {PARTITION_ORDER}; "
            f"received {tuple(partition_dates)}."
        )

    previous_last = None
    for partition in PARTITION_ORDER:
        dates = partition_dates[partition]
        validate_chronological_order(dates, column_name=f"{partition}.date")

        first_date = dates[0]
        last_date = dates[-1]
        if previous_last is not None:
            try:
                if first_date <= previous_last:
                    raise ProtocolError(
                        f"{partition} does not start strictly after the "
                        "previous partition ends."
                    )
            except TypeError as exc:
                raise ProtocolError(
                    "Partition date values are not mutually comparable."
                ) from exc
        previous_last = last_date


def assert_final_test_sealed(requested_partitions: Sequence[str]) -> None:
    """Reject any fitting/selection operation that requests FINAL_TEST."""
    normalised = {str(partition).strip().casefold() for partition in requested_partitions}
    forbidden = SEALED_PARTITIONS.intersection(normalised)
    if forbidden:
        raise ProtocolError(
            "FINAL_TEST is sealed and cannot be used for fitting, feature "
            f"selection, calibration, threshold selection, or policy tuning: "
            f"{sorted(forbidden)}"
        )


def validate_manifest(
    *,
    split_indices: Mapping[str, Sequence[int]],
    partition_dates: Mapping[str, Sequence[Any]],
) -> None:
    """Single entry point for validating a canonical upstream split manifest."""
    validate_split_indices(split_indices)
    validate_partition_dates(partition_dates)
