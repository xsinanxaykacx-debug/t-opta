"""Sealed experiment protocol placeholder.

Before implementation, verify the canonical chronological split manifest from the
existing pipeline. Never infer or silently recreate split boundaries here.

Hard constraints:
- Train/calibration/policy-selection/policy-check/final-test chronology is immutable.
- Fit preprocessing and models on permitted earlier partitions only.
- Select thresholds/weights without consulting FINAL TEST.
- Closing-price columns (AvgC, B365C, MaxC, PC) are audit-only, never features.
- All three models must be evaluated on the same eligible fixtures.
"""

# TODO: Load and validate the canonical split manifest from the existing pipeline.
# TODO: Add explicit feature allowlists and closing-odds denylist checks.
