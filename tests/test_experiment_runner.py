import numpy as np
import pandas as pd

from experiments.comparison import purge_same_calendar_day_boundaries
from experiments.history_features import add_xg18_historical_features
from validation.experiment_protocol import PARTITION_ORDER


def test_same_calendar_day_rows_are_purged_from_preceding_stage():
    dates = {
        "train": ["2024-01-01T10:00:00Z", "2024-01-02T12:00:00Z"],
        "calibration": ["2024-01-02T15:00:00Z", "2024-01-03T12:00:00Z"],
        "policy_select": ["2024-01-03T16:00:00Z", "2024-01-04T12:00:00Z"],
        "policy_check": ["2024-01-04T18:00:00Z", "2024-01-05T12:00:00Z"],
        "final_test": ["2024-01-06T12:00:00Z"],
    }
    partitions = {
        name: pd.DataFrame(
            {
                "Date": pd.to_datetime(values, utc=True),
                "_source_row": range(index * 10, index * 10 + len(values)),
            }
        )
        for index, (name, values) in enumerate(dates.items())
    }

    effective, purged = purge_same_calendar_day_boundaries(partitions)

    assert tuple(effective) == PARTITION_ORDER
    assert purged == {
        "train": 1,
        "calibration": 1,
        "policy_select": 1,
        "policy_check": 0,
        "final_test": 0,
    }
    assert all(len(effective[name]) > 0 for name in PARTITION_ORDER)

    for previous, current in zip(PARTITION_ORDER, PARTITION_ORDER[1:]):
        assert (
            effective[previous]["Date"].dt.normalize().max()
            < effective[current]["Date"].dt.normalize().min()
        )



def test_xgo_matches_join_to_audited_xg18_history():
    source = pd.read_csv(
        "xgo.csv",
        encoding="utf-8-sig",
        skiprows=range(1, 8500),
        nrows=3,
    )
    source["Date"] = pd.to_datetime(source["Date"], errors="coerce", utc=True)

    featured, metadata = add_xg18_historical_features(source, "xg18.csv")

    assert len(featured) == 3
    assert metadata["xgo_match_id_coverage"] == 1.0
    assert metadata["history_rows"] == 18509
    assert featured["p_poisson"].between(0.0, 1.0).all()
    assert featured["hist_home_xg_for_avg_5"].notna().any()


def test_policy_selection_prefers_hit_rate_when_minimum_sample_is_met():
    from experiments.comparison import choose_confidence_policy

    y_true = np.array([1] * 25 + [0] * 25 + [0] * 25 + [1] * 25)
    probability = np.array([0.90] * 25 + [0.10] * 25 + [0.60] * 25 + [0.40] * 25)
    market_valid = np.ones(100, dtype=bool)

    selected = choose_confidence_policy(y_true, probability, market_valid)

    assert selected is not None
    assert selected["bets"] == 50
    assert selected["hit_rate"] == 1.0
    assert selected["threshold"] > 0.6
