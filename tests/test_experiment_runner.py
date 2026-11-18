import pandas as pd

from experiments.comparison import purge_same_calendar_day_boundaries
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
