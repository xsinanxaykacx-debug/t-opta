import pandas as pd
import pytest
from validation.temporal_split import TemporalSplitter, rolling_origin_split

def sample_frame():
    return pd.DataFrame({"match_date": pd.date_range("2024-01-01", periods=12, freq="30D"), "value": range(12)})

def test_cutoff_rows_go_to_test_and_order_is_strict():
    train, test = TemporalSplitter("2024-07-01").split_data(sample_frame())
    assert train.match_date.max() < pd.Timestamp("2024-07-01", tz="UTC")
    assert test.match_date.min() >= pd.Timestamp("2024-07-01", tz="UTC")

def test_input_dataframe_is_not_mutated():
    frame = sample_frame(); original = frame.copy(deep=True)
    TemporalSplitter("2024-07-01").split_data(frame)
    pd.testing.assert_frame_equal(frame, original)

def test_invalid_date_is_rejected_not_dropped():
    frame = sample_frame(); frame.loc[2, "match_date"] = "bad-date"
    with pytest.raises(ValueError, match="invalid/missing dates"):
        TemporalSplitter("2024-07-01").split_data(frame)

def test_missing_date_column_is_rejected_even_if_empty():
    with pytest.raises(KeyError, match="Date column"):
        TemporalSplitter("2024-07-01").split_data(pd.DataFrame())

def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        TemporalSplitter("2024-07-01").split_data(pd.DataFrame(columns=["match_date"]))

def test_invalid_cutoff_is_rejected():
    with pytest.raises(ValueError, match="cutoff_date"):
        TemporalSplitter("bad-date")

def test_single_partition_is_rejected():
    with pytest.raises(ValueError, match="empty train or test"):
        TemporalSplitter("2030-01-01").split_data(sample_frame())

def test_walk_forward_has_no_temporal_overlap():
    folds = list(rolling_origin_split(sample_frame(), window_days=90, test_days=60, step_days=30))
    assert folds
    for train, test in folds:
        assert train.match_date.max() < test.match_date.min()

def test_fixed_window_is_supported():
    folds = list(rolling_origin_split(sample_frame(), window_days=90, test_days=60, step_days=60, expanding_window=False))
    assert folds
    assert (folds[0][0].match_date.max() - folds[0][0].match_date.min()).days <= 90

@pytest.mark.parametrize("kwargs", [{"window_days":0},{"test_days":-1},{"step_days":0},{"window_days":True}])
def test_invalid_window_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        list(rolling_origin_split(sample_frame(), **kwargs))
