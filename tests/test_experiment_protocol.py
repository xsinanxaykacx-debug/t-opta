"""Contract tests for the sealed chronological experiment protocol."""

import pytest

from validation.experiment_protocol import (
    EXPECTED_SPLIT_SIZES,
    EXPECTED_TOTAL_ROWS,
    PARTITION_ORDER,
    ProtocolError,
    assert_final_test_sealed,
    validate_chronological_order,
    validate_partition_dates,
    validate_split_indices,
)


def test_approved_partition_sizes_sum_to_dataset_rows():
    assert tuple(EXPECTED_SPLIT_SIZES) == PARTITION_ORDER
    assert EXPECTED_SPLIT_SIZES == {
        "train": 8500,
        "calibration": 900,
        "policy_select": 300,
        "policy_check": 300,
        "final_test": 5734,
    }
    assert EXPECTED_TOTAL_ROWS == 15734


def test_exact_contiguous_partition_indices_pass():
    cuts = (8500, 9400, 9700, 10000, 15734)
    start = 0
    split_indices = {}
    for name, end in zip(PARTITION_ORDER, cuts):
        split_indices[name] = list(range(start, end))
        start = end
    validate_split_indices(split_indices)


def test_partition_overlap_is_rejected():
    split_indices = {
        "train": list(range(8500)),
        "calibration": list(range(8499, 9399)),
        "policy_select": list(range(9399, 9699)),
        "policy_check": list(range(9699, 9999)),
        "final_test": list(range(9999, 15733)),
    }
    with pytest.raises(ProtocolError, match="overlaps"):
        validate_split_indices(split_indices)


def test_wrong_partition_size_is_rejected():
    split_indices = {
        "train": list(range(8499)),
        "calibration": list(range(8499, 9399)),
        "policy_select": list(range(9399, 9699)),
        "policy_check": list(range(9699, 9999)),
        "final_test": list(range(9999, 15733)),
    }
    with pytest.raises(ProtocolError, match="expected"):
        validate_split_indices(split_indices)


def test_chronological_order_is_checked_without_silent_sorting():
    validate_chronological_order(["2024-01-01", "2024-01-01", "2024-01-02"])
    with pytest.raises(ProtocolError, match="not chronological"):
        validate_chronological_order(["2024-01-02", "2024-01-01"])


def test_partition_dates_allow_equal_boundary_timestamp_but_not_regression():
    dates = {
        "train": ["2024-01-01", "2024-01-02"],
        "calibration": ["2024-01-02", "2024-01-03"],
        "policy_select": ["2024-01-03"],
        "policy_check": ["2024-01-04"],
        "final_test": ["2024-01-05"],
    }
    validate_partition_dates(dates)
    dates["policy_check"] = ["2024-01-02"]
    with pytest.raises(ProtocolError, match="starts before"):
        validate_partition_dates(dates)


def test_final_test_is_sealed_case_insensitively():
    assert_final_test_sealed(["train", "calibration", "policy_select", "policy_check"])
    with pytest.raises(ProtocolError, match="FINAL_TEST is sealed"):
        assert_final_test_sealed(["TRAIN", "FINAL_TEST"])
