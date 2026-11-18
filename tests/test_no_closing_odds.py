"""Regression tests for the pre-match feature leakage boundary."""

import pytest

from validation.experiment_protocol import (
    ProtocolError,
    validate_feature_columns,
)


@pytest.mark.parametrize(
    "column",
    [
        "AvgC",
        "B365C",
        "MaxC",
        "PC",
        "odds_AvgC>2.5",
        "odds_AvgC<2.5",
        "odds_B365C>2.5",
        "odds_B365C<2.5",
        "odds_MaxC>2.5",
        "odds_MaxC<2.5",
        "odds_PC>2.5",
        "odds_PC<2.5",
    ],
)
def test_closing_odds_are_rejected(column):
    with pytest.raises(ProtocolError, match="closing"):
        validate_feature_columns([column])


@pytest.mark.parametrize(
    "column",
    [
        "FTHG",
        "FTAG",
        "FTR",
        "HT_Home_Goals",
        "Home_xG",
        "Away_xG",
        "Home_Shots",
        "Attendance",
        "odds_FTR",
    ],
)
def test_post_match_fields_are_rejected(column):
    with pytest.raises(ProtocolError):
        validate_feature_columns([column])


@pytest.mark.parametrize(
    "column",
    [
        "odds_Avg>2.5",
        "odds_Avg<2.5",
        "odds_B365>2.5",
        "odds_B365<2.5",
        "odds_Max>2.5",
        "odds_Max<2.5",
        "odds_P>2.5",
        "odds_P<2.5",
        "odds_BFE>2.5",
        "odds_BFE<2.5",
        "hist_home_xg_for",
        "rolling_away_xg_against",
        "form_over_rate",
        "pre_match_home_strength",
    ],
)
def test_verified_pre_match_features_are_allowed(column):
    assert validate_feature_columns([column]) == (column,)


def test_unreviewed_odds_columns_are_not_allowed():
    with pytest.raises(ProtocolError, match="not on the pre-match allowlist"):
        validate_feature_columns(["odds_B365H"])


def test_denylist_wins_even_when_feature_is_explicitly_approved():
    with pytest.raises(ProtocolError):
        validate_feature_columns(
            ["odds_AvgC>2.5"],
            approved_features=["odds_AvgC>2.5"],
        )
