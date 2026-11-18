from __future__ import annotations
from collections.abc import Iterator
import pandas as pd

def _validated_frame(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if date_column not in df.columns:
        raise KeyError(f"Date column not found: {date_column}")
    if df.empty:
        raise ValueError("DataFrame is empty.")
    result = df.copy(deep=True)
    parsed = pd.to_datetime(result[date_column], errors="coerce", utc=True)
    invalid = parsed.isna()
    if invalid.any():
        rows = result.index[invalid].tolist()[:10]
        raise ValueError(f"{int(invalid.sum())} invalid/missing dates in '{date_column}'; sample indexes: {rows}")
    result[date_column] = parsed
    return result.sort_values(date_column, kind="stable").reset_index(drop=True)

class TemporalSplitter:
    """Rows at cutoff are assigned to test; train is strictly earlier."""
    def __init__(self, cutoff_date, date_column="match_date"):
        cutoff = pd.to_datetime(cutoff_date, errors="coerce", utc=True)
        if pd.isna(cutoff):
            raise ValueError("cutoff_date is not a valid date.")
        self.cutoff_date = cutoff
        self.date_column = date_column

    def split_data(self, df):
        data = _validated_frame(df, self.date_column)
        train = data.loc[data[self.date_column] < self.cutoff_date].copy()
        test = data.loc[data[self.date_column] >= self.cutoff_date].copy()
        if train.empty or test.empty:
            raise ValueError("Temporal split produced an empty train or test set.")
        if train[self.date_column].max() >= test[self.date_column].min():
            raise RuntimeError("Temporal ordering invariant violated.")
        return train.reset_index(drop=True), test.reset_index(drop=True)

def rolling_origin_split(df, date_column="match_date", window_days=365,
                         test_days=90, step_days=90, expanding_window=True) -> Iterator:
    """Walk-forward folds; test_days and step_days are independent."""
    for name, value in (("window_days", window_days), ("test_days", test_days), ("step_days", step_days)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    data = _validated_frame(df, date_column)
    first, last = data[date_column].min(), data[date_column].max()
    start = first + pd.Timedelta(days=window_days)
    while start <= last:
        end = start + pd.Timedelta(days=test_days)
        train_start = first if expanding_window else start - pd.Timedelta(days=window_days)
        train = data.loc[(data[date_column] >= train_start) & (data[date_column] < start)].copy()
        test = data.loc[(data[date_column] >= start) & (data[date_column] < end)].copy()
        if not train.empty and not test.empty:
            if train[date_column].max() >= test[date_column].min():
                raise RuntimeError("Walk-forward temporal ordering invariant violated.")
            yield train.reset_index(drop=True), test.reset_index(drop=True)
        start += pd.Timedelta(days=step_days)
