"""Lag-safe helpers. Every lag is computed by DATE ARITHMETIC (value at date − k days), never by
row position (`.shift(k)`), so:
  * gaps or unsorted rows cannot silently misalign a lag;
  * a daily lag can never reach a monthly aggregate — tables containing monthly rows are rejected,
    and lags before a series' first date are NaN, not borrowed from another granularity;
  * the stock-flow lag matrix refuses to build if any lag needed by a forecast date is missing
    (guards the 1 Aug 2025 train/test boundary).
"""
import numpy as np
import pandas as pd


class LagSafetyError(ValueError):
    pass


def _assert_daily(df: pd.DataFrame):
    for col in ("granularity", "source_granularity"):
        if col in df and (df[col] != "daily").any():
            raise LagSafetyError(f"table contains non-daily rows in `{col}` — daily lags are not allowed on it")


def daily_lag(df: pd.DataFrame, value_col: str, k: int, group_col: str | None = None,
              date_col: str = "date") -> pd.Series:
    """Value of `value_col` at (date − k days) for the same group; NaN if that date is not in the table."""
    _assert_daily(df)
    keys = ([group_col] if group_col else []) + [date_col]
    if df.duplicated(keys).any():
        raise LagSafetyError("duplicate (group, date) keys — lag would be ambiguous")
    src = df[keys + [value_col]].copy()
    src[date_col] = src[date_col] + pd.Timedelta(days=k)
    out = df[keys].merge(src, on=keys, how="left")[value_col]
    out.index = df.index
    return out


def daily_rolling_sum(df: pd.DataFrame, value_col: str, window: int, group_col: str | None = None,
                      date_col: str = "date") -> pd.Series:
    """Sum over the `window` days ending at each date; NaN unless all `window` days exist in the table."""
    total = sum(daily_lag(df, value_col, k, group_col, date_col) for k in range(window))
    return total  # NaN propagates if any day in the window is absent


def stock_flow_matrix(arrivals: pd.Series, target_dates: pd.DatetimeIndex, K: int) -> np.ndarray:
    """T × K matrix X[t, k] = arrivals on (t − k). `arrivals` must be a daily, date-indexed series.
    Raises LagSafetyError if any required (t − k) date is missing or NaN."""
    if not isinstance(arrivals.index, pd.DatetimeIndex):
        raise LagSafetyError("arrivals must be indexed by date")
    if arrivals.index.duplicated().any():
        raise LagSafetyError("duplicate dates in arrivals")
    need = pd.DatetimeIndex(np.unique(np.concatenate(
        [(target_dates - pd.Timedelta(days=k)).values for k in range(K)])))
    missing = need.difference(arrivals.index)
    if len(missing):
        raise LagSafetyError(f"{len(missing)} lag dates missing (first {missing.min().date()}); "
                             f"forecasts from {target_dates.min().date()} need arrivals back to {need.min().date()}")
    a = arrivals.reindex(need)
    if a.isna().any():
        raise LagSafetyError(f"{int(a.isna().sum())} lag dates have NaN arrivals (first {a[a.isna()].index.min().date()})")
    return np.column_stack([arrivals.reindex(target_dates - pd.Timedelta(days=k)).to_numpy() for k in range(K)])
