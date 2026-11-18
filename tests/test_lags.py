import numpy as np
import pandas as pd
import pytest
from atp.lags import daily_lag, daily_rolling_sum, stock_flow_matrix, LagSafetyError


def _series(dates, values, g="X"):
    return pd.DataFrame({"g": g, "date": pd.to_datetime(dates), "v": values, "granularity": "daily"})


def test_gap_does_not_misalign_lag():
    # 3 Jan is missing: positional shift(1) would wrongly give 4 Jan the value of 2 Jan
    df = _series(["2023-01-01", "2023-01-02", "2023-01-04"], [10, 20, 40])
    lag = daily_lag(df, "v", 1, "g")
    assert np.isnan(lag.iloc[2])                       # 3 Jan absent → NaN, not 20
    assert df.v.shift(1).iloc[2] == 20                 # the trap we are guarding against


def test_lag_before_series_start_is_nan():
    df = _series(pd.date_range("2023-01-01", periods=10), np.arange(10.0))
    lag = daily_lag(df, "v", 7, "g")
    assert lag.iloc[:7].isna().all() and (lag.iloc[7:].values == np.arange(3.0)).all()


def test_monthly_rows_rejected():
    df = _series(["2022-12-01", "2023-01-01"], [1000, 10]).assign(granularity=["monthly", "daily"])
    with pytest.raises(LagSafetyError):
        daily_lag(df, "v", 1, "g")


def test_duplicate_keys_rejected():
    df = _series(["2023-01-01", "2023-01-01"], [1, 2])
    with pytest.raises(LagSafetyError):
        daily_lag(df, "v", 1, "g")


def test_rolling_sum_requires_full_window():
    df = _series(pd.date_range("2023-01-01", periods=5), [1.0, 1, 1, 1, 1])
    r = daily_rolling_sum(df, "v", 3, "g")
    assert r.iloc[:2].isna().all() and (r.iloc[2:] == 3).all()


def test_stock_flow_matrix_values_and_guard():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2025-06-01", periods=100))
    t = pd.DatetimeIndex(["2025-08-01"])
    X = stock_flow_matrix(s, t, 45)
    assert X[0, 0] == s["2025-08-01"] and X[0, 1] == s["2025-07-31"] and X[0, 44] == s["2025-06-18"]
    with pytest.raises(LagSafetyError):
        stock_flow_matrix(s, pd.DatetimeIndex(["2025-06-10"]), 45)   # needs dates before the series
    s2 = s.copy(); s2["2025-07-15"] = np.nan
    with pytest.raises(LagSafetyError):
        stock_flow_matrix(s2, t, 45)
