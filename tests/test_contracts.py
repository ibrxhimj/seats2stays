"""Contracts on the processed tables (run after `make data`)."""
import pandas as pd
import pytest
from atp.config import load_config

cfg = load_config()
P = cfg["paths"]["processed"]


def load(name):
    p = P / f"{name}.parquet"
    if not p.exists():
        pytest.skip("run `make data` first")
    return pd.read_parquet(p)


def test_flights_daily_never_before_daily_start():
    fd = load("flights_country_daily")
    assert fd.date.min() == cfg["dates"]["flights_daily_start"] and (fd.granularity == "daily").all()
    assert not fd.duplicated(["nationality_key", "date"]).any()


def test_monthly_table_labels_granularity():
    fm = load("flights_country_monthly")
    assert set(fm.loc[fm.month < cfg["dates"]["flights_daily_start"], "source_granularity"]) == {"monthly"}
    assert set(fm.loc[fm.month >= cfg["dates"]["flights_daily_start"], "source_granularity"]) == {"daily"}


def test_hotel_grid_complete_and_targets_only_observed():
    hi = load("hotel_intl_daily")
    assert len(hi) == hi.nationality.nunique() * hi.date.nunique()
    assert hi.arrivals.notna().all()
    assert hi.loc[hi.is_target, "guests"].notna().all() and not hi.loc[hi.split == "test", "is_target"].any()


def test_calendar_single_source():
    cal = load("calendar")
    assert cal.date.is_unique and (cal.ramadan_day > 0).eq(cal.is_ramadan).all()


def test_trailing_7day_seats_are_lag_safe():
    fd = load("flights_country_daily").sort_values(["nationality_key", "date"])
    first_full = cfg["dates"]["flights_daily_start"] + pd.Timedelta(days=6)
    assert fd.loc[fd.date < first_full, "seats_7d"].isna().all()          # no borrowing from 2022 monthly data
    assert fd.loc[fd.date >= first_full, "seats_7d"].notna().all()
    # equals the plain trailing mean of days t-6..t, i.e. never uses later days
    manual = fd.groupby("nationality_key").seats.transform(lambda s: s.rolling(7).mean())
    ok = fd.date >= first_full
    assert (fd.loc[ok, "seats_7d"] - manual[ok]).abs().max() < 1e-9


def test_country_map_label():
    cm = load("country_map")
    assert "has_flight_data" in cm and "has_direct_route" not in cm
    assert cm.has_flight_data.sum() == 33
