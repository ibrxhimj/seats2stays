"""Flight data: clean row-level table, then SEPARATE daily and monthly country tables.

Key design rule (lag-contamination guard): the source file is MONTHLY for 2022 and DAILY from
2023-01-01. The daily table physically starts at `flights_daily_start`, so no daily lag or rolling
window can ever reach a monthly aggregate. Monthly rows only live in the monthly table.
"""
import numpy as np
import pandas as pd
from .io import to_num
from .ledger import Ledger

RENAME = {
    "Date": "date", "Departure Country Name": "country", "Departure City": "city",
    "Arrival City": "arrival_city", "Airline Name": "airline",
    "Average Weekly Frequency": "weekly_freq",
    "Business Class P2P Count": "p2p_business", "Business Class Seat Capacity": "seats_business",
    "Economy Class P2P Count": "p2p_economy", "Economy Class Seat Capacity": "seats_economy",
    "First Class P2P Count": "p2p_first", "First Class Seat Capacity": "seats_first",
    "Load Factor": "lf_file", "Total P2P": "p2p", "Total PAX": "pax",
    "Total PAX Excluding Infant": "pax_excl_infant", "Total Seats": "seats",
    "Total Transfer": "transfer", "Total Transit": "transit",
    "Transfer Business Class Count": "transfer_business", "Transfer Economy Class Count": "transfer_economy",
    "Transfer First Class Count": "transfer_first", "Transit Business Class Count": "transit_business",
    "Transit Economy Class Count": "transit_economy", "Transit First Count": "transit_first",
    "Destination": "destination",
}
STR_COLS = ["country", "city", "arrival_city", "airline", "destination"]
FLOW_COLS = ["seats", "pax", "pax_excl_infant", "p2p", "transfer", "transit"]


def clean_rows(raw: pd.DataFrame, cfg: dict, led: Ledger) -> pd.DataFrame:
    missing = set(RENAME) - set(raw.columns)
    led.check("flights:schema", not missing, f"missing columns: {sorted(missing)}" if missing else "26 expected columns present")
    df = raw.rename(columns=RENAME).copy()
    n0 = len(df)
    led.fact("flights_rows_raw", n0)

    # F1 dates
    df["date"] = pd.to_datetime(df["date"])
    led.check("flights:date_midnight", (df.date == df.date.dt.normalize()).all(), "all timestamps at 00:00")
    led.rule("F1", "flights_rows", "Parse `Date` to calendar date (all at 00:00:00)", n0)

    # F2 granularity
    d0 = cfg["dates"]["flights_daily_start"]
    df["granularity"] = np.where(df.date < d0, "monthly", "daily")
    mon = df[df.granularity == "monthly"]
    led.check("flights:monthly_rows_on_day1", (mon.date.dt.day == 1).all(),
              f"{len(mon)} pre-{d0.date()} rows all dated on the 1st of the month")
    led.rule("F2", "flights_rows",
             f"Tag granularity: rows before {d0.date()} are MONTHLY aggregates, rows from {d0.date()} are DAILY "
             "(data dictionary says monthly throughout — incorrect)", f"monthly {len(mon)}, daily {n0 - len(mon)}")

    # F3 numerics
    for c in df.columns:
        if c not in STR_COLS + ["date", "granularity"]:
            df[c] = to_num(df[c])
    for c in STR_COLS:
        df[c] = df[c].str.strip()
    class_cols = [c for c in df.columns if any(c.startswith(p) for p in ("p2p_", "seats_", "transfer_", "transit_"))] + ["weekly_freq"]
    na_daily = int(df.loc[df.granularity == "daily", class_cols].isna().sum().sum())
    na_monthly_share = df.loc[df.granularity == "monthly", class_cols].isna().mean().mean()
    led.rule("F3", "flights_rows",
             "Parse numerics; `*`/blank → missing. Class-level and frequency fields are missing on ALL monthly rows "
             f"({na_monthly_share:.0%}) and on {na_daily} daily cells", n0)

    # F4 flights per day from weekly frequency (daily rows only)
    dim = df.date.dt.days_in_month
    raw_n = df.weekly_freq * dim / 7
    daily = df.granularity == "daily"
    dev = (raw_n[daily] - raw_n[daily].round()).abs().max()
    led.check("flights:weekly_freq_integer_flights", dev < 1e-6,
              f"weekly_freq × days_in_month / 7 is integer on daily rows (max deviation {dev:.2e})")
    df["n_flights"] = raw_n.round().where(daily)
    df["seats_per_flight"] = (df.seats / df.n_flights).where(daily)
    led.rule("F4", "flights_rows",
             "Derive `n_flights` (flights that day) = weekly_freq × days_in_month / 7 and `seats_per_flight` "
             "(daily rows only)", int(daily.sum()))

    # F5 load factor
    lf = df.pax / df.seats
    led.check("flights:lf_identity", np.allclose(lf, df.lf_file, rtol=0, atol=1e-9), "Load Factor == PAX / Seats on every row")
    df["lf_raw"] = lf
    df["lf_capped"] = lf.clip(upper=cfg["constants"]["lf_cap"])
    led.fact("lf_gt1_rows", int((lf > 1).sum()))
    led.rule("F5", "flights_rows",
             f"Keep `lf_raw` (=PAX/Seats), add `lf_capped` (≤{cfg['constants']['lf_cap']}) and `lf_excl_infant`; "
             f"{int((lf > 1).sum())} rows have LF > 1 ({(lf > 1).mean():.1%})", int((lf > 1).sum()))

    # F6/F7 class-level P2P gap vs infants; the gap is also the evidence used to fill missing infant counts
    gap_all = df.p2p - df[["p2p_business", "p2p_economy", "p2p_first"]].sum(axis=1, min_count=3)
    miss = df.pax_excl_infant.isna()
    infer = gap_all[miss].fillna(0)
    df.loc[miss, "pax_excl_infant"] = df.loc[miss, "pax"] - infer
    df["infants"] = df.pax - df.pax_excl_infant
    df["lf_excl_infant"] = df.pax_excl_infant / df.seats
    led.rule("F7", "flights_rows", "Missing `pax_excl_infant` = PAX − inferred infants, where inferred infants = "
             "Total P2P − class P2P (equals infants on most rows; 0 if class data absent). Filled: "
             + ", ".join(f"{r.date.date()} {r.city}/{r.airline} → {int(v)} infant(s)" for (_, r), v in zip(df[miss].iterrows(), infer)),
             int(miss.sum()))

    led.fact("lf_excl_infant_gt1_rows", int((df.lf_excl_infant > 1).sum()))
    gap = gap_all[daily]
    inf = df.infants[daily]
    led.fact("p2p_class_gap_rows", int((gap > 0).sum()))
    led.check("flights:p2p_class_gap_is_infants", (gap >= 0).all() and (gap <= inf).all(),
              f"Total P2P − class P2P is ≥ 0 and ≤ infants on every daily row (equals infants exactly on {(gap == inf).mean():.1%})")
    led.rule("F6", "flights_rows",
             f"Use TOTAL columns only. Class-level P2P falls short of Total P2P by {gap[gap > 0].min():.0f}–{gap.max():.0f} pax on "
             f"{int((gap > 0).sum()):,} daily rows; the gap never exceeds the infant count (class counts exclude lap infants)",
             int((gap > 0).sum()))

    # F8 country key for joins with hotel nationality
    df["nationality_key"] = df.country.str.upper()
    led.rule("F8", "flights_rows", "Join key `nationality_key` = upper-case departure country", n0)
    led.fact("airline_unknown_rows", int((df.airline == "Unknown").sum()))

    # structural checks
    led.check("flights:pax_identity", (df.pax == df.p2p + df.transfer + df.transit).all(), "PAX == P2P + Transfer + Transit on every row")
    sc = df.loc[daily, ["seats_business", "seats_economy", "seats_first"]].sum(axis=1)
    led.check("flights:seat_class_identity", (sc == df.loc[daily, "seats"]).all(), "Seats == sum of class seats on daily rows")
    led.check("flights:unique_key", not df.duplicated(["date", "country", "city", "airline"]).any(), "unique (date, country, city, airline)")
    led.check("flights:arrival_constant", (df.arrival_city.eq("Abu Dhabi") & df.destination.eq("AUH")).all(), "all rows arrive at Abu Dhabi / AUH")
    led.check("flights:non_negative", (df[FLOW_COLS] >= 0).all().all() and (df.seats > 0).all(), "flows ≥ 0 and seats > 0")
    return df.drop(columns=["lf_file", "arrival_city", "destination"])


def day_quality(rows: pd.DataFrame, cfg: dict, led: Ledger) -> pd.DataFrame:
    d = rows[rows.granularity == "daily"].groupby("date").size().rename("n_rows").to_frame()
    d = d.reindex(pd.date_range(cfg["dates"]["flights_daily_start"], cfg["dates"]["test_end"]), fill_value=0)
    d.index.name = "date"
    d["trailing_median"] = d.n_rows.shift(1).rolling(28, min_periods=14).median()
    d["ratio"] = d.n_rows / d.trailing_median
    d["is_low_coverage"] = d.ratio < cfg["constants"]["partial_day_ratio"]
    fmt = lambda x: ", ".join(f"{i.date()} ({r.n_rows} rows, {r.ratio:.0%} of normal)" for i, r in x.iterrows()) or "none"
    led.rule("F9", "flights_day_quality",
             f"Flag days with route-row count < {cfg['constants']['partial_day_ratio']:.0%} of the trailing 28-day median "
             f"(`is_low_coverage`): {fmt(d[d.is_low_coverage])}. Both are real disruptions, not extraction errors: hotel "
             "check-ins ROSE on both days (stranded passengers) — see the external-events audit",
             int(d.is_low_coverage.sum()))
    return d.reset_index()


def country_daily(rows: pd.DataFrame, dq: pd.DataFrame, cfg: dict, led: Ledger) -> pd.DataFrame:
    d0, d1 = cfg["dates"]["flights_daily_start"], cfg["dates"]["test_end"]
    x = rows[rows.granularity == "daily"]
    agg = x.groupby(["nationality_key", "date"]).agg(
        **{c: (c, "sum") for c in FLOW_COLS}, n_flights=("n_flights", "sum"), n_route_rows=("city", "size")).reset_index()
    grid = pd.MultiIndex.from_product([sorted(x.nationality_key.unique()), pd.date_range(d0, d1)], names=["nationality_key", "date"])
    out = agg.set_index(["nationality_key", "date"]).reindex(grid, fill_value=0).reset_index()
    out = out.merge(dq[["date", "is_low_coverage"]], on="date", how="left")
    out["lf"] = (out.pax / out.seats).where(out.seats > 0)
    out["granularity"] = "daily"
    # trailing 7-day averages (days t-6..t): only information available up to day t; NaN until a full window
    # exists, so 1–6 Jan 2023 are empty rather than borrowing from the 2022 monthly table
    from .lags import daily_rolling_sum
    for c in ("seats", "p2p"):
        out[f"{c}_7d"] = daily_rolling_sum(out, c, 7, "nationality_key") / 7
    led.rule("F12", "flights_country_daily", "Trailing 7-day average seats and P2P (`seats_7d`, `p2p_7d`, days t−6…t) for "
             "Stage A: daily seats are zero on 3.7% of country-days (non-daily routes), the 7-day average almost never. "
             "Trailing, not centred, so a day never sees later cancellations; empty until a full window (first 7 Jan 2023)",
             int(out.seats_7d.notna().sum()))
    led.rule("F10", "flights_country_daily",
             f"Aggregate DAILY rows to country × day on a complete grid {d0.date()}…{d1.date()}; a missing "
             "country-day means no scheduled service → 0 seats (not missing)", int((out.seats == 0).sum()))
    led.check("flights_daily:starts_at_daily_start", out.date.min() == d0, f"daily table starts {out.date.min().date()} (no monthly contamination)")
    return out


def country_monthly(rows: pd.DataFrame, dq: pd.DataFrame, cfg: dict, led: Ledger) -> pd.DataFrame:
    x = rows.assign(month=rows.date.dt.to_period("M").dt.to_timestamp())
    agg = x.groupby(["nationality_key", "month", "granularity"]).agg(
        **{c: (c, "sum") for c in FLOW_COLS}, n_flights=("n_flights", lambda s: s.sum(min_count=1)),
        n_days_with_service=("date", "nunique")).reset_index().rename(columns={"granularity": "source_granularity"})
    agg.loc[agg.source_granularity == "monthly", "n_days_with_service"] = np.nan
    months = pd.period_range(rows.date.min(), cfg["dates"]["test_end"], freq="M").to_timestamp()
    grid = pd.MultiIndex.from_product([sorted(rows.nationality_key.unique()), months], names=["nationality_key", "month"])
    out = agg.set_index(["nationality_key", "month"]).reindex(grid).reset_index()
    out["source_granularity"] = np.where(out.month < cfg["dates"]["flights_daily_start"], "monthly", "daily")
    out[FLOW_COLS] = out[FLOW_COLS].fillna(0)
    out["lf"] = (out.pax / out.seats).where(out.seats > 0)
    low = dq.loc[dq.is_low_coverage, "date"].dt.to_period("M").dt.to_timestamp()
    out["contains_low_coverage_day"] = out.month.isin(set(low))
    led.rule("F11", "flights_country_monthly",
             "Country × month table: 2022 from native monthly rows, 2023+ summed from daily rows; "
             "`source_granularity` records which. Months with no service → 0", len(out))
    return out
