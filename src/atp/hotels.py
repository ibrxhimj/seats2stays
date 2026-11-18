"""Hotel guest data → complete daily grids (international by nationality, domestic) + monthly rollup.

Suppression handling (deliberately simple — no censored-likelihood machinery):
  * TRAIN rows are dropped from the file when Guests < 10. A blank arrivals value in a PRESENT train
    row means zero: the train files contain no literal 0 anywhere, yet many values 1–9 — so small
    values are not hidden, and '*' is the dictionary's "zero".
  * TEST rows are dropped when New Arrivals < 10 (test minimum present value is exactly 10).
  * Every hidden value is filled with ONE constant (median of observed train arrivals in 1–9),
    flagged in `arrivals_status`, and hidden guest counts are never used as targets.
  The bound on the guests involved is reported in the data card (< 0.01% of guests).
"""
import numpy as np
import pandas as pd
from .io import to_num
from .ledger import Ledger

RENAME = {"Date": "date", "Guests": "guests", "New Arrivals": "arrivals_raw",
          "Same-Day Guests": "same_day_raw", "Nationality": "nationality", "Residence (groups)": "residence"}


def _prep(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    df = raw.rename(columns=RENAME).copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("guests", "arrivals_raw", "same_day_raw"):
        if c in df:
            df[c] = to_num(df[c])
    df["split"] = split
    return df


def build_international(train_raw, test_raw, cfg: dict, led: Ledger) -> tuple[pd.DataFrame, float]:
    tr, te = _prep(train_raw, "train"), _prep(test_raw, "test")
    led.check("intl:residence", tr.residence.eq("International").all() and te.residence.eq("International").all(), "Residence == International")
    led.check("intl:test_has_no_guests", "guests" not in te, "test file carries no Guests column (target withheld)")
    led.check("intl:unique_key", not tr.duplicated(["date", "nationality"]).any() and not te.duplicated(["date", "nationality"]).any(), "unique (date, nationality)")
    led.check("intl:same_nationalities", set(tr.nationality) == set(te.nationality), f"{tr.nationality.nunique()} nationalities in both splits")
    thr = cfg["constants"]["suppression_threshold"]
    led.check("intl:train_guest_floor", tr.guests.min() >= thr, f"min train guests = {tr.guests.min():.0f} (rows < {thr} absent)")
    led.check("intl:test_arrival_floor", te.arrivals_raw.min() >= thr, f"min test arrivals = {te.arrivals_raw.min():.0f} (rows < {thr} absent)")
    led.check("intl:train_no_literal_zero_arrivals", (tr.arrivals_raw != 0).all(), "no literal 0 arrivals in train → '*' means zero")
    led.check("intl:arrivals_le_guests", (tr.arrivals_raw.dropna() <= tr.guests[tr.arrivals_raw.notna()]).mean() > 0.99,
              f"arrivals ≤ guests on {(tr.arrivals_raw.dropna() <= tr.guests[tr.arrivals_raw.notna()]).mean():.2%} of train rows", "warn")

    small = tr.arrivals_raw[(tr.arrivals_raw >= 1) & (tr.arrivals_raw <= thr - 1)]
    fill = float(small.median())
    led.fact("hidden_arrivals_fill_value", fill)
    led.fact("train_arrivals_1_9_rows", int(len(small)))

    nats = sorted(tr.nationality.unique())
    dates = pd.date_range(cfg["dates"]["hotel_start"], cfg["dates"]["test_end"])
    grid = pd.MultiIndex.from_product([nats, dates], names=["nationality", "date"]).to_frame(index=False)
    both = pd.concat([tr, te], ignore_index=True).drop(columns="residence")
    df = grid.merge(both, on=["nationality", "date"], how="left")
    df["split"] = np.where(df.date <= cfg["dates"]["hotel_train_end"], "train", "test")
    # a row is "present" if it appeared in the source file
    key = both.set_index(["nationality", "date"]).index
    df["row_present"] = df.set_index(["nationality", "date"]).index.isin(key)

    status = np.select(
        [df.row_present & df.arrivals_raw.notna(),
         df.row_present & df.arrivals_raw.isna() & (df.split == "train"),
         df.row_present & df.arrivals_raw.isna() & (df.split == "test"),
         ~df.row_present & (df.split == "train"),
         ~df.row_present & (df.split == "test")],
        ["observed", "star_zero_train", "star_hidden_test", "absent_train_lt10_guests", "absent_test_lt10_arrivals"],
        default="?")
    df["arrivals_status"] = status
    df["arrivals"] = np.select(
        [status == "observed", status == "star_zero_train"],
        [df.arrivals_raw, 0.0], default=fill)
    df["is_target"] = (df.split == "train") & df.guests.notna()
    df["same_day_status"] = np.where(df.same_day_raw.notna(), "observed", "star_or_absent")

    counts = pd.Series(status).value_counts().to_dict()
    for k, v in counts.items():
        led.fact(f"arrivals_status:{k}", int(v))
    led.check("intl:status_complete", "?" not in counts, "every grid cell has an arrivals status")
    led.rule("H1", "hotel_intl_daily", f"Complete nationality × day grid {dates[0].date()}…{dates[-1].date()} "
             f"({len(nats)} × {len(dates)} = {len(df)} cells)", len(df))
    led.rule("H2", "hotel_intl_daily", "Blank arrivals in a PRESENT train row → 0 (train contains no literal zeros "
             f"but {len(small)} values in 1–9, so '*' = zero)", counts.get("star_zero_train", 0))
    led.rule("H3", "hotel_intl_daily", f"Hidden small values (absent rows, blank test arrivals) → arrivals = {fill:g} "
             "(median of observed train arrivals in 1–9); flagged in `arrivals_status`",
             sum(v for k, v in counts.items() if k not in ("observed", "star_zero_train")))
    led.rule("H4", "hotel_intl_daily", "Target mask `is_target` = train split with observed Guests; absent rows are "
             "never targets (no imputed guests)", int(df.is_target.sum()))
    led.rule("H5", "hotel_intl_daily", "Same-Day Guests kept raw (not a model input); '*' kept missing", int(df.same_day_raw.isna().sum()))

    # bound on what suppression can move
    hidden = df.arrivals_status.isin(["absent_train_lt10_guests"]).sum()
    led.fact("suppressed_guest_share_upper_bound", float(9 * hidden / df.guests.sum()))
    hidden_test = df.arrivals_status.isin(["absent_test_lt10_arrivals", "star_hidden_test"]).sum()
    led.fact("test_hidden_arrivals_share_upper_bound", float(9 * hidden_test / df.loc[df.split == "test", "arrivals"].sum()))
    return df[["nationality", "date", "split", "guests", "arrivals", "arrivals_raw", "arrivals_status",
               "is_target", "same_day_raw", "same_day_status", "row_present"]], fill


def build_domestic(train_raw, test_raw, cfg: dict, led: Ledger) -> pd.DataFrame:
    tr, te = _prep(train_raw, "train"), _prep(test_raw, "test")
    led.check("dom:residence", tr.residence.eq("Domestic").all() and te.residence.eq("Domestic").all(), "Residence == Domestic")
    df = pd.concat([tr, te], ignore_index=True).drop(columns="residence").sort_values("date")
    dates = pd.date_range(cfg["dates"]["hotel_start"], cfg["dates"]["test_end"])
    led.check("dom:complete_dates", df.date.tolist() == list(dates), f"one row per day {dates[0].date()}…{dates[-1].date()}")
    led.check("dom:no_missing", df[["arrivals_raw", "same_day_raw"]].notna().all().all() and df.loc[df.split == "train", "guests"].notna().all(),
              "no missing arrivals / same-day / train guests")
    df["arrivals"] = df.arrivals_raw
    df["is_target"] = (df.split == "train") & df.guests.notna()
    df["nationality"] = "DOMESTIC"
    led.rule("H6", "hotel_dom_daily", "Domestic daily series: complete, no imputation needed", len(df))
    return df[["nationality", "date", "split", "guests", "arrivals", "arrivals_raw", "is_target", "same_day_raw"]]


def monthly(intl: pd.DataFrame, dom: pd.DataFrame, led: Ledger) -> pd.DataFrame:
    x = pd.concat([intl.assign(market="international"), dom.assign(market="domestic")], ignore_index=True)
    x["month"] = x.date.dt.to_period("M").dt.to_timestamp()
    g = x.groupby(["market", "nationality", "month"]).agg(
        arrivals=("arrivals", "sum"), guest_nights=("guests", "sum"),
        n_days=("date", "size"), n_days_guests_observed=("is_target", "sum"),
        n_days_arrivals_imputed=("arrivals_status", lambda s: s.isin(
            ["absent_train_lt10_guests", "absent_test_lt10_arrivals", "star_hidden_test"]).sum() if s.notna().any() else 0)).reset_index()
    g.loc[g.n_days_guests_observed == 0, "guest_nights"] = np.nan
    g["guest_nights_complete"] = g.n_days_guests_observed == g.n_days
    led.rule("H7", "hotel_monthly", "Monthly rollup: arrivals, guest nights (= sum of daily in-house guests); "
             "`guest_nights_complete` false if any day's guests suppressed or in test", len(g))
    return g
