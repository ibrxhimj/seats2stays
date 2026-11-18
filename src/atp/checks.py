"""Cross-table checks. Severity 'error' fails the build; 'warn' is reported in the data card."""
import numpy as np
import pandas as pd
from .flights import FLOW_COLS
from .lags import daily_lag, stock_flow_matrix, LagSafetyError
from .ledger import Ledger


def run_all(T: dict, raw_totals: dict, cfg: dict, led: Ledger):
    rows, fd, fm = T["flights_rows"], T["flights_country_daily"], T["flights_country_monthly"]
    hi, hd, cal = T["hotel_intl_daily"], T["hotel_dom_daily"], T["calendar"]

    # --- reconciliation with raw totals (nothing lost or double counted)
    for c in FLOW_COLS:
        ok = np.isclose(fm[c].sum(), raw_totals[c]) and np.isclose(fd[c].sum(), rows.loc[rows.granularity == "daily", c].sum())
        led.check(f"reconcile:flights:{c}", ok, f"monthly table total {fm[c].sum():,.0f} == raw {raw_totals[c]:,.0f}; daily table == raw daily rows")
    dm = fd.assign(month=fd.date.dt.to_period("M").dt.to_timestamp()).groupby(["nationality_key", "month"])[FLOW_COLS].sum()
    mm = fm[fm.source_granularity == "daily"].set_index(["nationality_key", "month"])[FLOW_COLS]
    led.check("reconcile:monthly_vs_daily", np.allclose(dm.reindex(mm.index).values, mm.values),
              "2023+ monthly table == daily table summed by month, every country")
    led.check("reconcile:intl_guests", np.isclose(hi.guests.sum(), raw_totals["intl_guests"]), f"intl train guests {hi.guests.sum():,.0f} == raw")
    led.check("reconcile:intl_arrivals_observed", np.isclose(hi.arrivals_raw.sum(), raw_totals["intl_arrivals"]), "observed intl arrivals == raw (train+test)")
    led.check("reconcile:dom_guests", np.isclose(hd.guests.sum(), raw_totals["dom_guests"]), "domestic train guests == raw")

    # --- granularity firewall
    lag = daily_lag(fd, "seats", 7, "nationality_key")
    early = fd.date < cfg["dates"]["flights_daily_start"] + pd.Timedelta(days=7)
    led.check("lag:no_monthly_contamination", lag[early].isna().all() and lag[~early].notna().all(),
              "7-day lag of daily seats is NaN for 1–7 Jan 2023 (never a 2022 monthly value) and complete afterwards")
    try:
        daily_lag(fm, "seats", 1, "nationality_key")
        led.check("lag:monthly_table_rejected", False, "daily_lag accepted a table with monthly rows")
    except LagSafetyError:
        led.check("lag:monthly_table_rejected", True, "daily_lag refuses the monthly table")

    # --- stock-flow lag window: every market, every fold, plus the official test window
    K = cfg["constants"]["max_stay_lag_days"]
    windows = {f"{k}:holdout": (v["holdout_start"], v["holdout_end"]) for k, v in cfg["folds"].items()}
    windows.update({f"{k}:train": (v["train_start"], v["train_end"]) for k, v in cfg["folds"].items()})
    series = {n: g.set_index("date").arrivals for n, g in hi.groupby("nationality")}
    series["DOMESTIC"] = hd.set_index("date").arrivals
    bad = []
    for name, (a, b) in windows.items():
        for n, s in series.items():
            try:
                stock_flow_matrix(s, pd.date_range(a, b), K)
            except LagSafetyError as e:
                bad.append(f"{name}/{n}: {e}")
    led.check("lag:stock_flow_windows_complete", not bad,
              f"{K}-day arrival lags fully populated for {len(series)} markets × {len(windows)} windows"
              + ("" if not bad else f" — {len(bad)} failures, e.g. {bad[0]}"))
    s = series["INDIA"]
    X = stock_flow_matrix(s, pd.DatetimeIndex([cfg["dates"]["test_start"]]), K)
    led.check("lag:boundary_uses_train_arrivals", X[0, 1] == s[cfg["dates"]["test_start"] - pd.Timedelta(days=1)],
              "on 1 Aug 2025 the 1-day lag equals 31 Jul 2025 train arrivals (continuous series across the split)")

    # --- coverage vs official annual totals (external validation)
    ot = pd.read_csv(cfg["paths"]["reference"] / "official_totals.csv").pivot(index="year", columns="metric", values="value")
    fm_y = fm.assign(y=fm.month.dt.year)
    cov, alos_rows = [], []
    for y in ot.index:
        i_, d_ = hi[hi.date.dt.year == y], hd[hd.date.dt.year == y]
        arr = i_.arrivals.sum() + d_.arrivals.sum()
        mov = ot.loc[y].get("abu_dhabi_airports_movements")
        pax = fm_y[fm_y.y == y].pax.sum()
        cov.append(f"{y}: hotel check-ins {arr / 1e6:.2f}M = {arr / ot.loc[y, 'dct_hotel_guests']:.0%} of DCT hotel guests; "
                   f"arriving PAX (33 countries) = {pax / (mov / 2):.0%} of Abu Dhabi airports arrivals (movements ÷ 2)")
        months = 12 if d_.date.max() <= cfg["dates"]["hotel_train_end"] else 7   # full year only if guests observed all year
        m = lambda df: df[df.date.dt.month <= months]
        ours = (m(i_).guests.sum() + m(d_).guests.sum()) / (m(i_).arrivals.sum() + m(d_).arrivals.sum())
        alos_rows.append((y, ours, ot.loc[y, "dct_alos_nights"], months))
    led.fact("coverage_vs_official", cov)
    led.fact("alos_vs_official", [f"{y}{'' if mo == 12 else ' (Jan–Jul)'}: ours {o:.2f} vs DCT {d:.1f}" for y, o, d, mo in alos_rows])
    match = [r for r in alos_rows if r[0] in (2023, 2025)]
    led.check("external:alos_matches_dct", all(abs(o - d) <= 0.1 for _, o, d, _ in match),
              "guests ÷ check-ins reproduces DCT average length of stay (2023 and 2025 within 0.1 night) — supports "
              "reading Guests as in-house guests per night; " + "; ".join(led.facts["alos_vs_official"]), "warn")
    g = hi[hi.date.dt.year.isin([2024, 2025])].groupby([hi.date.dt.year, "nationality"]).arrivals.sum().unstack(0)
    growth = g[2025].sum() / g[2024].sum() - 1
    top3 = set((g[2025] - g[2024]).sort_values(ascending=False).index[:3])
    led.check("external:2025_intl_growth", abs(growth - ot.loc[2025, "dct_intl_guest_growth"]) < 0.03 and
              top3 == {"INDIA", "RUSSIAN FEDERATION", "UNITED KINGDOM"},
              f"2025 international check-ins {growth:+.1%} vs DCT +10%; top-3 growth markets {sorted(x.title() for x in top3)} "
              "match DCT (India, Russia, UK)", "warn")
    tr = fm[(fm.month >= "2023-11-01") & (fm.month <= "2023-12-01")]
    share = tr.transfer.sum() / tr.pax.sum()
    led.check("external:terminal_a_transfer_share", abs(share - 0.45) < 0.03,
              f"Nov–Dec 2023 transfer share {share:.1%} vs official ≈45% (Terminal A first 60 days)", "warn")

    # --- calendar coverage and folds
    led.check("calendar:covers_data", cal.date.min() <= hi.date.min() - pd.Timedelta(days=K) and cal.date.max() >= cfg["dates"]["test_end"],
              "calendar spans all data dates plus the lag burn-in")
    ram_test = cal[(cal.date >= cfg["dates"]["test_start"]) & (cal.date <= cfg["dates"]["test_end"]) & cal.is_ramadan]
    led.fact("ramadan_days_in_test_window", len(ram_test))
    led.fact("ramadan_start_in_test_window", str(ram_test.date.min().date()) if len(ram_test) else None)
    for f, v in cfg["folds"].items():
        ok = v["train_start"] >= cfg["dates"]["hotel_start"] + pd.Timedelta(days=K) and v["train_end"] < v["holdout_start"]
        led.check(f"folds:{f}", ok, f"train {v['train_start'].date()}…{v['train_end'].date()} precedes holdout {v['holdout_start'].date()}…{v['holdout_end'].date()}")
    for f in ("A", "B"):
        v = cfg["folds"][f]
        tgt = hi[(hi.date >= v["holdout_start"]) & (hi.date <= v["holdout_end"])]
        led.check(f"folds:{f}_holdout_has_targets", tgt.is_target.mean() > 0.98, f"{tgt.is_target.mean():.1%} of fold-{f} holdout cells have observed guests")
