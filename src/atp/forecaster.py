"""Stage A daily forecaster — hotel check-ins per nationality per day (Step 2.4).

    log E[A_{n,t}] = calendar_n(t) + ε · log seats_7d_{n,t}            (ε fixed = the lever from Steps 2.2–2.3)

One Poisson regression per series (45 nationalities + domestic). No lagged check-ins: the forecast must move with
seats and the calendar, never with its own recent history. The 12 nationalities without flight data and the domestic
series are calendar-only (no seat term).

Rules fixed before any run:
  BASE   intercept + 11 month + 6 weekday dummies + Ramadan + MAIN UAE holiday breaks (Eid al-Fitr, Arafat/Eid al-Adha,
         National Day, with the weekend days of the same break) + ε·log seats_7d (seat term floored at 5% of the
         country's training mean, so a closure cannot force check-ins to zero).
  Candidates, each tested against the current model with the project rule (fold A daily×nationality WMAPE down
  ≥ 0.10 pts AND fold B not worse by > 0.05 pts), added in order of fold-A gain:
    C1 festival  the nationality's own main moving festival window (day −2…+7; festival_groups.csv)
    C2 yearend   20 Dec – 7 Jan (Western Christmas school break and Russia's 1–8 Jan holidays), own coefficient per series
    C3 events    recurring scheduled Abu Dhabi events: Grand Prix and ADIPEC (day before start … last day), separate terms
    C4 level     one of: (a) linear trend in the fit, frozen at its train-end value; (b) a "last 182 training days"
                 indicator in the fit, set to 1 in the forecast (the recent step in level, measured against the same
                 calendar months a year earlier). The better one on fold A is the C4 candidate.
                 [C4b was first drafted as an after-the-fit ratio actual/fitted over the last 182 days; a unit test showed
                 it recovers only about half of a level shift because the month terms absorb part of it, so it was
                 replaced by the in-fit indicator before any result was used.]
    S1 swap      all UAE public holidays (any long break) instead of main breaks — used only if it passes the rule.
  Estimation rows: 2023-01-07 (first full 7-day seat window) … train end, minus flight-disruption days and days whose
  7-day seats are below the floor (real closures, e.g. Israel 20–26 Jun 2025).
  ε: each fold uses its own default ε from Step 2.2 (honest); the final model uses 0.31. Sensitivity ε ∈ {0, 0.25,
  0.31, 0.44}; the default changes only if another value beats 0.31 by the rule on both folds.
  The official test window is forecast once with the final model and saved with a fingerprint; it is NOT scored here.

Amendments after the first run (declared, judged by the same thresholds):
  A1 Seasonal terms are judged on the fold where their days fall: C1's windows are Eid/Easter-heavy and fold A
     (Aug–Feb) contains no Eid or Easter, so C1 is judged on fold B (gain ≥ 0.10) with fold A not worse by > 0.05.
     C2 and C3 act mainly in fold A and keep fold A as their primary fold.
  A3 Level terms (C4a, C4b, C5) change how many people arrive over a period, not which day they arrive, so they are
     judged on monthly × nationality WMAPE. Daily WMAPE is minimised by forecasting the typical (median) day, which for
     spiky daily check-ins sits ~12% below the mean, so it rewards an under-forecast; judging level terms on it would
     select bias. Shape terms (C1, C2, C3, S1) stay on daily × nationality WMAPE. Level is chosen first, then shapes.
     Level variants are mutually exclusive: among those passing, the one with the smaller worst-fold error is kept
     (C4a and C4b tied within 0.11 pts on fold A while fold B favoured C4b by 0.75 pts).
  A4 External-factor audit (2 Oct 2026), candidates judged by the same rule:
     S2 split  the UAE main break into Eid breaks (Gulf-wide: Gulf-based expats and Gulf nationals travel) and the
               National Day break (UAE residents only). One switch made Filipino check-ins expect +79% on National
               Day, when they move ×0.95. Primary fold B (8 Eid-break days vs 4 National Day days), A not worse.
               Result: not kept (B unchanged). It is a trade-off, recorded in step24 JSON: the split fixes the expat
               National Day over-forecast (fold A) but over-forecasts the weaker 2025 Eids (fold B), monthly error no better.
     C6 home   main FIXED-DATE home holidays with a within-month spike the month terms cannot place: China and Japan
               Golden Week, Russia's May holidays, Korea's Chuseok, Saudi National and Founding Days, Kuwait and
               Bahrain national days (screened on data before fold A's holdout: uplift ≥ ×1.3 on average; Qatar and
               Oman national days, India's Independence Day and Egypt's Sham el-Nessim showed none and are left out).
     C7 school UAE school winter and summer breaks (spring added by A5), on the DOMESTIC series only (resident families), so it is judged on
               domestic daily WMAPE (fold A primary, B not worse); international forecasts are unchanged by construction.
               First run applied it to every series and judged it on international error: domestic improved on both folds
               (A 15.06→14.67, B 12.44→11.99) while international fold B worsened 0.40, as the hypothesis predicts.
  A5 Spring break (3 Oct 2026): C7 was declared with the winter and summer breaks only (C7ws). The UAE spring break
     (2–3 weeks every March/April; official dates in school_breaks.csv) is tested as an extension of the same term.
     Its days fall in fold B only (Jan–Jul), so it is judged on domestic daily WMAPE with fold B primary and fold A not
     worse; if it fails, C7ws (winter + summer) is kept. International forecasts are unchanged by construction.
  A2 C5 growth: the first run under-forecast international check-ins by 9% (fold A) and 7% (fold B), in line with DCT's
     reported ~10%/yr growth. C5 carries forward last year's growth NOT explained by seats:
     g = log(check-ins, last 182 training days / same days a year earlier) − ε·log(seats, same ratio), applied as
     exp(g · (h + 91)/365) on top of C4b (whose level is centred 91 days before the train end). Variants: one pooled
     rate for all international series (C5p) or one per series (C5s); the better on fold A is the candidate.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln
from . import stagea as sa

MAIN_UAE = {"eid_al_fitr", "arafat_day", "eid_al_adha", "national_day"}
SEAT_FLOOR = 0.05
LEVEL_DAYS = 182
YEAREND = ((12, 20), (1, 7))


# ----------------------------------------------------------------------------- features
def uae_breaks(cal: pd.DataFrame) -> pd.DataFrame:
    """Daily flags: all_break = any UAE long break or public holiday; main_break = breaks containing a main holiday;
    eid_break / nd_break = the same breaks split into Eid (Gulf-wide) and National Day (UAE residents only)."""
    c = cal[["date", "holiday_name", "is_long_break", "is_public_holiday"]].sort_values("date").reset_index(drop=True)
    on = c.is_long_break | c.is_public_holiday
    run = (on != on.shift()).cumsum().where(on)
    main_runs = set(run[c.holiday_name.isin(MAIN_UAE)].dropna())
    eid_runs = set(run[c.holiday_name.isin(MAIN_UAE - {"national_day"})].dropna())
    nd_runs = set(run[c.holiday_name == "national_day"].dropna())
    return pd.DataFrame({"date": c.date, "all_break": on.astype(float), "main_break": run.isin(main_runs).astype(float),
                         "eid_break": run.isin(eid_runs).astype(float), "nd_break": run.isin(nd_runs).astype(float)})


def home_holiday_daily(hh: pd.DataFrame) -> pd.DataFrame:
    """(nationality, date) inside a source market's main FIXED-DATE home holiday (data/reference/home_holidays.csv),
    from the day before it starts (travellers arrive the evening before) to its last day."""
    rows = [(r.nationality, d) for r in hh.itertuples() for d in pd.date_range(pd.Timestamp(r.start) - pd.Timedelta(days=1), r.end)]
    return pd.DataFrame(rows, columns=["nationality", "date"]).drop_duplicates().assign(home_holiday=1.0)


def festival_daily(groups: pd.DataFrame, fest: pd.DataFrame) -> pd.DataFrame:
    """(nationality, date) pairs inside the nationality's own main moving festival window (Ramadan excluded)."""
    days = [(f.festival, f.date + pd.Timedelta(days=k)) for f in fest.itertuples()
            for k in range(sa.FEST_WINDOW[0], sa.FEST_WINDOW[1] + 1)]
    win = pd.DataFrame(days, columns=["festival", "date"])
    g = groups.assign(festival=groups.festivals.fillna("").str.split(";")).explode("festival")
    return g.merge(win, on="festival")[["nationality", "date"]].drop_duplicates().assign(festival=1.0)


def event_daily(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in events.itertuples():
        kind = "f1" if e.event.startswith("abu_dhabi_grand_prix") else "adipec" if e.event.startswith("adipec") else None
        if kind:
            for d in pd.date_range(e.start_date - pd.Timedelta(days=1), e.end_date):
                rows.append((d, kind))
    return pd.DataFrame(rows, columns=["date", "kind"]).assign(v=1.0).pivot_table(index="date", columns="kind", values="v", fill_value=0).reset_index()


def build_frame(series: pd.DataFrame, cal, breaks, fest_d, ev_d, home_d=None) -> pd.DataFrame:
    """series: columns key, date, y, seats_7d (NaN for calendar-only), exclude (bool). Adds all feature columns."""
    cc = cal.assign(is_school_break=cal["is_school_break"] if "is_school_break" in cal else False)
    cc["is_school_break_ws"] = cc[[c for c in ("is_school_winter_break", "is_school_summer_break") if c in cc]].any(axis=1)
    d = series.merge(cc[["date", "month", "dow", "is_ramadan", "is_school_break", "is_school_break_ws"]], on="date", how="left").merge(breaks, on="date", how="left")
    d = d.merge(fest_d.rename(columns={"nationality": "key"}), on=["key", "date"], how="left").merge(ev_d, on="date", how="left")
    if home_d is not None:
        d = d.merge(home_d.rename(columns={"nationality": "key"}), on=["key", "date"], how="left")
    else:
        d["home_holiday"] = 0.0
    d[["festival", "f1", "adipec", "home_holiday"]] = d[["festival", "f1", "adipec", "home_holiday"]].fillna(0.0)
    # UAE school breaks move resident families, i.e. the DOMESTIC series only (international guests do not follow them)
    d["school_break"] = (d.is_school_break.fillna(False) & (d.key == "DOMESTIC")).astype(float)
    d["school_break_ws"] = (d.is_school_break_ws.fillna(False) & (d.key == "DOMESTIC")).astype(float)   # winter + summer only (A5)
    for c in ("eid_break", "nd_break"):
        if c not in d:
            d[c] = 0.0
    md = d.date.dt.month * 100 + d.date.dt.day
    d["yearend"] = ((md >= YEAREND[0][0] * 100 + YEAREND[0][1]) | (md <= YEAREND[1][0] * 100 + YEAREND[1][1])).astype(float)
    d["ramadan"] = d.is_ramadan.astype(float)
    for mth in range(2, 13):
        d[f"m{mth}"] = (d.month == mth).astype(float)
    for w in range(1, 7):
        d[f"w{w}"] = (d.dow == w).astype(float)
    return d


def columns(spec: set) -> list:
    cols = ["const"] + [f"m{m}" for m in range(2, 13)] + [f"w{w}" for w in range(1, 7)] + ["ramadan"]
    cols += ["all_break"] if "S1" in spec else (["eid_break", "nd_break"] if "S2" in spec else ["main_break"])
    if "C1" in spec: cols += ["festival"]
    if "C2" in spec: cols += ["yearend"]
    if "C3" in spec: cols += ["f1", "adipec"]
    if "C4a" in spec: cols += ["trend"]
    if "C4b" in spec: cols += ["recent"]
    if "C6" in spec: cols += ["home_holiday"]
    if "C7" in spec: cols += ["school_break_ws" if "C7ws" in spec else "school_break"]
    return cols


# ----------------------------------------------------------------------------- fit / predict
def _offset(d, eps, floor):
    if eps == 0 or d.seats_7d.isna().all():
        return np.zeros(len(d))
    return eps * np.log(np.maximum(d.seats_7d.to_numpy(), floor))


def growth_rates(D: pd.DataFrame, train_end, eps: float, pooled: bool, intl: set) -> dict:
    """C5: last year's check-in growth beyond what seats explain, per series (or pooled over international series)."""
    w = pd.Timedelta(days=LEVEL_DAYS); yr = pd.Timedelta(days=364)
    rec = D[(D.date > train_end - w) & (D.date <= train_end)]; prv = D[(D.date > train_end - w - yr) & (D.date <= train_end - yr)]
    def g(r, p):
        gy = np.log(r.y.sum() / p.y.sum())
        has = r.seats_7d.notna().any() and p.seats_7d.notna().any()
        return gy - (eps * np.log(r.seats_7d.sum() / p.seats_7d.sum()) if has else 0.0), gy
    out = {}
    if pooled:
        ri, pi = rec[rec.key.isin(intl)], prv[prv.key.isin(intl)]
        fl = ri.seats_7d.notna().groupby(ri.key).any(); fl = set(fl[fl].index)
        net, _ = g(ri[ri.key.isin(fl)], pi[pi.key.isin(fl)]); _, gross = g(ri, pi)
        for k in D.key.unique():
            out[k] = net if k in fl else gross if k in intl else g(rec[rec.key == k], prv[prv.key == k])[0]
    else:
        for k in D.key.unique():
            out[k] = g(rec[rec.key == k], prv[prv.key == k])[0]
    return out


def fit_predict(d: pd.DataFrame, train_end, start, horizon_end, spec: set, eps: float, growth: float = 0.0,
                pred_start=None) -> tuple:
    """Fit one series on [start, train_end]; predict (pred_start or train_end+1) … horizon_end.
    Returns (pred frame, params, info). pred_start ≤ train_end gives in-sample (oracle) fitted values."""
    d = d.assign(const=1.0, trend=(d.date - start).dt.days / 365.25,
                 recent=(d.date > train_end - pd.Timedelta(days=LEVEL_DAYS)).astype(float))
    tr = d[(d.date >= start) & (d.date <= train_end)]
    has_seats = tr.seats_7d.notna().any()
    floor = SEAT_FLOOR * tr.seats_7d.mean() if has_seats else np.nan
    fitrows = tr[~tr.exclude & (~has_seats | (tr.seats_7d >= floor))]
    cols = [c for c in columns(spec) if c == "const" or fitrows[c].std() > 0]     # drop flags never seen in training
    off = _offset(fitrows, eps if has_seats else 0, floor)
    res = sm.GLM(fitrows.y.to_numpy(), fitrows[cols].to_numpy(), family=sm.families.Poisson(), offset=off).fit()
    beta = pd.Series(res.params, index=cols)
    p0 = train_end + pd.Timedelta(days=1) if pred_start is None else pred_start
    te = d[(d.date >= p0) & (d.date <= horizon_end)].copy()
    X = te[cols].copy()
    if "trend" in cols:                                                            # C4a: trend frozen at train end
        X["trend"] = (train_end - start).days / 365.25
    mu = np.exp(X.to_numpy() @ beta.to_numpy() + _offset(te, eps if has_seats else 0, floor))
    level = float(np.exp(beta["recent"])) if "recent" in beta else 1.0           # C4b: recent step, already in mu
    if "C5" in spec:                                                               # carry last year's unexplained growth
        h = (te.date - train_end).dt.days.to_numpy() + LEVEL_DAYS / 2
        mu = mu * np.exp(growth * h / 365)
    te["pred"] = mu
    return te[["key", "date", "y", "pred"]], beta, {"converged": bool(res.converged), "n_fit": int(len(fitrows)),
                                                    "level": float(level), "growth": float(growth) if "C5" in spec else 0.0,
                                                    "seat_floor": float(floor) if has_seats else None}


def poisson_mad(mu: np.ndarray) -> np.ndarray:
    """E|Y − μ| for Y ~ Poisson(μ): 2 μ^{k+1} e^{−μ} / k!, k = ⌊μ⌋ (the error even a perfect mean would make)."""
    k = np.floor(mu)
    return np.where(mu > 0, 2 * np.exp((k + 1) * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(k + 1)), 0.0)


def wmape(y, p) -> float:
    return float(np.abs(np.asarray(y) - np.asarray(p)).sum() / np.asarray(y).sum() * 100)


def metrics(pr: pd.DataFrame, intl_keys: set, flight_keys: set) -> dict:
    """WMAPE (%) at the levels used throughout the project. pr: key, date, y, pred."""
    i = pr[pr.key.isin(intl_keys)]; dom = pr[~pr.key.isin(intl_keys)]
    mo = i.assign(m=i.date.dt.to_period("M")).groupby(["key", "m"])[["y", "pred"]].sum()
    tot_d = i.groupby("date")[["y", "pred"]].sum(); tot_m = i.assign(m=i.date.dt.to_period("M")).groupby("m")[["y", "pred"]].sum()
    out = {"intl_daily_nat": wmape(i.y, i.pred), "intl_daily_total": wmape(tot_d.y, tot_d.pred),
           "intl_monthly_nat": wmape(mo.y, mo.pred), "intl_monthly_total": wmape(tot_m.y, tot_m.pred),
           "flight_daily_nat": wmape(i[i.key.isin(flight_keys)].y, i[i.key.isin(flight_keys)].pred),
           "noflight_daily_nat": wmape(i[~i.key.isin(flight_keys)].y, i[~i.key.isin(flight_keys)].pred)}
    if len(dom):
        dm = dom.assign(m=dom.date.dt.to_period("M")).groupby("m")[["y", "pred"]].sum()
        out |= {"dom_daily": wmape(dom.y, dom.pred), "dom_monthly": wmape(dm.y, dm.pred)}
    return out
