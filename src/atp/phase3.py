"""Phase 3 — scenario engine.

Step 3.1 (`world`): the forward world for the scenario horizon (config: dates.horizon_start … horizon_end).
  1. Horizon calendar audit: every dated input the models use inside the horizon (UAE holidays, Ramadan, school
     breaks, source-market festivals, home holidays, Grand Prix / ADIPEC), with its source and status
     (official / scheduled / estimated / rule window / fixed date). All dates live in data/reference/*.csv.
  2. Baseline flight schedule = the network as it stood at the end of the data, carried forward month by month:
       established routes  same calendar month of the last 12 months (seasonality kept)
       seasonal routes     same (they return in season even if not flying at the data end)
       new / resumed       months before their (re)launch take their current level (last 90 days, per day); resumed =
                         flying again after a non-seasonal pause (year-round, or absent, the year before)
       ended routes        dropped: the airline left AUH (no flight in the last 90 days — Wizz Air Abu Dhabi,
                           1 Sep 2025) or the route stopped without being seasonal
     Per-day rates use the same days as the Phase 2 airline defaults (low-coverage and closure days excluded), so the
     unadjusted route sum reproduces `stagea.airline_defaults` exactly (tested).

Step 3.2 (`refit`): the production models behind the baseline ("normal year") world.
  Stage A  the frozen v2 forecaster spec (Step 2.4: chosen terms, ε = final default, seat floor) refitted per series on
           all check-ins to the data end (28 Feb 2026, itself excluded as a low-coverage day, as everywhere), then run
           over the horizon on the Step 3.1 baseline seats. Same code path as the Phase 2 final model: the function
           reproduces the archived Phase 2 test-window forecast exactly when given the Phase 2 train end (tested).
  Stage B  kept as fitted in Phase 1 (guests are hidden after Jul 2025, so it cannot be refitted later); parameters
           are persisted here, and the fit reproduces the Phase 1 test-window predictions exactly. Its 45-day memory
           holds the actual check-ins up to 27 Feb; 28 Feb (first war day, stranded transit passengers) takes the
           refitted forecaster's normal-world value, because the baseline is the pre-war world.

Step 3.3 (`inputs`): the inputs a scenario can change (atp.scenario) and the sourced presets.
  Aircraft seats per flight come from the data (Etihad route averages, narrowbody vs widebody by natural breaks).
  "March 2026 observed" = SCAD's March 2026 guests by world region relative to the normal world (region shock shared by
  its markets; Domestic = remainder to SCAD's total). After March no market-level data are published: the August total
  and the India/China/Russia year-to-date statement are recorded as checks, not turned into an invented monthly path.

Step 3.4 (`engine`): atp.engine checked against the validated pieces (Stage B and the Phase 2 seat lever reproduced
exactly, empty scenario = no change, cancellation and restoration symmetric, short changes not diluted, stay lever
exact, markets independent) and run on the sourced Etihad preset and four illustrative scenarios.

Step 3.6 (`sensitivity`): atp.sensitivity — equal ±10% lever ranking, guest nights per 1,000 added seats by market and
month, the flights needed for a target, and which uncertainty matters most; plus the data check that check-ins respond
to local passengers per seat (the load-factor / transfer-share channel).

Step 3.5 (`bands`): atp.bands — the effect band (how sure the change is) and the level band (how sure a monthly level
is), calibrated and stored (effect_band.json, level_bands.csv, market_tiers.csv) and applied to the baseline and demos.
"""
import inspect
import json
import sys
import numpy as np
import pandas as pd
from .config import load_config
from . import stagea as sa

ACTIVE_DAYS = 28            # flew in the last 4 weeks of the data → active at the data end
AIRLINE_GONE_DAYS = 90      # airline with no AUH flight in the last 90 days → exited (all its routes ended)
SEASONAL_MAX_MONTHS = 8     # seasonal: ≤ 8 operating months in the prior year …
SEASONAL_MIN_JACCARD = 0.5  # … in the same months as this year (Jaccard overlap of calendar months ≥ 0.5)
FILL_DAYS = 90              # a new route's current level: per-day rates over its last 90 days
FLOWS = ["seats", "pax", "p2p", "transfer", "transit", "n_flights"]


def _out(cfg):
    d = cfg["paths"]["processed"] / "phase3"; d.mkdir(parents=True, exist_ok=True); return d


def _status(src: str) -> str:
    s = str(src).lower()
    if "estimated" in s or "algorithmic" in s or "approximate" in s:
        return "estimated"
    if s.startswith("scheduled"):
        return "scheduled"
    if s.startswith("rule window"):
        return "rule window"
    if "fixed" in s:
        return "fixed date"
    return "official"


# ----------------------------------------------------------------------------- 1. horizon calendar audit
def horizon_events(cfg) -> pd.DataFrame:
    P, RD = cfg["paths"]["processed"], cfg["paths"]["reference"]
    h0, h1 = cfg["dates"]["horizon_start"], cfg["dates"]["horizon_end"]
    inside = lambda s, e: (e >= h0) & (s <= h1)
    rows = []
    ev = pd.read_parquet(P / "calendar_events.parquet")
    ev = ev[inside(ev.start_date, ev.end_date)]                          # inactive rows are already removed in Phase 0
    for r in ev.itertuples():
        kind = "school break" if r.event.startswith("school_") else "Ramadan" if r.event == "ramadan" else "UAE public holiday"
        term = {"school break": "Domestic school-break term; Stage B calendar", "Ramadan": "Ramadan term (forecaster, Stage B)"}.get(
            kind, "UAE breaks (forecaster), long break (Stage B)")
        rows.append((kind, r.event, "UAE", r.start_date, r.end_date, _status(r.source), r.source, term))
    off = pd.read_csv(RD / "calendar_events_official.csv", parse_dates=["start_date", "end_date"])
    off["active"] = off.active.astype(str).str.lower().isin(["true", "1"])
    fest = sa.festival_dates(RD, off)
    srcs = pd.read_csv(RD / "source_market_festivals.csv", parse_dates=["date"]).set_index(["festival", "date"]).source
    for r in fest.itertuples():
        s, e = r.date + pd.Timedelta(days=sa.FEST_WINDOW[0]), r.date + pd.Timedelta(days=sa.FEST_WINDOW[1])
        if inside(s, e):
            src = srcs.get((r.festival, r.date), "UAE official calendar (Eid)" if r.festival.startswith("eid") else
                           "exact computus (dateutil)")
            rows.append(("main moving festival", r.festival, "festival group", s, e,
                         "estimated" if "ESTIMATED" in str(src) else "official", src, "own-festival term C1"))
    hh = pd.read_csv(RD / "home_holidays.csv", parse_dates=["start", "end"])
    for r in hh[inside(hh.start, hh.end)].itertuples():
        st = "rule window" if r.source.startswith("Rule window") else "fixed date" if "fixed" in r.source.lower() else "official"
        rows.append(("home holiday", r.holiday, r.nationality, r.start - pd.Timedelta(days=1), r.end, st, r.source,
                     "home-holiday term C6 (from the day before)"))
    ex = pd.read_csv(RD / "external_events.csv", parse_dates=["start_date", "end_date"])
    ex = ex[ex.event.str.match(r"abu_dhabi_grand_prix|adipec") & inside(ex.start_date, ex.end_date)]
    for r in ex.itertuples():
        rows.append(("event", r.event, "all", r.start_date - pd.Timedelta(days=1), r.end_date, "official", r.source,
                     "event term C3 (from the day before)"))
    t = pd.DataFrame(rows, columns=["kind", "name", "applies_to", "start", "end", "status", "source", "model_term"])
    return t.sort_values(["start", "kind"]).reset_index(drop=True)


REQUIRED = {  # what the horizon must contain for every calendar term the models use
    "UAE public holiday": {"eid_al_fitr", "arafat_day", "eid_al_adha", "islamic_new_year", "mawlid", "national_day", "new_year"},
    "Ramadan": {"ramadan"},
    "school break": {"school_summer_break", "school_winter_break", "school_spring_break"},
    "main moving festival": {"eid_al_fitr", "eid_al_adha", "easter", "orthodox_easter", "diwali", "lunar_new_year"},
    "event": {"abu_dhabi_grand_prix_2026", "adipec_2026"},
}


def check_horizon(t: pd.DataFrame, hh: pd.DataFrame) -> dict:
    missing = {k: sorted(v - set(t[t.kind == k].name)) for k, v in REQUIRED.items()}
    homes = set(hh.nationality)
    missing["home holiday markets"] = sorted(homes - set(t[t.kind == "home holiday"].applies_to))
    return {k: v for k, v in missing.items() if v}


# ----------------------------------------------------------------------------- 2. baseline flight schedule
def _rows(cfg):
    """Daily flight rows with the country-day exclusions of the Phase 2 airline defaults."""
    P = cfg["paths"]["processed"]
    f = pd.read_parquet(P / "flights_rows.parquet")
    f = f[f.granularity == "daily"].copy()
    fd = pd.read_parquet(P / "flights_country_daily.parquet")
    return f, fd


def route_baseline(cfg):
    """Route × calendar-month baseline (per-day flows) and the route status table."""
    f, fd = _rows(cfg)
    end = cfg["dates"]["test_end"]; w0 = end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    key = ["nationality_key", "city", "airline"]
    # included country-days (identical to stagea.airline_defaults): not low coverage, not a closure day
    cl = sa.closure_days(fd, w0, end)
    days = fd[(fd.date >= w0) & (fd.date <= end) & ~fd.is_low_coverage]
    days = days[~pd.MultiIndex.from_frame(days[["nationality_key", "date"]]).isin(cl)][["nationality_key", "date"]]
    days = days.assign(moy=days.date.dt.month)
    ndays = days.groupby(["nationality_key", "moy"]).size().rename("days")
    w = f.merge(days, on=["nationality_key", "date"])                      # inner join applies the exclusions
    m = w.groupby(key + ["moy"])[FLOWS].sum().join(ndays, on=["nationality_key", "moy"])
    rate = m[FLOWS].div(m.days, axis=0)                                   # per-day flows, zero-service days included
    # ---- route status
    st = f.groupby(key).date.agg(ever_first="min").join(
        f[(f.date >= w0) & (f.date <= end)].groupby(key).date.agg(first="min", last="max"), how="inner")
    airline_last = f.groupby("airline").date.max()
    st["airline_exited"] = st.index.get_level_values("airline").map(airline_last) < end - pd.Timedelta(days=AIRLINE_GONE_DAYS)
    st["active"] = st["last"] >= end - pd.Timedelta(days=ACTIVE_DAYS)
    st["new"] = st.ever_first >= w0
    prior = f[(f.date >= w0 - pd.DateOffset(years=1)) & (f.date < w0)]
    pm = prior.groupby(key).date.agg(lambda s: frozenset(s.dt.month))
    cm = f[(f.date >= w0) & (f.date <= end)].groupby(key).date.agg(lambda s: frozenset(s.dt.month))
    def seasonal(k):
        a, b = pm.get(k, frozenset()), cm.get(k, frozenset())
        return 0 < len(a) <= SEASONAL_MAX_MONTHS and len(b) <= SEASONAL_MAX_MONTHS and len(a & b) / len(a | b) >= SEASONAL_MIN_JACCARD
    st["seasonal"] = [seasonal(k) for k in st.index]
    st["prior_months"] = [len(pm.get(k, frozenset())) for k in st.index]
    # resumed: active again after a pause that is not its seasonal pattern (year-round last year, or absent last year)
    st["resumed"] = (st.active & ~st.new & (st["first"] > w0 + pd.Timedelta(days=14))
                     & ((st.prior_months >= 9) | (st.prior_months == 0)))
    st["status"] = np.select(
        [st.airline_exited, st.active & st.new, st.resumed, st.active, st.seasonal],
        ["ended (airline exited)", "new", "resumed", "established", "seasonal"], "ended (route stopped)")
    ov = pd.read_csv(cfg["paths"]["reference"] / "route_overrides.csv", dtype=str)
    h0 = cfg["dates"]["horizon_start"]
    st["status_from_data"] = st.status
    st["override_source"] = ""
    for o in ov.itertuples():
        k = (o.nationality_key, o.city, o.airline)
        assert k in st.index and pd.Timestamp(o.published) <= end, f"override must name a pack route, published by the data end: {k}"
        st.loc[k, ["status", "override_source"]] = [o.status, o.source]
    # ---- current level of new routes: per-day flows over their last FILL_DAYS (or since launch)
    cur0 = end - pd.Timedelta(days=FILL_DAYS - 1)
    nk = st[st.status.isin(["new", "resumed"])].index
    cur = {}
    for k in nk:
        d0 = max(cur0, st.loc[k, "first"])
        dd = days[(days.nationality_key == k[0]) & (days.date >= d0)]
        x = w[(w.nationality_key == k[0]) & (w.city == k[1]) & (w.airline == k[2]) & (w.date >= d0)][FLOWS].sum()
        cur[k] = x / max(len(dd), 1)
    # ---- baseline: full route × month grid
    grid = pd.MultiIndex.from_product([st.index.unique(), range(1, 13)]).to_frame(index=False)
    grid.columns = ["route", "moy"]
    grid[key] = pd.DataFrame(grid.route.tolist(), index=grid.index)
    b = grid.drop(columns="route").set_index(key + ["moy"]).join(rate).fillna(0.0)
    b["naive"] = b.seats                                                   # 'same month last year', no adjustment
    b = b.join(st[["status", "first"]], on=key)
    b["filled"] = False
    for k in nk:                                                           # months before launch → current level
        launch = st.loc[k, "first"].replace(day=1)
        window_month = lambda mo: pd.Timestamp(year=w0.year if mo >= w0.month else w0.year + 1, month=mo, day=1)
        for mo in [mo for mo in range(1, 13) if window_month(mo) < launch]:
            if b.loc[k + (mo,), "seats"] == 0:
                b.loc[k + (mo,), FLOWS] = cur[k][FLOWS].to_numpy(); b.loc[k + (mo,), "filled"] = True
    b.loc[b.status.str.startswith("ended"), FLOWS] = 0.0
    # published schedules (before the data end) that the data rule cannot see: inside the operating window a route
    # keeps last year's same-month flows; a partly covered month gets the nearest fully covered month's per-day
    # level × the share of its days in the window (last year's partial months are not reused); nothing outside it
    for o in ov.itertuples():
        k = (o.nationality_key, o.city, o.airline)
        frac = {}
        for mo in range(1, 13):
            yr = h0.year if mo >= h0.month else h0.year + 1
            days_m = pd.date_range(pd.Timestamp(yr, mo, 1), periods=pd.Timestamp(yr, mo, 1).days_in_month)
            a, z = pd.Timestamp(f"{yr}-{o.active_from}"), pd.Timestamp(f"{yr}-{o.active_to}")
            frac[mo] = float(((days_m >= a) & (days_m <= z)).mean())
        full = [m for m, f in frac.items() if f == 1.0]
        lvl = lambda mo: rate.reindex([k + (mo,)]).fillna(0.0).to_numpy()[0]
        for mo, f in frac.items():
            src = mo if f == 1.0 else min(full, key=lambda m: min((m - mo) % 12, (mo - m) % 12))
            b.loc[k + (mo,), FLOWS] = lvl(src) * f
    b = b.reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        b["flights_per_week"] = b.n_flights * 7
        b["seats_per_flight"] = (b.seats / b.n_flights).where(b.n_flights > 0)
        b["load_factor"] = (b.pax / b.seats).where(b.seats > 0)
        b["transfer_transit_share"] = ((b.transfer + b.transit) / b.pax).where(b.pax > 0)
        b["local_share"] = (b.p2p / b.pax).where(b.pax > 0)
    return b, st.reset_index(), ndays


def _month_index(h0, h1):
    return pd.date_range(h0, h1, freq="D")


def daily_baseline(b: pd.DataFrame, cfg) -> pd.DataFrame:
    """Country × day baseline over the horizon (+6 lead-in days so the trailing 7-day mean starts on the baseline,
    not on the 28 Feb 2026 disruption): each day carries its calendar month's per-day flows."""
    h0, h1 = cfg["dates"]["horizon_start"], cfg["dates"]["horizon_end"]
    c = b.groupby(["nationality_key", "moy"])[FLOWS].sum().reset_index()
    dates = pd.DataFrame({"date": pd.date_range(h0 - pd.Timedelta(days=6), h1)}); dates["moy"] = dates.date.dt.month
    d = c.merge(dates, on="moy").sort_values(["nationality_key", "date"]).drop(columns="moy")
    d["seats_7d"] = d.groupby("nationality_key").seats.transform(lambda s: s.rolling(7).mean())
    d["p2p_7d"] = d.groupby("nationality_key").p2p.transform(lambda s: s.rolling(7).mean())
    return d[d.date >= h0].reset_index(drop=True)


def _zero_seat_check(b, fd, cfg) -> dict:
    """Country-months with no baseline seats: what the frozen forecaster gives there (seat term floored at 5% of the
    country's training mean, so never log 0) vs the scenario lever's full-cancellation rule, both measured against the
    country's seats in the nearest months that have service."""
    from . import forecaster as fc
    P = cfg["paths"]["processed"]
    el = pd.read_csv(P / "phase2" / "elasticities.csv").set_index(["fold", "parameter"])
    eps = float(el.loc[("final", "eps_total_default"), "estimate"])
    cm = b.groupby(["nationality_key", "moy"]).seats.sum()
    out = {}
    for (k, mo), v in cm.items():
        if v > 0:
            continue
        pos = cm.loc[k][cm.loc[k] > 0]
        dist = pos.index.map(lambda m: min((m - mo) % 12, (mo - m) % 12))
        ref = float(pos[dist == dist.min()].mean())
        tr = fd[(fd.nationality_key == k) & (fd.date >= "2023-01-07") & (fd.date <= cfg["dates"]["test_end"])]
        floor = fc.SEAT_FLOOR * float(tr.seats_7d.mean())
        dA, _ = sa.seat_lever(1.0, ref, -ref, eps, np.inf)
        out[f"{k.title()} {pd.Timestamp(2000, mo, 1):%b}"] = {
            "reference_seats_per_day": round(ref, 1), "forecaster_floor_seats_per_day": round(floor, 1),
            "forecaster_keeps": float((floor / ref) ** eps), "lever_full_cancellation_keeps": float(1 + dA)}
    return out


# ----------------------------------------------------------------------------- 3.2 production refit
START = pd.Timestamp("2023-01-07")          # first full 7-day seat window (as in Step 2.4)
WAR_DAY = pd.Timestamp("2026-02-28")


def forecaster_frame(cfg) -> tuple:
    """Step 2.4's daily frame extended over the horizon: actual check-ins and seats to the data end, Step 3.1 baseline
    seats after it; 28 Feb 2026 carries 27 Feb's 7-day seats (the normal-world counterfactual of the war day)."""
    from . import forecaster as fc, phase2 as p2
    P, RD = cfg["paths"]["processed"], cfg["paths"]["reference"]
    fd, _, hi = p2._load(cfg)
    S = p2._series(cfg, fd, hi)
    h1 = cfg["dates"]["horizon_end"]
    bd = pd.read_parquet(P / "phase3" / "baseline_seats_daily.parquet")[["nationality_key", "date", "seats_7d"]]
    keys = S.key.unique()
    fut = pd.MultiIndex.from_product([keys, pd.date_range(cfg["dates"]["test_end"] + pd.Timedelta(days=1), h1)],
                                     names=["key", "date"]).to_frame(index=False)
    fut = fut.merge(bd.rename(columns={"nationality_key": "key"}), on=["key", "date"], how="left").assign(y=np.nan, exclude=False)
    S = pd.concat([S, fut[S.columns]], ignore_index=True)
    prev = S[S.date == WAR_DAY - pd.Timedelta(days=1)].set_index("key").seats_7d
    war = S.date == WAR_DAY
    S.loc[war, "seats_7d"] = S.loc[war, "key"].map(prev).to_numpy()
    cal = pd.read_parquet(P / "calendar.parquet")
    off = pd.read_csv(RD / "calendar_events_official.csv", parse_dates=["start_date", "end_date"])
    ev = pd.read_csv(RD / "external_events.csv", parse_dates=["start_date", "end_date"])
    hh = pd.read_csv(RD / "home_holidays.csv", parse_dates=["start", "end"])
    D = fc.build_frame(S, cal, fc.uae_breaks(cal), fc.festival_daily(pd.read_csv(RD / "festival_groups.csv"), sa.festival_dates(RD, off)),
                       fc.event_daily(ev), fc.home_holiday_daily(hh))
    return D.sort_values(["key", "date"]).reset_index(drop=True), set(hi.nationality), set(fd.nationality_key)


def forecast(D, train_end, horizon_end, spec, eps, pred_start=None) -> tuple:
    """The Phase 2 final-model procedure for any train end: one Poisson fit per series, check-ins after the train end
    never seen. Returns (predictions, coefficients, info)."""
    from . import forecaster as fc
    Dm = D.assign(y=D.y.where(D.date <= train_end))
    rows, params, info = [], [], {}
    for key in sorted(Dm.key.unique()):
        p_, beta, inf = fc.fit_predict(Dm[Dm.key == key], train_end, START, horizon_end, set(spec), eps, pred_start=pred_start)
        rows.append(p_.drop(columns="y")); info[key] = inf
        params += [{"series": key, "term": t, "coef": float(b), "multiplier": float(np.exp(b))} for t, b in beta.items()]
    return pd.concat(rows, ignore_index=True), pd.DataFrame(params), info


def stageb_final(cfg):
    """Phase 1's final Stage B fit (selection from final_summary.json), returned with everything needed to predict."""
    from . import stageb as sb, phase1 as p1
    hi, hd, cal, gp_days, idx = p1.load(cfg)
    f = cfg["folds"]["final"]; ok = (idx >= f["train_start"]) & (idx <= f["train_end"])
    sel = json.load(open(cfg["paths"]["processed"] / "phase1" / "final_summary.json", encoding="utf-8"))["selection"]["chosen"]
    ci, cd = sel["intl"], sel["dom"]
    mk = sb.build_markets(hi, hd, idx)
    intl = {n: m for n, m in mk.items() if n != "DOMESTIC"}
    X, names = sb.design(cal, idx, f["train_end"], {"long_break": ci["long_break"], "grand_prix": ci["grand_prix"], "gp_days": gp_days})
    Xd, dnames = sb.design(cal, idx, f["train_end"], {"long_break": cd["long_break"]})
    ci.setdefault("drift_shape", "trend"); cd.setdefault("drift_shape", "trend")
    ri = sb.fit_group(intl, X, ok, drift=ci["drift"], idx=idx, drift_shape=ci["drift_shape"])
    rd = sb.fit_group({"DOMESTIC": mk["DOMESTIC"]}, Xd, ok, cd["arrival_dow"], drift=cd["drift"], idx=idx, max_nfev=3000, drift_shape=cd["drift_shape"])
    spec = {"intl": {**ci, "names": names}, "dom": {**cd, "names": dnames}, "train_end": str(f["train_end"].date())}
    return ri, rd, spec, gp_days


def stageb_predict(cfg, ri, rd, spec, gp_days, arrivals: pd.DataFrame, idx: pd.DatetimeIndex) -> dict:
    """Guests in-house per market on idx from a check-in table (market, date, arrivals) with the persisted Stage B."""
    from . import stageb as sb
    cal = pd.read_parquet(cfg["paths"]["processed"] / "calendar.parquet")
    a = arrivals.assign(guests=np.nan)
    mk = sb.build_markets(a[a.market != "DOMESTIC"].rename(columns={"market": "nationality"}),
                          a[a.market == "DOMESTIC"].drop(columns="market"), idx)
    te = pd.Timestamp(spec["train_end"])
    X, _ = sb.design(cal, idx, te, {"long_break": spec["intl"]["long_break"], "grand_prix": spec["intl"]["grand_prix"], "gp_days": gp_days})
    Xd, _ = sb.design(cal, idx, te, {"long_break": spec["dom"]["long_break"]})
    e, ed = np.exp(X @ ri.beta), np.exp(Xd @ rd.beta)
    out = {}
    for n, m in mk.items():
        th = (rd if n == "DOMESTIC" else ri).markets[n].theta
        if len(th) == 1 + sb.NH + 1:                                       # drift: frozen at 0 after the train end
            m.tau = sb.make_tau(idx, te, spec["dom" if n == "DOMESTIC" else "intl"].get("drift_shape", "trend"))
        out[n] = sb.predict(m, th, ed if n == "DOMESTIC" else e, rd.gamma if n == "DOMESTIC" else ri.gamma)
    return out


def step_refit():
    from . import forecaster as fc, phase1 as p1
    cfg = load_config(); out = _out(cfg); P = cfg["paths"]["processed"]
    h0, h1, data_end = cfg["dates"]["horizon_start"], cfg["dates"]["horizon_end"], cfg["dates"]["test_end"]
    e24 = json.load(open(P / "phase2" / "step24_forecaster.json", encoding="utf-8")); spec = set(e24["chosen_spec"]); eps = float(e24["eps_default"])
    D, intl, flight = forecaster_frame(cfg)
    j = {"spec": sorted(spec), "eps": eps, "train_start": str(START.date()), "train_end": str(data_end.date())}

    # 1. same code path: the Phase 2 train end reproduces the archived Phase 2 test-window forecast
    arch = pd.read_csv(P / "phase2" / "stagea_test_forecast.csv", parse_dates=["date"])
    rp, _, _ = forecast(D, cfg["folds"]["final"]["train_end"], data_end, spec, eps)
    m = arch.merge(rp.rename(columns={"key": "series"}), on=["series", "date"])
    d_ = (m.pred / m.checkins_forecast - 1).abs(); wd = m.date == WAR_DAY
    j["reproduces_phase2_final"] = {"rows": int(len(m)), "of": int(len(arch)), "max_rel_diff": float(d_[~wd].max()),
                                    "war_day_max_rel_diff": float(d_[wd].max()),
                                    "war_day_note": "28 Feb now carries 27 Feb's 7-day seats (normal-world counterfactual), by design"}

    # 2. production fit: all check-ins to the data end; predictions from the war day (its normal-world value) on
    pr, params, info = forecast(D, data_end, h1, spec, eps, pred_start=WAR_DAY)
    params.to_csv(out / "stagea_production_params.csv", index=False)
    old = pd.read_csv(P / "phase2" / "stagea_final_params.csv")
    cmp_ = old.merge(params, on=["series", "term"], suffixes=("_jul25", "_feb26"))
    cmp_["log_ratio"] = cmp_.coef_feb26 - cmp_.coef_jul25
    big = D[D.date <= data_end].groupby("key").y.sum().nlargest(10).index
    calt = ~cmp_.term.isin(["const", "recent"])
    j["coefficient_stability"] = {
        "median_abs_change_pct_calendar_terms_top10_series": float((np.exp(cmp_[calt & cmp_.series.isin(big)].log_ratio.abs()) - 1).median() * 100),
        "recent_level_multiplier": {k: [round(float(np.exp(old.set_index(["series", "term"]).coef.get((k, "recent"), 0))), 3), round(info[k]["level"], 3)]
                                    for k in big}}
    j["non_converged"] = sorted(k for k, v in info.items() if not v["converged"])
    base = pr[pr.date >= h0].rename(columns={"key": "series", "pred": "checkins"})
    s7 = D.set_index(["key", "date"]).seats_7d
    base["seats_7d"] = s7.reindex(pd.MultiIndex.from_frame(base[["series", "date"]].rename(columns={"series": "key"}))).to_numpy()
    floor = {k: v["seat_floor"] for k, v in info.items()}
    base["seat_floor_active"] = base.seats_7d.notna() & (base.seats_7d < base.series.map(floor))
    base["months_after_data"] = (base.date.dt.year - data_end.year) * 12 + base.date.dt.month - data_end.month
    base["beyond_7_months"] = base.months_after_data > 7
    base.to_parquet(out / "baseline_checkins_daily.parquet", index=False)

    # 3. Stage B: persisted Phase 1 fit; memory = actual check-ins to 27 Feb + the war day's normal-world value
    ri, rd, sbspec, gp_days = stageb_final(cfg)
    hi, hd, _, _, _ = p1.load(cfg)
    act = pd.concat([hi[["nationality", "date", "arrivals"]].rename(columns={"nationality": "market"}),
                     hd[["date", "arrivals"]].assign(market="DOMESTIC")], ignore_index=True)
    war = pr[pr.date == WAR_DAY].set_index("key").pred
    a28 = act[act.date == WAR_DAY].set_index("market").arrivals
    j["war_day_checkins"] = {"actual": float(a28.sum()), "normal_world": float(war.sum())}
    arr = pd.concat([act[act.date < WAR_DAY], war.rename("arrivals").rename_axis("market").reset_index().assign(date=WAR_DAY),
                     base.rename(columns={"series": "market", "checkins": "arrivals"})[["market", "date", "arrivals"]]], ignore_index=True)
    idx = pd.date_range(cfg["dates"]["hotel_start"], h1)
    from . import stageb as sb
    arr[(arr.date >= h0 - pd.Timedelta(days=sb.K - 1)) & (arr.date < h0)].sort_values(["market", "date"]) \
        .to_parquet(out / "stageb_memory.parquet", index=False)              # check-ins still in hotels on 1 Mar (engine)
    g = stageb_predict(cfg, ri, rd, sbspec, gp_days, arr, idx)
    # same parameters + actual check-ins reproduce the Phase 1 test-window predictions
    t1 = pd.read_csv(P / "phase1" / "test_predictions_daily.csv", parse_dates=["date"])
    gi = pd.DataFrame({n: v for n, v in g.items()}, index=idx).stack().rename("p").rename_axis(["date", "market"]).reset_index()
    t1 = t1[t1.date < WAR_DAY].merge(gi, on=["date", "market"])
    j["stageb_reproduces_phase1_test"] = {"rows": int(len(t1)), "max_abs_diff_guests": float((t1.p - t1.pred_guests).abs().max())}
    th = {n: m.theta.tolist() for n, m in {**ri.markets, **rd.markets}.items()}
    json.dump({"spec": sbspec, "theta": th, "intl_beta": ri.beta.tolist(), "dom_beta": rd.beta.tolist(),
               "intl_gamma": None if ri.gamma is None else ri.gamma.tolist(), "dom_gamma": None if rd.gamma is None else rd.gamma.tolist()},
              open(out / "stageb_params.json", "w", encoding="utf-8"), indent=1)
    gb = gi[gi.date >= h0].rename(columns={"p": "guests"})
    gb.to_parquet(out / "baseline_guests_daily.parquet", index=False)

    # 4. checks: boundary, year on year, capacity
    A = act.assign(dom=act.market == "DOMESTIC")
    mon = lambda d, c: d.assign(m=d.date.dt.to_period("M")).groupby("m")[c].sum()
    a_int, a_dom = mon(A[~A.dom], "arrivals"), mon(A[A.dom], "arrivals")
    b_int = mon(base[base.series != "DOMESTIC"], "checkins"); b_dom = mon(base[base.series == "DOMESTIC"], "checkins")
    P_ = pd.Period
    j["boundary_mar_over_feb"] = {"baseline_2026": float((b_int[P_("2026-03")] + b_dom[P_("2026-03")]) / (a_int[P_("2026-02")] + a_dom[P_("2026-02")])),
                                  "actual_2025": float((a_int[P_("2025-03")] + a_dom[P_("2025-03")]) / (a_int[P_("2025-02")] + a_dom[P_("2025-02")])),
                                  "actual_2024": float((a_int[P_("2024-03")] + a_dom[P_("2024-03")]) / (a_int[P_("2024-02")] + a_dom[P_("2024-02")]))}
    yo = pd.DataFrame({"intl_baseline": b_int.values, "intl_last_year": a_int.reindex([p - 12 for p in b_int.index]).values,
                       "dom_baseline": b_dom.values, "dom_last_year": a_dom.reindex([p - 12 for p in b_dom.index]).values}, index=b_int.index.astype(str))
    yo["intl_yoy_pct"] = (yo.intl_baseline / yo.intl_last_year - 1) * 100; yo["dom_yoy_pct"] = (yo.dom_baseline / yo.dom_last_year - 1) * 100
    yo.to_csv(out / "baseline_yoy_monthly.csv")
    j["yoy_total_pct"] = {"international": float((yo.intl_baseline.sum() / yo.intl_last_year.sum() - 1) * 100),
                          "domestic": float((yo.dom_baseline.sum() / yo.dom_last_year.sum() - 1) * 100)}
    # per market: year-on-year change vs the part the seat change explains (ε · log seat ratio, same 12 months)
    fdc = pd.read_parquet(P / "flights_country_daily.parquet")
    ly = act[(act.date >= h0 - pd.DateOffset(years=1)) & (act.date < h0)].groupby("market").arrivals.sum()
    seats_ly = fdc[(fdc.date >= h0 - pd.DateOffset(years=1)) & (fdc.date < h0) & ~fdc.is_low_coverage].groupby("nationality_key").seats.mean()
    seats_h = pd.read_parquet(P / "phase3" / "baseline_seats_daily.parquet").groupby("nationality_key").seats.mean()
    pm = pd.DataFrame({"baseline": base.groupby("series").checkins.sum(), "last_year": ly})
    pm["yoy_pct"] = (pm.baseline / pm.last_year - 1) * 100
    pm["seat_change_pct"] = (seats_h / seats_ly - 1).reindex(pm.index) * 100
    pm["seat_explained_pct"] = ((seats_h / seats_ly) ** eps - 1).reindex(pm.index) * 100
    rec = D[(D.date > data_end - pd.Timedelta(days=182)) & (D.date <= data_end) & ~D.exclude].groupby("key").y.sum()
    prv = D[(D.date > data_end - pd.Timedelta(days=182 + 364)) & (D.date <= data_end - pd.Timedelta(days=364)) & ~D.exclude].groupby("key").y.sum()
    pm["last_6m_yoy_pct"] = (rec / prv - 1).reindex(pm.index) * 100
    pm.sort_values("last_year", ascending=False).to_csv(out / "baseline_yoy_by_market.csv")
    j["markets_yoy_beyond_25pct"] = pm[pm.yoy_pct.abs() > 25].round(1).to_dict("index")
    tot = gi.groupby("date").p.sum()
    hist = tot[(tot.index <= cfg["folds"]["final"]["train_end"])]
    j["capacity"] = {"baseline_peak_guests": float(tot[tot.index >= h0].max()), "on": str(tot[tot.index >= h0].idxmax().date()),
                     "historic_peak_model": float(hist.max()), "historic_peak_on": str(hist.idxmax().date())}
    j["beyond_7_months_from"] = str((data_end + pd.offsets.MonthBegin(1) + pd.DateOffset(months=7)).date())   # 8th month after the data
    json.dump(j, open(out / "step32_refit.json", "w", encoding="utf-8"), indent=1, default=float)
    # print
    r = j["reproduces_phase2_final"]; sbr = j["stageb_reproduces_phase1_test"]
    print(f"SAME CODE PATH: Phase 2 final forecast reproduced on {r['rows']:,}/{r['of']:,} rows (max rel diff {r['max_rel_diff']:.1e}; "
          f"28 Feb differs by up to {r['war_day_max_rel_diff']:.0%} only because its seats are now the pre-war value); "
          f"Stage B reproduces Phase 1 test predictions on {sbr['rows']:,} rows (max abs diff {sbr['max_abs_diff_guests']:.1e} guests)")
    print(f"PRODUCTION FIT {START.date()} … {data_end.date()} (28 Feb excluded), spec {sorted(spec)}, ε {eps:.3f}; non-converged: {j['non_converged'] or 'none'}")
    cs = j["coefficient_stability"]
    print(f"  calendar coefficients vs the Jul 2025 fit (top-10 series): median change {cs['median_abs_change_pct_calendar_terms_top10_series']:.1f}%")
    print("  recent-level multiplier Jul 2025 → Feb 2026: " + ", ".join(f"{k.title()} {a}→{b}" for k, (a, b) in cs["recent_level_multiplier"].items()))
    w = j["war_day_checkins"]; print(f"  28 Feb check-ins: actual {w['actual']:,.0f} → normal-world {w['normal_world']:,.0f} in Stage B memory")
    b = j["boundary_mar_over_feb"]; print(f"BOUNDARY total check-ins Mar/Feb: baseline 2026 {b['baseline_2026']:.3f} | actual 2025 {b['actual_2025']:.3f}, 2024 {b['actual_2024']:.3f}")
    print(f"YEAR ON YEAR (Mar 2026–Feb 2027 vs Mar 2025–Feb 2026 actual): international {j['yoy_total_pct']['international']:+.1f}%, domestic {j['yoy_total_pct']['domestic']:+.1f}%")
    print(yo[["intl_yoy_pct", "dom_yoy_pct"]].round(1).T.to_string())
    print("  markets beyond ±25%: " + "; ".join(f"{k.title()} {v['yoy_pct']:+.0f}% (seats {v['seat_change_pct']:+.0f}% → explains {v['seat_explained_pct']:+.0f}%; last 6 months {v['last_6m_yoy_pct']:+.0f}%)"
                                               for k, v in j["markets_yoy_beyond_25pct"].items()))
    c = j["capacity"]; print(f"CAPACITY baseline peak in-house {c['baseline_peak_guests']:,.0f} on {c['on']} (model historic peak {c['historic_peak_model']:,.0f} on {c['historic_peak_on']})")
    print(f"OUTPUT baseline_checkins_daily ({len(base):,} rows, floor active on {int(base.seat_floor_active.sum())} rows), baseline_guests_daily ({len(gb):,} rows); "
          f"beyond 7 months from {j['beyond_7_months_from']}")


# ----------------------------------------------------------------------------- 3.3 scenario inputs
def aircraft_seats(cfg) -> pd.DataFrame:
    """Etihad seats per flight by aircraft class, from the baseline year's routes (route_baseline: the last 12 months of
    data, ended routes left out, a new route's months before launch at its current level, published overrides applied):
    per route Σ per-day seats ÷ Σ per-day flights over the months, split into two classes by natural breaks (the split
    that minimises within-class variance: narrowbody A320 family vs widebody), median per class. The routes a planner
    can still fly, not every route in the data: counting the 4 ended routes and the raw months instead gives 173 / 302
    (independent review, Step 5.3)."""
    rb = pd.read_csv(cfg["paths"]["processed"] / "phase3" / "route_baseline.csv")
    e = rb[(rb.airline == "Etihad Airways") & (rb.n_flights > 0)].groupby(["nationality_key", "city"])[["seats", "n_flights"]].sum()
    spf = (e.seats / e.n_flights).sort_values()
    v = spf.to_numpy()
    sse = [v[:k].var() * k + v[k:].var() * (len(v) - k) for k in range(2, len(v) - 1)]
    cut = v[int(np.argmin(sse)) + 1]                                      # last value of the lower class
    cls = np.where(spf <= cut, "narrowbody", "widebody")
    t = spf.groupby(cls).agg(seats_per_flight="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), routes="size")
    t.index.name = "aircraft_class"
    return t.assign(split_between=f"{cut:.0f}–{spf[spf > cut].min():.0f} seats").reset_index()


def march_2026_observed(cfg) -> tuple[dict, dict]:
    """Demand preset for March 2026 = the observed level (all causes) relative to the normal world: each nationality
    takes its SCAD region's Mar 2026 / Mar 2025 guest ratio applied to its own Mar 2025 check-ins; Domestic (residents,
    no SCAD region) is the remainder that makes the total match SCAD's total ratio."""
    P, RD = cfg["paths"]["processed"], cfg["paths"]["reference"]
    # compiled SCAD workbook; its March 2026 figures were checked cell by cell against the original release (271 fields)
    t4 = pd.read_excel(cfg["paths"]["external"] / "scad" / "SCAD_Hotel_Master_Database_2022_2026.xlsx", sheet_name="T4_Guests_By_Nationality")
    t4["date"] = pd.to_datetime(t4.date)
    T = t4.pivot(index="date", columns="nationality_region", values="guests_thousands")
    ratio = T.loc["2026-03-01"] / T.loc["2025-03-01"]
    reg = pd.read_csv(RD / "scad_region_map.csv").set_index("nationality").scad_region
    hi = pd.read_parquet(P / "hotel_intl_daily.parquet"); hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    a25 = pd.concat([hi[hi.date.dt.strftime("%Y-%m") == "2025-03"].groupby("nationality").arrivals.sum(),
                     pd.Series({"DOMESTIC": hd[hd.date.dt.strftime("%Y-%m") == "2025-03"].arrivals.sum()})])
    b = pd.read_parquet(P / "phase3" / "baseline_checkins_daily.parquet")
    b26 = b[b.date.dt.strftime("%Y-%m") == "2026-03"].groupby("series").checkins.sum()
    # the region's change relative to the normal world (same region total as SCAD's ratio), shared by its markets,
    # so each market keeps its own normal-world dynamics (e.g. the Wizz Air exit) under the regional shock
    intl = [n for n in a25.index if n != "DOMESTIC"]
    by_reg = pd.DataFrame({"r": reg.reindex(intl), "a25": a25[intl], "b26": b26.reindex(intl)})
    war = (ratio[by_reg.groupby("r").a25.sum().index] * by_reg.groupby("r").a25.sum() / by_reg.groupby("r").b26.sum())
    lvl = {n: b26[n] * war[reg[n]] for n in intl}
    lvl["DOMESTIC"] = a25.sum() * ratio["Total"] - sum(lvl.values())
    mult = {n: float(lvl[n] / b26[n]) for n in lvl}
    changes = [{"type": "demand", "markets": n, "start": "2026-03-01", "end": "2026-03-31", "pct": round((m - 1) * 100, 2),
                "scad_region": reg.get(n) if n != "DOMESTIC" else "remainder to SCAD total"} for n, m in sorted(mult.items())]
    preset = {"name": "march_2026_observed", "title": "March 2026 as observed (all causes: flight cuts and demand), relative to the normal world",
              "method": "each SCAD region's Mar 2026/Mar 2025 guest ratio, divided by the normal world's ratio for the same "
                        "markets, applied to every market of the region; Domestic = remainder to SCAD's total ratio",
              "kind": "observed level", "changes": changes,
              "notes": ["Source: SCAD Hotel Establishments Statistics, March 2026 release (Table 4, guests by world region), "
                        "vs March 2025; data marked preliminary.",
                        "SCAD regions mix visitors and resident expats of that nationality; our nationality series are visitors "
                        "(residents are in Domestic), so long-haul visitor markets may have fallen more than their region.",
                        "Domestic is the remainder that makes the total match SCAD's total ratio (SCAD 'UAE' = Emirati nationals only).",
                        "Use for March 2026 only; no market-level data are published after March."]}
    check = {"scad_ratio_by_region": ratio.round(4).to_dict(), "region_vs_normal": war.round(4).to_dict(),
             "pack_total_mar26_over_mar25": float(sum(lvl.values()) / a25.sum()),
             "normal_world_mar26_over_mar25": float(b26.sum() / a25.sum()),
             "all_causes_vs_normal_total": float(sum(lvl.values()) / b26.sum()),
             "domestic_vs_mar25": float(lvl["DOMESTIC"] / a25["DOMESTIC"])}
    return preset, check


def published_anchors(cfg) -> dict:
    """Published 2026 figures after March, checked against the normal world (context; not used as inputs)."""
    P = cfg["paths"]["processed"]
    hi = pd.read_parquet(P / "hotel_intl_daily.parquet"); hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    act = pd.concat([hi[["nationality", "date", "arrivals"]].rename(columns={"nationality": "m"}), hd[["date", "arrivals"]].assign(m="DOMESTIC")])
    act["mo"] = act.date.dt.to_period("M"); A = act.groupby(["m", "mo"]).arrivals.sum().unstack()
    b = pd.read_parquet(P / "phase3" / "baseline_checkins_daily.parquet"); b["mo"] = b.date.dt.to_period("M")
    B = b.groupby(["series", "mo"]).checkins.sum().unstack()
    Pp = pd.Period
    aug_normal = float(B[Pp("2026-08")].sum() / A[Pp("2025-08")].sum() - 1)
    out = {"august_2026_total": {"published_yoy": -0.052, "normal_world_yoy": aug_normal,
                                 "actual_vs_normal": float(0.948 / (1 + aug_normal) - 1),
                                 "source": "Skift 16 Sep 2026 (DCT): 512,000 guests, -5.2%; matches SCAD's Aug 2025 base exactly"}}
    imp = {}
    for n in ["INDIA", "CHINA", "RUSSIAN FEDERATION"]:
        jf = A.loc[n, [Pp("2026-01"), Pp("2026-02")]].sum(); ly = A.loc[n, [Pp(f"2025-0{k}") for k in range(1, 9)]].sum()
        nm = B.loc[n, [Pp(f"2026-0{k}") for k in range(3, 9)]].sum(); ma = A.loc[n, [Pp(f"2025-0{k}") for k in range(3, 9)]].sum()
        m = (2 / 3 * ly - jf) / nm
        imp[n] = {"mar_aug_level_vs_normal_at_most": float(m), "mar_aug_vs_last_year_at_most": float(m * nm / ma - 1)}
    out["india_china_russia_ytd"] = {"statement": "each down more than one-third, Jan-Aug 2026 (Skift 16 Sep 2026, DCT)",
                                     "implied_if_same_basis": imp,
                                     "caveat": "DCT's nationality counts may include residents; our series are visitors"}
    out["occupancy"] = {"h1_2026": 0.668, "june_2026": 0.652, "source": "JLL via The National 11 Aug 2026 (JLL definitions differ from SCAD's)"}
    return out


def step_inputs():
    from . import scenario as sc
    cfg = load_config(); out = _out(cfg); RD = cfg["paths"]["reference"]
    ac = aircraft_seats(cfg); ac.to_csv(out / "aircraft_seats.csv", index=False)
    mar, mcheck = march_2026_observed(cfg)
    (out / "presets").mkdir(exist_ok=True)
    json.dump(mar, open(out / "presets" / "march_2026_observed.json", "w", encoding="utf-8"), indent=1)
    w = sc.World(cfg)
    j = {"aircraft_seats": ac.round(1).to_dict("records"), "march_2026_observed": mcheck, "published_anchors": published_anchors(cfg)}
    # null scenario = the baseline, exactly
    base, fl = sc.resolve({"name": "null", "changes": []}, w)
    bs = pd.read_parquet(out / "baseline_seats_daily.parquet").set_index(["nationality_key", "date"]).seats
    fb = base[base.has_flight_data].set_index(["market", "date"])
    j["null_scenario"] = {"max_seat_gap_vs_baseline": float((fb.seats_base - bs.reindex(fb.index)).abs().max()),
                          "scenario_equals_base": bool(np.allclose(base[["seats_scen", "pax_scen", "p2p_scen"]].to_numpy(),
                                                                   base[["seats_base", "pax_base", "p2p_base"]].to_numpy())),
                          "rows": int(len(base)), "markets": int(base.market.nunique())}
    # presets
    res = {}
    for name, s in [("march_2026_observed", mar), ("etihad_2026_launches", sc.load(RD / "scenarios" / "etihad_2026_launches.json"))]:
        r, flags = sc.resolve(s, w)
        m3 = r[r.date.dt.strftime("%Y-%m") == "2026-03"]
        res[name] = {"flags": flags, "changes": len(s["changes"]),
                     "added_seats": float((r.seats_scen - r.seats_base).sum()), "new_route_seats": float(r.new_route_seats.sum()),
                     "march_checkins_ratio_to_normal": float((m3.checkins_base * m3.demand_mult).sum() / m3.checkins_base.sum())}
        if name == "etihad_2026_launches":
            res[name]["new_seats_by_market"] = r.groupby("market").new_route_seats.sum().loc[lambda x: x > 0].round(0).to_dict()
            uz = r[(r.market == "UZBEKISTAN") & (r.date.dt.month == 8)]
            res[name]["uzbekistan_aug_seats_base_vs_scen"] = [float(uz.seats_base.sum()), float(uz.seats_scen.sum())]
    j["presets"] = res
    json.dump(j, open(out / "step33_inputs.json", "w", encoding="utf-8"), indent=1, default=float)
    # print
    print("AIRCRAFT SEATS (Etihad, from the data): " + "; ".join(f"{r.aircraft_class} {r.seats_per_flight:.0f} (IQR {r.q25:.0f}–{r.q75:.0f}, {r.routes} routes)"
                                                                  for r in ac.itertuples()) + f"; split {ac.split_between[0]}")
    n = j["null_scenario"]; print(f"NULL SCENARIO: {n['rows']:,} market-days, {n['markets']} markets; equals baseline {n['scenario_equals_base']}; "
                                  f"route sum vs baseline seats max gap {n['max_seat_gap_vs_baseline']:.1e}")
    print(f"MARCH 2026 OBSERVED: total {mcheck['pack_total_mar26_over_mar25']:.3f} × Mar 2025 (SCAD total {mcheck['scad_ratio_by_region']['Total']:.3f}); "
          f"normal world {mcheck['normal_world_mar26_over_mar25']:.3f} → all causes {mcheck['all_causes_vs_normal_total'] - 1:+.1%} vs normal; "
          f"Domestic {mcheck['domestic_vs_mar25'] - 1:+.1%} vs Mar 2025")
    print("  by market (vs normal): " + ", ".join(f"{c['markets'].title()} {c['pct']:+.0f}%" for c in sorted(mar["changes"], key=lambda c: c["pct"])))
    pa = j["published_anchors"]
    print(f"AUGUST 2026: published {pa['august_2026_total']['published_yoy']:+.1%}, normal world {pa['august_2026_total']['normal_world_yoy']:+.1%} "
          f"→ actual {pa['august_2026_total']['actual_vs_normal']:+.1%} vs normal (total back to normal by August)")
    print("INDIA/CHINA/RUSSIA YTD > −1/3 implies Mar–Aug at most: " + ", ".join(
        f"{k.title()} {v['mar_aug_level_vs_normal_at_most']:.2f}× normal ({v['mar_aug_vs_last_year_at_most']:+.0%} vs 2025)" for k, v in pa["india_china_russia_ytd"]["implied_if_same_basis"].items()))
    e = res["etihad_2026_launches"]
    print(f"ETIHAD 2026 LAUNCHES: {e['changes']} changes, {e['new_route_seats']:,.0f} seats over the horizon; by market " +
          ", ".join(f"{k.title()} {v:,.0f}" for k, v in e["new_seats_by_market"].items()))
    print(f"  Uzbekistan August seats: baseline {e['uzbekistan_aug_seats_base_vs_scen'][0]:,.0f} → with Tashkent {e['uzbekistan_aug_seats_base_vs_scen'][1]:,.0f}")
    for f in e["flags"]:
        print("  flag:", f)


# ----------------------------------------------------------------------------- 3.4 engine
DEMOS = {
    "network +10% seats, Nov 2026–Feb 2027": [{"type": "route", "markets": "flight markets", "start": "2026-11-01", "end": "2027-02-28", "seats_pct": 10}],
    "China: all flights cancelled, Dec 2026": [{"type": "route", "markets": "CHINA", "start": "2026-12-01", "end": "2026-12-31", "seats_pct": -100}],
    "India: transfer share −10 points, Oct–Dec 2026": [{"type": "transfer_share", "markets": "INDIA", "start": "2026-10-01", "end": "2026-12-31", "points": -10}],
    "Domestic stays +10%, Jul–Aug 2026": [{"type": "stay_length", "markets": "DOMESTIC", "start": "2026-07-01", "end": "2026-08-31", "pct": 10}],
}


def step_engine():
    from . import engine as en, scenario as sc, stagea as sa
    cfg = load_config(); out = _out(cfg); RD = cfg["paths"]["reference"]
    E = en.Engine(cfg); P = E.params; w = E.world
    mk = lambda r, m: r.daily[r.daily.market == m]
    chk = {}
    # 1. Stage B from the persisted parameters reproduces the Step 3.2 baseline guests
    gb = w.guests.set_index(["market", "date"]).guests
    chk["stageb_reproduces_baseline_max_abs"] = float(max(
        np.abs(E.stageb.guests(m, g.sort_values("date").checkins.to_numpy())[:E.stageb.n_h] - gb.loc[m].reindex(g.sort_values("date").date).to_numpy()).max()
        for m, g in w.checkins.groupby("series")))
    # 2. empty scenario = no change at all
    r0 = E.run({"changes": []})
    chk["null_max_abs_change"] = float(max(abs(v) for k, v in r0.summary.items() if k.startswith("d_")))
    # 3. the lever is exactly the Phase 2 seat lever when only seats change
    r = E.run({"changes": [{"type": "route", "market": "UNITED KINGDOM", "start": "2026-11-01", "end": "2026-12-31", "seats_pct": 20}]})
    d = mk(r, "UNITED KINGDOM"); d = d[d.date.dt.month.isin([11, 12])]
    ref, _ = sa.seat_lever(d.checkins_base.to_numpy(), d.seats_base.to_numpy(), 0.2 * d.seats_base.to_numpy(), P.eps,
                           (d.p2p_base / d.seats_base).to_numpy(), (P.lo, P.hi))
    chk["lever_equals_phase2_seat_lever_max_abs"] = float(np.abs(d.d_checkins_flights.to_numpy() - ref).max())
    # 4. full cancellation keeps (1 + g(−1)) unless the cap binds; cancel-and-restore from a zero-seat month is symmetric
    G0 = float(en.lever_G(0.0, P.eps, P.lo, P.hi))
    keeps = {}
    for m in ["UNITED KINGDOM", "INDIA", "CHINA"]:
        r = E.run({"changes": [{"type": "route", "market": m, "start": "2026-11-01", "end": "2026-11-30", "seats_pct": -100}]})
        d = mk(r, m); d = d[d.date.dt.month == 11]
        keeps[m] = float(d.checkins_scen.sum() / d.checkins_base.sum())
    chk["full_cancellation_keeps"] = {"rule": G0, **keeps}
    rb = w.routes; uz = rb[(rb.nationality_key == "UZBEKISTAN") & rb.moy.isin([6, 9])]
    seats_ref = uz.seats.sum() / 2
    r = E.run({"changes": [{"type": "new_route", "market": "UZBEKISTAN", "city": "Tashkent", "airline": "Air Arabia Abu Dhabi",
                            "start": "2026-07-01", "end": "2026-07-31", "flights_per_week": seats_ref * 7 / 174, "seats_per_flight": 174}]})
    d = mk(r, "UZBEKISTAN"); d = d[d.date.dt.month == 7]
    chk["zero_seat_restore"] = {"scenario_over_base": float(d.checkins_scen.sum() / d.checkins_base.sum()), "expected_1_over_G0": 1 / G0,
                                "local_scen_over_reference": float((d.p2p_scen / E.ref_local[("UZBEKISTAN", 7)]).mean())}
    # 5. a ten-day change moves ten days; guests follow for the length of those stays
    r = E.run({"changes": [{"type": "route", "market": "INDIA", "start": "2026-11-10", "end": "2026-11-19", "seats_pct": -50}]})
    d = mk(r, "INDIA"); ch = d[d.d_checkins_flights != 0]; gd = d[d.d_guest_nights != 0]
    chk["ten_day_change"] = {"checkin_days": int(len(ch)), "first": str(ch.date.min().date()), "last": str(ch.date.max().date()),
                             "guest_effect_until": str(gd.date.max().date())}
    # 6. stay lever is exact for the cohorts it targets; markets are independent; demand is proportional
    r = E.run({"changes": [{"type": "stay_length", "markets": "DOMESTIC", "start": "2026-12-01", "end": "2026-12-31", "pct": 10}]})
    d = mk(r, "DOMESTIC"); dec = (d.date.dt.month == 12) & (d.date.dt.year == 2026)
    coh = E.stageb.guests("DOMESTIC", np.where(dec, d.checkins_base, 0.0), memory=False).sum()
    chk["stay_plus_10pct_realised_pct"] = float(r.summary["d_guest_nights"] / coh * 100)
    a = E.run({"changes": [DEMOS["China: all flights cancelled, Dec 2026"][0]]}).summary
    b = E.run({"changes": [{"type": "route", "market": "GERMANY", "start": "2026-06-01", "end": "2026-08-31", "seats_pct": -10}]}).summary
    c = E.run({"changes": [DEMOS["China: all flights cancelled, Dec 2026"][0],
                           {"type": "route", "market": "GERMANY", "start": "2026-06-01", "end": "2026-08-31", "seats_pct": -10}]}).summary
    chk["additivity_max_abs"] = float(max(abs(a[k] + b[k] - c[k]) for k in ("d_checkins", "d_guest_nights")))
    r = E.run({"changes": [{"type": "demand", "markets": "Europe", "start": "2026-11-01", "end": "2026-11-30", "pct": -20}]})
    eu = r.daily[r.daily.market.isin(w.regions["Europe"]) & (r.daily.date.dt.month == 11)]
    chk["demand_minus_20pct_ratio"] = float(eu.checkins_scen.sum() / eu.checkins_base.sum())
    # demos: sourced preset + four illustrations
    demos = {"Etihad 2026 launches (sourced)": sc.load(RD / "scenarios" / "etihad_2026_launches.json")}
    demos.update({k: {"name": k, "changes": v} for k, v in DEMOS.items()})
    res = {}
    (out / "engine_demos").mkdir(exist_ok=True)
    for name, s in demos.items():
        r = E.run(s)
        res[name] = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.summary.items()}, "flags": r.flags}
        slug = "".join(ch_ if ch_.isalnum() else "_" for ch_ in name.lower())[:48].strip("_")
        r.monthly().loc[lambda x: (x.d_checkins != 0) | (x.d_guest_nights != 0)].round(1).to_csv(out / "engine_demos" / f"{slug}.csv")
    j = {"params": {"eps": P.eps, "trusted_range": [P.lo, P.hi], "new_route_checkins_per_seat": P.per_seat,
                    "eps_band": list(P.eps_band), "per_seat_band": list(P.per_seat_band), "peak_guests_observed": P.peak_guests},
         "checks": chk, "demos": res}
    json.dump(j, open(out / "step34_engine.json", "w", encoding="utf-8"), indent=1, default=float)
    # print
    print(f"ENGINE ε {P.eps:.3f}, trusted range {P.lo:+.0%} … {P.hi:+.0%}, new-route {P.per_seat:.3f} check-ins/seat, capacity flag above {P.peak_guests:,.0f} guests")
    print(f"CHECKS Stage B reproduces baseline (max {chk['stageb_reproduces_baseline_max_abs']:.1e}); empty scenario changes nothing "
          f"(max {chk['null_max_abs_change']:.1e}); lever = Phase 2 seat lever (max {chk['lever_equals_phase2_seat_lever_max_abs']:.1e})")
    k = chk["full_cancellation_keeps"]; print("  full cancellation keeps " + ", ".join(f"{m.title()} {v:.3f}" for m, v in k.items() if m != "rule") + f" (rule {k['rule']:.3f})")
    z = chk["zero_seat_restore"]; print(f"  Uzbekistan July restore at reference seats: ×{z['scenario_over_base']:.3f} (symmetric value ×{z['expected_1_over_G0']:.3f}; local {z['local_scen_over_reference']:.3f} of reference)")
    t = chk["ten_day_change"]; print(f"  10-day change: check-ins move on {t['checkin_days']} days ({t['first']}…{t['last']}); guests until {t['guest_effect_until']}")
    print(f"  stay +10% → cohort guest nights {chk['stay_plus_10pct_realised_pct']:+.3f}%; additivity gap {chk['additivity_max_abs']:.1e}; demand −20% → ×{chk['demand_minus_20pct_ratio']:.4f}")
    print("DEMOS (Δ over the horizon; guest nights include stays ending after it)")
    for n, v in res.items():
        cap = v["checkins_per_extra_local_passenger"]; npc = v["guest_nights_per_extra_checkin"]
        print(f"  {n}: seats {v['d_seats']:+,.0f}, local passengers {v['d_local_passengers']:+,.0f} → check-ins {v['d_checkins']:+,.0f} "
              f"({v['pct_checkins']:+.2f}%), guest nights {v['d_guest_nights']:+,.0f} ({v['pct_guest_nights']:+.2f}%)"
              + (f"; {cap:.3f} check-ins per extra local passenger" if cap else "") + (f"; {npc:.2f} nights per extra check-in" if npc else "")
              + f"; {len(v['flags'])} flags")


# ----------------------------------------------------------------------------- 3.5 bands
def step_bands():
    from . import bands as bd, engine as en, scenario as sc
    cfg = load_config(); out = _out(cfg); RD = cfg["paths"]["reference"]
    eb = bd.effect_band(cfg)
    json.dump(eb, open(out / "effect_band.json", "w", encoding="utf-8"), indent=1)
    lv, ld = bd.calibrate_level(cfg)
    lv.to_csv(out / "level_bands.csv", index=False)
    pd.Series(ld["tiers"], name="tier").rename_axis("market").to_csv(out / "market_tiers.csv")
    B = bd.Bands(cfg); E = en.Engine(cfg)
    # level band on the normal-world baseline: per market and month, and the all-market monthly total
    base = E.run({"changes": []})
    m = base.monthly()[["checkins_base", "guests_base"]]
    bm = B.level_monthly(m)
    bm.round(1).to_csv(out / "baseline_level_monthly.csv")
    tot = B.level_total(base.monthly(by_market=False)[["checkins_base", "guests_base"]])
    tot.round(1).to_csv(out / "baseline_level_total.csv")
    # effect band on the demos
    demos = {"Etihad 2026 launches (sourced)": sc.load(RD / "scenarios" / "etihad_2026_launches.json")}
    demos.update({k: {"name": k, "changes": v} for k, v in DEMOS.items()})
    res = {}
    for name, s in demos.items():
        r = B.run(E, s)
        eb_ = r.effect_band["summary"]
        res[name] = {"d_checkins": r.summary["d_checkins"], "d_checkins_band": eb_["d_checkins"],
                     "d_guest_nights": r.summary["d_guest_nights"], "d_guest_nights_band": eb_["d_guest_nights"],
                     "parts": r.effect_band["parts"], "note": r.effect_band["note"],
                     "months_beyond_7": r.effect_band["months_beyond_7"], "share_beyond_7_months": r.effect_band["share_beyond_7_months"]}
    # checks: the effect grows with ε (seat increase), assumption levers carry no effect band
    up = {"changes": [{"type": "route", "market": "INDIA", "start": "2026-10-01", "end": "2026-12-31", "seats_pct": 20}]}
    lo_, mid_, hi_ = (E.run(up, eps=e).summary["d_checkins"] for e in (eb["eps_lo"], eb["eps"], eb["eps_hi"]))
    chk = {"effect_increases_with_eps": bool(lo_ < mid_ < hi_),
           "assumption_only_lever_part": res["Domestic stays +10%, Jul–Aug 2026"]["parts"]["d_guest_nights"]["lever"],
           "level_band_months_calibrated": int(tot.level_band_calibrated.sum()), "level_band_months_flagged": int((~tot.level_band_calibrated).sum())}
    j = {"effect_band": eb, "level_bands": lv.round(4).to_dict("records"), "level_diagnostics": ld["diagnostics"],
         "tier_counts": pd.Series(ld["tiers"]).value_counts().to_dict(), "checks": chk, "demos": res}
    json.dump(j, open(out / "step35_bands.json", "w", encoding="utf-8"), indent=1, default=float)
    # print
    print(f"EFFECT BAND ε {eb['eps']:.3f} → 80% [{eb['eps_lo']:.3f}, {eb['eps_hi']:.3f}] from the official test window (slope "
          f"{eb['source']['test_window_slope']:.2f}, s.e. {eb['source']['slope_se_cluster']:.2f}, {eb['source']['n_countries']} countries); "
          f"cross-checks: train bootstrap 80% [{eb['cross_checks']['bootstrap_80pct_train_window'][0]:.3f}, {eb['cross_checks']['bootstrap_80pct_train_window'][1]:.3f}], "
          f"Phase 2 band {[round(x, 3) for x in eb['cross_checks']['phase2_simulator_band']]}; new-route per seat [{eb['per_seat_lo']:.3f}, {eb['per_seat_hi']:.3f}]")
    for kind, d in ld["diagnostics"].items():
        tc = [v for x in d["coverage_by_tier"].values() for v in x.values()]
        lead = [v for x in d["median_abs_log_error_by_months_ahead"].values() for v in x.values()]
        print(f"LEVEL BAND {kind}: q {d['q_market_month']:.3f}, cross-fold coverage {d['coverage_A_to_B']:.0%} / {d['coverage_B_to_A']:.0%} "
              f"(per tier {min(tc):.0%}–{max(tc):.0%}; median error by month ahead {min(lead):.2f}–{max(lead):.2f}, no growth to month 7) "
              f"(each market's own band would cover only {d['own_market_bands_coverage_A_to_B']:.0%} / {d['own_market_bands_coverage_B_to_A']:.0%}); "
              f"monthly total: q {d['q_total']:.3f}, cross-fold coverage {d['total_coverage_A_to_B']:.0%} / {d['total_coverage_B_to_A']:.0%}")
    print(lv.round(3).to_string(index=False))
    print(f"  tiers: {j['tier_counts']}; baseline months calibrated {chk['level_band_months_calibrated']}, flagged beyond 7 months {chk['level_band_months_flagged']}")
    g = tot.loc["2026-12"]
    print(f"  e.g. Dec 2026 all-market guest nights {g.guests_base:,.0f} (80%: {g.guests_base_lo:,.0f} – {g.guests_base_hi:,.0f}; flagged {not g.level_band_calibrated})")
    print(f"CHECKS effect grows with ε: {chk['effect_increases_with_eps']}; assumption-only lever part {chk['assumption_only_lever_part']}")
    print("DEMOS (change, effect band)")
    for n, v in res.items():
        print(f"  {n}: check-ins {v['d_checkins']:+,.0f} [{v['d_checkins_band'][0]:+,.0f}, {v['d_checkins_band'][1]:+,.0f}]; "
              f"guest nights {v['d_guest_nights']:+,.0f} [{v['d_guest_nights_band'][0]:+,.0f}, {v['d_guest_nights_band'][1]:+,.0f}]"
              + (f"; {v['share_beyond_7_months']:.0%} of it beyond 7 months (flagged)" if v["months_beyond_7"] else ""))


# ----------------------------------------------------------------------------- 3.6 sensitivity
def local_passenger_evidence(cfg) -> dict:
    """Do check-ins respond to local passengers per seat (load factor × local share) as they do to seats? Same training
    panel and trusted range as ε (Step 2.2), plain specification with both terms."""
    from . import phase2 as p2
    fd, _, hi = p2._load(cfg)
    m, _, _ = p2._panel_with_festivals(cfg, fd, hi)
    p, _ = p2._primary(m, cfg["folds"]["final"]["train_end"])
    e = sa.fe_elasticity(p.assign(d_share=p.d_p2p - p.d_seats), "d_checkins", ["d_seats", "d_share"] + p2.FEST, n_boot=0)
    eb = json.load(open(_out(cfg) / "effect_band.json", encoding="utf-8"))
    return {"seats": [e["estimate"], e["ci_cluster"]], "local_per_seat": [e["other_coefs"]["d_share"], e["other_ci_cluster"]["d_share"]],
            "ratio": e["other_coefs"]["d_share"] / e["estimate"], "band_low_ratio": eb["eps_lo"] / eb["eps"], "n_obs": e["n_obs"]}


QUERIES = [("INDIA", "2026-12-01", "2026-12-31", 10_000), ("UNITED KINGDOM", "2026-11-01", "2026-11-30", 5_000),
           ("CHINA", "2026-12-01", "2026-12-31", 5_000)]


def step_sensitivity():
    from . import bands as bd, engine as en, sensitivity as sn
    cfg = load_config(); out = _out(cfg)
    E = en.Engine(cfg); B = bd.Bands(cfg)
    spf = float(pd.read_csv(out / "aircraft_seats.csv").set_index("aircraft_class").seats_per_flight["narrowbody"])
    base = E.run({"changes": []})
    gn = base.daily.groupby("market").guests_base.sum()
    share = {"flight markets": float(gn[E.world.flight].sum() / gn.sum()),
             "nationalities without flight data": float(gn.drop(E.world.flight + ["DOMESTIC"]).sum() / gn.sum()),
             "domestic": float(gn["DOMESTIC"] / gn.sum())}
    lv = sn.lever_ranking(E, B); lv.to_csv(out / "sens_levers.csv", index=False)
    ms = sn.market_season(E, B, spf); ms.to_csv(out / "sens_market_month.csv", index=False)
    per = ms.groupby("market").apply(lambda g: pd.Series({"guest_nights_per_1000_seats": g.d_guest_nights.sum() / g.seats_added.sum() * 1000,
                                                          "lo": g.lo.sum() / g.seats_added.sum() * 1000, "hi": g.hi.sum() / g.seats_added.sum() * 1000,
                                                          "rule": g.rule.mode().iloc[0],
                                                          "other_months": "; ".join(f"{k} ×{v}" for k, v in g.rule.value_counts().items()
                                                                                    if k != g.rule.mode().iloc[0])}),
                                   include_groups=False).sort_values("guest_nights_per_1000_seats", ascending=False)
    per.to_csv(out / "sens_market_ranking.csv")
    season = ms[ms.rule != "new-route lever (no flight data)"].groupby("month").apply(
        lambda g: g.d_guest_nights.sum() / g.seats_added.sum() * 1000, include_groups=False).rename("guest_nights_per_1000_seats")
    season.to_csv(out / "sens_season.csv")
    q = [sn.seats_needed(E, B, m, a, b, t, spf) for m, a, b, t in QUERIES]
    boot = json.load(open(out / "effect_band.json", encoding="utf-8"))["cross_checks"]["bootstrap_80pct_train_window"]
    ref = {"name": "network +10% seats, Nov 2026–Feb 2027", "changes": DEMOS["network +10% seats, Nov 2026–Feb 2027"]}
    tor = sn.uncertainty_tornado(E, B, ref, boot)
    ev = local_passenger_evidence(cfg)
    rank = lv.groupby("lever").elasticity_small_change.apply(lambda x: float(x.abs().mean()))
    # rank on 6 decimals, ties by name: levers equal to rounding (demand and stay length, both 0.999…) keep a fixed order
    rank = rank.loc[sorted(rank.index, key=lambda k: (-round(rank[k], 6), k))]
    j = {"guest_night_shares": share, "seats_per_flight": spf, "local_passenger_evidence": ev, "levers": lv.round(4).to_dict("records"),
         "lever_ranking_by_small_change_elasticity": rank.round(4).to_dict(),
         "market_ranking": per.round(2).reset_index().to_dict("records"), "season": season.round(2).to_dict(),
         "seats_needed": q, "uncertainty_tornado": tor}
    json.dump(j, open(out / "step36_sensitivity.json", "w", encoding="utf-8"), indent=1, default=float)
    # print
    print("GUEST NIGHTS BY SOURCE: " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
    print("1. LEVERS ±10% over the year (guest nights, 80% effect band, elasticity at ±10% and at ±1%)")
    for r in lv.itertuples():
        print(f"  {r.lever:17s} {r.change_pct:+.0f}%: local passengers {r.local_passengers_pct:+5.1f}% → guest nights {r.d_guest_nights:+11,.0f} "
              f"[{r.lo:+,.0f}, {r.hi:+,.0f}] ({r.guest_nights_pct:+.2f}%), elasticity {r.elasticity:.3f} (±1%: {r.elasticity_small_change:.3f})"
              + (f"; {r.pct_of_checkin_change_beyond_trusted_range:.0f}% of the check-in change beyond the trusted range" if abs(r.pct_of_checkin_change_beyond_trusted_range) >= 1 else "")
              + ("; capped at full planes" if "full planes" in str(r.capped_or_flagged) else ""))
    print("  ranking (|elasticity| at ±1%): " + ", ".join(f"{k} {v:.3f}" for k, v in rank.items()))
    print(f"  evidence for the load-factor / transfer-share channel (training panel, plain spec): seats {ev['seats'][0]:.3f} "
          f"{[round(x, 3) for x in ev['seats'][1]]}, local passengers per seat {ev['local_per_seat'][0]:.3f} {[round(x, 3) for x in ev['local_per_seat'][1]]} "
          f"(ratio {ev['ratio']:.2f}; the effect band's low end is {ev['band_low_ratio']:.2f}× the central ε)")
    print(f"2. ONE EXTRA DAILY NARROWBODY ({spf:.0f} seats) — guest nights per 1,000 added seats, year average (80% band)")
    for m, r in per.head(12).iterrows():
        print(f"  {m.title():26s} {r.guest_nights_per_1000_seats:7.0f} [{r.lo:5.0f}, {r.hi:5.0f}]  {r.rule}" + (f" (other months: {r.other_months})" if r.other_months else ""))
    print("  … lowest: " + ", ".join(f"{m.title()} {r.guest_nights_per_1000_seats:.0f}" for m, r in per.tail(5).iterrows()))
    print("  season (flight markets): " + ", ".join(f"{k} {v:.0f}" for k, v in season.items()))
    print("3. SEATS NEEDED")
    for x in q:
        fr, fl = x["flights_per_week_range"], x["flights_per_week_range_lever_only"]
        f_ = lambda v: f"{v:.1f}" if v else "out of reach"
        print(f"  {x['market'].title()} +{x['target_guest_nights']:,} guest nights {x['window'][0]}…{x['window'][1]}: "
              + (f"{x['flights_per_week']:.1f} flights/week ({x['seats_added']:,.0f} seats); 80% range {f_(fr[0])}–{f_(fr[1])} "
                 f"(lever only {f_(fl[0])}–{f_(fl[1])})" if x["flights_per_week"] else "out of reach"))
    t = tor
    print(f"4. UNCERTAINTY for '{ref['name']}': central {t['central']:+,.0f} guest nights; lever (test-window ε) "
          f"{t['lever_test_window_eps'][0]:+,.0f}…{t['lever_test_window_eps'][1]:+,.0f}; lever (training bootstrap ε, cross-check) "
          f"{t['lever_training_bootstrap_eps'][0]:+,.0f}…{t['lever_training_bootstrap_eps'][1]:+,.0f}; baseline level "
          f"{t['baseline_level'][0]:+,.0f}…{t['baseline_level'][1]:+,.0f}; combined {t['combined'][0]:+,.0f}…{t['combined'][1]:+,.0f}")


# ----------------------------------------------------------------------------- 3.7 validation
def step_validate():
    from . import bands as bd, engine as en, validation as va
    cfg = load_config(); out = _out(cfg); RD = cfg["paths"]["reference"]
    E = en.Engine(cfg); B = bd.Bands(cfg)
    # 1. historical replay beyond the trusted range
    rep, rows = va.beyond_range_replay(cfg, B.effect)
    rows[rows.side != "inside"][["nationality_key", "month", "window", "side", "rel_d_seats", "rel_d_p2p", "rel_d_checkins_adj",
                                 "line", "log", "band_a", "band_b"]].round(4).to_csv(out / "replay_beyond_range.csv", index=False, date_format="%Y-%m")
    g = {(x["window"][:4], x["side"], x["excluding_war_months"]): x for x in rep["groups"]}
    tc, tn = g[("test", "cut", False)], g[("trai", "cut", True)]
    rep_checks = {"test_cuts_log_closer_than_line": tc["log"]["mae_log"] < tc["line"]["mae_log"],
                  "test_cuts_mean_inside_widened_band": tc["lever_band_pct"][0] <= tc["actual_mean_pct"] <= tc["lever_band_pct"][1],
                  "test_cuts_mean_inside_line_only_band": tc["lever_band_line_only_pct"][0] <= tc["actual_mean_pct"] <= tc["lever_band_line_only_pct"][1],
                  "training_non_war_cuts_mean_inside_band": tn["lever_band_pct"][0] <= tn["actual_mean_pct"] <= tn["lever_band_pct"][1]}
    # 2. engine properties on random scenarios
    props = va.properties(E, B, n=200, n_pairs=40, n_bands=30, seed=37)
    # 3. March 2026 replay, flights only vs SCAD
    sched = pd.read_csv(RD / "scenarios" / "march_2026_etihad_schedule.csv")
    obs = json.load(open(out / "presets" / "march_2026_observed.json", encoding="utf-8"))
    mr = va.march_replay(E, B, cfg, sched, obs)
    T = mr["by_region"]
    T.assign(**{c: T[c].map(lambda v: [round(x, 4) for x in v]) for c in T.columns if c.endswith("range") or c.startswith("share")}) \
        .round(4).to_csv(out / "march_2026_replay_by_region.csv", index=False)
    for k in ("low", "high"):
        json.dump(mr["scenarios"][k] | {"name": f"march_2026_flights_{k}", "notes": [inspect.cleandoc(va.march_scenarios.__doc__).split("\n\n")[0]]},
                  open(out / "presets" / f"march_2026_flights_{k}.json", "w", encoding="utf-8"), indent=1)
        json.dump(mr["net_of_flights"][k], open(out / "presets" / f"march_2026_demand_net_of_flights_{k}.json", "w", encoding="utf-8"), indent=1)
    # flights + net-of-flights demand must reproduce SCAD's March by region (and the scenario layer must not flag it)
    regions = pd.read_csv(RD / "scad_region_map.csv").set_index("nationality").scad_region
    rep_ok = {}
    for k in ("low", "high"):
        r = E.run({"changes": mr["scenarios"][k]["changes"] + mr["net_of_flights"][k]["changes"]})
        x = r.daily[r.daily.date.dt.strftime("%Y-%m") == "2026-03"].groupby("market")[["checkins_base", "checkins_scen"]].sum()
        x["region"] = [regions.get(m) if m != "DOMESTIC" else "UAE residents (Domestic)" for m in x.index]
        gx = x.groupby("region").sum(); want = T.set_index("region").observed_vs_normal
        rep_ok[k] = {"max_abs_gap": float((gx.checkins_scen / gx.checkins_base - want.reindex(gx.index)).abs().max()),
                     "double_count_flag": any("counted twice" in f for f in r.flags)}
    m_checks = {"flights_plus_net_demand_reproduce_scad": all(v["max_abs_gap"] < 1e-4 for v in rep_ok.values()),   # pct rounded to 0.01 points
                "no_double_count_flag": not any(v["double_count_flag"] for v in rep_ok.values()),
                "flights_never_over_explain_beyond_lever_band": bool(all(
                    row.observed_vs_normal <= max(row.flights_only_low_seats_lever_range[1], row.flights_only_high_seats_lever_range[1]) + 1e-9
                    for row in T.itertuples())),
                "regions_where_one_for_one_over_explains": [row.region for row in T.itertuples()
                                                            if min(row.one_for_one_low_seats, row.one_for_one_high_seats) < row.observed_vs_normal]}
    j = {"beyond_range_replay": rep, "beyond_range_checks": rep_checks, "properties": props,
         "march_2026": {"seats_vs_normal": mr["march_seats_vs_normal"], "total": mr["total"], "by_region": T.to_dict("records"),
                        "checks": m_checks, "reproduction": rep_ok, "etihad_network_share_restored":
                            {f"{b} {p}": v["etihad_network"] for (b, p), v in mr["etihad_ratios"].items()},
                        "other_airlines_suspended_all_march": va.SUSPENDED_ALL_MARCH}}
    json.dump(j, open(out / "step37_validation.json", "w", encoding="utf-8"), indent=1, default=float)
    # print
    print(f"1. BEYOND THE TRUSTED RANGE ({rep['trusted_range_pct'][0]:+.0%} … {rep['trusted_range_pct'][1]:+.0%}), relative to other countries, festival-adjusted:")
    for x in rep["groups"]:
        print(f"  {x['window'][:5]} {x['side']:8s}{' excl. war months' if x['excluding_war_months'] else '':17s} n={x['n']:2d} ({x['countries']} countries): "
              f"actual {x['actual_mean_pct']:+4.0f}% | straight line {x['line']['pred_mean_pct']:+4.0f}% (MAE {x['line']['mae_log']:.3f}, slope {x['line']['slope']:.2f} ± {x['line']['slope_se']:.2f}) "
              f"| log form {x['log']['pred_mean_pct']:+4.0f}% (MAE {x['log']['mae_log']:.3f}, slope {x['log']['slope']:.2f} ± {x['log']['slope_se']:.2f}) "
              f"| lever band {x['lever_band_pct'][0]:+.0f}…{x['lever_band_pct'][1]:+.0f}% (straight line only {x['lever_band_line_only_pct'][0]:+.0f}…{x['lever_band_line_only_pct'][1]:+.0f}%)")
    print("  test-window cuts by country (Wizz Air Abu Dhabi exit, 1 Sep 2025, dominates):")
    for x in rep["test_window_cuts_by_country"]:
        print(f"    {x['nationality_key'].title():11s} {x['first']}…{x['last']}: local passengers {x['local_passengers']:+.0f}% → actual {x['actual']:+.0f}%, "
              f"line {x['line']:+.0f}%, log {x['log']:+.0f}%, band {x['band_b']:+.0f}…{x['band_a']:+.0f}% ({'inside' if x['inside_lever_band'] else 'outside'})")
    s_ = rep["inside_range_test_slope_local_passengers"]
    print(f"  inside the range, test window via local passengers: slope {s_['slope']:.2f} ± {s_['slope_se']:.2f} ({s_['n']} rows, {s_['countries']} countries)")
    print(f"  checks: {rep_checks}")
    print(f"2. PROPERTIES on {props['scenarios']} random scenarios ({props['refused_by_validation']} refused by validation): all passed {props['all_passed']}")
    print("   " + "; ".join(f"{k} {v}" for k, v in props["passed"].items()) + f"; bands ordered {props['band_ordered']}; market additivity {props['market_additivity']}")
    print(f"3. MARCH 2026 REPLAY: seats vs normal {mr['march_seats_vs_normal']['low']:.0%}–{mr['march_seats_vs_normal']['high']:.0%}; "
          f"check-ins observed {mr['total']['observed_vs_normal'] - 1:+.0%} vs normal; flights only {mr['total']['flights_only_low_seats'] - 1:+.0%} … "
          f"{mr['total']['flights_only_high_seats'] - 1:+.0%} (one-for-one {mr['total']['one_for_one_low_seats'] - 1:+.0%} … {mr['total']['one_for_one_high_seats'] - 1:+.0%}); "
          f"flights explain {mr['total']['share_of_drop_explained_by_flights'][0]:.0%}–{mr['total']['share_of_drop_explained_by_flights'][1]:.0%} of the drop "
          f"({mr['total']['share_of_drop_explained_by_flights_lever_range'][0]:.0%}–{mr['total']['share_of_drop_explained_by_flights_lever_range'][1]:.0%} across the lever's range)")
    for row in T.itertuples():
        print(f"   {row.region:34s} observed {row.observed_vs_normal - 1:+5.0%} | flights only {row.flights_only_low_seats - 1:+4.0%} … {row.flights_only_high_seats - 1:+4.0%} "
              f"(lever {min(row.flights_only_low_seats_lever_range[0], row.flights_only_high_seats_lever_range[0]) - 1:+.0%} … "
              f"{max(row.flights_only_low_seats_lever_range[1], row.flights_only_high_seats_lever_range[1]) - 1:+.0%}) | demand residual "
              f"×{min(row.residual_demand_low_seats, row.residual_demand_high_seats):.2f}–{max(row.residual_demand_low_seats, row.residual_demand_high_seats):.2f}")
    print(f"   checks: {m_checks}")


# ----------------------------------------------------------------------------- 3.8 showcase
RECOVERY = {"window": ("2026-12-01", "2026-12-31"), "shortfall_pct": 10.0, "markets": 6}     # illustrative assumption


def step_showcase():
    from . import bands as bd, engine as en, showcase as sh
    cfg = load_config(); out = _out(cfg); RD = cfg["paths"]["reference"]; PR = out / "presets"
    E = en.Engine(cfg); B = bd.Bands(cfg)
    spf = float(pd.read_csv(out / "aircraft_seats.csv").set_index("aircraft_class").seats_per_flight["narrowbody"])
    # 1. March 2026 replayed (flights + demand net of flights), guest nights checked against SCAD, stays fitted
    m = sh.march_2026(E, cfg)
    for k in ("low", "high"):
        t_ = m["bounds"][k]["total"]
        rep = {"name": f"march_2026_replay_{k}", "kind": "replay", "title": f"March 2026 replayed: flights ({k} seats) + war demand shock net of flights + longer stays",
               "notes": ["Flights: Etihad interim schedules (AeroRoutes 5, 10, 16 Mar 2026); other airlines " + ("none" if k == "low" else "at Etihad's restored share")
                         + "; demand: SCAD guests by region net of flights; stays: one change for March check-ins fitted to SCAD's March guest "
                         f"nights (+{t_['stay_change_fitted_pct']:.1f}%; it also absorbs the 28 Feb check-in spike, which the normal world excludes)."],
               "changes": m["bounds"][k].pop("replay_changes")}
        json.dump(rep, open(PR / f"march_2026_replay_{k}.json", "w", encoding="utf-8"), indent=1)
        m["bounds"][k]["by_region"].round(4).to_csv(out / f"showcase_march_2026_{k}.csv", index=False)
    # 2. Etihad launches route by route
    L = sh.etihad_launches(E, B, RD / "scenarios" / "etihad_2026_launches.json")
    nf = set(E.world.markets) - set(E.world.flight)
    for x in L["routes"]:
        x["lever"] = "new-route lever (no flight data)" if x["market"] in nf else "seat lever"
        if x["market"] in nf:
            x["load_factor"] = x["local_share"] = None
    pd.DataFrame(L["routes"]).drop(columns=["flags"]).to_csv(out / "showcase_etihad_routes.csv", index=False)
    # 3. European carriers' return
    eu = sh.european_return(E, B, RD / "scenarios" / "european_carriers_return.json")
    # 4. recovery planner
    g = E.world.guests
    a, b = RECOVERY["window"]
    top = (g[g.date.between(a, b) & g.market.isin(E.world.flight)].groupby("market").guests.sum()
           .sort_values(ascending=False).index[:RECOVERY["markets"]].tolist())
    rp = sh.recovery_planner(E, B, spf, top, a, b, RECOVERY["shortfall_pct"])
    rr = E.world.routes
    for x in rp:
        cur_seats = float(rr[(rr.nationality_key == x["market"]) & rr.moy.isin(pd.date_range(a, b).month.unique())].groupby("moy").seats.sum().mean() * 7)
        x["current_seats_per_week"] = cur_seats
        x["extra_seats_pct"] = x["flights_per_week_needed"] * spf / cur_seats * 100 if x["flights_per_week_needed"] else None
    pd.DataFrame(rp).to_csv(out / "showcase_recovery_planner.csv", index=False)
    rule = sh.offset_rule({"central": B.effect["eps"], "eps_low": B.effect["eps_lo"], "eps_high": B.effect["eps_hi"]}, E.params.lo, E.params.hi)
    j = {"march_2026": {k: v for k, v in m.items() if k != "bounds"} | {"bounds": {k: {"total": v["total"], "chain_flights": v["chain_flights"],
                                                                                    "by_region": v["by_region"].to_dict("records")} for k, v in m["bounds"].items()}},
         "etihad_launches": L, "european_return": eu, "recovery_planner": {"assumption": RECOVERY, "seats_per_flight": spf, "rows": rp, "offset_rule": rule}}
    json.dump(j, open(out / "step38_showcase.json", "w", encoding="utf-8"), indent=1, default=float)
    # print
    lo_, hi_ = m["bounds"]["low"]["total"], m["bounds"]["high"]["total"]
    print(f"1. MARCH 2026 REPLAY vs the normal world: check-ins {lo_['checkins_replay_vs_normal'] - 1:+.1%} (= SCAD by region, by construction); "
          f"flights alone {lo_['checkins_flights_only_vs_normal'] - 1:+.1%} … {hi_['checkins_flights_only_vs_normal'] - 1:+.1%}")
    fs = m["feb28_spike"]
    print(f"   guest nights: SCAD {lo_['guest_nights_scad_vs_normal'] - 1:+.1%}; replay with normal stays {lo_['guest_nights_replay_vs_normal'] - 1:+.1%} "
          f"(nights per check-in vs normal: replay {lo_['nights_per_checkin_vs_normal_replay_normal_stays'] - 1:+.1%}, SCAD {lo_['nights_per_checkin_vs_normal_scad'] - 1:+.1%}); "
          f"28 Feb spike ({fs['checkins_actual']:,.0f} vs {fs['checkins_normal']:,.0f} check-ins) adds {fs['points_of_normal_march_guest_nights']:.1%} of March nights; "
          f"fitted March stays {lo_['stay_change_fitted_pct']:+.1f}% / {hi_['stay_change_fitted_pct']:+.1f}% → {lo_['guest_nights_with_fitted_stays_vs_normal'] - 1:+.1%}")
    print(f"   guest nights lost (normal stays) {lo_['guest_nights_lost']:,.0f}, of which flights {hi_['guest_nights_lost_flights']:,.0f}–{lo_['guest_nights_lost_flights']:,.0f}")
    print(f"   by region (low seats): " + "; ".join(f"{r.region.split(' (')[0]} SCAD {r.guest_nights_scad_vs_normal - 1:+.0%} vs replay {r.guest_nights_replay_vs_normal - 1:+.0%}"
                                              for r in m["bounds"]["low"]["by_region"].itertuples()))
    t = L["total"]
    print(f"2. ETIHAD 2026 LAUNCHES: seats {t['seats']:+,.0f} → passengers {t['passengers']:+,.0f} (load factor {t['load_factor']:.0%}) → local passengers "
          f"{t['local_passengers']:+,.0f} (local share {t['local_share']:.0%}) → check-ins {t['checkins']:+,.0f} [{t['checkins_band'][0]:+,.0f}, {t['checkins_band'][1]:+,.0f}] "
          f"({t['checkins_per_local_passenger']:.2f} per local passenger) → guest nights {t['guest_nights']:+,.0f} [{t['guest_nights_band'][0]:+,.0f}, {t['guest_nights_band'][1]:+,.0f}]")
    for x in L["routes"]:
        print(f"   {x['route']:28s} {x['market'].title():24s} from {x['start']} {x['flights_per_week']:>4s}/wk: guest nights {x['guest_nights']:+8,.0f} "
              f"[{x['guest_nights_band'][0]:+,.0f}, {x['guest_nights_band'][1]:+,.0f}] ({x['lever']})")
    s_, sl = eu["stay_away"], eu["one_month_slip"]
    print(f"3. EUROPEAN CARRIERS RETURN 24–25 Oct (Wego, provisional): worth {-s_['checkins']:+,.0f} check-ins and {-s_['guest_nights']:+,.0f} guest nights "
          f"[{-s_['guest_nights_band'][1]:+,.0f}, {-s_['guest_nights_band'][0]:+,.0f}] to 28 Feb; a one-month slip costs {sl['guest_nights']:,.0f} guest nights "
          f"[{sl['guest_nights_band'][0]:,.0f}, {sl['guest_nights_band'][1]:,.0f}]; " + "; ".join(f"{k} {-v['guest_nights']:+,.0f}" for k, v in eu["by_carrier"].items()))
    print("4. RECOVERY PLANNER: extra local passengers that offset a demand shortfall of "
          + ", ".join(f"{k}: +{v['central']:.0%} [{v['eps_high']:+.0%}, {v['eps_low']:+.0%}]" for k, v in rule.items())
          + f"; illustrative {RECOVERY['shortfall_pct']:.0f}% shortfall in December:")
    for x in rp:
        print(f"   {x['market'].title():24s} Dec: lose {-x['guest_nights_lost']:,.0f} guest nights; offset = {x['flights_per_week_needed']:.0f} extra narrowbody flights/week "
              f"[{x['range'][0]:.0f}–{x['range'][1]:.0f}] = +{x['extra_seats_pct']:.0f}% seats (now {x['current_flights_per_week']:.0f} flights/week)")


# ----------------------------------------------------------------------------- report
def step_report():
    """reports/phase3_report.md from the saved Phase 3 outputs (no number typed by hand)."""
    cfg = load_config(); out = _out(cfg); P2 = cfg["paths"]["processed"] / "phase2"
    J = {k: json.load(open(out / f"step{k}_{n}.json", encoding="utf-8")) for k, n in (("31", "world"), ("32", "refit"), ("33", "inputs"), ("34", "engine"),
                                                                  ("35", "bands"), ("36", "sensitivity"), ("37", "validation"), ("38", "showcase"))}
    e26 = json.load(open(P2 / "step26_endtoend.json", encoding="utf-8"))
    w, rf, inp, eng, bd_, se, va, sh = (J[k] for k in ("31", "32", "33", "34", "35", "36", "37", "38"))
    eb = bd_["effect_band"]; tw = e26["test_window"]
    rgn = va["march_2026"]["by_region"]
    o11_both = sum(max(r["one_for_one_low_seats"], r["one_for_one_high_seats"]) < r["observed_vs_normal"] for r in rgn if r["flight_markets"])
    o11_any = sum(min(r["one_for_one_low_seats"], r["one_for_one_high_seats"]) < r["observed_vs_normal"] for r in rgn if r["flight_markets"])
    T = lambda m: m.title().replace("Of America", "of America").replace("Russian Federation", "Russia")
    n0 = lambda x: f"{x:,.0f}"; sg = lambda x: f"{x:+,.0f}"; pc = lambda x: f"{x:+.0%}"
    # model estimates (check-ins, guest nights, their ranges) at 3 significant figures; schedule inputs stay exact
    r3 = lambda x: 0.0 if x == 0 else round(x, 2 - int(np.floor(np.log10(abs(x)))))
    e0 = lambda x: f"{r3(x):,.0f}"; es = lambda x: f"{r3(x):+,.0f}"
    band = lambda b: f"{es(b[0])} … {es(b[1])}"
    L = ["# Phase 3 report — scenario engine: flight changes → hotel guests", "",
         "_Generated by `make phase3` from the saved outputs in `data/processed/phase3/`; do not edit by hand. Model estimates "
         "(check-ins, guest nights and their ranges) are rounded to 3 significant figures; schedule inputs (seats, passengers) are exact._", ""]
    # ---------------------------------------------------------------- explainer
    ch = next(x for x in sh["etihad_launches"]["routes"] if x["route"].endswith("Charlotte"))
    L += ["## In plain words", "",
          "You describe a change in flights — a route added or cancelled, more or fewer flights, bigger planes, fuller planes, more "
          "passengers connecting onwards — or a change that has nothing to do with flights (an advisory, a visa rule, an event, longer stays). "
          "The simulator answers: how many more or fewer hotel check-ins and guest nights Abu Dhabi gets, for which nationalities and on "
          "which days, with an honest range. It compares your scenario with a **normal world**: the year March 2026 – February 2027 as it "
          "would have been with last year's schedules and no war (the 2026 war is itself a scenario).", "",
          "**The chain, one step at a time** (every scenario reports each number):", "",
          "1. **Seats** — flights × seats per flight, from each airline's own schedule at Abu Dhabi (33 countries).",
          "2. **Passengers** = seats × load factor (each route's own; a plane cannot fill beyond its seats).",
          "3. **Local passengers** = passengers minus those connecting onwards (transfer and transit): only they can check in to a hotel.",
          f"4. **Check-ins**: a 10% rise in a country's local passengers raises that nationality's hotel check-ins by about "
          f"{(1.1 ** eb['eps'] - 1) * 100:.0f}% (ε = {eb['eps']:.2f}, estimated across 33 countries and confirmed on months the model never saw). "
          "The resulting **capture rate** (check-ins per extra local passenger) is reported for every scenario. It is a market response, not a "
          "head count: it also carries guests who arrive by other routes as connectivity changes, and it can never exceed one check-in per extra local passenger.",
          "5. **Guest nights** = check-ins × how long that nationality stays in that season (the stay model of Phase 1): the "
          "**average length of stay** of the extra guests is reported too.", "",
          f"**Worked example — Etihad's new Abu Dhabi–Charlotte route** (4 weekly, from {ch['start']}, to the end of the horizon):", "",
          "| Step | Value | Ratio |", "|---|---|---|",
          f"| Seats | {n0(ch['seats'])} | |",
          f"| Passengers | {n0(ch['passengers'])} | load factor {ch['load_factor']:.0%} |",
          f"| Local passengers | {n0(ch['local_passengers'])} | {ch['local_share']:.0%} of passengers (most connect onwards) |",
          f"| Check-ins | {e0(ch['checkins'])} | {ch['checkins_per_local_passenger']:.2f} per local passenger |",
          f"| Guest nights | {e0(ch['guest_nights'])} (80%: {e0(ch['guest_nights_band'][0])} – {e0(ch['guest_nights_band'][1])}) | {ch['nights_per_checkin']:.1f} nights per check-in |", "",
          "**Why not \"one passenger = one guest\"?** Scaling check-ins one-for-one with seats predicted worse than no change at all on unseen "
          f"months (skill {tw['lever_check']['skill']['naive one-for-one (1.0)']:+.0%}, against {tw['lever_check']['skill']['default 0.31']:+.0%} for the lever), and in the "
          f"March 2026 replay flights alone would then remove more guests than the war removed in total in {o11_both} of the 5 world regions with flight data "
          f"under either seat assumption ({o11_any} with the lower seat count).", "",
          "**How sure are we?** Two ranges, never mixed:", "",
          f"- **Effect range (80%)** for the change a scenario causes: the lever's uncertainty (ε {eb['eps_lo']:.2f}–{eb['eps_hi']:.2f}, from months the "
          "model never saw), for very deep cuts also a second way of extending the lever (Step 3.7), and the uncertainty of the baseline it scales "
          "(an extra guest night needs a traveller: if the baseline is lower, so is the gain). When comparing two plans for the same markets and "
          "months, the baseline part is shared, so the lever part is what separates them; both parts are saved with every result.",
          "- **Level range (80%)** for a monthly number (baseline or scenario): the forecast error measured on two validation folds by market size "
          f"(cross-fold coverage {bd_['level_diagnostics']['guests']['coverage_A_to_B']:.0%} / {bd_['level_diagnostics']['guests']['coverage_B_to_A']:.0%}).",
          f"- From {rf['beyond_7_months_from'][:7]} (more than 7 months after the data) both are flagged: validation reached 7 months.", ""]
    # ---------------------------------------------------------------- steps
    cc = w["countries_changed_ge_2pct"]
    L += ["## Results by step", "",
          f"**3.1 Normal world.** Horizon {w['horizon'][0]} … {w['horizon'][1]}. Each route keeps last year's schedule by calendar month unless the "
          f"data show it ended or is new: {w['routes']['established']['routes']} established routes carry {w['routes']['established']['share_of_last_12m_seats_pct']:.0f}% "
          f"of seats; {w['routes']['ended (airline exited)']['routes']} routes ended with their airline (mainly the Wizz Air Abu Dhabi exit). Seats "
          f"{w['seats_vs_same_month_last_year_pct']:+.1f}% vs a year earlier; biggest changes: "
          + ", ".join(f"{T(k)} {v:+.0f}%" for k, v in list(cc.items())[:4]) + ", "
          + ", ".join(f"{T(k)} {v:+.0f}%" for k, v in list(cc.items())[-2:])
          + f". {w['calendar']['events_in_horizon']} dated calendar inputs (Ramadan, Eids, school breaks, National Day, F1, ADIPEC, home holidays).", "",
          f"**3.2 Production refit.** Stage A refitted through 28 Feb 2026; it reproduces the Phase 2 model exactly (max difference "
          f"{rf['reproduces_phase2_final']['max_rel_diff']:.0e}) and Stage B reproduces Phase 1 ({rf['stageb_reproduces_phase1_test']['max_abs_diff_guests']:.0e} guests). "
          f"Normal world vs last year: international check-ins {rf['yoy_total_pct']['international']:+.1f}%, domestic {rf['yoy_total_pct']['domestic']:+.1f}%. "
          f"28 Feb 2026, the first war day, is excluded (actual {n0(rf['war_day_checkins']['actual'])} check-ins vs {e0(rf['war_day_checkins']['normal_world'])} normal).", "",
          f"**3.3 Scenario inputs.** Six change types (route, new route, load factor, transfer share, demand, stay length) on any nationality, "
          f"list, SCAD world region or group; aircraft classes from the data (narrowbody {inp['aircraft_seats'][0]['seats_per_flight']:.0f}, "
          f"widebody {inp['aircraft_seats'][1]['seats_per_flight']:.0f} seats). Inputs are validated (errors stop a scenario, flags travel with it). "
          f"Published reality check: DCT's August 2026 guests ({inp['published_anchors']['august_2026_total']['published_yoy']:+.1%} year on year) are "
          f"{abs(inp['published_anchors']['august_2026_total']['actual_vs_normal']):.1%} {'above' if inp['published_anchors']['august_2026_total']['actual_vs_normal'] > 0 else 'below'} the normal world.", "",
          f"**3.4 Engine.** Day by day and market by market; every check exact: empty scenario = baseline ({eng['checks']['null_max_abs_change']:.0f}), lever = "
          f"Phase 2 lever ({eng['checks']['lever_equals_phase2_seat_lever_max_abs']:.0f}), markets add up ({eng['checks']['additivity_max_abs']:.0f}), a +10% stay change "
          f"realised as {eng['checks']['stay_plus_10pct_realised_pct']:.1f}%.", "",
          f"**3.5 Ranges.** ε {eb['eps']:.2f} → 80% {eb['eps_lo']:.2f}–{eb['eps_hi']:.2f}; level ranges by market size "
          + ", ".join(f"{r['tier']} {r['ratio_lo']:.2f}–{r['ratio_hi']:.2f}×" for r in sorted([r for r in bd_["level_bands"] if r["kind"] == "guests" and r["level"] == "market-month"],
                                                                                              key=lambda r: r["ratio_hi"])) + " (guest nights, actual ÷ forecast).", ""]
    lv = {(r["lever"], r["change_pct"]): r for r in se["levers"]}
    rk = se["lever_ranking_by_small_change_elasticity"]
    mr = [r for r in se["market_ranking"] if r["rule"] == "seat lever"]
    sea = se["season"]
    L += [f"**3.6 What moves guests most.** Guest nights move per 1% change in each lever: " + ", ".join(f"{k} {v:.2f}%" + (" (opposite direction)" if k == "transfer share" else "") for k, v in rk.items())
          + f". Flight levers act only through local passengers, and flight markets hold {se['guest_night_shares']['flight markets']:.0%} of guest nights "
          f"(domestic {se['guest_night_shares']['domestic']:.0%}). Guest nights per 1,000 added seats (one extra daily narrowbody): "
          + ", ".join(f"{T(r['market'])} {r['guest_nights_per_1000_seats']:.0f}" for r in mr[:5]) + " … "
          + ", ".join(f"{T(r['market'])} {r['guest_nights_per_1000_seats']:.0f}" for r in mr[-3:])
          + f"; by month {max(sea, key=sea.get)} highest ({max(sea.values()):.0f}), {min(sea, key=sea.get)} lowest ({min(sea.values()):.0f}). "
          "Markets whose answer relies on a rule beyond the tested range are flagged in the table (e.g. "
          + ", ".join([T(r["market"]) for r in se["market_ranking"] if r["rule"] != "seat lever" and "new-route" not in r["rule"]][:3]) + ").", ""]
    br = {(g["window"][:4], g["side"], g["excluding_war_months"]): g for g in va["beyond_range_replay"]["groups"]}
    tc = br[("test", "cut", False)]; tn = br[("trai", "cut", True)]
    assert abs(tn["line"]["pred_mean_pct"] - tn["actual_mean_pct"]) < abs(tn["log"]["pred_mean_pct"] - tn["actual_mean_pct"]), "3.7 text: line no longer closer on training cuts"
    assert abs(tc["log"]["pred_mean_pct"] - tc["actual_mean_pct"]) < abs(tc["line"]["pred_mean_pct"] - tc["actual_mean_pct"]), "3.7 text: log no longer closer on test cuts"
    pr = va["properties"]; m7 = va["march_2026"]
    L += [f"**3.7 Validation.** (a) Every seat shock outside the tested range was replayed. On unseen months (mainly the Wizz Air exit, "
          f"{tc['n']} country-months) check-ins fell {-tc['actual_mean_pct']:.0f}% relative to other countries, where the adopted straight line predicted "
          f"{-tc['line']['pred_mean_pct']:.0f}% and the log form {-tc['log']['pred_mean_pct']:.0f}%. In training months without war ({tn['n']} country-months) both "
          f"over-predicted and the straight line was closer (check-ins fell {-tn['actual_mean_pct']:.0f}%; line {-tn['line']['pred_mean_pct']:.0f}%, log {-tn['log']['pred_mean_pct']:.0f}%). "
          f"Neither form wins on both, so the straight line fixed before the test stays central and the effect range for deep cuts spans both "
          f"(test mean inside {tc['lever_band_pct'][0]:.0f}…{tc['lever_band_pct'][1]:.0f}%). "
          f"(b) {pr['scenarios']} random scenarios pass all {len(pr['passed'])} engine rules (plus range ordering {pr['band_ordered']}, market additivity "
          f"{pr['market_additivity']}). (c) March 2026: seats were {m7['seats_vs_normal']['low']:.0%}–{m7['seats_vs_normal']['high']:.0%} of normal; flights explain "
          f"{m7['total']['share_of_drop_explained_by_flights'][0]:.0%}–{m7['total']['share_of_drop_explained_by_flights'][1]:.0%} of the {1 - m7['total']['observed_vs_normal']:.0%} "
          f"drop in check-ins at the central lever ({m7['total']['share_of_drop_explained_by_flights_lever_range'][0]:.0%}–"
          f"{m7['total']['share_of_drop_explained_by_flights_lever_range'][1]:.0%} across its 80% range; see showcase 1).", ""]
    # ---------------------------------------------------------------- showcases
    m = sh["march_2026"]; lo_, hi_ = m["bounds"]["low"]["total"], m["bounds"]["high"]["total"]
    rg = {r["region"]: r for r in m["bounds"]["low"]["by_region"]}; DOMR = "UAE residents (Domestic)"
    am = m["scad_alos_americas"]
    rgh = {r["region"]: r for r in m["bounds"]["high"]["by_region"]}
    L += ["## Showcases (Step 3.8)", "",
          "### 1. March 2026, replayed", "",
          "Flights from Etihad's published interim schedules (AeroRoutes filings of 5, 10 and 16 March; nothing flew 1–5 March; other airlines "
          "between none and Etihad's restored share), plus the war's demand shock net of flights fitted to SCAD's March guests by region. "
          "Guest nights were not fitted: they test the stay model.", "",
          "| Group | Check-ins vs normal (SCAD) | Flights alone would give (low … high seats) | Guest nights vs normal: SCAD | replay, normal stays |", "|---|---|---|---|---|"]
    for k in [x for x in rg if rg[x]["checkins_normal"] > 5000]:
        L.append(f"| {k} | {pc(rg[k]['checkins_replay_vs_normal'] - 1)} | {pc(rg[k]['checkins_flights_only_vs_normal'] - 1)} … {pc(rgh[k]['checkins_flights_only_vs_normal'] - 1)} "
                 f"| {pc(rg[k]['guest_nights_scad_vs_normal'] - 1)} | {pc(rg[k]['guest_nights_replay_vs_normal'] - 1)} |")
    L += [f"| **All** | **{pc(lo_['checkins_replay_vs_normal'] - 1)}** | **{pc(lo_['checkins_flights_only_vs_normal'] - 1)} … {pc(hi_['checkins_flights_only_vs_normal'] - 1)}** "
          f"| **{pc(lo_['guest_nights_scad_vs_normal'] - 1)}** | **{pc(lo_['guest_nights_replay_vs_normal'] - 1)}** |", "",
          "_UAE residents = the remainder to SCAD's all-nationality total (SCAD's 'UAE' row counts Emirati nationals only). Africa and "
          "Australia/New Zealand (under 3,000 check-ins each) are in `showcase_march_2026_*.csv`._", "",
          f"- Flights explain only part of March: {e0(hi_['guest_nights_lost_flights'])}–{e0(lo_['guest_nights_lost_flights'])} of {e0(lo_['guest_nights_lost'])} "
          f"lost guest nights at the central lever. UAE residents, who need no flight, fell {1 - rg[DOMR]['checkins_replay_vs_normal']:.0%} (derived as the "
          "remainder to SCAD's total): the war hit demand directly (fear, advisories, Dubai also disrupted).",
          f"- Guest nights were the independent test, and the stay model with normal stays misses part of March: {pc(lo_['guest_nights_replay_vs_normal'] - 1)} "
          f"vs SCAD {pc(lo_['guest_nights_scad_vs_normal'] - 1)}. SCAD's average stay rose from {m['scad_alos']['2025-03']:.2f} to {m['scad_alos']['2026-03']:.2f} nights. "
          f"Part of that is mechanical — fewer new arrivals next to guests already in-house — and the replay reproduces it (nights per check-in "
          f"{pc(lo_['nights_per_checkin_vs_normal_replay_normal_stays'] - 1)} vs normal; SCAD {pc(lo_['nights_per_checkin_vs_normal_scad'] - 1)}). The 28 February check-in spike "
          f"({n0(m['feb28_spike']['checkins_actual'])} vs {e0(m['feb28_spike']['checkins_normal'])} normal: stranded travellers), which the normal world excludes by design, adds "
          f"{m['feb28_spike']['points_of_normal_march_guest_nights'] * 100:.1f} points. The rest is longer stays. The replay presets fit one stay change for March check-ins, "
          f"{lo_['stay_change_fitted_pct']:+.0f}%, that reproduces SCAD's March guest nights; it also carries the 28 February spike and longer stays of guests who "
          "arrived before March, which no March setting can reach separately. By region stays rose most for the Americas "
          f"(SCAD average stay {am['alos25']:.1f} → {am['alos26']:.1f} nights).",
          "- Presets `march_2026_replay_low/high` (flights + demand + stays) and `march_2026_demand_net_of_flights_*` let a user combine "
          "the war's demand shock with any flight plan without counting flights twice.", ""]
    t = sh["etihad_launches"]["total"]
    L += ["### 2. Etihad's 2026 launches into our markets (sourced)", "",
          f"Seats {sg(t['seats'])} → passengers {sg(t['passengers'])} (load factor {t['load_factor']:.0%}) → local passengers {sg(t['local_passengers'])} "
          f"({t['local_share']:.0%}) → check-ins {es(t['checkins'])} (80%: {band(t['checkins_band'])}) → **guest nights {es(t['guest_nights'])}** "
          f"(80%: {band(t['guest_nights_band'])}), in the normal world.", "",
          "| Route | From | Flights/week | Guest nights | 80% range | Note |", "|---|---|---|---|---|---|"]
    for x in sh["etihad_launches"]["routes"]:
        L.append(f"| {x['route']} ({T(x['market'])}) | {x['start']} | {x['flights_per_week']} | {es(x['guest_nights'])} | {band(x['guest_nights_band'])} | "
                 + ("no flight data: new-route lever" if "new-route" in x["lever"] else ("no baseline seats in August: zero-seat rule" if x["market"] == "UZBEKISTAN" else "")) + " |")
    eu = sh["european_return"]; sa_, sl = eu["stay_away"], eu["one_month_slip"]
    L += ["", "### 3. European carriers' return on 24–25 October (Wego, provisional)", "",
          f"If British Airways (London), Eurowings (Berlin) and Wizz Air (Katowice, Krakow) return as announced, Abu Dhabi gains "
          f"**{es(-sa_['guest_nights'])} guest nights** from arrivals up to 28 February (80%: {es(-sa_['guest_nights_band'][1])} … {es(-sa_['guest_nights_band'][0])}; "
          f"{es(-sa_['checkins'])} check-ins) compared with them staying away: "
          + ", ".join(f"{k} {es(-v['guest_nights'])}" for k, v in eu["by_carrier"].items())
          + f". A one-month slip costs {e0(-sl['guest_nights'])} guest nights (80%: {e0(-sl['guest_nights_band'][1])} – {e0(-sl['guest_nights_band'][0])}). "
          "Computed with normal-world demand: if European demand stays below normal, the value shrinks in the same proportion.", ""]
    rp = sh["recovery_planner"]; rule = rp["offset_rule"]
    L += ["### 4. Recovery planner: can flights make up for weaker demand?", "",
          "Extra local passengers needed to offset a demand shortfall (inside the tested range: (1/(1−shortfall))^(1/ε) − 1): "
          + "; ".join(f"{k} shortfall → +{v['central']:.0%} (80%: +{v['eps_high']:.0%} – +{v['eps_low']:.0%})" for k, v in rule.items())
          + f". Illustrative case — a {rp['assumption']['shortfall_pct']:.0f}% shortfall in December in the six largest flight markets, offset by extra "
          f"narrowbody flights ({rp['seats_per_flight']:.0f} seats, the market's own passenger mix):", "",
          "| Market | Guest nights lost | Extra flights/week needed | 80% range | Flights/week today |", "|---|---|---|---|---|"]
    for x in rp["rows"]:
        L.append(f"| {T(x['market'])} | {e0(-x['guest_nights_lost'])} | {x['flights_per_week_needed']:.0f} | {x['range'][0]:.0f} – {x['range'][1]:.0f} | {x['current_flights_per_week']:.0f} |")
    L += ["", f"Flights are a slow lever for a demand problem: a 10% shortfall needs about {rule['10%']['central']:.0%} more seats; recovering the demand "
          "itself (advisories, visas, campaigns, events) is the direct route.", ""]
    # ---------------------------------------------------------------- limitations
    L += ["## Limitations (verified)", "",
          "- **Abu Dhabi airport only.** Guests arriving through Dubai or by road are inside the market response, not counted passenger by passenger; "
          "12 nationalities without flight data use the new-route lever.",
          "- **Sustained changes.** ε comes from monthly year-on-year changes; changes shorter than 28 days are flagged.",
          "- **Deep cuts.** Below −29% local passengers the lever is extended, not estimated; on the Wizz Air exit check-ins fell more than either extension "
          "predicts at the central ε, and the effect range (both extensions, both ends of ε) covers it.",
          f"- **Demand shocks are not flights.** War, advisories, visas and currency move guests directly (March 2026: flights explained about a third at the central "
          f"lever, {m7['total']['share_of_drop_explained_by_flights_lever_range'][0]:.0%}–{m7['total']['share_of_drop_explained_by_flights_lever_range'][1]:.0%} across its range); use the demand lever.",
          f"- **Stays change in shocks.** In March 2026 guests stayed longer (fitted {lo_['stay_change_fitted_pct']:+.0f}% for March check-ins, which also carries "
          "the 28 February spike); Stage B uses normal "
          "stays unless the stay lever is set.",
          "- **Seats partly follow demand.** Airlines add seats where demand rises; the lead control lowers ε for this, but a pure capacity decision may achieve less.",
          "- **No prices and no room limit.** Hotel rates and airfares are not modelled; scenarios above the record in-house level are flagged.",
          "- **Normal world after September 2026** is less certain (flagged), and SCAD's world regions mix visitors and residents.", ""]
    # ---------------------------------------------------------------- summary
    fa, fb = (e26["folds"][f]["end_to_end"]["intl_monthly_total"] for f in "AB")
    summ = (f"Our simulator turns flight changes into Abu Dhabi hotel demand through a transparent chain: seats → passengers (load factor) → local "
            f"passengers (minus transfer/transit) → check-ins → guest nights, by nationality and day. A 10% rise in a country's local passengers "
            f"lifts its check-ins about {(1.1 ** eb['eps'] - 1) * 100:.0f}% (ε {eb['eps']:.2f}, 33 countries, confirmed on unseen months; one-for-one fails). "
            f"Back-tested from flights alone, monthly international guest nights are within {min(fa, fb):.0f}–{max(fa, fb):.0f}% (WMAPE). Every result has an 80% range. Validation: {pr['scenarios']} random scenarios pass {len(pr['passed'])} rule checks; replaying March 2026, flights "
            f"explain about a third of the {1 - m7['total']['observed_vs_normal']:.0%} drop ({m7['total']['share_of_drop_explained_by_flights_lever_range'][0]:.0%}–"
            f"{m7['total']['share_of_drop_explained_by_flights_lever_range'][1]:.0%} across the lever's range); the rest is war-driven demand, a separate lever. Etihad's 2026 launches add "
            f"~{t['guest_nights'] / 1000:.0f}k guest nights ({t['guest_nights_band'][0] / 1000:.0f}k–{t['guest_nights_band'][1] / 1000:.0f}k); Europe's carriers returning in late "
            f"October are worth ~{-sa_['guest_nights'] / 1000:.0f}k by February; offsetting a 10% demand dip needs ~{rule['10%']['central']:.0%} more seats. "
            "Limits: Abu Dhabi airport only, deep cuts uncertain, prices not modelled.")
    assert len(summ) <= 1000, len(summ)
    L += ["## Draft summary (≤ 1,000 characters)", "", summ, "", f"_{len(summ)} characters._", "",
          "## Files and reproduction", "",
          "`data/processed/phase3/`: `step31…step38*.json` (one per step), baseline tables (`baseline_*`), `route_baseline.csv`, ranges "
          "(`effect_band.json`, `level_bands.csv`), sensitivity (`sens_*.csv`), validation (`replay_beyond_range.csv`, `march_2026_replay_by_region.csv`), "
          "showcases (`showcase_*.csv`), presets (`presets/*.json`). Scenarios with sources: `data/reference/scenarios/`. "
          "Reproduce: `make phase3` then `make test`."]
    (cfg["paths"]["processed"].parent.parent / "reports").mkdir(exist_ok=True)
    rp_path = cfg["paths"]["processed"].parent.parent / "reports" / "phase3_report.md"
    open(rp_path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {rp_path} ({len(L)} lines); summary {len(summ)} characters")


# ----------------------------------------------------------------------------- step
def step_world():
    cfg = load_config(); out = _out(cfg); RD = cfg["paths"]["reference"]
    h0, h1 = cfg["dates"]["horizon_start"], cfg["dates"]["horizon_end"]
    t = horizon_events(cfg); hh = pd.read_csv(RD / "home_holidays.csv")
    missing = check_horizon(t, hh)
    assert not missing, f"horizon calendar incomplete: {missing}"
    t.to_csv(out / "horizon_events.csv", index=False, date_format="%Y-%m-%d")
    b, st, ndays = route_baseline(cfg)
    # consistency: the unadjusted route sum reproduces the Phase 2 airline defaults (same days, same exclusions)
    _, fd = _rows(cfg); end = cfg["dates"]["test_end"]
    ad = sa.airline_defaults(fd, end - pd.DateOffset(years=1) + pd.Timedelta(days=1), end).set_index(["nationality_key", "moy"])
    naive = b.groupby(["nationality_key", "moy"]).naive.sum()
    gap = float((naive - ad.seats / ad.days).abs().max())
    assert gap < 1e-6, f"route sum differs from airline defaults by {gap}"
    b.drop(columns=["naive"]).to_csv(out / "route_baseline.csv", index=False, date_format="%Y-%m-%d")
    st.to_csv(out / "route_status.csv", index=False, date_format="%Y-%m-%d")
    d = daily_baseline(b, cfg)
    d.to_parquet(out / "baseline_seats_daily.parquet", index=False)
    # summaries
    tot = b.groupby("status").naive.sum() * 365 / 12
    by_status = {s: {"routes": int((st.status == s).sum()), "share_of_last_12m_seats_pct": float(b[b.status == s].naive.sum() / b.naive.sum() * 100)}
                 for s in ["established", "seasonal", "new", "resumed", "ended (airline exited)", "ended (route stopped)"]}
    cty = b.groupby("nationality_key")[["naive", "seats"]].sum()
    cty["change_pct"] = (cty.seats / cty.naive - 1) * 100
    movers = cty[cty.change_pct.abs() >= 2].sort_values("change_pct")
    total_change = float((b.seats.sum() / b.naive.sum() - 1) * 100)
    est = t[t.status.isin(["estimated", "scheduled"])]
    j = {"horizon": [str(h0.date()), str(h1.date())],
         "calendar": {"events_in_horizon": int(len(t)), "by_status": t.status.value_counts().to_dict(),
                      "to_recheck": [f"{r.name} {r.start.date()}…{r.end.date()} ({r.status}): {r.source}" for r in est.itertuples()],
                      "corrections_to_history": ["national_day 2023: 2–4 Dec (MoHRE 24 Nov 2023), was 2–3 Dec",
                                                 "national_day 2025: 1–2 Dec (MoHRE/FAHR Nov 2025), was 2–3 Dec"]},
         "routes": by_status, "seats_vs_same_month_last_year_pct": total_change,
         "countries_changed_ge_2pct": {k: round(v, 1) for k, v in movers.change_pct.items()},
         "filled_route_months": int(b.filled.sum()), "airline_defaults_reproduced_max_gap": gap,
         "zero_seat_country_months": [f"{k[0].title()} {pd.Timestamp(2000, k[1], 1):%b}" for k, v in
                                      b.groupby(["nationality_key", "moy"]).seats.sum().items() if v == 0],
         "zero_seat_check": _zero_seat_check(b, fd, cfg),
         "route_overrides": {f"{r.city} ({r.airline})": f"{r.status_from_data} → {r.status}" for r in st[st.override_source != ""].itertuples()},
         "news_after_data_end": pd.read_csv(RD / "horizon_news.csv").groupby("kind").size().to_dict(),
         "rules": {"active_days": ACTIVE_DAYS, "airline_gone_days": AIRLINE_GONE_DAYS, "seasonal_max_months": SEASONAL_MAX_MONTHS,
                   "seasonal_min_jaccard": SEASONAL_MIN_JACCARD, "fill_days": FILL_DAYS}}
    json.dump(j, open(out / "step31_world.json", "w", encoding="utf-8"), indent=1, default=str)
    # print
    print(f"HORIZON {h0.date()} … {h1.date()}: {len(t)} dated inputs — " + ", ".join(f"{k} {v}" for k, v in j["calendar"]["by_status"].items()))
    for x in j["calendar"]["to_recheck"]:
        print("  re-check:", x)
    print("ROUTES (share of last-12-month seats): " + "; ".join(f"{k} {v['routes']} ({v['share_of_last_12m_seats_pct']:.1f}%)" for k, v in by_status.items()))
    print(f"  baseline seats vs same month last year: {total_change:+.1f}%; filled route-months (new routes before launch): {j['filled_route_months']}")
    print("  countries changed ≥ 2%: " + ", ".join(f"{k.title()} {v:+.0f}%" for k, v in j["countries_changed_ge_2pct"].items()))
    print("  country-months with no baseline seats: " + (", ".join(j["zero_seat_country_months"]) or "none"))
    for k, v in j["zero_seat_check"].items():
        print(f"    {k}: forecaster seat floor keeps {v['forecaster_keeps']:.0%} of the level at the adjacent months' seats "
              f"(the lever's full-cancellation rule keeps {v['lever_full_cancellation_keeps']:.0%}) — finite, no log(0)")
    print("  published schedules applied (data rule overridden): " + "; ".join(f"{k}: {v}" for k, v in j["route_overrides"].items()))
    print(f"  airline defaults reproduced by the unadjusted route sum (max gap {gap:.1e} seats/day)")
    print(f"DAILY BASELINE: {len(d):,} rows, {d.nationality_key.nunique()} countries, {d.date.min().date()} … {d.date.max().date()}")


def observed_peak(cfg) -> tuple[float, pd.Timestamp]:
    """Highest number of guests in-house on one night in the hotel data (international target rows + domestic) — the
    engine's capacity flag. Saved by `peak` so the simulator does not need the full daily hotel tables."""
    P = cfg["paths"]["processed"]
    hi = pd.read_parquet(P / "hotel_intl_daily.parquet"); hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    tot = hi[hi.is_target].groupby("date").guests.sum() + hd.set_index("date").guests
    return float(tot.max()), tot.idxmax()


def step_peak():
    cfg = load_config(); v, d = observed_peak(cfg)
    j = {"peak_guests": v, "date": str(d.date()),
         "rule": "highest guests in-house on one night: international target rows + domestic, Jan 2022 – Jul 2025 "
                 "(guests are withheld after Jul 2025)",
         "used_for": "capacity flag: scenarios above this level are flagged (engine.Params.peak_guests)"}
    json.dump(j, open(_out(cfg) / "peak_guests.json", "w", encoding="utf-8"), indent=1)
    print(f"PEAK: {v:,.0f} guests in-house on {d.date()} → phase3/peak_guests.json")


if __name__ == "__main__":
    {"world": step_world, "refit": step_refit, "inputs": step_inputs, "engine": step_engine, "bands": step_bands, "peak": step_peak,
     "sensitivity": step_sensitivity, "validate": step_validate, "showcase": step_showcase, "report": step_report}[sys.argv[1]]()
