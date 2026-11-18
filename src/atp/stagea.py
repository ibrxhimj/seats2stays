"""Stage A — flights → hotel check-ins. Shared building blocks.

Step 2.1 (this module, first half): airline-side defaults and ε₁ (seats → local passengers).
Conventions fixed before results were seen:
  * Elasticities come from MONTHLY year-on-year log changes with a common month-year effect
    (identification = a country's seats growing faster/slower than other countries' in the same month).
  * Primary sample: both months ≥ 2023-01 (post-COVID; 2023-vs-2022 comparisons carry country-specific
    reopening jumps, e.g. China seats +700%). Robustness: include 2022 base, add 1–2 month seat leads,
    weight by route size.
  * Flight-disruption days (Phase 0 `is_low_coverage`) are removed from monthly sums.
  * Uncertainty: 95% interval from a country-cluster bootstrap (2,000 resamples) + clustered analytic SE.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- panels
def monthly_panel(fd: pd.DataFrame, hi: pd.DataFrame, fm: pd.DataFrame | None = None) -> pd.DataFrame:
    """Country × month: seats, pax, p2p, transfer, transit, flights, hotel check-ins (same-named nationality).
    2023+ built from DAILY data without disruption days; 2022 (optional) from the native monthly table."""
    d = fd[~fd.is_low_coverage].assign(month=lambda x: x.date.dt.to_period("M").dt.to_timestamp())
    m = d.groupby(["nationality_key", "month"]).agg(seats=("seats", "sum"), pax=("pax", "sum"), p2p=("p2p", "sum"),
                                                   transfer=("transfer", "sum"), transit=("transit", "sum"),
                                                   n_flights=("n_flights", "sum"), days=("date", "nunique")).reset_index()
    dq_days = set(fd.loc[fd.is_low_coverage, "date"])
    h = hi[~hi.date.isin(dq_days)].assign(month=lambda x: x.date.dt.to_period("M").dt.to_timestamp())
    ci = h.groupby(["nationality", "month"]).arrivals.sum().rename("checkins").reset_index() \
          .rename(columns={"nationality": "nationality_key"})
    m = m.merge(ci, on=["nationality_key", "month"], how="left")
    m["source"] = "daily"
    if fm is not None:
        old = fm[fm.source_granularity == "monthly"][["nationality_key", "month", "seats", "pax", "p2p", "transfer", "transit"]]
        old = old.merge(ci, on=["nationality_key", "month"], how="left").assign(source="monthly")
        m = pd.concat([old, m], ignore_index=True)
    return m.sort_values(["nationality_key", "month"]).reset_index(drop=True)


def truncate(m: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Keep panel months up to the month containing `end` (applied BEFORE yoy, so leads past `end` are missing)."""
    return m[m.month <= end].reset_index(drop=True)


FEST_WINDOW = (-2, 7)              # festival travel window: 2 days before to 7 days after the festival day (10 days)


def festival_dates(ref_dir, official: pd.DataFrame) -> pd.DataFrame:
    """Main moving festival days, one row per (festival, year). Eids come from the official UAE calendar
    (Eid al-Fitr = day after Ramadan ends; Eid al-Adha = first official holiday day), Easter and Orthodox Easter
    from the exact algorithms (dateutil), Lunar New Year and Diwali from data/reference/source_market_festivals.csv."""
    from dateutil.easter import easter
    o = official[official.active]
    rows = [("eid_al_fitr", r.end_date + pd.Timedelta(days=1)) for r in o[o.event == "ramadan"].itertuples()]
    rows += [("eid_al_adha", r.start_date) for r in o[o.event == "eid_al_adha"].itertuples()]
    years = range(o.start_date.dt.year.min(), o.end_date.dt.year.max() + 1)                # every year the calendar covers
    rows += [("easter", pd.Timestamp(easter(y))) for y in years]
    rows += [("orthodox_easter", pd.Timestamp(easter(y, 2))) for y in years]                  # 2 = EASTER_ORTHODOX
    other = pd.read_csv(ref_dir / "source_market_festivals.csv", parse_dates=["date"])
    rows += list(zip(other.festival, other.date))
    d = pd.DataFrame(rows, columns=["festival", "date"])
    return d.assign(year=d.date.dt.year).sort_values(["festival", "date"]).reset_index(drop=True)


def festival_panel(groups: pd.DataFrame, fest: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Country × month: `festival_days` = days of the country's own main festival travel window(s) in the month;
    `ramadan_days` = Ramadan days in the month (Muslim-majority markets only, 0 elsewhere)."""
    days = []
    for f in fest.itertuples():
        for k in range(FEST_WINDOW[0], FEST_WINDOW[1] + 1):
            days.append((f.festival, f.date + pd.Timedelta(days=k)))
    win = pd.DataFrame(days, columns=["festival", "date"]).drop_duplicates()
    g = groups.assign(festival=groups.festivals.fillna("").str.split(";")).explode("festival")
    g = g[g.festival.isin(fest.festival.unique())]
    fd_ = g.merge(win, on="festival").drop_duplicates(["nationality", "date"])     # overlapping windows count once
    fd_ = fd_.assign(month=fd_.date.dt.to_period("M").dt.to_timestamp()).groupby(["nationality", "month"]).size()
    ram = cal[cal.is_ramadan].assign(month=lambda x: x.date.dt.to_period("M").dt.to_timestamp()).groupby("month").size()
    months = pd.date_range("2022-01-01", "2026-12-01", freq="MS")
    out = pd.MultiIndex.from_product([groups.nationality, months], names=["nationality_key", "month"]).to_frame(index=False)
    out["festival_days"] = fd_.reindex(pd.MultiIndex.from_frame(out)).fillna(0).to_numpy()
    musl = set(groups.loc[groups.group == "muslim_majority", "nationality"])
    out["ramadan_days"] = np.where(out.nationality_key.isin(musl), ram.reindex(out.month).fillna(0).to_numpy(), 0.0)
    return out


def yoy(m: pd.DataFrame, cols=("seats", "pax", "p2p", "checkins"), leads=(1, 2), lags=(1, 2)) -> pd.DataFrame:
    """12-month log differences by country (date arithmetic on month, never positional)."""
    x = m.set_index(["nationality_key", "month"])
    out = m[["nationality_key", "month", "source"]].copy()
    prev = m.assign(month=m.month + pd.DateOffset(years=1)).set_index(["nationality_key", "month"])
    for c in cols:
        cur = x[c].reindex(pd.MultiIndex.from_frame(out[["nationality_key", "month"]]))
        pre = prev[c].reindex(pd.MultiIndex.from_frame(out[["nationality_key", "month"]]))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"d_{c}"] = np.log(cur.to_numpy()) - np.log(pre.to_numpy())
        out[f"base_{c}"] = pre.to_numpy()
    out["base_month"] = out.month - pd.DateOffset(years=1)
    key = out.set_index(["nationality_key", "month"])["d_seats"]
    for L in leads:
        lead_idx = pd.MultiIndex.from_arrays([out.nationality_key, out.month + pd.DateOffset(months=L)])
        out[f"d_seats_lead{L}"] = key.reindex(lead_idx).to_numpy()
    for L in lags:
        lag_idx = pd.MultiIndex.from_arrays([out.nationality_key, out.month - pd.DateOffset(months=L)])
        out[f"d_seats_lag{L}"] = key.reindex(lag_idx).to_numpy()
    for c in ("festival_days", "ramadan_days"):            # day counts: plain year-on-year differences (not logs)
        if c in m:
            out[f"d_{c}"] = x[c].reindex(pd.MultiIndex.from_frame(out[["nationality_key", "month"]])).to_numpy() - \
                prev[c].reindex(pd.MultiIndex.from_frame(out[["nationality_key", "month"]])).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------- out-of-sample lever test
def _rel(d: pd.DataFrame, cols, w: str | None = None) -> pd.DataFrame:
    """Country change minus the same month's (weighted) average change across countries (what month effects remove)."""
    if w is None:
        return d[cols] - d.groupby("month")[cols].transform("mean")
    num = d[cols].mul(d[w], axis=0).groupby(d.month).transform("sum")
    return d[cols] - num.div(d.groupby("month")[w].transform("sum"), axis=0)


def lever_skill(test: pd.DataFrame, eps: dict, n_boot: int = 2000, seed: int = 0, w: str | None = None) -> dict:
    """Out-of-sample skill of the seat lever on relative check-in growth.
    For each candidate ε: SSE(ε) = Σ (rel Δlog check-ins − ε · rel Δlog seats)²; skill = 1 − SSE(ε)/SSE(0),
    i.e. the share of cross-country check-in growth differences the lever explains beyond 'everyone grows alike'.
    Interval from a country bootstrap (month averages recomputed inside each draw)."""
    d = test.dropna(subset=["d_seats", "d_checkins"] + ([w] if w else [])).reset_index(drop=True)
    g = d.nationality_key.to_numpy(); countries = np.unique(g); by = {c: np.where(g == c)[0] for c in countries}

    def sse(dd):
        r = _rel(dd, ["d_seats", "d_checkins"], w); ww = dd[w].to_numpy() if w else np.ones(len(dd))
        return {k: float(np.sum(ww * (r.d_checkins - e * r.d_seats) ** 2)) for k, e in {"none": 0.0, **eps}.items()}
    s0 = sse(d); point = {k: 1 - v / s0["none"] for k, v in s0.items() if k != "none"}
    rng = np.random.default_rng(seed); draws = {k: [] for k in eps}
    for _ in range(n_boot):
        idx = np.concatenate([by[c] for c in rng.choice(countries, len(countries), replace=True)])
        s = sse(d.iloc[idx].reset_index(drop=True))
        for k in eps:
            draws[k].append(1 - s[k] / s["none"])
    draws = {k: np.array(v) for k, v in draws.items()}
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()} if n_boot else None
    return {"skill": point, "ci": ci, "draws": draws, "n_obs": int(len(d)), "n_countries": int(len(countries))}


def trusted_range(y: pd.DataFrame, q: float = 0.05) -> tuple:
    """q and 1−q quantiles of the YoY log seat change (uses seats only — never the outcome)."""
    lo, hi = y.d_seats.quantile([q, 1 - q])
    return float(lo), float(hi)


def trim(y: pd.DataFrame, rng: tuple) -> pd.DataFrame:
    """Keep rows whose YoY log seat change lies inside the trusted range (selection on the regressor only)."""
    return y[y.d_seats.between(*rng)].reset_index(drop=True)


def ratio(num: dict, den: dict) -> dict:
    """Implied ratio num/den with a PAIRED country bootstrap: valid only when both estimates resampled the
    same country list with the same seed and draw count (so draw i used the same countries in both)."""
    assert num["countries"] == den["countries"] and len(num["boot_draws"]) == len(den["boot_draws"]) > 0, \
        "paired bootstrap needs identical country sets and draw counts"
    r = num["boot_draws"] / den["boot_draws"]
    return {"estimate": num["estimate"] / den["estimate"], "ci_boot": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
            "boot_draws": r}


def difference(a: dict, b: dict) -> dict:
    """a − b for estimates on DISJOINT country groups (independent, stratified bootstrap)."""
    assert not set(a["countries"]) & set(b["countries"]), "groups must not share countries"
    dd = a["boot_draws"] - b["boot_draws"]
    return {"estimate": a["estimate"] - b["estimate"], "ci_boot": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}


# ----------------------------------------------------------------------------- estimator
def _demean(y, X, t, w=None):
    """Within-transformation for one-way time fixed effects (exact for one-way FE, unbalanced OK). With weights the
    month means are WEIGHTED, which makes weighted least squares on the result identical to WLS with month dummies."""
    df = pd.DataFrame(np.column_stack([y, X]), index=t)
    if w is None:
        return (df - df.groupby(level=0).transform("mean")).to_numpy()
    w = pd.Series(np.asarray(w, dtype=float), index=t)
    mean = df.mul(w, axis=0).groupby(level=0).transform("sum").div(w.groupby(level=0).transform("sum"), axis=0)
    return (df - mean).to_numpy()


def fe_elasticity(df: pd.DataFrame, y: str, x: list, w: str | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """OLS of y on x with month-year fixed effects; country-clustered SE and cluster-bootstrap interval for x[0]."""
    d = df.dropna(subset=[y] + x + ([w] if w else [])).reset_index(drop=True)
    t = d.month.astype(str).to_numpy(); g = d.nationality_key.to_numpy()
    # drop CONTROL columns with no variation left after month demeaning (e.g. Ramadan days in a Jul–Dec sample)
    Zx = _demean(np.zeros(len(d)), d[x].to_numpy(), t)[:, 1:]
    dropped = [c for i, c in enumerate(x) if i > 0 and np.allclose(Zx[:, i], 0)]
    x = [c for c in x if c not in dropped]
    wt = d[w].to_numpy() if w else np.ones(len(d))

    def est(dd, tt, ww):
        Z = _demean(dd[y].to_numpy(), dd[x].to_numpy(), tt, ww if w else None)
        sw = np.sqrt(ww)
        b, *_ = np.linalg.lstsq(Z[:, 1:] * sw[:, None], Z[:, 0] * sw, rcond=None)
        return b
    b = est(d, t, wt)
    import statsmodels.api as sm                                                    # imported here: the app (which only reads ε's interval) does not need statsmodels
    # analytic clustered SE (statsmodels on demeaned data; df correction for FE is minor with 30+ months)
    Z = _demean(d[y].to_numpy(), d[x].to_numpy(), t, wt if w else None)
    res = sm.WLS(Z[:, 0], Z[:, 1:], weights=wt).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    ci = np.asarray(res.conf_int()); ci_a = ci[0]
    wald_p = None
    if len(x) > 1:                                          # joint test that all coefficients after x[0] are zero
        R = np.eye(len(x))[1:]
        wald_p = float(res.wald_test(R, scalar=True).pvalue)
    # country-cluster bootstrap (month FE are re-estimated inside each draw; duplicated countries are fine
    # because the fixed effects are by month only)
    rng = np.random.default_rng(seed); countries = np.unique(g); boots = []
    by = {c: np.where(g == c)[0] for c in countries}
    for _ in range(n_boot):
        pick = rng.choice(countries, len(countries), replace=True)
        idx = np.concatenate([by[c] for c in pick])
        boots.append(est(d.iloc[idx].reset_index(drop=True), t[idx], wt[idx])[0])
    boots = np.array(boots)
    ci_b = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if n_boot else None
    return {"estimate": float(b[0]), "ci_boot": ci_b,
            "ci_cluster": [float(ci_a[0]), float(ci_a[1])], "n_obs": int(len(d)), "n_countries": int(len(countries)),
            "n_months": int(len(np.unique(t))), "first_month": str(d.month.min().date()), "last_month": str(d.month.max().date()),
            "countries": [str(c) for c in countries], "other_coefs": dict(zip(x[1:], map(float, b[1:]))),
            "other_ci_cluster": {k: [float(ci[i + 1, 0]), float(ci[i + 1, 1])] for i, k in enumerate(x[1:])},
            "se_cluster": float(res.bse[0]), "wald_p_others": wald_p, "dropped_controls": dropped,
            "boot_draws": boots}


# ----------------------------------------------------------------------------- airline defaults
def closure_days(fd: pd.DataFrame, start, end, frac: float = 0.05, min_run: int = 3) -> pd.MultiIndex:
    """(country, date) pairs in [start, end] inside a CLOSURE: a run of ≥ `min_run` consecutive days whose seats are below
    `frac` of the median of the same weekday in BOTH the 4 weeks before and the 4 weeks after. Found: Israel 8–10 Oct 2023
    and 14–26 Jun 2025; Armenia 24–26 and Azerbaijan 23–26 Jun 2025 (routes over Iran). Weekdays a route never flies, seasonal
    gaps, launches and terminations never qualify (one side's median is zero); single cancelled flights stay in the
    schedule as part of normal operations."""
    s = fd.pivot_table(index="date", columns="nationality_key", values="seats", aggfunc="sum").asfreq("D").fillna(0.0)
    med = lambda sign: pd.concat([s.shift(sign * 7 * k) for k in range(1, 5)]).groupby(level=0).median()
    ref = np.minimum(med(1), med(-1))                                         # NaN at the data edges → never flagged
    low = (s < frac * ref) & (ref > 0)
    run_len = low.apply(lambda c: c.groupby((c != c.shift()).cumsum()).transform("size")).where(low, 0)
    low = (run_len >= min_run).loc[pd.Timestamp(start):pd.Timestamp(end)].stack()
    idx = low[low].index
    return pd.MultiIndex.from_arrays([idx.get_level_values(1), idx.get_level_values(0)], names=["nationality_key", "date"])


def airline_defaults(fd: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Country × calendar month defaults from the most recent 12 months of flights (disruption days and a country's
    own airspace-closure days excluded, so a war-time month does not become the 'normal' schedule).
    These are the starting values the Phase 3 levers adjust."""
    cl = closure_days(fd, start, end)
    keep = ~pd.MultiIndex.from_frame(fd[["nationality_key", "date"]]).isin(cl)
    d = fd[keep & (fd.date >= start) & (fd.date <= end) & ~fd.is_low_coverage].assign(moy=lambda x: x.date.dt.month)
    g = d.groupby(["nationality_key", "moy"]).agg(seats=("seats", "sum"), pax=("pax", "sum"), p2p=("p2p", "sum"),
                                                 transfer=("transfer", "sum"), transit=("transit", "sum"),
                                                 n_flights=("n_flights", "sum"), days=("date", "nunique")).reset_index()
    g["flights_per_week"] = g.n_flights / g.days * 7
    g["seats_per_flight"] = g.seats / g.n_flights
    g["load_factor"] = g.pax / g.seats
    g["transfer_share"] = g.transfer / g.pax
    g["transit_share"] = g.transit / g.pax
    g["local_share"] = g.p2p / g.pax                      # = 1 − transfer − transit (identity, tested)
    g["local_pax_per_seat"] = g.p2p / g.seats             # = load_factor × local_share
    return g


# ----------------------------------------------------------------------------- scenario levers (used by Phase 3)
Z80 = 1.2815515655446004                     # two-sided 80% normal quantile


def lever_interval(step26: dict) -> tuple[float, float]:
    """ε's 80% interval from the official test-window lever check (Step 2.6): ε × (1 ± z₈₀ · s.e.) of the calibration
    slope (country-clustered). The one ε range used everywhere from Step 2.7 on (replaces the provisional 2.3 band,
    whose low end was picked on partly in-sample months)."""
    lc = step26["test_window"]["lever_check"]; se = lc["calibration_slope"]["se_cluster"]; eps = lc["eps"]
    return eps * (1 - Z80 * se), eps * (1 + Z80 * se)


def seat_lever(A, S, dS, eps: float, local_per_seat, trusted=(-0.29, 1.09)):
    """Change in check-ins when a market with seats S gets dS extra seats (dS < 0 = cut), over the same period.

    Inside the trusted range of seat changes r = dS/S (the range ε was estimated and validated on) the rule is the
    estimated log form:            ΔA = A · [(1 + r)^ε − 1]
    Outside it, the effect continues in a straight line from the range edge with the same slope (no extrapolation of
    the curve beyond the data). A full cancellation therefore keeps (1 − ε)(1 + r_lo)^ε · A (≈ 62% at ε = 0.31): travellers
    who re-route. Cap: the change never exceeds the local passengers those seats carry (|dS| · local_per_seat).
    Returns (ΔA, inside_trusted_range)."""
    A, S, dS, lps = (np.asarray(v, dtype=float) for v in (A, S, dS, local_per_seat))
    if np.any(S <= 0):
        raise ValueError("seat_lever needs existing seats (S > 0); use new_route_lever for markets without seat data")
    if np.any(dS < -S):
        raise ValueError("cannot cut more seats than exist")
    lo, hi = trusted
    r = dS / S
    rc = np.clip(r, lo, hi)
    g = (1 + rc) ** eps - 1                                           # log form inside the range
    slope = eps * (1 + rc) ** (eps - 1)                               # its slope at the range edge
    g = g + np.where((r < lo) | (r > hi), slope * (r - rc), 0.0)      # straight line beyond the edge
    dA = np.sign(g) * np.minimum(np.abs(A * g), np.abs(dS) * lps)
    return dA, (r >= lo) & (r <= hi)


def new_route_lever(dS, per_seat: float, local_per_seat: float):
    """Extra check-ins from dS seats added where the pack has no seat history (Step 2.5): per-seat rate × seats,
    never above the local passengers those seats carry. Only additions are defined (there is nothing to cut)."""
    dS = np.asarray(dS, dtype=float)
    if np.any(dS < 0):
        raise ValueError("new_route_lever only adds seats")
    return dS * min(per_seat, local_per_seat)
