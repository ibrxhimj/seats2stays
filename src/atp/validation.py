"""Phase 3, Step 3.7 — validation of the scenario engine.

  1. Historical replay of seat shocks beyond the trusted range. ε was estimated on year-on-year changes inside the
     range (−29% … +109%); beyond it the engine continues in a straight line (Step 2.7). Every market-month outside
     the range is replayed — the training months (never used to fit ε: the range trims them) and the official test
     window (Aug 2025–Feb 2026, unseen), whose large cuts are the Wizz Air Abu Dhabi exit (1 Sep 2025). Changes are
     relative to the other countries in the same month and festival-adjusted, exactly as in the Phase 2 lever check,
     and the engine's lever is applied to the change in local passengers (what the engine uses).
  2. Engine properties on random scenarios: determinism, decomposition, locality (only changed market-days move),
     sign, the local-passenger cap, ε = 0 removes every flight effect, Stage B keeps the sign, market additivity,
     band ordering, no invalid numbers.
  3. The March 2026 war replay (flights only, against SCAD's observed March) — see march_replay.
"""
import numpy as np
import pandas as pd
from . import engine as en, scenario as sc

WAR_MONTHS = {"ISRAEL", "LEBANON"}           # Gaza and Hezbollah wars: demand collapsed with the seats (Phase 2.7)


# ----------------------------------------------------------------------------- 1. beyond-range replay
def _slope(act, pred, cl):
    """No-intercept calibration slope of actual on predicted, country-clustered s.e. (small-sample factor)."""
    b = float((act * pred).sum() / (pred ** 2).sum()); e = act - b * pred
    G = len(np.unique(cl))
    s = np.sqrt(sum(((pred[cl == c] * e[cl == c]).sum()) ** 2 for c in np.unique(cl))) / (pred ** 2).sum() * np.sqrt(G / max(G - 1, 1))
    return b, float(s)


def replay_rows(cfg) -> tuple[pd.DataFrame, dict]:
    from . import phase2 as p2, stagea as sa
    fd, _, hi = p2._load(cfg)
    m, _, _ = p2._panel_with_festivals(cfg, fd, hi)
    F = cfg["folds"]; tr_end, te_end = F["final"]["train_end"], cfg["dates"]["test_end"]
    p_tr, rng_ = p2._primary(m, tr_end)
    g = sa.fe_elasticity(p_tr, "d_checkins", ["d_seats"] + p2.FEST, n_boot=0)          # festival effects from training
    y = sa.yoy(sa.truncate(m, te_end))
    y = y[y.base_month >= p2.BASE_START].dropna(subset=["d_seats", "d_checkins", "d_p2p"]).copy()
    y["window"] = np.where(y.month > tr_end, "test (Aug 2025–Feb 2026)", "training (Jan 2024–Jul 2025)")
    y["d_checkins_adj"] = (y.d_checkins - g["other_coefs"]["d_festival_days"] * y.d_festival_days
                           - g["other_coefs"]["d_ramadan_days"] * y.d_ramadan_days)
    parts = []
    for _, w in y.groupby("window"):                                   # relative to other countries, same window
        parts.append(w.join(sa._rel(w, ["d_seats", "d_p2p", "d_checkins_adj"]).add_prefix("rel_")))
    y = pd.concat(parts)
    y["side"] = np.where(y.d_seats < rng_[0], "cut", np.where(y.d_seats > rng_[1], "increase", "inside"))
    return y.reset_index(drop=True), {"trusted_range_log": [float(x) for x in rng_]}


def beyond_range_replay(cfg, eff: dict) -> dict:
    P = en.Params.load(cfg); eps = P.eps
    y, meta = replay_rows(cfg)
    x = np.exp(y.rel_d_p2p.to_numpy())
    y["line"] = np.log(en.lever_G(x, eps, P.lo, P.hi))
    y["log"] = np.log(en.lever_G(x, eps, P.lo, P.hi, "log", eff["cut_floor"]))
    # lever band on the same rows: least change (ε low, straight line) … most change (ε high, log form for cuts)
    y["band_a"] = np.log(en.lever_G(x, eff["eps_lo"], P.lo, P.hi))
    y["band_b"] = np.log(en.lever_G(x, eff["eps_hi"], P.lo, P.hi, "log", eff["cut_floor"]))
    y["band_b_line_only"] = np.log(en.lever_G(x, eff["eps_hi"], P.lo, P.hi))
    out = {"trusted_range_pct": [float(np.expm1(v)) for v in meta["trusted_range_log"]], "cut_floor": eff["cut_floor"],
           "eps": eps, "groups": []}
    for (win, side), z in y[y.side != "inside"].groupby(["window", "side"]):
        for excl in (False, True):
            zz = z[~z.nationality_key.isin(WAR_MONTHS)] if excl else z
            if len(zz) == 0 or (excl and len(zz) == len(z)):
                continue
            act = zz.rel_d_checkins_adj.to_numpy(); cl = zz.nationality_key.to_numpy()
            row = {"window": win, "side": side, "excluding_war_months": excl, "n": int(len(zz)), "countries": int(zz.nationality_key.nunique()),
                   "actual_mean_pct": float(np.expm1(act.mean()) * 100)}
            for r in ("line", "log"):
                pr = zz[r].to_numpy(); b, s = _slope(act, pr, cl)
                row[r] = {"pred_mean_pct": float(np.expm1(pr.mean()) * 100), "mae_log": float(np.abs(act - pr).mean()),
                          "bias_log": float((act - pr).mean()), "slope": b, "slope_se": s}
            # the lever band at the group mean (a single month also carries its own noise, which the level band covers)
            row["lever_band_pct"] = sorted([float(np.expm1(zz.band_a.mean()) * 100), float(np.expm1(zz.band_b.mean()) * 100)])
            row["lever_band_line_only_pct"] = sorted([float(np.expm1(zz.band_a.mean()) * 100), float(np.expm1(zz.band_b_line_only.mean()) * 100)])
            out["groups"].append(row)
    # the out-of-sample event: test-window cuts by country (the Wizz Air exit markets dominate)
    t = y[(y.side == "cut") & y.window.str.startswith("test")]
    ev = t.groupby("nationality_key").agg(months=("month", "size"), first=("month", "min"), last=("month", "max"),
                                          seats=("rel_d_seats", "mean"), local_passengers=("rel_d_p2p", "mean"),
                                          actual=("rel_d_checkins_adj", "mean"), line=("line", "mean"), log=("log", "mean"),
                                          band_a=("band_a", "mean"), band_b=("band_b", "mean"), band_b_line=("band_b_line_only", "mean"))
    for c in ("seats", "local_passengers", "actual", "line", "log", "band_a", "band_b", "band_b_line"):
        ev[c] = np.expm1(ev[c]) * 100
    ev["inside_lever_band"] = (ev.actual <= ev.band_a) & (ev.actual >= ev.band_b)
    ev["inside_line_only_band"] = (ev.actual <= ev.band_a) & (ev.actual >= ev.band_b_line)
    ev["first"] = ev["first"].dt.strftime("%Y-%m"); ev["last"] = ev["last"].dt.strftime("%Y-%m")
    out["test_window_cuts_by_country"] = ev.round(1).reset_index().to_dict("records")
    # inside the range, the engine's input path (local passengers) on the unseen test window
    z = y[(y.side == "inside") & y.window.str.startswith("test")]
    b, s = _slope(z.rel_d_checkins_adj.to_numpy(), eps * z.rel_d_p2p.to_numpy(), z.nationality_key.to_numpy())
    out["inside_range_test_slope_local_passengers"] = {"slope": b, "slope_se": s, "n": int(len(z)), "countries": int(z.nationality_key.nunique())}
    return out, y


# ----------------------------------------------------------------------------- 2. engine properties
def random_scenario(w, rng) -> dict:
    """1–3 random changes of any type, window 1–90 days inside the horizon."""
    intl = [m for m in w.markets if m != "DOMESTIC"]
    ch = []
    for _ in range(rng.integers(1, 4)):
        a = w.h0 + pd.Timedelta(days=int(rng.integers(0, len(w.days))))
        b = min(a + pd.Timedelta(days=int(rng.integers(0, 90))), w.h1)
        win = {"start": str(a.date()), "end": str(b.date())}
        k = rng.choice(["route", "route", "new_route", "load_factor", "transfer_share", "demand", "stay_length"])
        if k == "route":
            m = str(rng.choice(w.flight))
            r = w.routes[(w.routes.nationality_key == m) & (w.routes.seats > 0)]
            if rng.random() < 0.5 and len(r):
                x = r.iloc[int(rng.integers(0, len(r)))]
                ch.append({"type": "route", "market": m, "city": x.city, "airline": x.airline, **win,
                           "flights_per_week": float(np.round(rng.uniform(0, 2) * x.flights_per_week, 1))})
            else:
                ch.append({"type": "route", "market": m, **win, "seats_pct": float(np.round(rng.uniform(-100, 300), 1))})
        elif k == "new_route":
            ch.append({"type": "new_route", "market": str(rng.choice(intl)), "city": "test city", "airline": "test airline", **win,
                       "flights_per_week": int(rng.integers(1, 15)), "aircraft_class": str(rng.choice(["narrowbody", "widebody"]))})
        elif k == "load_factor":
            ch.append({"type": "load_factor", "markets": str(rng.choice(w.flight)), **win, "pct": float(np.round(rng.uniform(-40, 20), 1))})
        elif k == "transfer_share":
            ch.append({"type": "transfer_share", "markets": str(rng.choice(w.flight)), **win, "points": float(np.round(rng.uniform(-10, 10), 1))})
        elif k == "demand":
            ch.append({"type": "demand", "markets": str(rng.choice(w.markets + ["Europe", "GCC", "all"])), **win,
                       "pct": float(np.round(rng.uniform(-60, 60), 1))})
        else:
            ch.append({"type": "stay_length", "markets": str(rng.choice(w.markets)), **win, "pct": float(np.round(rng.uniform(-20, 20), 1))})
    return {"name": "random", "changes": ch}


def _check(r: en.Result, r2: en.Result, r0: en.Result, per_seat: float) -> dict:
    d, d0 = r.daily, r0.daily
    tol = 1e-6
    std = d.has_flight_data & (d.p2p_base > 0)
    unchanged = np.isclose(d.p2p_scen, d.p2p_base) & (d.new_route_seats == 0) & (d.demand_mult == 1)
    dL = d.p2p_scen - d.p2p_base
    dAf = d.d_checkins_flights / d.demand_mult
    first = d[~(unchanged & (d.stay_mult == 1))].groupby("market").date.min()
    before = d.date < pd.to_datetime(d.market.map(first.to_dict())).fillna(pd.Timestamp("2262-01-01"))
    by_m = d.groupby("market")
    all_up = by_m.d_checkins.transform(lambda v: (v >= -tol).all()) & by_m.stay_mult.transform(lambda v: (v == 1).all())
    all_dn = by_m.d_checkins.transform(lambda v: (v <= tol).all()) & by_m.stay_mult.transform(lambda v: (v == 1).all())
    nf = ~d.has_flight_data & (d.new_route_seats > 0)
    return {
        "deterministic": bool(np.array_equal(d.d_checkins, r2.daily.d_checkins) and np.array_equal(d.d_guest_nights, r2.daily.d_guest_nights)),
        "finite_and_non_negative": bool(np.isfinite(d[["checkins_scen", "guests_scen", "d_guest_nights"]]).all().all()
                                        and (d.checkins_scen >= -tol).all() and (d.guests_scen >= -1e-3).all()),
        "decomposition_exact": bool(np.allclose(d.d_checkins, d.d_checkins_flights + d.d_checkins_demand, atol=tol)),
        "unchanged_inputs_no_checkin_change": bool((d.d_checkins[unchanged].abs() <= tol).all()),
        "no_guest_change_before_first_change": bool((d.d_guest_nights[before].abs() <= 1e-6).all()),
        "sign_follows_local_passengers": bool((np.sign(dAf[std & (dL.abs() > 1e-3)]) == np.sign(dL[std & (dL.abs() > 1e-3)])).all()),
        "cap_respected": bool((dAf[std].abs() <= dL[std].abs() + 1e-6).all() and (dAf[d.zero_baseline_rule] <= d.p2p_scen[d.zero_baseline_rule] + 1e-6).all()
                              and (dAf[nf] <= per_seat * d.new_route_seats[nf] + 1e-6).all()),
        "guests_keep_checkin_sign": bool((d.d_guest_nights[all_up] >= -1e-6).all() and (d.d_guest_nights[all_dn] <= 1e-6).all()),
        "eps_zero_removes_flight_effect": bool((d0.d_checkins_flights[d0.has_flight_data].abs() <= tol).all()),
        "demand_part_exact": bool(np.allclose(d.d_checkins_demand, d.checkins_base * (d.demand_mult - 1), atol=tol)),
    }


def properties(E: en.Engine, B, n: int = 150, n_pairs: int = 40, n_bands: int = 30, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); w = E.world
    ok, refused, tried = {}, 0, 0
    fails = []
    scen = []
    while len(scen) < n:
        s = random_scenario(w, rng); tried += 1
        try:
            r = E.run(s)
        except sc.ScenarioError:
            refused += 1
            continue
        res = _check(r, E.run(s), E.run(s, eps=0.0), E.params.per_seat)
        for k, v in res.items():
            ok[k] = ok.get(k, 0) + int(v)
            if not v:
                fails.append({"property": k, "scenario": s})
        scen.append((s, r))
    # bands: lo ≤ central ≤ hi
    band_ok = 0
    for s, r in scen[:n_bands]:
        b = B.run(E, s).effect_band["summary"]
        band_ok += int(all(b[k][0] - 1e-6 <= r.summary[k] <= b[k][1] + 1e-6 for k in ("d_checkins", "d_guest_nights")))
    # additivity across markets: single-market scenarios on different markets add up exactly
    add_ok, pairs = 0, 0
    singles = [(s, r) for s, r in scen if len({c.get("market", c.get("markets")) for c in s["changes"]}) == 1
               and all(isinstance(c.get("market", c.get("markets")), str) and c.get("market", c.get("markets")) in w.markets for c in s["changes"])]
    for i in range(min(n_pairs, len(singles) // 2)):
        (s1, r1), (s2, r2) = singles[2 * i], singles[2 * i + 1]
        m1, m2 = (s1["changes"][0].get("market", s1["changes"][0].get("markets")), s2["changes"][0].get("market", s2["changes"][0].get("markets")))
        if m1 == m2:
            continue
        rr = E.run({"changes": s1["changes"] + s2["changes"]})
        pairs += 1
        add_ok += int(np.allclose(rr.daily.d_checkins, r1.daily.d_checkins + r2.daily.d_checkins, atol=1e-6)
                      and np.allclose(rr.daily.d_guest_nights, r1.daily.d_guest_nights + r2.daily.d_guest_nights, atol=1e-6)
                      and abs(rr.summary["d_guest_nights"] - r1.summary["d_guest_nights"] - r2.summary["d_guest_nights"]) < 1e-4)
    types = pd.Series([c["type"] for s, _ in scen for c in s["changes"]]).value_counts().to_dict()
    return {"scenarios": n, "generated": tried, "refused_by_validation": refused, "change_types": types,
            "passed": {k: f"{v}/{n}" for k, v in ok.items()}, "band_ordered": f"{band_ok}/{min(n_bands, n)}",
            "market_additivity": f"{add_ok}/{pairs}", "failures": fails[:10],
            "all_passed": bool(all(v == n for v in ok.values()) and band_ok == min(n_bands, n) and add_ok == pairs)}


# ----------------------------------------------------------------------------- 3. March 2026 war replay
SUSPENDED_ALL_MARCH = {"Qatar Airways": "reported suspended (Gulf News 1 Apr 2026)", "Gulf Air": "reported suspended (Gulf News 1 Apr 2026)",
                       "FLYNAS": "reported suspended (Gulf News 1 Apr 2026)",
                       "British Airways": "not back until 25 Oct (Wego 4 Sep 2026); suspension from the war's start assumed",
                       "Wizz Air": "not back until 24 Oct (Wego 4 Sep 2026); suspension from the war's start assumed",
                       "Eurowings": "Lufthansa Group not back until 24 Oct (Wego 4 Sep 2026); suspension from the war's start assumed"}
EY = "Etihad Airways"


def march_scenarios(w, sched: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Flights-only scenarios for March 2026, low and high seats. 1–5 Mar: no scheduled flights (Etihad suspended all
    flights 28 Feb–5 Mar; UAE airspace restricted). Etihad afterwards: AeroRoutes interim schedules, each route at its
    listed weekly frequency with last year's aircraft size; routes not listed are cancelled. 6–12 Mar: low = the 25
    destinations of the 5 Mar filing (frequencies from the 13–19 Mar filing), high = the 13–19 Mar schedule. Other
    airlines: low = none flew; high = the same share of their seats as Etihad restored in that market (Etihad's network
    share where Etihad does not fly); carriers reported suspended, or not back until late October, fly nothing in both."""
    r = w.routes[(w.routes.moy == 3) & (w.routes.seats > 0)]
    ey = r[r.airline == EY]; other = r[r.airline != EY]
    periods = {"B": ("2026-03-06", "2026-03-12"), "C": ("2026-03-13", "2026-03-19"), "D": ("2026-03-20", "2026-03-31")}
    lists = {("B", "low"): "B_low", ("B", "high"): "C", ("C", "low"): "C", ("C", "high"): "C", ("D", "low"): "D", ("D", "high"): "D"}
    out, ratios = {}, {}
    for bound in ("low", "high"):
        ch = [{"type": "route", "markets": "flight markets", "start": "2026-03-01", "end": "2026-03-05", "seats_pct": -100}]
        for p, (a, b) in periods.items():
            lst = sched[sched.period == lists[(p, bound)]]
            fpw = {(x.market, x.city_baseline): float(x.flights_per_week) for x in lst.itertuples() if isinstance(x.city_baseline, str)}
            scen_seats = {}
            for x in ey.itertuples():
                f = fpw.get((x.nationality_key, x.city))
                if f is None:
                    ch.append({"type": "route", "market": x.nationality_key, "city": x.city, "airline": EY, "start": a, "end": b, "seats_pct": -100})
                    scen_seats[x.nationality_key] = scen_seats.get(x.nationality_key, 0.0)
                else:
                    ch.append({"type": "route", "market": x.nationality_key, "city": x.city, "airline": EY, "start": a, "end": b, "flights_per_week": f})
                    scen_seats[x.nationality_key] = scen_seats.get(x.nationality_key, 0.0) + f / 7 * x.seats_per_flight
            for x in lst[lst.city_baseline.isna()].itertuples():                       # Etihad routes new in March 2026
                ch.append({"type": "new_route", "market": x.market, "city": x.city_source, "airline": EY, "start": max(a, x.start), "end": b,
                           "flights_per_week": float(x.flights_per_week), "aircraft_class": "widebody"})
            base = ey.groupby("nationality_key").seats.sum()
            ratio = {m: scen_seats.get(m, 0.0) / v for m, v in base.items()}
            net = sum(scen_seats.values()) / base.sum()
            ratios[(bound, p)] = {"etihad_network": net, **ratio}
            for (m, al), g in other.groupby(["nationality_key", "airline"]):
                pct = -100.0 if (bound == "low" or al in SUSPENDED_ALL_MARCH) else (ratio.get(m, net) - 1) * 100
                ch.append({"type": "route", "market": m, "airline": al, "start": a, "end": b, "seats_pct": round(pct, 3)})
        out[bound] = {"name": f"March 2026 flights only ({bound} seats)", "kind": "flights only", "changes": ch}
    return out["low"], out["high"], ratios


def _net_demand(daily: pd.DataFrame, markets: list, target: float) -> float:
    """The demand multiplier m that, applied with the flight changes, gives `target` check-ins over these markets' rows.
    The engine takes the flight effect in the scenario's demand and then caps it (Engine.run, step 4):
    A(m) = Σ A₀·m + sign(r)·min(|r|·m, cap). A(m) is continuous, piecewise linear and increasing in m (|r| ≤ A₀ for a cut), so
    the root is unique; Brent's method finds it. With no binding cap it is observed ÷ flights-only, as before."""
    from scipy.optimize import brentq
    x = daily[daily.market.isin(markets)]
    A0, r, cap = x.checkins_base.to_numpy(), x.flight_raw.to_numpy(), x.flight_cap.to_numpy()
    f = lambda m: float((A0 * m + np.sign(r) * np.minimum(np.abs(r) * m, cap)).sum()) - target
    lo, hi = 1e-9, 1.0
    while f(hi) < 0:
        hi *= 2
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)) if f(lo) < 0 else lo


def march_replay(E, B, cfg, sched: pd.DataFrame, observed: dict) -> dict:
    """Flights-only March 2026 vs SCAD's observed March (all causes, relative to the normal world), by SCAD region.
    The residual is what flights cannot explain: the war's demand shock (fear, advisories, closed alternative gateways)."""
    w = E.world
    reg = pd.read_csv(cfg["paths"]["reference"] / "scad_region_map.csv").set_index("nationality").scad_region
    low, high, ratios = march_scenarios(w, sched)
    res = {}
    for k, s in (("low", low), ("high", high)):
        mid = E.run(s)
        runs = [mid] + B.end_runs(E, s, mid)
        mo, daily = [], None
        for r in runs:
            x = r.daily[r.daily.date.dt.month.eq(3) & r.daily.date.dt.year.eq(2026)]
            if daily is None:
                daily = x[["market", "checkins_base", "flight_raw", "flight_cap"]]
            x = x.assign(one_for_one=np.where(x.p2p_base > 0, x.checkins_base * x.p2p_scen / x.p2p_base.where(x.p2p_base > 0, 1), x.checkins_base))
            mo.append(x.groupby("market")[["checkins_base", "checkins_scen", "seats_base", "seats_scen", "one_for_one"]].sum())
        res[k] = {"mid": mo[0], "ends": mo[1:], "flags": mid.flags, "summary": mid.summary, "daily": daily}
    regions = {**{m: reg.get(m) for m in w.markets if m != "DOMESTIC"}, "DOMESTIC": "UAE residents (Domestic)"}
    tab = []
    preset = {c["markets"]: 1 + c["pct"] / 100 for c in observed["changes"]}
    for rg in sorted(set(regions.values()), key=str):
        mk = [m for m, x in regions.items() if x == rg]
        base = float(res["low"]["mid"].loc[mk, "checkins_base"].sum())
        o = float(sum(preset[m] * res["low"]["mid"].loc[m, "checkins_base"] for m in mk)) / base
        row = {"region": rg, "markets": len(mk), "flight_markets": int(sum(m in w.flight for m in mk)), "normal_checkins": base,
               "observed_vs_normal": o}
        for k in ("low", "high"):
            f = float(res[k]["mid"].loc[mk, "checkins_scen"].sum()) / base
            fe = [float(x.loc[mk, "checkins_scen"].sum()) / base for x in res[k]["ends"]]
            row[f"flights_only_{k}_seats"] = f
            row[f"flights_only_{k}_seats_lever_range"] = [min(fe + [f]), max(fe + [f])]
            row[f"residual_demand_{k}_seats"] = _net_demand(res[k]["daily"], mk, o * base)
            row[f"one_for_one_{k}_seats"] = float(res[k]["mid"].loc[mk, "one_for_one"].sum()) / base
        row["share_of_drop_explained_by_flights"] = sorted([(1 - row["flights_only_high_seats"]) / (1 - o) if o < 1 else np.nan,
                                                           (1 - row["flights_only_low_seats"]) / (1 - o) if o < 1 else np.nan])
        tab.append(row)
    T = pd.DataFrame(tab)
    tot_base = T.normal_checkins.sum()
    total = {"observed_vs_normal": float((T.observed_vs_normal * T.normal_checkins).sum() / tot_base)}
    for k in ("low", "high"):
        total[f"flights_only_{k}_seats"] = float((T[f"flights_only_{k}_seats"] * T.normal_checkins).sum() / tot_base)
        total[f"residual_demand_{k}_seats"] = _net_demand(res[k]["daily"], list(regions), total["observed_vs_normal"] * tot_base)
    total["share_of_drop_explained_by_flights"] = sorted([(1 - total[f"flights_only_{k}_seats"]) / (1 - total["observed_vs_normal"]) for k in ("high", "low")])
    for k in ("low", "high"):
        total[f"one_for_one_{k}_seats"] = float((T[f"one_for_one_{k}_seats"] * T.normal_checkins).sum() / tot_base)
    # the same share across the lever's 80% range (ε ends; log form for deep cuts), both seat bounds
    ends = [x.checkins_scen.sum() / x.checkins_base.sum() for k in ("low", "high") for x in [res[k]["mid"]] + res[k]["ends"]]
    sh_ = [(1 - v) / (1 - total["observed_vs_normal"]) for v in ends]
    total["share_of_drop_explained_by_flights_lever_range"] = [float(min(sh_)), float(max(sh_))]
    seats = {k: float(res[k]["mid"].seats_scen.sum() / res[k]["mid"].seats_base.sum()) for k in res}
    # the war's demand shock net of flights: one multiplier per region (Domestic its own), so that flights + this
    # demand change reproduce SCAD's March by region — use it with flight changes without counting flights twice
    net = {}
    for k in ("low", "high"):
        dm = dict(zip(T.region, T[f"residual_demand_{k}_seats"]))
        net[k] = {"name": f"march_2026_demand_net_of_flights_{k}", "kind": "demand net of flights",
                  "title": f"March 2026 demand shock, net of flight cuts ({k}-seats flight replay)",
                  "notes": ["Region multiplier = SCAD's observed March (relative to the normal world) ÷ the flights-only replay; "
                            "combine with the matching flights-only preset to reproduce SCAD's March by region.",
                            "Includes everything flights do not: fear, advisories, alternative gateways closed (Dubai was also "
                            "disrupted, while the seat lever assumes travellers can reroute)."],
                  "changes": [{"type": "demand", "markets": m, "start": "2026-03-01", "end": "2026-03-31",
                               "pct": round((dm[regions[m]] - 1) * 100, 2), "basis": f"{regions[m]} net of flights"} for m in sorted(regions)]}
    return {"by_region": T, "total": total, "march_seats_vs_normal": seats, "etihad_ratios": ratios,
            "scenarios": {"low": low, "high": high}, "net_of_flights": net, "flags": {k: res[k]["flags"] for k in res}}
