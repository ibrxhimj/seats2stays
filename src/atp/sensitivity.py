"""Phase 3, Step 3.6 — sensitivity: which factors move hotel guests most, where and when, and what it takes to hit a target.

Four answers, all from the Step 3.4 engine with Step 3.5 bands (no separate surrogate model):

  1. Lever ranking (equal changes): each lever moved ±10% over the whole horizon on the whole network — seats, load
     factor, transfer share (each market's own share ±10% of itself, capped at 99%), stay length, and non-flight demand
     (benchmark).
     Reported as guest nights moved and as an elasticity (% guest nights per % lever); the ranking uses the elasticity
     at ±1%, because at ±10% transfer share pushes hub markets beyond the lever's trusted range and load factor hits
     full planes (the share of the change that comes from beyond the trusted range is reported).
  2. Where and when seats matter: one extra daily narrowbody flight (Etihad's median narrowbody seats, the market's own
     passenger mix) added to each nationality in each month, one month at a time → guest nights per 1,000 added seats
     by market and season.
  3. Reverse query: flights per week needed for a target of extra guest nights in a market and window (root-finding on
     the engine, which is monotone in seats), with the range where the effect band's ends reach the target.
  4. Which uncertainty matters most for a typical result: the lever (ε), cross-checked with the training bootstrap,
     against the baseline level.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

GENERIC = {"city": "added daily flight", "airline": "generic"}   # no airline history → the market's own passenger mix


def _band(B, r, k="d_guest_nights"):
    return r.effect_band["summary"][k]


def lever_ranking(E, B, pct: float = 10.0, small: float = 1.0) -> pd.DataFrame:
    """±pct on each lever (guest nights with the effect band), plus the elasticity at a ±small change. The small change
    is the ranking number: at ±10% a relative change in transfer share moves hub markets' local passengers far outside
    the range the lever was estimated on (a 10% rise in an 88% share removes 73% of local passengers), and load factor
    hits full planes, so the ±10% elasticities partly measure the extrapolation and cap rules rather than the lever."""
    w = E.world; h0, h1 = str(w.h0.date()), str(w.h1.date())
    levers = {
        "seats": lambda x: [{"type": "route", "markets": "flight markets", "start": h0, "end": h1, "seats_pct": x}],
        "load factor": lambda x: [{"type": "load_factor", "markets": "flight markets", "start": h0, "end": h1, "pct": x}],
        "transfer share": lambda x: [{"type": "transfer_share", "markets": "flight markets", "start": h0, "end": h1, "pct": x}],
        "stay length": lambda x: [{"type": "stay_length", "markets": "all", "start": h0, "end": h1, "pct": x}],
        "non-flight demand": lambda x: [{"type": "demand", "markets": "all", "start": h0, "end": h1, "pct": x}],
    }
    base_gn = float(E.run({"changes": []}).summary["baseline_guest_nights"])
    rows = []
    for name, f in levers.items():
        for sign in (+1, -1):
            res = B.run(E, {"name": f"{name} {sign * pct:+.0f}%", "changes": f(sign * pct)})
            s = res.summary; lo, hi = _band(B, res); d = res.daily
            sm = E.run({"changes": f(sign * small)}).summary["d_guest_nights"]
            beyond = float(d.loc[d.outside_trusted_range, "d_checkins"].sum() / s["d_checkins"] * 100) if s["d_checkins"] else 0.0
            rows.append({"lever": name, "change_pct": sign * pct, "d_local_passengers": s["d_local_passengers"],
                         "local_passengers_pct": s["d_local_passengers"] / float(d.p2p_base.sum()) * 100,
                         "d_guest_nights": s["d_guest_nights"], "lo": lo, "hi": hi,
                         "guest_nights_pct": s["d_guest_nights"] / base_gn * 100,
                         "elasticity": s["d_guest_nights"] / base_gn * 100 / (sign * pct),
                         "elasticity_small_change": sm / base_gn * 100 / (sign * small),
                         "pct_of_checkin_change_beyond_trusted_range": beyond,
                         "capped_or_flagged": "; ".join(fl for fl in res.flags if "capped" in fl or "outside the range" in fl)})
    return pd.DataFrame(rows)


def market_season(E, B, seats_per_flight: float) -> pd.DataFrame:
    """One extra daily flight in every nationality, one calendar month at a time (markets are independent in the engine,
    so all markets are run together for a month and read separately)."""
    w = E.world
    intl = [m for m in w.markets if m != "DOMESTIC"]
    rows = []
    for mo in pd.period_range(w.h0, w.h1, freq="M"):
        a, b = mo.start_time.date(), mo.end_time.date()
        ch = [{"type": "new_route", "market": m, **GENERIC, "start": str(a), "end": str(b), "flights_per_week": 7,
               "seats_per_flight": seats_per_flight} for m in intl]
        res = B.run(E, {"name": f"+1 daily flight {mo}", "changes": ch})
        days = mo.days_in_month
        ends = [r.summary["by_market"] for r in B.end_runs(E, {"changes": ch}, res)]
        d = res.daily
        flagged = d[d.outside_trusted_range | d.zero_baseline_rule | d.new_route_lever | d.cap_binds].groupby("market").size()
        for m in intl:
            v = res.summary["by_market"].get(m, {}); gn = v.get("d_guest_nights", 0.0)
            ev = [gn] + [x.get(m, {}).get("d_guest_nights", 0.0) for x in ends]
            e_lo, e_hi = min(ev), max(ev)
            ldn, lup = B._level_dev("guests", {m: gn})
            rule = ("new-route lever (no flight data)" if m not in w.flight else
                    "from full cancellation (no baseline seats)" if d[(d.market == m) & d.zero_baseline_rule].shape[0] else
                    "beyond trusted range (straight line)" if d[(d.market == m) & d.outside_trusted_range].shape[0] else
                    "capped by local passengers" if d[(d.market == m) & d.cap_binds].shape[0] else "seat lever")
            rows.append({"market": m, "month": str(mo), "seats_added": seats_per_flight * days, "d_checkins": v.get("d_checkins", 0.0),
                         "d_guest_nights": gn, "lo": gn - np.hypot(gn - e_lo, ldn), "hi": gn + np.hypot(e_hi - gn, lup),
                         "guest_nights_per_1000_seats": gn / (seats_per_flight * days) * 1000, "rule": rule,
                         "flagged_days": int(flagged.get(m, 0))})
    return pd.DataFrame(rows)


TOP_FPW = 700.0      # Phase 3 search limit: 700 flights a week (100 a day) on the added route. The app's own reverse
                     # question uses tighter limits (Simulator.reverse: double today's flights, or 100 a week on a new route)


def seats_needed(E, B, market: str, start: str, end: str, target_guest_nights: float, seats_per_flight: float,
                 base_changes: list | None = None) -> dict:
    """Flights per week (one generic route with the market's own passenger mix) for a target of extra guest nights.
    Two ranges are returned and labelled: `flights_per_week_range`, the flights at which the combined band's upper end
    (fewest flights) and lower end (most flights) reach the target — lever and baseline level together, as every other
    result — and `flights_per_week_range_lever_only`, the same from the lever's runs alone. Each end is solved on its own
    and returns None when it cannot reach the target within the limit.
    base_changes (e.g. a demand shortfall) are part of every run, so a target of 0 means "offset them" (Step 3.8)."""
    base = list(base_changes or [])

    def scen(fpw):
        return {"changes": base + ([{"type": "new_route", "market": market, **GENERIC, "start": start, "end": end,
                                     "flights_per_week": fpw, "seats_per_flight": seats_per_flight}] if fpw > 0 else [])}

    def gain(fpw):
        return E.run(scen(fpw)).summary["d_guest_nights"]

    def lever_end(fpw, f):                                           # f = max (most favourable) or min over the lever's ends
        s = scen(fpw); mid = E.run(s)
        return f([mid.summary["d_guest_nights"]] + [r.summary["d_guest_nights"] for r in B.end_runs(E, s, mid)])

    def band_end(fpw, i):
        return B.run(E, scen(fpw)).effect_band["summary"]["d_guest_nights"][i]

    def solve(fn, top=TOP_FPW):
        """Root of the rising fn on [0, top]: 0 when no added flight is needed (fn(0) already reaches the target, e.g. a
        target of 0 with nothing to offset), None when even `top` flights a week fall short."""
        if fn(0.0) >= target_guest_nights:
            return 0.0
        if fn(top) < target_guest_nights:
            return None
        return brentq(lambda x: fn(x) - target_guest_nights, 0.0, top, xtol=0.01)

    fpw = solve(gain)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    out = {"market": market, "window": [start, end], "target_guest_nights": target_guest_nights, "seats_per_flight": seats_per_flight,
           "flights_per_week": fpw,
           "flights_per_week_range": [solve(lambda x: band_end(x, 1)), solve(lambda x: band_end(x, 0))],
           "flights_per_week_range_lever_only": [solve(lambda x: lever_end(x, max)), solve(lambda x: lever_end(x, min))]}
    if fpw is not None:
        out["seats_added"] = fpw / 7 * days * seats_per_flight
    return out


def uncertainty_tornado(E, B, scenario: dict, boot80: list) -> dict:
    """For one result: how far each source of uncertainty moves it (80% ranges). The bootstrap cross-check varies ε
    only (the reference scenario changes seats in flight markets, so the new-route rate plays no part)."""
    r = B.run(E, scenario)
    c = r.summary["d_guest_nights"]; parts = r.effect_band["parts"]["d_guest_nights"]
    bo = [E.run(scenario, eps=e).summary["d_guest_nights"] for e in boot80]
    return {"central": c, "lever_test_window_eps": [c + parts["lever"][0], c + parts["lever"][1]],
            "lever_training_bootstrap_eps": [min(bo), max(bo)],
            "baseline_level": [c + parts["baseline_level"][0], c + parts["baseline_level"][1]],
            "combined": r.effect_band["summary"]["d_guest_nights"]}
