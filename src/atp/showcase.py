"""Phase 3, Step 3.8 — showcase scenarios: the simulator answering four real questions, each with the full chain
(seats → passengers → local passengers → check-ins → guest nights) and its 80% effect band.

  1. March 2026 replayed: Etihad's interim schedules (flights) plus the war's demand shock net of flights reproduce
     SCAD's March check-ins by region (Step 3.7); guest nights are then an independent check of Stage B against SCAD.
  2. Etihad's 2026 launches into our markets (sourced, route by route).
  3. European carriers' return on 24–25 Oct 2026 (Wego, provisional): what their return is worth, and what a one-month
     slip would cost.
  4. Recovery planner: weekly flights needed to offset a demand shortfall in a market (illustrative 10% in December),
     and the general rule behind it.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from . import engine as en, scenario as sc, sensitivity as sn

DOM = "UAE residents (Domestic)"


def chain(r: en.Result) -> dict:
    """The transparent chain for a result (flight part), with the ratios the brief asks for."""
    s = r.summary
    div = lambda a, b: float(a / b) if abs(b) > 1e-9 else None
    out = {"seats": s["d_seats"], "passengers": s["d_passengers"], "local_passengers": s["d_local_passengers"],
           "checkins_from_flights": s["d_checkins_flights"], "checkins_from_demand": s["d_checkins_demand"], "checkins": s["d_checkins"],
           "guest_nights": s["d_guest_nights"], "load_factor": div(s["d_passengers"], s["d_seats"]),
           "local_share": div(s["d_local_passengers"], s["d_passengers"]),
           "checkins_per_local_passenger": div(s["d_checkins_flights"], s["d_local_passengers"]),
           "nights_per_checkin": div(s["d_guest_nights"], s["d_checkins"])}
    if r.effect_band:
        out["checkins_band"] = r.effect_band["summary"]["d_checkins"]; out["guest_nights_band"] = r.effect_band["summary"]["d_guest_nights"]
        out["months_beyond_7"] = r.effect_band["months_beyond_7"]
    out["flags"] = [f for f in r.flags if "assumption lever" not in f and "shorter than" not in f]
    return out


# ----------------------------------------------------------------------------- 1. March 2026
def _march(r: en.Result) -> pd.DataFrame:
    x = r.daily[r.daily.date.dt.strftime("%Y-%m") == "2026-03"]
    return x.groupby("market")[["checkins_base", "checkins_scen", "guests_base", "guests_scen"]].sum()


def march_2026(E, cfg) -> dict:
    P, RD = cfg["paths"]["processed"], cfg["paths"]["reference"]
    O = P / "phase3" / "presets"
    reg = pd.read_csv(RD / "scad_region_map.csv").set_index("nationality").scad_region
    region = {m: (reg.get(m) if m != "DOMESTIC" else DOM) for m in E.world.markets}
    # SCAD guest nights, March 2026 vs March 2025, applied to our March 2025 guest nights → observed vs normal world
    t6 = pd.read_excel(cfg["paths"]["external"] / "scad" / "SCAD_Hotel_Master_Database_2022_2026.xlsx", sheet_name="T6_Nights_By_Nationality")
    t6["date"] = pd.to_datetime(t6.date)
    N = t6.pivot(index="date", columns="nationality_region", values="guest_nights_thousands")
    nr = N.loc["2026-03-01"] / N.loc["2025-03-01"]
    t1 = pd.read_excel(cfg["paths"]["external"] / "scad" / "SCAD_Hotel_Master_Database_2022_2026.xlsx", sheet_name="T1_Emirate_Key_Indicators")
    t1["date"] = pd.to_datetime(t1.date); t1 = t1.set_index("date")
    total_nights_ratio = float(t1.loc["2026-03-01", "guest_nights_thousands"] / t1.loc["2025-03-01", "guest_nights_thousands"])
    hi = pd.read_parquet(P / "hotel_intl_daily.parquet"); hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    g25 = pd.concat([hi[hi.date.dt.strftime("%Y-%m") == "2025-03"].groupby("nationality").guests.sum(),
                     pd.Series({"DOMESTIC": hd[hd.date.dt.strftime("%Y-%m") == "2025-03"].guests.sum()})])
    base = _march(E.run({"changes": []}))
    obs_gn = {}
    for rg in sorted(set(region.values()) - {DOM}, key=str):            # fixed order: identical sums on every run
        mk = [m for m, x in region.items() if x == rg]
        obs_gn[rg] = float(nr[rg] * g25[mk].sum())
    obs_gn[DOM] = float(total_nights_ratio * g25.sum() - sum(obs_gn.values()))
    t4 = pd.read_excel(cfg["paths"]["external"] / "scad" / "SCAD_Hotel_Master_Database_2022_2026.xlsx", sheet_name="T4_Guests_By_Nationality")
    t4["date"] = pd.to_datetime(t4.date)
    G = t4.pivot(index="date", columns="nationality_region", values="guests_thousands")
    am = "North and South America"
    out = {"scad_total_nights_ratio": total_nights_ratio,
           "scad_alos_americas": {"alos25": float(N.loc["2025-03-01", am] / G.loc["2025-03-01", am]), "alos26": float(N.loc["2026-03-01", am] / G.loc["2026-03-01", am])},
           "scad_alos": {"2025-03": float(t1.loc["2025-03-01", "avg_length_of_stay_nights"]), "2026-03": float(t1.loc["2026-03-01", "avg_length_of_stay_nights"])},
           "bounds": {}}
    for k in ("low", "high"):
        fl = json.load(open(O / f"march_2026_flights_{k}.json", encoding="utf-8")); dm = json.load(open(O / f"march_2026_demand_net_of_flights_{k}.json", encoding="utf-8"))
        rf, ra = E.run(fl), E.run({"changes": fl["changes"] + dm["changes"]})
        f, a = _march(rf), _march(ra)
        rows = []
        for rg in sorted(set(region.values()), key=str):
            mk = [m for m, x in region.items() if x == rg]
            b = base.loc[mk].sum(); ff = f.loc[mk].sum(); aa = a.loc[mk].sum()
            rows.append({"region": rg, "checkins_normal": b.checkins_base, "checkins_replay_vs_normal": aa.checkins_scen / b.checkins_base,
                         "checkins_flights_only_vs_normal": ff.checkins_scen / b.checkins_base,
                         "guest_nights_normal": b.guests_base, "guest_nights_replay_vs_normal": aa.guests_scen / b.guests_base,
                         "guest_nights_flights_only_vs_normal": ff.guests_scen / b.guests_base,
                         "guest_nights_scad_vs_normal": obs_gn[rg] / b.guests_base})
        T = pd.DataFrame(rows)
        tb = base.sum(); tf = f.sum(); ta = a.sum()
        tot = {"checkins_replay_vs_normal": ta.checkins_scen / tb.checkins_base, "checkins_flights_only_vs_normal": tf.checkins_scen / tb.checkins_base,
               "guest_nights_replay_vs_normal": ta.guests_scen / tb.guests_base, "guest_nights_flights_only_vs_normal": tf.guests_scen / tb.guests_base,
               "guest_nights_scad_vs_normal": sum(obs_gn.values()) / tb.guests_base,
               "guest_nights_lost": float(tb.guests_base - ta.guests_scen), "guest_nights_lost_flights": float(tb.guests_base - tf.guests_scen),
               "checkins_lost": float(tb.checkins_base - ta.checkins_scen), "checkins_lost_flights": float(tb.checkins_base - tf.checkins_scen)}
        # stays: with normal stays the replay already lifts nights per check-in (fewer arrivals meet the guests already
        # in-house); SCAD shows more. Fit one stay change for March check-ins that reproduces SCAD's March guest nights
        # (it also absorbs the 28 Feb check-in spike and extensions by earlier arrivals, which no March setting reaches).
        tgt = tot["guest_nights_scad_vs_normal"]
        stays = lambda p_: [{"type": "stay_length", "markets": "all", "start": "2026-03-01", "end": "2026-03-31", "pct": float(p_)}]
        gn = lambda p_: float(_march(E.run({"changes": fl["changes"] + dm["changes"] + stays(p_)})).guests_scen.sum() / tb.guests_base)
        p_fit = brentq(lambda p_: gn(p_) - tgt, 0.0, 60.0, xtol=0.01)
        tot.update({"stay_change_fitted_pct": p_fit, "guest_nights_with_fitted_stays_vs_normal": gn(round(p_fit, 2)),
                    "nights_per_checkin_vs_normal_scad": tgt / tot["checkins_replay_vs_normal"],
                    "nights_per_checkin_vs_normal_replay_normal_stays": tot["guest_nights_replay_vs_normal"] / tot["checkins_replay_vs_normal"]})
        out["bounds"][k] = {"by_region": T, "total": {kk: float(v) for kk, v in tot.items()}, "chain_flights": chain(rf),
                            "replay_changes": fl["changes"] + dm["changes"] + stays(round(p_fit, 2))}
    out["feb28_spike"] = feb28_spike(E, cfg, float(base.guests_base.sum()))
    return out


def feb28_spike(E, cfg, march_normal_gn: float) -> dict:
    """Guest nights in March from the extra check-ins of 28 Feb 2026 (first war day, stranded travellers), which the
    normal world excludes by design (Step 3.2): Stage B with the actual 28 Feb check-ins in its memory."""
    P = cfg["paths"]["processed"]; d = pd.Timestamp("2026-02-28")
    hi = pd.read_parquet(P / "hotel_intl_daily.parquet"); hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    act = hi[hi.date == d].set_index("nationality").arrivals
    act.loc["DOMESTIC"] = float(hd[hd.date == d].arrivals.iloc[0])
    mem = E.stageb.memory; extra = 0.0; normal = float(mem.loc[d].sum())
    ck = E.world.checkins.sort_values("date")
    for m in mem.columns:
        if m not in act.index or not np.isfinite(act[m]) or act[m] == mem.loc[d, m]:
            continue
        a = ck[ck.series == m].checkins.to_numpy()
        g0 = E.stageb.guests(m, a)
        keep = mem.loc[d, m]
        try:
            mem.loc[d, m] = act[m]; g1 = E.stageb.guests(m, a)
        finally:
            mem.loc[d, m] = keep
        extra += float((g1 - g0)[:31].sum())
    return {"checkins_actual": float(act.reindex(mem.columns).fillna(mem.loc[d]).sum()), "checkins_normal": normal,
            "extra_march_guest_nights": extra, "points_of_normal_march_guest_nights": extra / march_normal_gn}


# ----------------------------------------------------------------------------- 2. Etihad launches, route by route
def etihad_launches(E, B, path) -> dict:
    s = sc.load(path)
    total = B.run(E, s)
    routes = []
    for city, ch in pd.Series(s["changes"]).groupby([c["city"] for c in s["changes"]]):
        ch = list(ch)
        r = B.run(E, {"name": city, "changes": ch})
        routes.append({"route": f"Abu Dhabi–{city}", "market": ch[0]["market"], "start": min(c["start"] for c in ch),
                       "flights_per_week": "/".join(f"{c['flights_per_week']:g}" for c in ch), "status": ch[-1]["status"], **chain(r)})
    routes.sort(key=lambda x: -x["guest_nights"])
    return {"total": chain(total), "routes": routes, "sum_of_routes_guest_nights": float(sum(x["guest_nights"] for x in routes))}


# ----------------------------------------------------------------------------- 3. European carriers' return
def european_return(E, B, path) -> dict:
    s = sc.load(path)
    away = B.run(E, s)
    slip = {"name": "one-month slip", "changes": [c | {"end": str((pd.Timestamp(c["start"]) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date())}
                                                 for c in s["changes"]]}
    sl = B.run(E, slip)
    by = {}
    for c in s["changes"]:
        r = B.run(E, {"changes": [c]})
        by[f"{c['airline']} ({c['market'].title()})"] = chain(r)
    return {"stay_away": chain(away), "one_month_slip": chain(sl), "by_carrier": by,
            "return_dates": {c["airline"]: c["start"] for c in s["changes"]}}


# ----------------------------------------------------------------------------- 4. recovery planner
def offset_rule(eps_values: dict, lo: float, hi: float, shortfalls=(5, 10, 20)) -> dict:
    """Increase in local passengers that offsets a demand shortfall d: G(x) = 1/(1−d), inside the trusted range
    x = (1/(1−d))^(1/ε); beyond it the engine's straight line."""
    out = {}
    for d in shortfalls:
        tgt = 1 / (1 - d / 100)
        out[f"{d}%"] = {k: float(brentq(lambda x: en.lever_G(x, e, lo, hi) - tgt, 1.0, 1e3) - 1) for k, e in eps_values.items()}
    return out


def recovery_planner(E, B, spf: float, markets: list, start: str, end: str, shortfall_pct: float) -> list:
    rows = []
    r_ = E.world.routes
    for m in markets:
        dem = [{"type": "demand", "markets": m, "start": start, "end": end, "pct": -shortfall_pct}]
        lost = E.run({"changes": dem}).summary["d_guest_nights"]
        q = sn.seats_needed(E, B, m, start, end, 0.0, spf, base_changes=dem)
        mo = pd.date_range(start, end).month.unique()
        cur = float(r_[(r_.nationality_key == m) & r_.moy.isin(mo)].groupby("moy").flights_per_week.sum().mean())
        rows.append({"market": m, "shortfall_pct": shortfall_pct, "guest_nights_lost": lost, "current_flights_per_week": cur,
                     "flights_per_week_needed": q["flights_per_week"], "range": q["flights_per_week_range"],
                     "range_lever_only": q["flights_per_week_range_lever_only"],
                     "extra_vs_current": q["flights_per_week"] / cur if q["flights_per_week"] else None})
    return rows
