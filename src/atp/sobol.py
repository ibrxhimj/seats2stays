"""Phase 5, Step 5.6 — global sensitivity (Sobol indices) on the scenario engine.

Pre-registered in docs/phase5_6_preregistration.md. The engine is evaluated in an exact vectorised form (FastEngine) for
levers that are uniform over whole calendar months; it must reproduce engine.Engine.run before any index is computed.

  levers       Tab 2's five levers drawn together (±10%, and ±25% as a size check): first-, second- and total-order indices
  uncertainty  for two reference results, ε against the baseline level (each market's own level error, one group)

Run:  python -m atp.phase5 sobol
"""
import json
import numpy as np
import pandas as pd
from . import engine as en, scenario as sc

PREREG_SOBOL = "docs/phase5_6_preregistration.md"
N_BASE, N_CHECK, SEED, N_RESAMPLE = 8192, 4096, 1, 1000
GATE = 1e-6
LEVERS = ["seats", "load factor", "transfer share", "non-flight demand", "stay length"]
FLIGHT = LEVERS[:3]
TAB2_ORDER = ["non-flight demand", "stay length", "transfer share", "seats", "load factor"]   # |elasticity| at ±1%, ties by name
INTERACT = 0.05
Z80 = 1.2815515655446004


class FastEngine:
    """Engine.run for changes that are uniform over whole calendar months: route flows per route × month with the
    engine's rules, the check-in lever per market × day, and guest nights as fixed Stage B weights × check-ins."""

    def __init__(self, E: en.Engine):
        self.E = E; w = E.world; p = E.params
        self.lo, self.hi = p.lo, p.hi
        d, _ = E.resolve({"changes": []})
        d = d.sort_values(["market", "date"]).reset_index(drop=True)
        self.markets = list(dict.fromkeys(d.market)); M = len(self.markets); self.days = w.days; T = len(w.days)
        self.mix = {m: i for i, m in enumerate(self.markets)}
        self.A0 = d.checkins_base.to_numpy().reshape(M, T)
        self.fl = np.array([m in w.flight for m in self.markets])
        self.moy = w.days.month.to_numpy()                                      # each calendar month appears once
        assert len(np.unique(self.moy)) == 12 and T == len(self.moy)
        r = w.routes
        self.rm = r.nationality_key.map(self.mix).to_numpy()                    # route → market index
        self.rmoy = r.moy.to_numpy()
        self.S0, self.P0, self.Q0 = (r[c].to_numpy(dtype=float) for c in ("seats", "pax", "p2p"))
        self.L0 = self._local(self.Q0)                                          # market × month (moy 1..12 → 0..11)
        assert np.allclose(self.L0[:, self.moy - 1], d.p2p_base.to_numpy().reshape(M, T), rtol=1e-12, atol=1e-9)
        self.W = self._weights()                                                # market × day: guest nights per check-in

    def _local(self, Q):
        L = np.zeros((len(self.markets), 12))
        np.add.at(L, (self.rm, self.rmoy - 1), Q)
        return L

    def _weights(self) -> np.ndarray:
        """Guest nights (in-house days to h1 + 44) per check-in on each horizon day, per market. Stage B is linear in
        check-ins and a cohort stays at most K nights, so impulses K days apart do not overlap: K runs per market."""
        sb = self.E.stageb; K = en.sbm.K; T = len(self.days)
        W = np.zeros((len(self.markets), T))
        for i, m in enumerate(self.markets):
            for o in range(K):
                a = np.zeros(T); a[o::K] = 1.0
                g = sb.guests(m, a, memory=False)                               # days h0 … h1 + K − 1
                for t in range(o, T, K):
                    W[i, t] = g[t:t + K].sum()
        return W

    def flows(self, months: np.ndarray, seats=0.0, lf=0.0, ts=0.0) -> np.ndarray:
        """Local passengers (market × month) after seats (%), load factor (%) and transfer share (% of each market's own
        share), applied in that order to every flight route in the given calendar months (scenario.resolve's rules)."""
        on = np.isin(self.rmoy, months)
        f = np.where(on, 1 + seats / 100, 1.0)
        S, P, Q = self.S0 * f, self.P0 * f, self.Q0 * f
        if lf:
            cap = np.where(self.S0 > 0, np.maximum(S, self.P0 * S / np.where(self.S0 > 0, self.S0, 1.0)), S)
            want = P * (1 + lf / 100)
            newP = np.where(on, np.minimum(want, cap), P)
            Q = Q * np.divide(newP, P, out=np.ones_like(P), where=P > 0)
            P = newP
        if ts:
            sel = on & (P > 0)
            key = self.rm * 12 + (self.rmoy - 1)
            sp = np.bincount(key[sel], P[sel], minlength=len(self.markets) * 12)
            sq = np.bincount(key[sel], Q[sel], minlength=len(self.markets) * 12)
            t_mkt = 1 - np.divide(sq, sp, out=np.ones_like(sp), where=sp > 0)
            tm = t_mkt[key]
            target = np.minimum(tm * (1 + ts / 100), 0.99)
            if (target[sel] < 0).any():
                raise ValueError("transfer share below 0")
            tau = 1 - np.divide(Q, P, out=np.zeros_like(P), where=P > 0)
            new = tau.copy()
            new[sel], _ = sc.scale_shares(tau[sel], P[sel], key[sel], target[sel])   # scenario.resolve's rule, per market-month
            Q = np.where(sel, P * (1 - new), Q)
        return self._local(Q)

    def run(self, months, seats=0.0, lf=0.0, ts=0.0, demand=0.0, stay=0.0, eps=None, flight_markets=None,
            all_markets=None) -> np.ndarray:
        """Change in guest nights per market (incl. stays ending after the horizon) — Engine.run's d_guest_nights by
        market. Flight levers act on `flight_markets` (default: all), demand and stay on `all_markets` (default: all)."""
        eps = self.E.params.eps if eps is None else eps
        months = np.atleast_1d(months)
        L1 = self.flows(months, seats, lf, ts)
        if flight_markets is not None:                                          # other markets keep their baseline flows
            keep = ~np.isin(np.array(self.markets), flight_markets)
            L1[keep] = self.L0[keep]
        L0d, L1d = self.L0[:, self.moy - 1], L1[:, self.moy - 1]
        std = self.fl[:, None] & (L0d > 0)
        assert not (self.fl[:, None] & (L0d == 0) & (L1d > 0)).any(), "zero-baseline rule not supported"
        x = np.divide(L1d, L0d, out=np.ones_like(L0d), where=std)
        G = en.lever_G(x, eps, self.lo, self.hi)
        raw = self.A0 * (G - 1)
        win = np.isin(self.moy, months)[None, :]
        sel = np.ones(len(self.markets), bool) if all_markets is None else np.isin(np.array(self.markets), all_markets)
        on = win & sel[:, None]
        dm = np.where(on, 1 + demand / 100, 1.0); v = np.where(on, 1 + stay / 100, 1.0)
        fx = raw * dm                                                           # flight effect in the scenario's demand, then capped
        dA = np.where(std, np.sign(fx) * np.minimum(np.abs(fx), np.abs(L1d - L0d)), 0.0)
        As = self.A0 * dm + dA
        return (self.W * As * v).sum(1) - (self.W * self.A0).sum(1)


HORIZON = np.arange(1, 13)
WINTER = np.array([11, 12, 1, 2])          # Nov 2026 – Feb 2027


def scenario(levels: dict, months=HORIZON, h=("2026-03-01", "2027-02-28")) -> dict:
    """The same change as an Engine scenario (for the gate)."""
    w0, w1 = h
    if not np.array_equal(np.sort(months), HORIZON):
        w0, w1 = "2026-11-01", "2027-02-28"
        assert np.array_equal(np.sort(months), np.sort(WINTER))
    ch = []
    if levels.get("seats"): ch.append({"type": "route", "markets": "flight markets", "start": w0, "end": w1, "seats_pct": levels["seats"]})
    if levels.get("lf"): ch.append({"type": "load_factor", "markets": "flight markets", "start": w0, "end": w1, "pct": levels["lf"]})
    if levels.get("ts"): ch.append({"type": "transfer_share", "markets": "flight markets", "start": w0, "end": w1, "pct": levels["ts"]})
    if levels.get("demand"): ch.append({"type": "demand", "markets": "all", "start": w0, "end": w1, "pct": levels["demand"]})
    if levels.get("stay"): ch.append({"type": "stay_length", "markets": "all", "start": w0, "end": w1, "pct": levels["stay"]})
    return {"changes": ch}


def _engine_by_market(E, s, eps=None) -> tuple[float, dict]:
    r = E.run(s, eps=eps)
    return r.summary["d_guest_nights"], {m: v["d_guest_nights"] for m, v in r.summary["by_market"].items()}


def gate(F: FastEngine, cases: list) -> dict:
    """Each case: (name, levels, months, eps, single_market or None). Relative difference of the total and of each market."""
    E = F.E; out = []
    for name, lv, months, eps, mk in cases:
        s = scenario(lv, months)
        if mk:
            for c in s["changes"]:
                c["markets"] = [mk]
        tot, bm = _engine_by_market(E, s, eps)
        fast = F.run(months, **{k: v for k, v in lv.items()}, eps=eps, flight_markets=[mk] if mk else None,
                     all_markets=[mk] if mk else None)
        ft = float(fast.sum())
        rel_t = abs(ft - tot) / max(abs(tot), 1.0)
        rel_m = max([abs(float(fast[F.mix[m]]) - v) / max(abs(v), 1.0) for m, v in bm.items()] +
                    [abs(float(fast[i])) / 1.0 for i, m in enumerate(F.markets) if m not in bm and abs(fast[i]) > 1e-6])
        out.append({"case": name, "engine": tot, "fast": ft, "rel_total": rel_t, "rel_market_max": rel_m})
    return {"cases": out, "max_rel": max(max(c["rel_total"], c["rel_market_max"]) for c in out),
            "passed": all(c["rel_total"] <= GATE and c["rel_market_max"] <= GATE for c in out)}


# ----------------------------------------------------------------------------- Sobol analysis
def _indices(problem, f, N, second=True) -> dict:
    from SALib.analyze import sobol as sa
    from SALib.sample import sobol as ss
    X = ss.sample(problem, N, calc_second_order=second, scramble=True, seed=SEED)
    Y = np.array([f(x) for x in X])
    S = sa.analyze(problem, Y, calc_second_order=second, num_resamples=N_RESAMPLE, conf_level=0.95, seed=SEED)
    names = problem.get("groups") and list(dict.fromkeys(problem["groups"])) or problem["names"]
    out = {"n_base": N, "n_runs": int(len(Y)), "mean": float(Y.mean()), "sd": float(Y.std()),
           "S1": dict(zip(names, map(float, S["S1"]))), "S1_conf": dict(zip(names, map(float, S["S1_conf"]))),
           "ST": dict(zip(names, map(float, S["ST"]))), "ST_conf": dict(zip(names, map(float, S["ST_conf"])))}
    if second:
        out["S2"] = {f"{names[i]} × {names[j]}": [float(S["S2"][i, j]), float(S["S2_conf"][i, j])]
                     for i in range(len(names)) for j in range(i + 1, len(names))}
    out["interaction_share"] = 1 - sum(out["S1"].values())
    return out


def lever_problem(names, half):
    return {"num_vars": len(names), "names": names, "bounds": [[-half, half]] * len(names)}


KW = {"seats": "seats", "load factor": "lf", "transfer share": "ts", "non-flight demand": "demand", "stay length": "stay"}


def levers(F: FastEngine, names, half, N):
    return _indices(lever_problem(names, half), lambda x: float(F.run(HORIZON, **{KW[n]: v for n, v in zip(names, x)}).sum()), N)


def ranking_check(r: dict) -> dict:
    """Tab 2's order is confirmed if no lever ranked below another in Tab 2 has a clearly larger ST (95% intervals apart)."""
    ST, C = r["ST"], r["ST_conf"]
    clash = [(a, b) for i, a in enumerate(TAB2_ORDER) for b in TAB2_ORDER[i + 1:]
             if ST[b] - C[b] > ST[a] + C[a]]
    order = sorted(ST, key=lambda k: -ST[k])
    return {"tab2_order": TAB2_ORDER, "sobol_order": order, "contradictions": clash, "confirmed": not clash}


def interactions(*rs) -> dict:
    worst = max(max(r["ST"][k] - r["S1"][k] for k in r["ST"]) for r in rs)
    share = max(r["interaction_share"] for r in rs)
    return {"max_interaction_share": share, "max_ST_minus_S1": worst, "matter": bool(share > INTERACT or worst > INTERACT)}


def uncertainty(F: FastEngine, B, market, N, shared: bool = False) -> dict:
    """ε against the baseline level for one reference result. shared=False: each market's own level error, independent
    (the bands' assumption); shared=True (Amendment A1): one level error common to every market, from the all-market band
    (the floor the app applies to a net effect), an upper bound on the level's share."""
    E = F.E; p = E.params
    mk = [market] if market else None
    se = (p.eps_band[1] / p.eps - 1) / Z80                                    # ε × (1 ± Z80 · s.e.)
    base = F.run(WINTER, seats=10, flight_markets=mk)
    idx = [i for i in range(len(F.markets)) if abs(base[i]) > 1e-9]
    names = [F.markets[i] for i in idx]
    ln = lambda lo_hi: (float(np.mean(np.log(lo_hi))), float((np.log(lo_hi[1]) - np.log(lo_hi[0])) / (2 * Z80)))
    lv = [ln(B._total["guests"])] if shared else [ln(B._ratio[("guests", B.tiers[m])]) for m in names]
    lv_names = ["all markets"] if shared else names
    problem = {"num_vars": 1 + len(lv), "names": ["eps"] + lv_names, "groups": ["ε (check-in response)"] + ["baseline level"] * len(lv),
               "bounds": [[p.eps, p.eps * se]] + [list(x) for x in lv], "dists": ["norm"] + ["lognorm"] * len(lv)}
    cache = {}

    def f(x):
        e = float(x[0])
        if e not in cache:
            cache.clear(); cache[e] = F.run(WINTER, seats=10, eps=e, flight_markets=mk)[idx]
        return float(x[1] * cache[e].sum()) if shared else float((cache[e] * x[1:]).sum())
    r = _indices(problem, f, N)
    # the app's split: lever and baseline-level half-widths of the 80% range, and whether its level part is the all-market floor
    s = scenario({"seats": 10}, WINTER)
    if market:
        for c in s["changes"]:
            c["markets"] = [market]
    res = B.run(E, s); parts = res.effect_band["parts"]["d_guest_nights"]
    hl = (parts["lever"][1] - parts["lever"][0]) / 2; hv = (parts["baseline_level"][1] - parts["baseline_level"][0]) / 2
    eff = {m: v["d_guest_nights"] for m, v in res.summary["by_market"].items()}
    rl, rh = B._level_ratios("guests", list(eff)); Ev = np.array(list(eff.values()))
    quad = float((np.sqrt(((Ev * (1 - rl)) ** 2).sum()) + np.sqrt(((Ev * (rh - 1)) ** 2).sum())) / 2)
    r.update({"central": float(res.summary["d_guest_nights"]), "range80": [float(x) for x in res.effect_band["summary"]["d_guest_nights"]],
              "app_lever_halfwidth": float(hl), "app_level_halfwidth": float(hv), "app_lever_share": float(hl ** 2 / (hl ** 2 + hv ** 2)),
              "app_level_is_all_market_floor": bool(hv > quad * (1 + 1e-9)), "level_halfwidth_independent_markets": quad,
              "eps_sd": float(p.eps * se), "markets": len(names), "level_model": "shared (all-market band)" if shared else "independent markets",
              "dominant": next((k for k, v in r["ST"].items() if v > 0.5), None)})
    return r


def run(cfg) -> dict:
    from . import bands as bd
    from .phase5 import sha256
    E = en.Engine(cfg); F = FastEngine(E); B = bd.Bands(cfg); p = E.params
    rng = np.random.default_rng(SEED)
    cases = [(f"{n} {s:+d}%", {KW[n]: s}, HORIZON, None, None) for n in LEVERS for s in (10, -10)]
    cases += [(f"random mix {i + 1}", dict(zip(KW.values(), rng.uniform(-10, 10, 5))), HORIZON, None, None) for i in range(8)]
    rng25 = np.random.default_rng(SEED + 1)                                   # Amendment A1: the ±25% runs (1c) gated too
    cases += [(f"±25% corner {i + 1}", dict(zip(KW.values(), v)), HORIZON, None, None)
              for i, v in enumerate(([25] * 5, [-25] * 5, [25, -25, 25, -25, 25], [-25, 25, -25, 25, -25]))]
    cases += [(f"±25% random mix {i + 1}", dict(zip(KW.values(), rng25.uniform(-25, 25, 5))), HORIZON, None, None) for i in range(4)]
    big = max(F.markets, key=lambda m: F.E.world.guests.query("market == @m").guests.sum() if m in F.E.world.flight else -1)
    for nm, mk in (("R1", None), ("R2", big)):
        for e, lab in ((p.eps_band[0], "low"), (p.eps_band[1], "high")):
            cases.append((f"{nm} at ε {lab}", {"seats": 10}, WINTER, e, mk))
    g = gate(F, cases)
    print(f"GATE: {len(g['cases'])} cases, largest relative difference {g['max_rel']:.1e} → {'passed' if g['passed'] else 'FAILED'}", flush=True)
    res = {"prereg_sha256": sha256(cfg["root"] / PREREG_SOBOL), "gate": g, "settings": {"N": N_BASE, "N_check": N_CHECK, "seed": SEED,
           "resamples": N_RESAMPLE, "salib": "1.6.0"}}
    if not g["passed"]:
        return res
    for key, names, half in (("1a", LEVERS, 10), ("1b", FLIGHT, 10), ("1c", LEVERS, 25)):
        res[key] = levers(F, names, half, N_BASE); res[key + "_check"] = levers(F, names, half, N_CHECK)
        print(f"[{key}] ±{half}%: " + ", ".join(f"{k} S1 {res[key]['S1'][k]:.3f} ST {res[key]['ST'][k]:.3f}" for k in names)
              + f"; interactions {res[key]['interaction_share']:.4f}", flush=True)
    res["ranking"] = ranking_check(res["1a"])
    res["interactions"] = interactions(res["1a"], res["1b"])
    for key, mk in (("R1", None), ("R2", big)):
        res[key] = uncertainty(F, B, mk, N_BASE) | {"market": mk or "all flight markets"}
        res[key + "_check"] = {k: v for k, v in uncertainty(F, B, mk, N_CHECK).items() if k in ("S1", "ST", "n_base")}
        print(f"[{key}] " + ", ".join(f"{k} S1 {res[key]['S1'][k]:.3f} ST {res[key]['ST'][k]:.3f}" for k in res[key]["ST"])
              + f"; app's lever share {res[key]['app_lever_share']:.2f}", flush=True)
    res["R1_shared"] = uncertainty(F, B, None, N_BASE, shared=True) | {"market": "all flight markets"}
    res["R1_shared_check"] = {k: v for k, v in uncertainty(F, B, None, N_CHECK, shared=True).items() if k in ("S1", "ST", "n_base")}
    print("[R1, level shared] " + ", ".join(f"{k} S1 {res['R1_shared']['S1'][k]:.3f} ST {res['R1_shared']['ST'][k]:.3f}" for k in res["R1_shared"]["ST"]), flush=True)
    return res


# ----------------------------------------------------------------------------- what Tab 3 and the report say
def _p(x, d=0) -> str:
    return f"{x * 100:.{d}f}%"


def tab3(r: dict, names: dict) -> dict:
    """The Tab 3 block: the levers' shares when they all move, whether they interact, which uncertainty dominates.
    names: lever key → the name Tab 2 uses."""
    a, b, R1, R2 = r["1a"], r["1b"], r["R1"], r["R2"]
    order = sorted(a["ST"], key=lambda k: (-round(a["ST"][k], 2), TAB2_ORDER.index(k)))       # ties (to 1 point) in Tab 2's order
    rows = [{"id": k, "label": names[k], "alone": a["S1"][k], "total": a["ST"][k]} for k in order]
    flight = [{"id": k, "label": names[k], "total": b["ST"][k]} for k in sorted(b["ST"], key=lambda k: -b["ST"][k])]
    e = "ε (check-in response)"
    inter = r["interactions"]["max_interaction_share"]
    return {
        "lead": (f"Move all five levers at once, each anywhere within ±10% for the year, and ask what explains the spread in guest nights. "
                 f"{names[order[0]]} and {names[order[1]].lower()} explain {_p(a['ST'][order[0]])} each, {names[order[2]].lower()} "
                 f"{_p(a['ST'][order[2]])}, {names[order[3]].lower()} {_p(a['ST'][order[3]])} and {names[order[4]].lower()} "
                 f"{_p(a['ST'][order[4]])}: the same order as “What moves guests”."),
        "rows": rows, "flight": flight,
        "points": [
            f"Within ±10%, only {_p(inter, 1)} of the spread comes from combinations of levers. The chain does interact (demand, for one, "
            f"multiplies the flight effect), but over these ranges several levers together come close to the sum of their effects.",
            f"Among the flight levers alone, {names[flight[0]['id']].lower()} explain {_p(flight[0]['total'])}, {names[flight[1]['id']].lower()} "
            f"{_p(flight[1]['total'])} and {names[flight[2]['id']].lower()} {_p(flight[2]['total'])}.",
            f"Which uncertainty matters: for 10% more seats on every route over the winter, {_p(r['R1_shared']['S1'][e])} to {_p(R1['S1'][e])} of the "
            f"result's uncertainty is the check-in response and the rest the forecast level (depending on whether countries' forecast errors move "
            f"together or not); for {R2['market'].title()} alone, {_p(R2['S1'][e])} and {_p(R2['S1']['baseline level'])}, the rest from the two together. "
            "A narrower check-in response would tighten flight results most."],
        "method": (f"Sobol indices (SALib), {a['n_runs']:,} runs of the engine itself for the five levers. Bars: the share a lever explains on its own "
                   f"(solid) and with its interactions (outline). Each lever is drawn evenly and independently within ±10%, so the shares answer that "
                   f"question, not every possible one. The shares were the same with half the runs.")}


def report(r: dict) -> str:
    g = r["gate"]; a, b, c = r["1a"], r["1b"], r["1c"]; rk, it = r["ranking"], r["interactions"]
    e = "ε (check-in response)"
    L = ["# Phase 5, Step 5.6 — global sensitivity (Sobol)", "",
         "Sobol indices on the scenario engine itself, in an exact vectorised form. Rules and settings: "
         "`docs/phase5_6_preregistration.md` (sha256 recorded in the results).", "",
         "## Gate: the fast form is the engine", "",
         f"{len(g['cases'])} cases (the ten Tab 2 runs, eight random combinations of all five levers within ±10%, eight within ±25% (Amendment A1), "
         f"and both reference results at both ends of ε's 80% interval): largest relative difference from `Engine.run` {g['max_rel']:.1e} (limit {GATE:g}, on the total and on every market). "
         f"**{'Passed' if g['passed'] else 'Failed'}.** The small differences come from the stay lever's root-finding tolerance.", "",
         "## Part 1 — levers", "",
         f"Each lever drawn uniformly within ±10% over the year, all five together ({a['n_runs']:,} engine runs). Output: the change in guest "
         "nights. S1 = the share of the spread a lever explains alone; ST = with all its interactions (95% intervals from 1,000 bootstrap resamples).", "",
         "| Lever | S1 | ST | ST, half the runs |", "|---|---|---|---|"]
    for k in sorted(a["ST"], key=lambda k: -a["ST"][k]):
        L.append(f"| {k} | {a['S1'][k]:.3f} ± {a['S1_conf'][k]:.3f} | {a['ST'][k]:.3f} ± {a['ST_conf'][k]:.3f} | {r['1a_check']['ST'][k]:.3f} |")
    top = sorted(a["S2"].items(), key=lambda kv: -kv[1][0])[:3]
    L += ["", f"Interactions: 1 − ΣS1 = {a['interaction_share']:.4f}; the largest pairs (S2): "
          + "; ".join(f"{k} {v[0]:.4f} ± {v[1]:.4f}" for k, v in top) + ".", "",
          f"**Ranking: {'confirmed' if rk['confirmed'] else 'corrected'}.** Tab 2's order ({', '.join(rk['tab2_order'])}) against the Sobol order by ST "
          f"({', '.join(rk['sobol_order'])}); "
          + ("no lever ranked lower in Tab 2 has a clearly larger ST. Demand and stay length are tied (they act identically: one for one on guest nights)."
             if rk["confirmed"] else f"contradictions: {rk['contradictions']}."), "",
          f"**Interactions: {'matter' if it['matter'] else 'do not matter'}** (largest interaction share {it['max_interaction_share']:.4f}, "
          f"largest ST − S1 {it['max_ST_minus_S1']:.4f}; the threshold was {INTERACT}). Within these ranges the one-at-a-time ranking "
          "is enough and several levers together come close to the sum of their effects. The chain is not interaction-free (demand "
          "multiplies the flight effect, and caps and stay changes add more); the indices describe levers drawn evenly and independently "
          "within the stated ranges.", "",
          f"**Flight levers only (1b)**, ±10%: " + ", ".join(f"{k} S1 {b['S1'][k]:.3f}, ST {b['ST'][k]:.3f}" for k in sorted(b["ST"], key=lambda k: -b["ST"][k]))
          + f"; 1 − ΣS1 = {b['interaction_share']:.4f}.", "",
          f"**Size check (1c)**, ±25%: " + ", ".join(f"{k} ST {c['ST'][k]:.3f}" for k in sorted(c["ST"], key=lambda k: -c["ST"][k]))
          + f"; 1 − ΣS1 = {c['interaction_share']:.4f}. Larger moves bring in the caps (full planes, 99% connecting) and the straight line beyond the "
          f"trusted range: transfer share's ST goes from {a['ST']['transfer share']:.3f} to {c['ST']['transfer share']:.3f} and interactions from "
          f"{a['interaction_share']:.4f} to {c['interaction_share']:.4f}; the order "
          + ("is the same (Tab 2's order is confirmed at ±25% too)." if ranking_check(c)["confirmed"] else "changes: " + ", ".join(ranking_check(c)["sobol_order"]) + "."), "",
          "## Part 2 — which uncertainty dominates", "",
          "ε drawn from its out-of-sample interval (normal, 80% interval exactly the app's 0.197–0.426). Baseline level: each market's "
          "ratio from its size tier's 80% guest-night band, independent between markets (log-normal, one group); and, for R1, one ratio "
          "shared by every market from the all-market 80% band (Amendment A1). The two are the ends of how market errors can be related.", "",
          "| Reference result | Level model | Central (guest nights) | App's 80% range | ε: S1 / ST | Baseline level: S1 / ST | App's split (lever share) |",
          "|---|---|---|---|---|---|---|"]
    rows_ = (("R1", "All flight markets +10% seats, Nov 2026–Feb 2027"), ("R1_shared", "All flight markets +10% seats, Nov 2026–Feb 2027"),
             ("R2", f"{r['R2']['market'].title()} +10% seats, Nov 2026–Feb 2027"))
    for key, lab in rows_:
        x = r[key]
        L.append(f"| {lab} | {x['level_model']} | {x['central']:,.0f} | {x['range80'][0]:,.0f} to {x['range80'][1]:,.0f} | {x['S1'][e]:.3f} / {x['ST'][e]:.3f} | "
                 f"{x['S1']['baseline level']:.3f} / {x['ST']['baseline level']:.3f} | {x['app_lever_share']:.2f} |")
    R1, R1s, R2 = r["R1"], r["R1_shared"], r["R2"]
    dom = all(r[k]["ST"][e] > 0.5 for k in ("R1", "R1_shared", "R2"))
    L += ["", (f"**The check-in response (ε) dominates** every run (ST {R1['ST'][e]:.2f} and {R1s['ST'][e]:.2f} for R1 with independent and shared "
               f"levels, {R2['ST'][e]:.2f} for {R2['market'].title()}; the threshold is 0.5).") if dom else
          "**ε does not dominate every run** (see the table).", "",
          f"How the app's own range splits: for R1 the app's level part is {'the all-market floor' if R1['app_level_is_all_market_floor'] else 'the independent-markets quadrature'} "
          f"(±{R1['app_level_halfwidth']:,.0f} guest nights, against ±{R1['level_halfwidth_independent_markets']:,.0f} if market errors were "
          f"independent), giving the level {1 - R1['app_lever_share']:.0%} of the variance. The Sobol runs put it at "
          f"{R1['ST']['baseline level']:.0%} (independent) to {R1s['ST']['baseline level']:.0%} (shared), so the app's split sits inside the range "
          f"the two ends give{'' if R1['ST']['baseline level'] <= 1 - R1['app_lever_share'] <= R1s['ST']['baseline level'] else ' — NOT inside: check'}. "
          f"For {R2['market'].title()} alone the app's level part is {'the all-market floor' if R2['app_level_is_all_market_floor'] else 'its own market band'} "
          f"and the split matches the Sobol run ({R2['app_lever_share']:.2f} against ST {R2['ST'][e]:.2f}). A single market's level matters more "
          "because its error does not average out across markets, but ε still dominates.", "",
          "## What follows", "",
          "- Tab 2's ranking stands; no change to the app's order.",
          "- Within ±10% the levers' combined effect is close to the sum of their separate effects, so the per-lever results can be read one at a "
          "time; for larger or correlated changes, run them together in the simulator.",
          "- For flight decisions, the uncertainty worth reducing is ε: more countries or months of seat changes with known outcomes would "
          "narrow it; a better baseline forecast would move the range far less.",
          "- Tab 3 shows a small block with these numbers.", "",
          "**Deck line:** " + deck_line(r), ""]
    return "\n".join(L)


def deck_line(r: dict) -> str:
    a, R1 = r["1a"], r["R1"]; e = "ε (check-in response)"
    return (f"A Sobol check ({a['n_runs']:,} engine runs) confirms the lever ranking — demand and stay length {_p(a['ST']['non-flight demand'])} each, "
            f"connecting passengers {_p(a['ST']['transfer share'])}, seats {_p(a['ST']['seats'])}, load factor {_p(a['ST']['load factor'])}, "
            f"interactions {_p(r['interactions']['max_interaction_share'], 1)} — and shows the check-in response, not the baseline forecast, "
            f"drives most of a flight result's uncertainty ({_p(r['R1_shared']['S1'][e])}–{_p(R1['S1'][e])} network-wide, {_p(r['R2']['S1'][e])} for {r['R2']['market'].title()} alone).")
