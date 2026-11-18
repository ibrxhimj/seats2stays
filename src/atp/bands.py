"""Phase 3, Step 3.5 — uncertainty bands. Two bands, never mixed:

  EFFECT band  how sure we are about the CHANGE a scenario causes (extra or lost check-ins and guest nights).
               Two independent parts, combined in quadrature:
               (1) the lever: ε's 80% interval from the official test window (Aug 2025–Feb 2026), where the default ε
               was checked out of sample against actual check-in changes in 31 countries (calibration slope 1.00,
               country-clustered s.e. 0.29) → ε × (1 ± 1.2816 · s.e.); new routes in nationalities without flight data
               use Step 2.5's 80% band per added seat. The engine is re-run at both ends. Cuts beyond the trusted range
               also span the alternative extrapolation (log form continued, flat below the deepest training cut): the
               adopted straight line under-predicted the out-of-sample Wizz Air exit (Step 3.7). Assumption levers
               (demand, stay length) have no lever part: they are exactly what the user assumed.
               (2) the baseline level: every change scales with a baseline that is itself forecast, so each market's
               level band (below) applies to its share of the change; markets add in quadrature (their errors are
               close to independent on the folds), never below the all-market band on the net change.

  LEVEL band   how sure we are about a LEVEL (baseline or scenario check-ins and guest nights in a month), i.e. the
               end-to-end forecast error flights → check-ins → guests. Calibrated on Phase 2's two validation folds
               (A: Aug 2024–Feb 2025, B: Jan–Jul 2025, each 1–7 months ahead): monthly ratios actual / predicted, pooled
               within market-size tiers (a single market has only 7 months per fold, too few for its own band — those
               bands covered only 39–50% across folds, step35 level_diagnostics). The band is the narrowest symmetric quantile pair whose coverage
               is ≥ 80% when learned on one fold and tested on the other, both ways (Phase 1's rule). The all-market
               monthly total has its own band.

Beyond 7 months after the data the level band — and the baseline-level part of an effect band — is flagged as
uncalibrated (the folds reach 7 months; within them the error did not grow with lead). The level band is conditional
on the flight schedule (the folds were forecast from the seats actually flown), which is the tool's input. Both bands
describe the normal world: the 2026 war is a scenario (demand presets), not forecast error.
"""
import json
import numpy as np
import pandas as pd
from .config import load_config

Z80 = 1.2815515655446004            # two-sided 80% normal quantile
TIERS = [(1, 10, "top 10"), (11, 25, "11–25"), (26, 99, "26–46")]
GRID = np.round(np.arange(0.10, 0.0049, -0.005), 3)


def effect_band(cfg) -> dict:
    P = cfg["paths"]["processed"] / "phase2"
    j23 = json.load(open(P / "step23_lever_test.json", encoding="utf-8")); j25 = json.load(open(P / "step25_new_route.json", encoding="utf-8"))
    s26 = json.load(open(P / "step26_endtoend.json", encoding="utf-8")); sl = s26["test_window"]["lever_check"]["calibration_slope"]
    eps = float(j23["decision"]["default_eps_total"])
    from .stagea import lever_interval
    e_lo, e_hi = lever_interval(s26)                                   # the project's one ε interval (Step 2.6)
    assert abs(s26["test_window"]["lever_check"]["eps"] - eps) < 1e-12
    boot = np.load(P / "eps_boot_final.npz")["eps_total_lead"]
    return {"eps": eps, "eps_lo": e_lo, "eps_hi": e_hi, "per_seat": float(j25["checkins_per_added_seat"]),
            "per_seat_lo": float(j25["band"][0]), "per_seat_hi": float(j25["band"][1]),
            "cut_floor": cut_floor(cfg),
            "source": {"test_window_slope": sl["estimate"], "slope_se_cluster": sl["se_cluster"],
                       "slope_ci95_cluster": sl["ci_cluster"], "n_countries": sl["n_countries"]},
            "cross_checks": {"bootstrap_80pct_train_window": [float(np.percentile(boot, 10)), float(np.percentile(boot, 90))],
                             "phase2_simulator_band": j23["decision"]["simulator_band"]}}


def cut_floor(cfg) -> float:
    """Deepest year-on-year cut in local passengers in the training data (both months ≥ 2023): below it the
    alternative cut rule (log form) is held flat — no extrapolation beyond what was observed."""
    from . import phase2 as p2, stagea as sa
    fd, _, hi = p2._load(cfg)
    y = sa.yoy(sa.truncate(sa.monthly_panel(fd, hi), cfg["folds"]["final"]["train_end"]))
    return float(np.exp(y.loc[y.base_month >= p2.BASE_START, "d_p2p"].min()))


def _fold_monthly(cfg, fold: str, kind: str) -> pd.DataFrame:
    P = cfg["paths"]["processed"] / "phase2"
    if kind == "guests":
        p = pd.read_parquet(P / f"endtoend_pred_{fold}.parquet").rename(columns={"market": "m", "p": "pred"})
    else:
        p = pd.read_parquet(P / f"stagea_pred_{fold}.parquet").rename(columns={"key": "m"})
    p = p.dropna(subset=["y"])
    return p.assign(month=p.date.dt.to_period("M")).groupby(["m", "month"])[["y", "pred"]].sum().reset_index().assign(fold=fold)


def market_tiers(cfg) -> pd.Series:
    """Size tier of each market: rank of its actual check-ins over both folds (fixed once, used everywhere)."""
    v = pd.concat([_fold_monthly(cfg, f, "checkins") for f in "AB"]).groupby("m").y.sum().rank(ascending=False, method="first")
    return v.map(lambda r: next(name for a, b, name in TIERS if a <= r <= b)).rename("tier")


def _band(r: pd.Series, q: float) -> tuple:
    return float(r.quantile(q)), float(r.quantile(1 - q))


def calibrate_level(cfg) -> tuple[pd.DataFrame, dict]:
    tiers = market_tiers(cfg)
    rows, diag = [], {}
    for kind in ("guests", "checkins"):
        F = {f: _fold_monthly(cfg, f, kind).assign(tier=lambda x: x.m.map(tiers), r=lambda x: x.y / x.pred) for f in "AB"}

        def cover(learn, test, q):
            b = {t: _band(g.r, q) for t, g in learn.groupby("tier")}
            lo, hi = test.tier.map(lambda t: b[t][0]), test.tier.map(lambda t: b[t][1])
            return float(((test.r >= lo) & (test.r <= hi)).mean())
        q = next((q for q in GRID if min(cover(F["A"], F["B"], q), cover(F["B"], F["A"], q)) >= 0.80), 0.0)
        both = pd.concat(F.values())
        for t, g in both.groupby("tier"):
            lo, hi = _band(g.r, q)
            rows.append({"kind": kind, "level": "market-month", "tier": t, "q": q, "ratio_lo": lo, "ratio_hi": hi, "n": len(g)})
        # all-market monthly total (months in both folds counted once per fold)
        T = {f: d.groupby("month")[["y", "pred"]].sum().assign(r=lambda x: x.y / x.pred) for f, d in F.items()}
        qt = next((q_ for q_ in GRID if min(
            ((T["B"].r >= _band(T["A"].r, q_)[0]) & (T["B"].r <= _band(T["A"].r, q_)[1])).mean(),
            ((T["A"].r >= _band(T["B"].r, q_)[0]) & (T["A"].r <= _band(T["B"].r, q_)[1])).mean()) >= 0.80), 0.0)
        tr = pd.concat(T.values()).r
        lo, hi = _band(tr, qt)
        tcov = [float(((T[b].r >= _band(T[a].r, qt)[0]) & (T[b].r <= _band(T[a].r, qt)[1])).mean()) for a, b in (("A", "B"), ("B", "A"))]
        rows.append({"kind": kind, "level": "total-month", "tier": "all markets", "q": qt, "ratio_lo": lo, "ratio_hi": hi, "n": len(tr)})
        diag[kind] = {"q_market_month": float(q), "coverage_A_to_B": cover(F["A"], F["B"], q), "coverage_B_to_A": cover(F["B"], F["A"], q),
                      "q_total": float(qt), "total_coverage_A_to_B": tcov[0], "total_coverage_B_to_A": tcov[1],
                      "total_note": "7 months per fold: no quantile pair reached 80% across folds, so the total band is the "
                                    "full observed range of the 14 fold-months" if qt == 0 else "",
                      "own_market_bands_coverage_A_to_B": _own_market_cover(F["A"], F["B"]),
                      "own_market_bands_coverage_B_to_A": _own_market_cover(F["B"], F["A"]),
                      "coverage_by_tier": {f"{a}_to_{b}": _tier_cover(F[a], F[b], q) for a, b in (("A", "B"), ("B", "A"))},
                      "median_abs_log_error_by_months_ahead": {f: _by_lead(F[f]) for f in "AB"}}
    return pd.DataFrame(rows), {"tiers": tiers.to_dict(), "diagnostics": diag}


def _tier_cover(learn, test, q) -> dict:
    b = {t: _band(g.r, q) for t, g in learn.groupby("tier")}
    ok = (test.r >= test.tier.map(lambda t: b[t][0])) & (test.r <= test.tier.map(lambda t: b[t][1]))
    return {t: float(v) for t, v in ok.groupby(test.tier).mean().items()}


def _by_lead(d) -> dict:
    """Market-month median |log(actual/predicted)| by months ahead (1 = first fold month): does error grow with lead?"""
    lead = (d.month - d.month.min()).map(lambda x: x.n) + 1
    return {int(k): float(v) for k, v in np.log(d.y / d.pred).abs().groupby(lead).median().items()}


def _own_market_cover(learn, test, q=0.10):
    b = learn.groupby("m").r.quantile([q, 1 - q]).unstack()
    t = test.join(b, on="m")
    return float(((t.r >= t[q]) & (t.r <= t[1 - q])).mean())


class Bands:
    """Applies the calibrated bands to engine results."""

    def __init__(self, cfg=None):
        cfg = cfg or load_config(); P = cfg["paths"]["processed"] / "phase3"
        self.effect = json.load(open(P / "effect_band.json", encoding="utf-8"))
        self.level = pd.read_csv(P / "level_bands.csv")
        self.tiers = pd.read_csv(P / "market_tiers.csv", index_col=0).tier
        self.beyond_from = pd.Period(json.load(open(P / "step32_refit.json", encoding="utf-8"))["beyond_7_months_from"][:7], "M")
        mm = self.level[self.level.level == "market-month"]
        self._ratio = {(r.kind, r.tier): (r.ratio_lo, r.ratio_hi) for r in mm.itertuples()}   # (kind, tier) → (lo, hi)
        tt = self.level[self.level.level == "total-month"]
        self._total = {r.kind: (r.ratio_lo, r.ratio_hi) for r in tt.itertuples()}                # kind → all-market (lo, hi)

    def level_monthly(self, monthly: pd.DataFrame, value_cols=("checkins", "guests")) -> pd.DataFrame:
        """monthly: rows market × month (string 'YYYY-MM'), columns e.g. checkins_scen / guests_scen. Adds lo/hi."""
        out = monthly.copy()
        for col in out.columns:
            kind = "guests" if col.startswith("guests") else "checkins" if col.startswith("checkins") else None
            if kind is None:
                continue
            rl, rh = self._level_ratios(kind, out.index.get_level_values("market"))
            out[f"{col}_lo"] = out[col] * rl
            out[f"{col}_hi"] = out[col] * rh
        out["level_band_calibrated"] = [pd.Period(m, "M") < self.beyond_from for m in out.index.get_level_values("month")]
        return out

    def level_total(self, total: pd.DataFrame) -> pd.DataFrame:
        out = total.copy()
        tt = self.level[self.level.level == "total-month"].set_index("kind")
        for col in out.columns:
            kind = "guests" if col.startswith("guests") else "checkins" if col.startswith("checkins") else None
            if kind:
                out[f"{col}_lo"] = out[col] * tt.loc[kind, "ratio_lo"]; out[f"{col}_hi"] = out[col] * tt.loc[kind, "ratio_hi"]
        out["level_band_calibrated"] = [pd.Period(m, "M") < self.beyond_from for m in out.index]
        return out

    def _level_ratios(self, kind: str, markets) -> tuple[np.ndarray, np.ndarray]:
        tier = {m: self.tiers[m] for m in set(markets)}
        lohi = np.array([self._ratio[(kind, tier[m])] for m in markets]).reshape(-1, 2)
        return lohi[:, 0], lohi[:, 1]

    def _level_dev(self, kind: str, eff: dict) -> tuple[float, float]:
        """Downward and upward deviation of a summed effect from the baseline-level error of each market (effects scale
        with the baseline). Markets' errors are close to independent (validation folds, guest nights: all-market monthly
        total error RMS 4.8% / 6.6% observed vs 5.3% / 4.5% predicted under independence, 17.6% / 16.8% under full
        correlation; scripts/audit_checks.py), so deviations add in quadrature,
        never below the all-market monthly band applied to the net effect."""
        if not eff:
            return 0.0, 0.0
        mk = list(eff); E = np.array([eff[m] for m in mk])
        rl, rh = self._level_ratios(kind, mk)
        down = np.where(E >= 0, E * (1 - rl), -E * (rh - 1)); up = np.where(E >= 0, E * (rh - 1), -E * (1 - rl))
        t_lo, t_hi = self._total[kind]
        net = E.sum()
        fl_down = net * (1 - t_lo) if net >= 0 else -net * (t_hi - 1)
        fl_up = net * (t_hi - 1) if net >= 0 else -net * (1 - t_lo)
        return max(float(np.sqrt((down ** 2).sum())), fl_down), max(float(np.sqrt((up ** 2).sum())), fl_up)

    def end_runs(self, engine, scenario: dict, mid) -> list:
        """Engine runs over the lever's interval: both ends of ε (and of the new-route rate); where the scenario cuts local
        passengers beyond the trusted range (or restores a market from zero), also the alternative cut rule (log form
        continued) at both ends — the straight line under-predicted the out-of-sample Wizz Air exit (Step 3.7).

        When every flight effect in the scenario has the same sign, the result moves monotonically with ε and the new-route
        rate, so the two ends contain it over the whole interval. When a scenario adds and cuts at once, the total need not
        be monotone (gains and losses grow at different rates): then ε is also run at interior points, the two rates are
        crossed at the ends (ε low with the new-route rate high and the reverse), and an interior extreme of the total is
        refined by golden-section search, so the range spans the extremes over the interval, not just its ends."""
        e = self.effect; p = engine.params
        ends = [("lo", "line"), ("hi", "line")]
        d = mid.daily
        log_rule = bool(((d.outside_trusted_range & (d.p2p_scen < d.p2p_base)) | d.zero_baseline_rule).any())
        if log_rule:
            ends += [("lo", "log"), ("hi", "log")]
        run = lambda eps, ps, c="line": engine.run(scenario, eps=eps, per_seat=ps, cuts=c, floor=e["cut_floor"])
        runs = [run(e[f"eps_{k}"], e[f"per_seat_{k}"], c) for k, c in ends]
        f = d.d_checkins_flights.to_numpy()
        mixed = bool((f > 1e-9).any() and (f < -1e-9).any())
        self.last_mixed = mixed
        if not mixed:
            return runs
        lo, hi = e["eps_lo"], e["eps_hi"]
        if (d.new_route_lever | d.zero_baseline_rule).any():                       # the two rates are separate sources
            runs += [run(lo, e["per_seat_hi"]), run(hi, e["per_seat_lo"])]
        grid = {lo + (hi - lo) * q: None for q in (0.25, 0.5, 0.75)}
        for x in grid:
            grid[x] = run(x, p.per_seat); runs.append(grid[x])
        k = "d_guest_nights"
        pts = sorted([(lo, runs[0].summary[k]), (hi, runs[1].summary[k])] + [(x, r.summary[k]) for x, r in grid.items()])
        xs = [x for x, _ in pts]; ys = [y for _, y in pts]
        for sgn in (1, -1):                                                         # refine an interior minimum and maximum
            i = int(np.argmin([sgn * y for y in ys]))
            if 0 < i < len(xs) - 1:
                a_, b_ = xs[i - 1], xs[i + 1]; g = (np.sqrt(5) - 1) / 2
                c_, d_ = b_ - g * (b_ - a_), a_ + g * (b_ - a_)
                rc, rd = run(c_, p.per_seat), run(d_, p.per_seat); runs += [rc, rd]
                for _ in range(6):
                    if sgn * rc.summary[k] < sgn * rd.summary[k]:
                        b_, d_, rd = d_, c_, rc; c_ = b_ - g * (b_ - a_); rc = run(c_, p.per_seat); runs.append(rc)
                    else:
                        a_, c_, rc = c_, d_, rd; d_ = a_ + g * (b_ - a_); rd = run(d_, p.per_seat); runs.append(rd)
        return runs

    def _breakdown(self, runs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Guest-night change with its 80% range for each month (all markets together) and each market (whole horizon,
        incl. stays ending after it) — the summary's rule applied to that part: lever = spread over the lever's end runs,
        baseline level = the markets' level errors in quadrature, the two combined in quadrature."""
        k, kind = "d_guest_nights", "guests"
        per = [r.monthly()[k] for r in runs]                                   # (market, month) for each run
        rows = []
        for mo, v in per[0].groupby(level="month"):
            c = float(v.sum()); ends = [float(p.xs(mo, level="month").sum()) for p in per]
            dn, up = self._level_dev(kind, {m: x for (m, _), x in v.items() if x != 0})
            rows.append({"month": mo, "d_guest_nights": c, "lo": c - float(np.hypot(c - min(ends), dn)),
                         "hi": c + float(np.hypot(max(ends) - c, up))})
        by_month = pd.DataFrame(rows).set_index("month")
        rows = []
        for m, v in runs[0].summary["by_market"].items():
            c = float(v[k]); ends = [float(r.summary["by_market"].get(m, {}).get(k, 0.0)) for r in runs]
            dn, up = self._level_dev(kind, {m: c})
            rows.append({"market": m, "d_guest_nights": c, "d_checkins": float(v["d_checkins"]),
                         "lo": c - float(np.hypot(c - min(ends), dn)), "hi": c + float(np.hypot(max(ends) - c, up))})
        by_market = pd.DataFrame(rows, columns=["market", "d_guest_nights", "d_checkins", "lo", "hi"]).set_index("market")
        return by_month, by_market

    def run(self, engine, scenario: dict, eps: float | None = None, per_seat: float | None = None):
        """Central result plus the effect band: the lever part (engine re-run at the lever's ends, see end_runs) and
        the baseline-level part (the change scales with a baseline that is itself forecast), combined in quadrature
        (independent 80% half-widths). eps / per_seat: the response the result is shown at (default: the central
        estimate; the app's "check-in response" setting picks an end of their 80% interval). The range is always taken
        around the result shown, over the same end runs."""
        e = self.effect
        mid = engine.run(scenario, eps=eps, per_seat=per_seat)
        runs = [mid] + self.end_runs(engine, scenario, mid)
        d = mid.daily
        log_rule = bool(((d.outside_trusted_range & (d.p2p_scen < d.p2p_base)) | d.zero_baseline_rule).any())
        summ, parts = {}, {}
        for k, kind in (("d_checkins", "checkins"), ("d_guest_nights", "guests")):
            c = mid.summary[k]
            lev_dn, lev_up = self._level_dev(kind, {m: v[k] for m, v in mid.summary["by_market"].items()})
            eps_dn = c - min(r.summary[k] for r in runs); eps_up = max(r.summary[k] for r in runs) - c
            summ[k] = [c - float(np.hypot(eps_dn, lev_dn)), c + float(np.hypot(eps_up, lev_up))]
            parts[k] = {"lever": [-eps_dn, eps_up], "baseline_level": [-lev_dn, lev_up]}
        # per market and month: the market's own level band (a single market-month: no cancellation)
        mm = [r.monthly()[["d_checkins", "d_guest_nights"]] for r in runs]
        monthly = mm[0].copy()
        mk = monthly.index.get_level_values("market")
        for c, kind in (("d_checkins", "checkins"), ("d_guest_nights", "guests")):
            stack = np.stack([m[c].to_numpy() for m in mm]); v = monthly[c].to_numpy()
            rl, rh = self._level_ratios(kind, mk)
            dn = np.where(v >= 0, v * (1 - rl), -v * (rh - 1)); up = np.where(v >= 0, v * (rh - 1), -v * (1 - rl))
            monthly[f"{c}_lo"] = v - np.hypot(v - stack.min(0), dn); monthly[f"{c}_hi"] = v + np.hypot(stack.max(0) - v, up)
        # months beyond 7 after the data: the baseline-level part uses the 1–7-month validation errors (flagged, as for levels)
        mo = monthly.index.get_level_values("month")
        monthly["level_band_calibrated"] = [pd.Period(m, "M") < self.beyond_from for m in mo]
        gn = monthly.d_guest_nights.abs().groupby(mo).sum() + monthly.d_checkins.abs().groupby(mo).sum()
        unc = [m for m in gn.index[gn > 0] if pd.Period(m, "M") >= self.beyond_from]
        unc_share = float(monthly.d_guest_nights.abs()[~monthly.level_band_calibrated].sum() / max(monthly.d_guest_nights.abs().sum(), 1e-9))
        if unc:
            mid.flags.append(f"{unc_share:.0%} of the guest-night change falls {unc[0]}…{unc[-1]}, more than 7 months after the data: "
                             "its baseline-level band uses the 1–7-month validation errors (they did not grow with lead in the folds; "
                             "8–12 months untested)")
        by_month, by_market = self._breakdown(runs)
        assumption_only = all(c["type"] in {"demand", "stay_length"} for c in scenario.get("changes", []))
        mid.effect_band = {"summary": summ, "parts": parts, "monthly": monthly, "eps_range": [e["eps_lo"], e["eps_hi"]],
                           "by_month": by_month, "by_market": by_market,
                           "months_beyond_7": unc, "share_beyond_7_months": unc_share if unc else 0.0,
                           "rule_uncertainty": log_rule,
                           "interior_runs": bool(getattr(self, "last_mixed", False)),
                           "note": ("assumption levers only: the lever is as assumed; the band is the baseline-level part only"
                                    if assumption_only else "80%: lever uncertainty (ε's out-of-sample interval, new-route band"
                                    + ("; beyond-range cuts: straight line and log form" if log_rule else "")
                                    + ("; adds and cuts together: extremes over the whole ε interval" if getattr(self, "last_mixed", False) else "")
                                    + ") and baseline-level uncertainty, combined in quadrature")}
        return mid
