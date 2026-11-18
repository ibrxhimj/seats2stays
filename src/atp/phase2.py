"""Phase 2 — Stage A (flights → hotel check-ins). Run step by step (or `make phase2`):

    PYTHONPATH=src python -m atp.phase2 airline      # Step 2.1: airline defaults + ε₁ (seats → local passengers)
    PYTHONPATH=src python -m atp.phase2 checkins     # Step 2.2: ε_total (seats → check-ins), ε₂ = ε_total / ε₁

Estimation window for every Stage A elasticity = the final training window (YoY months 2024-01…2025-07; both
months of each comparison ≥ 2023-01). Official test-window check-ins (Aug 2025–Feb 2026) are never used here —
they are reserved for the one-time frozen evaluation in Step 2.6. Fold A/B versions use the same rule with their
own cut-off, and the panel is cut BEFORE year-on-year changes are formed, so no lead can reach past a cut-off.

Trusted-range estimation (adopted in Step 2.2, applied to every Stage A elasticity): only country-months whose
YoY seat change lies inside that window's 5th–95th percentile are used. Rows are selected on the seat change alone
(never on check-ins), so the slope stays unbiased under the model's own assumption, and the lever is measured on
exactly the range the simulator allows. Reason: on all rows, war/reopening swings (Israel −81%…+518%) move demand
and supply together, one country shifts ε_total by 0.09 and the estimate drifts 0.87 → 0.54 across windows; on the
trusted range it is stable (0.56 → 0.52 → 0.44). All-row results are kept as a documented sensitivity.
"""
import json
import sys
import numpy as np
import pandas as pd
from .config import load_config
from . import stagea as sa

N_BOOT = 2000
BASE_START = "2023-01-01"          # both months of every YoY comparison are post-COVID (2023+)
TRIM_Q = 0.05                      # trusted range = 5th–95th percentile of the window's YoY seat changes


def _load(cfg):
    P = cfg["paths"]["processed"]
    return (pd.read_parquet(P / "flights_country_daily.parquet"), pd.read_parquet(P / "flights_country_monthly.parquet"),
            pd.read_parquet(P / "hotel_intl_daily.parquet"))


def _out(cfg):
    o = cfg["paths"]["processed"] / "phase2"; o.mkdir(exist_ok=True); return o


def _slim(r):
    return {k: v for k, v in r.items() if k not in ("boot_draws", "countries")}


def _window(m, end):
    """All YoY rows of one estimation window (untrimmed) and its trusted range in log points."""
    y = sa.yoy(sa.truncate(m, end)); y = y[y.base_month >= BASE_START].reset_index(drop=True)
    return y, sa.trusted_range(y, TRIM_Q)


def _primary(m, end):
    y, rng_ = _window(m, end)
    return sa.trim(y, rng_), rng_


def _fmt(v):
    ci = v.get("ci_boot") or v["ci_cluster"]
    return f"{v['estimate']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"




LEADS = ["d_seats", "d_seats_lead1", "d_seats_lead2"]


# ============================================================================ Step 2.1
def step_airline():
    cfg = load_config(); out = _out(cfg); fd, fm, hi = _load(cfg)
    train_end = cfg["folds"]["final"]["train_end"]
    # --- defaults: latest 12 months of flight data in the pack. Flights are an INPUT (never a scored target) and
    # these values only translate "add N weekly flights" into seats, so using the newest schedule is leak-free.
    end = fd.date.max(); start = end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    dfl = sa.airline_defaults(fd, start, end)
    dfl.to_csv(out / "airline_defaults.csv", index=False)
    ident = np.allclose(dfl.local_share, 1 - dfl.transfer_share - dfl.transit_share) and \
        np.allclose(dfl.local_pax_per_seat, dfl.load_factor * dfl.local_share)
    # --- ε₁ panels: primary = final training window, trusted range (shared rule for every Stage A elasticity)
    m23 = sa.monthly_panel(fd, hi); m22 = sa.monthly_panel(fd, hi, fm)
    full, rng_ = _window(m23, train_end)
    prim = sa.trim(full, rng_).pipe(lambda d: d.assign(w=d.base_seats))
    y_all = sa.yoy(m23); y_all = sa.trim(y_all[y_all.base_month >= BASE_START], rng_)
    y22 = sa.yoy(sa.truncate(m22, train_end)); y22 = sa.trim(y22[y22.base_month >= "2022-01-01"], rng_)
    res = {
        "primary (train window, trusted range)": sa.fe_elasticity(prim, "d_p2p", ["d_seats"], n_boot=N_BOOT),
        "sensitivity: all rows (no trim)": sa.fe_elasticity(full, "d_p2p", ["d_seats"], n_boot=N_BOOT),
        "robust: all flight months to Feb 2026": sa.fe_elasticity(y_all, "d_p2p", ["d_seats"], n_boot=N_BOOT),
        "robust: include 2023-vs-2022": sa.fe_elasticity(y22, "d_p2p", ["d_seats"], n_boot=N_BOOT),
        "robust: + seats 1–2 months ahead": sa.fe_elasticity(prim, "d_p2p", LEADS, n_boot=N_BOOT),
        "robust: weighted by route size": sa.fe_elasticity(prim, "d_p2p", ["d_seats"], w="w", n_boot=N_BOOT),
    }
    # tourist-heavy vs resident-heavy (pre-specified: 2023 check-ins per local passenger above/below median)
    c23 = m23[m23.month.dt.year == 2023].groupby("nationality_key")[["checkins", "p2p"]].sum()
    ratio_ = (c23.checkins / c23.p2p).dropna(); tour = set(ratio_[ratio_ > ratio_.median()].index)
    for lab, sel in (("tourist-heavy routes", prim.nationality_key.isin(tour)), ("resident-heavy routes", ~prim.nationality_key.isin(tour))):
        res[f"by type: {lab}"] = sa.fe_elasticity(prim[sel], "d_p2p", ["d_seats"], n_boot=N_BOOT)
    summary = {k: _slim(v) for k, v in res.items()}
    P1 = "primary (train window, trusted range)"
    np.save(out / "eps1_boot_primary.npy", res[P1]["boot_draws"])

    # --- diagnostics on the primary sample
    # (1) decomposition: log p2p = log seats + log LF + log local_share  ⇒  ε₁ = 1 + β_LF + β_share exactly
    p = prim.assign(d_lf=prim.d_pax - prim.d_seats, d_share=prim.d_p2p - prim.d_pax,
                    d_up=prim.d_seats.clip(lower=0), d_down=prim.d_seats.clip(upper=0))
    dec = {k: sa.fe_elasticity(p, k, ["d_seats"], n_boot=0) for k in ("d_lf", "d_share")}
    decomp = {"load_factor": dec["d_lf"]["estimate"], "load_factor_ci": dec["d_lf"]["ci_cluster"],
              "local_share": dec["d_share"]["estimate"], "local_share_ci": dec["d_share"]["ci_cluster"],
              "identity_gap": 1 + dec["d_lf"]["estimate"] + dec["d_share"]["estimate"] - summary[P1]["estimate"]}
    # (2) leave-one-country-out
    loo = pd.Series({c: sa.fe_elasticity(prim[prim.nationality_key != c], "d_p2p", ["d_seats"], n_boot=0)["estimate"]
                     for c in sorted(prim.nationality_key.unique())})
    # (3) seat increases vs cuts
    asy = sa.fe_elasticity(p, "d_p2p", ["d_up", "d_down"], n_boot=0)
    asym = {"increases": asy["estimate"], "increases_ci": asy["ci_cluster"],
            "cuts": asy["other_coefs"]["d_down"], "cuts_ci": asy["other_ci_cluster"]["d_down"]}
    # (4) observed range of seat changes in the (untrimmed) window = the simulator's trusted range
    chg = np.expm1(full.d_seats.dropna())
    obs = {"p5": float(np.expm1(rng_[0])), "p95": float(np.expm1(rng_[1])), "min": float(chg.min()), "max": float(chg.max())}
    diag = {"decomposition": decomp, "loo_min": [loo.idxmin(), float(loo.min())], "loo_max": [loo.idxmax(), float(loo.max())],
            "asymmetry": asym, "observed_seat_change": obs, "rows_kept": [len(prim), int(full.d_seats.notna().sum())]}
    json.dump({"eps1": summary, "diagnostics": diag, "tourist_heavy": sorted(tour), "defaults_identity_ok": bool(ident),
               "defaults_window": [str(start.date()), str(end.date())], "estimation_window_end": str(train_end.date()),
               "trusted_range_log": list(rng_)}, open(out / "step21_airline.json", "w", encoding="utf-8"), indent=1)
    print(f"defaults: {len(dfl)} country×month rows, window {start.date()}…{end.date()}, identities hold: {ident}, "
          f"missing values: {int(dfl[['seats_per_flight', 'load_factor', 'transfer_share']].isna().sum().sum())}")
    tot = dfl[["seats", "pax", "p2p", "transfer", "transit", "n_flights"]].sum()
    print(f"  overall: LF {tot.pax / tot.seats:.3f}, transfer {tot.transfer / tot.pax:.3f}, transit {tot.transit / tot.pax:.3f}, "
          f"local {tot.p2p / tot.pax:.3f}, seats/flight {tot.seats / tot.n_flights:.0f}")
    print(f"  trusted range {obs['p5']:+.0%} … {obs['p95']:+.0%} (observed {obs['min']:+.0%} … {obs['max']:+.0%}); rows kept {len(prim)} of {diag['rows_kept'][1]}")
    print(f"  decomposition: ε₁ = 1 + LF {decomp['load_factor']:+.3f} {np.round(decomp['load_factor_ci'], 3)} "
          f"+ local share {decomp['local_share']:+.3f} {np.round(decomp['local_share_ci'], 3)} (gap {decomp['identity_gap']:.1e})")
    print(f"  leave-one-country-out: {loo.min():.3f} (without {loo.idxmin()}) … {loo.max():.3f} (without {loo.idxmax()})")
    print(f"  increases {asym['increases']:.3f} {np.round(asym['increases_ci'], 3)} | cuts {asym['cuts']:.3f} {np.round(asym['cuts_ci'], 3)}")
    for k, v in summary.items():
        print(f"  ε₁ {k:40s} {_fmt(v)}  cluster [{v['ci_cluster'][0]:.3f}, {v['ci_cluster'][1]:.3f}]  "
              f"n={v['n_obs']} ({v['n_countries']} countries, {v['first_month'][:7]}…{v['last_month'][:7]})"
              + "".join(f"  {kk} {vv:+.3f} {np.round(v['other_ci_cluster'][kk], 3)}" for kk, vv in v["other_coefs"].items()))


# ============================================================================ Step 2.2
def _fold_estimates(m, end):
    """ε_total (plain + lead-controlled), ε₁, default rule, ε₂ — one window, trusted range, paired bootstrap."""
    p, rng_ = _primary(m, end)
    R = {"eps_total_plain": sa.fe_elasticity(p, "d_checkins", ["d_seats"], n_boot=N_BOOT),
         "eps_total_lead": sa.fe_elasticity(p, "d_checkins", LEADS, n_boot=N_BOOT)}
    # pre-registered: the default is the SMALLER of the two ε_total estimates (conservative lever)
    dkey = min(("eps_total_plain", "eps_total_lead"), key=lambda k: R[k]["estimate"])
    R["eps_total_default"] = dict(R[dkey], spec=dkey)
    # ε₁ on exactly the rows of the default ε_total (the lead spec loses the last two months), so ε₂ = ε_total / ε₁
    # compares like with like (audit fix, 2 Oct 2026)
    rows = p.dropna(subset=LEADS) if dkey == "eps_total_lead" else p
    R["eps1"] = sa.fe_elasticity(rows, "d_p2p", ["d_seats"], n_boot=N_BOOT)
    assert R["eps1"]["n_obs"] == R["eps_total_default"]["n_obs"]
    R["eps2"] = sa.ratio(R["eps_total_default"], R["eps1"])
    return p, rng_, R, dkey


def step_checkins():
    cfg = load_config(); out = _out(cfg); fd, fm, hi = _load(cfg)
    tour = set(json.load(open(out / "step21_airline.json", encoding="utf-8"))["tourist_heavy"])
    m = sa.monthly_panel(fd, hi); m22 = sa.monthly_panel(fd, hi, fm)
    test_start = cfg["folds"]["final"]["holdout_start"]

    folds, table = {}, []
    for fold in ("A", "B", "final"):
        end = cfg["folds"][fold]["train_end"]
        p, rng_, R, dkey = _fold_estimates(m, end)
        assert pd.Timestamp(R["eps_total_plain"]["last_month"]) < test_start       # never touches test check-ins
        folds[fold] = (p, rng_, R, dkey)
        for k in ("eps_total_plain", "eps_total_lead", "eps_total_default", "eps1", "eps2"):
            v = R[k]; base = R["eps_total_default"] if k == "eps2" else v
            table.append({"fold": fold, "train_end": str(end.date()), "parameter": k, "spec": v.get("spec", k),
                          "estimate": v["estimate"], "ci_lo": v["ci_boot"][0], "ci_hi": v["ci_boot"][1],
                          "ci_cluster_lo": v.get("ci_cluster", [np.nan] * 2)[0], "ci_cluster_hi": v.get("ci_cluster", [np.nan] * 2)[1],
                          "n_obs": base["n_obs"], "n_countries": base["n_countries"],
                          "first_month": base["first_month"], "last_month": base["last_month"],
                          "trusted_range_lo_pct": float(np.expm1(rng_[0])), "trusted_range_hi_pct": float(np.expm1(rng_[1]))})
    tab = pd.DataFrame(table); tab.to_csv(out / "elasticities.csv", index=False)

    # ---------------- final-window robustness (cluster intervals)
    p, rng_, R, dkey = folds["final"]
    end = cfg["folds"]["final"]["train_end"]
    xs = LEADS if dkey == "eps_total_lead" else ["d_seats"]
    full, _ = _window(m, end)
    y22 = sa.yoy(sa.truncate(m22, end)); y22 = sa.trim(y22[y22.base_month >= "2022-01-01"], rng_)
    q = p.assign(w=p.base_checkins, d_up=p.d_seats.clip(lower=0), d_down=p.d_seats.clip(upper=0))
    rob = {
        "sensitivity: all rows (no trim), plain": sa.fe_elasticity(full, "d_checkins", ["d_seats"], n_boot=0),
        "sensitivity: all rows (no trim), lead-controlled": sa.fe_elasticity(full, "d_checkins", LEADS, n_boot=0),
        "sensitivity: all rows, lead-controlled, without Israel": sa.fe_elasticity(full[full.nationality_key != "ISRAEL"], "d_checkins", LEADS, n_boot=0),
        "plain + seats 1–2 months earlier (lags)": sa.fe_elasticity(p, "d_checkins", ["d_seats", "d_seats_lag1", "d_seats_lag2"], n_boot=0),
        "lead-controlled + lags": sa.fe_elasticity(p, "d_checkins", LEADS + ["d_seats_lag1", "d_seats_lag2"], n_boot=0),
        "default spec, weighted by check-in volume": sa.fe_elasticity(q, "d_checkins", xs, w="w", n_boot=0),
        "default spec, include 2023-vs-2022": sa.fe_elasticity(y22, "d_checkins", xs, n_boot=0),
    }
    asy = sa.fe_elasticity(q, "d_checkins", ["d_up", "d_down"] + xs[1:], n_boot=0)
    asym = {"increases": asy["estimate"], "increases_ci": asy["ci_cluster"],
            "cuts": asy["other_coefs"]["d_down"], "cuts_ci": asy["other_ci_cluster"]["d_down"]}
    loo = pd.Series({c: sa.fe_elasticity(p[p.nationality_key != c], "d_checkins", xs, n_boot=0)["estimate"]
                     for c in sorted(p.nationality_key.unique())})
    # ---------------- tourist-heavy vs resident-heavy (default spec; disjoint groups → stratified bootstrap)
    grp = {lab: sa.fe_elasticity(p[sel], "d_checkins", xs, n_boot=N_BOOT, seed=s)
           for s, (lab, sel) in enumerate((("tourist-heavy", p.nationality_key.isin(tour)), ("resident-heavy", ~p.nationality_key.isin(tour))), 1)}
    diff = sa.difference(grp["tourist-heavy"], grp["resident-heavy"])
    split_forward = bool(diff["ci_boot"][0] > 0 or diff["ci_boot"][1] < 0)          # pre-registered rule
    # ---------------- average vs marginal check-ins per local passenger / per seat (last 12 training months)
    last12 = m[(m.month > end - pd.DateOffset(years=1)) & (m.month <= end)]
    avg_p, avg_s = last12.checkins.sum() / last12.p2p.sum(), last12.checkins.sum() / last12.seats.sum()
    marg = {"avg_checkins_per_local_pax": float(avg_p), "marginal_checkins_per_local_pax": float(R["eps2"]["estimate"] * avg_p),
            "avg_checkins_per_seat": float(avg_s), "marginal_checkins_per_seat": float(R["eps_total_default"]["estimate"] * avg_s),
            "window": [str((end - pd.DateOffset(years=1)).date()), str(end.date())]}

    np.savez(out / "eps_boot_final.npz", **{k: R[k]["boot_draws"] for k in ("eps_total_plain", "eps_total_lead", "eps1", "eps2")})
    j = {"window_rule": "YoY months with both months ≥ 2023-01 and month ≤ fold train_end; test-window check-ins never used",
         "trim_rule": f"rows kept only if the YoY seat change is inside the window's {TRIM_Q:.0%}–{1 - TRIM_Q:.0%} percentile range (seats only)",
         "default_rule": "eps_total default = smaller of plain and lead-controlled; the other is the upper end of the range",
         "split_rule": "tourist/resident split goes forward to Step 2.4 only if the 95% interval of the difference excludes 0",
         "folds": {f: {"default_spec": d, "trusted_range_pct": [float(np.expm1(r[0])), float(np.expm1(r[1]))],
                       **{k: _slim(v) for k, v in R_.items()}} for f, (_, r, R_, d) in folds.items()},
         "final_range": sorted([R["eps_total_plain"]["estimate"], R["eps_total_lead"]["estimate"]]),
         "robustness": {k: _slim(v) for k, v in rob.items()}, "asymmetry": asym,
         "loo_min": [loo.idxmin(), float(loo.min())], "loo_max": [loo.idxmax(), float(loo.max())],
         "split": {k: _slim(v) for k, v in grp.items()}, "split_difference": diff, "split_goes_forward": split_forward,
         "per_passenger": marg}
    json.dump(j, open(out / "step22_checkins.json", "w", encoding="utf-8"), indent=1)

    # ---------------- print
    print("window  trusted range     ε_total plain          ε_total lead-ctrl      default  ε₁                     ε₂ = default/ε₁")
    for f, (_, r, R_, d) in folds.items():
        print(f"  {f:6s} {np.expm1(r[0]):+.0%}…{np.expm1(r[1]):+.0%}  {_fmt(R_['eps_total_plain'])}  {_fmt(R_['eps_total_lead'])}  "
              f"{'lead ' if d == 'eps_total_lead' else 'plain'}  {_fmt(R_['eps1'])}  {_fmt(R_['eps2'])}   n={R_['eps_total_plain']['n_obs']}")
    lc = R["eps_total_lead"]
    print(f"  final lead coefficients: " + ", ".join(f"{k} {v:+.3f} {np.round(lc['other_ci_cluster'][k], 3)}" for k, v in lc["other_coefs"].items()))
    for k, v in rob.items():
        print(f"  {k:55s} {v['estimate']:.3f} [{v['ci_cluster'][0]:.3f}, {v['ci_cluster'][1]:.3f}]"
              + "".join(f"  {kk} {vv:+.3f}" for kk, vv in v["other_coefs"].items()))
    print(f"  increases {asym['increases']:.3f} {np.round(asym['increases_ci'], 3)} | cuts {asym['cuts']:.3f} {np.round(asym['cuts_ci'], 3)}")
    print(f"  leave-one-country-out: {loo.min():.3f} (without {loo.idxmin()}) … {loo.max():.3f} (without {loo.idxmax()})")
    print(f"  tourist-heavy {_fmt(grp['tourist-heavy'])} | resident-heavy {_fmt(grp['resident-heavy'])} | "
          f"difference {diff['estimate']:+.3f} [{diff['ci_boot'][0]:+.3f}, {diff['ci_boot'][1]:+.3f}] → split forward: {split_forward}")
    print(f"  per local passenger: average {marg['avg_checkins_per_local_pax']:.3f} check-ins, marginal {marg['marginal_checkins_per_local_pax']:.3f} | "
          f"per seat: average {marg['avg_checkins_per_seat']:.3f}, marginal {marg['marginal_checkins_per_seat']:.3f}")


# ============================================================================ Step 2.3
"""Out-of-sample test of the seat lever, with main moving festivals and real-world shocks.

Rules fixed before the full run:
  * External factors in a YoY monthly panel with month effects: shared shocks (UAE holidays, F1, floods, oil, global
    travel) are removed by the month effect; fixed-date holidays sit in the same month every year and cancel in YoY;
    so only MOVING festivals of specific source markets can distort the lever. One main festival per market group
    (data/reference/festival_groups.csv): Ramadan + both Eids (Muslim-majority), Easter (countries where Good Friday or
    Easter Monday is a public holiday), Lunar New Year (China, South Korea), Diwali (India). Window = festival day −2…+7.
  * Festivals enter ε only if they shift it by more than 1 cluster SE in the final window (either spec); otherwise
    ε from Step 2.2 stands. They pass to Step 2.4 if jointly significant for check-ins (Wald p < 0.05).
  * Primary lever test: ε from fold B (data to Dec 2024) applied to Jan–Jul 2025, rows inside fold B's trusted range,
    festival controls in the test regression. Passes if the calibration slope's 95% interval contains 1 and the lever's
    out-of-sample skill (festival-adjusted, fold-B festival coefficients) is above 0. Secondary: fold A → Aug 2024–Jul
    2025. Sensitivity: all rows. Official test-window check-ins (Aug 2025+) are not touched.
"""
FEST = ["d_festival_days", "d_ramadan_days"]


def _panel_with_festivals(cfg, fd, hi):
    RD, P = cfg["paths"]["reference"], cfg["paths"]["processed"]
    off = pd.read_csv(RD / "calendar_events_official.csv", parse_dates=["start_date", "end_date"])
    fest = sa.festival_dates(RD, off); grp = pd.read_csv(RD / "festival_groups.csv")
    fp = sa.festival_panel(grp, fest, pd.read_parquet(P / "calendar.parquet"))
    m = sa.monthly_panel(fd, hi).merge(fp, on=["nationality_key", "month"], how="left")
    assert m[["festival_days", "ramadan_days"]].notna().all().all(), "every flight country needs a festival group"
    return m, fest, grp


def _test_rows(m, train_end, test_end, rng_=None):
    """YoY rows strictly after train_end up to test_end (both months ≥ 2023-01). Leads are not used here."""
    y = sa.yoy(sa.truncate(m, test_end)); y = y[(y.base_month >= BASE_START) & (y.month > train_end)].reset_index(drop=True)
    return sa.trim(y, rng_) if rng_ else y


def step_lever():
    cfg = load_config(); out = _out(cfg); fd, fm, hi = _load(cfg)
    m, fest, grp = _panel_with_festivals(cfg, fd, hi)
    el = pd.read_csv(out / "elasticities.csv").set_index(["fold", "parameter"])
    F = cfg["folds"]; test_start = F["final"]["holdout_start"]
    fest.assign(window_start=fest.date + pd.Timedelta(days=sa.FEST_WINDOW[0]), window_end=fest.date + pd.Timedelta(days=sa.FEST_WINDOW[1])) \
        .to_csv(out / "main_festival_days.csv", index=False)

    # ---------------- (1) do the main festivals distort the lever?
    fest_res = {}
    for fold in ("A", "B", "final"):
        p, rng_ = _primary(m, F[fold]["train_end"])
        for spec, xs in (("plain", ["d_seats"]), ("lead", LEADS)):
            a = sa.fe_elasticity(p, "d_checkins", xs, n_boot=0); b = sa.fe_elasticity(p, "d_checkins", xs + FEST, n_boot=0)
            fest_res[f"{fold}/{spec}"] = {"eps": a["estimate"], "eps_with_festivals": b["estimate"],
                                          "shift_in_se": (b["estimate"] - a["estimate"]) / a["se_cluster"],
                                          "festival_day": b["other_coefs"]["d_festival_days"], "festival_day_ci": b["other_ci_cluster"]["d_festival_days"],
                                          "ramadan_day": b["other_coefs"]["d_ramadan_days"], "ramadan_day_ci": b["other_ci_cluster"]["d_ramadan_days"]}
        s = sa.fe_elasticity(p, "d_seats", FEST, n_boot=0)            # do airlines move seats with the festivals?
        fest_res[f"{fold}/seats_on_festivals"] = {"festival_day": s["estimate"], "festival_day_ci": s["ci_cluster"],
                                                  "ramadan_day": s["other_coefs"]["d_ramadan_days"], "ramadan_day_ci": s["other_ci_cluster"]["d_ramadan_days"]}
    pf, _ = _primary(m, F["final"]["train_end"])
    joint_p = sa.fe_elasticity(pf, "d_checkins", ["d_seats"] + FEST, n_boot=0)["wald_p_others"]
    by_fest = sa.fe_elasticity(pf.assign(**{f"d_fest_{g}": np.where(pf.nationality_key.isin(grp.loc[grp.group == g, "nationality"]), pf.d_festival_days, 0.0)
                                             for g in ("muslim_majority", "easter", "lunar_new_year", "diwali")}),
                               "d_checkins", ["d_seats", "d_ramadan_days"] + [f"d_fest_{g}" for g in ("muslim_majority", "easter", "lunar_new_year", "diwali")], n_boot=0)
    adopt = any(abs(fest_res[f"final/{s}"]["shift_in_se"]) > 1 for s in ("plain", "lead"))
    to_24 = bool(joint_p < 0.05)

    # ---------------- (2) out-of-sample lever test
    def run(fold, test_end, trim=True):
        tr_end = F[fold]["train_end"]
        p_tr, rng_ = _primary(m, tr_end)
        g = sa.fe_elasticity(p_tr, "d_checkins", ["d_seats"] + FEST, n_boot=0)      # fold-trained festival effects
        te = _test_rows(m, tr_end, test_end, rng_ if trim else None).dropna(subset=["d_seats", "d_checkins"])
        assert te.month.max() < test_start and te.month.min() > tr_end
        eps_f = float(el.loc[(fold, "eps_total_default"), "estimate"])
        te = te.assign(pred=eps_f * te.d_seats,
                       d_checkins_adj=te.d_checkins - g["other_coefs"]["d_festival_days"] * te.d_festival_days
                       - g["other_coefs"]["d_ramadan_days"] * te.d_ramadan_days)
        slope = sa.fe_elasticity(te, "d_checkins", ["pred"] + FEST, n_boot=N_BOOT)
        slope_nf = sa.fe_elasticity(te, "d_checkins", ["pred"], n_boot=0)
        cands = {"final default 0.31": float(el.loc[("final", "eps_total_default"), "estimate"]),
                 "final upper 0.44": float(el.loc[("final", "eps_total_plain"), "estimate"]),
                 f"fold {fold} estimate (honest)": eps_f, "naive one-for-one (1.0)": 1.0}
        sk = sa.lever_skill(te.assign(d_checkins=te.d_checkins_adj), cands, n_boot=N_BOOT)
        skw = sa.lever_skill(te.assign(d_checkins=te.d_checkins_adj), cands, n_boot=0, w="base_checkins")
        grid = np.round(np.arange(0, 1.01, 0.05), 2)
        prof = sa.lever_skill(te.assign(d_checkins=te.d_checkins_adj), {str(e): float(e) for e in grid}, n_boot=0)["skill"]
        best = max(prof, key=prof.get)
        diff_naive = np.percentile(sk["draws"][f"fold {fold} estimate (honest)"] - sk["draws"]["naive one-for-one (1.0)"], [2.5, 97.5])
        return te, {"train_end": str(tr_end.date()), "test_months": [str(te.month.min().date()), str(te.month.max().date())],
                    "n_obs": int(len(te)), "n_countries": int(te.nationality_key.nunique()), "trimmed": trim,
                    "eps_used": eps_f, "calibration_slope": _slim(slope), "calibration_slope_no_festivals": _slim(slope_nf),
                    "skill": sk["skill"], "skill_ci": sk["ci"], "skill_volume_weighted": skw["skill"],
                    "honest_minus_naive_ci": [float(diff_naive[0]), float(diff_naive[1])],
                    "best_eps_on_test": float(best), "skill_profile": prof,
                    "festival_effects_trained": {k: g["other_coefs"][k] for k in FEST}}
    te_B, primary = run("B", F["final"]["train_end"])
    _, second = run("A", F["final"]["train_end"])
    _, allrows = run("B", F["final"]["train_end"], trim=False)
    passed = bool(primary["calibration_slope"]["ci_boot"][0] <= 1 <= primary["calibration_slope"]["ci_boot"][1]
                  and primary["skill"]["fold B estimate (honest)"] > 0)
    rel = te_B.assign(**{f"rel_{c}": sa._rel(te_B, ["d_seats", "d_checkins_adj"])[c] for c in ("d_seats", "d_checkins_adj")})
    rel.assign(rel_pred=primary["eps_used"] * rel.rel_d_seats)[
        ["nationality_key", "month", "d_seats", "d_checkins", "d_festival_days", "d_ramadan_days", "d_checkins_adj",
         "rel_d_seats", "rel_d_checkins_adj", "rel_pred"]].to_csv(out / "lever_test_rows.csv", index=False)

    # ---------------- (3) real-world shocks (all before the official test window)
    full = sa.yoy(sa.truncate(m, F["final"]["train_end"])); full = full[full.base_month >= BASE_START].reset_index(drop=True)
    r = sa._rel(full, ["d_seats", "d_checkins", "d_festival_days", "d_ramadan_days"])
    fe = fest_res["final/plain"]; lo, hi_ = sa.trusted_range(full, TRIM_Q)
    eps_lo, eps_hi = float(el.loc[("final", "eps_total_default"), "estimate"]), float(el.loc[("final", "eps_total_plain"), "estimate"])
    events = [("ISRAEL", "2024-01-01", "2024-03-01", "Gaza war: most foreign carriers suspended Tel Aviv after 7 Oct 2023"),
              ("ISRAEL", "2024-12-01", "2025-03-01", "Capacity rebuilt after the 2023–24 cuts"),
              ("LEBANON", "2024-10-01", "2024-11-01", "Israel–Hezbollah war (ground operation 1 Oct, ceasefire 27 Nov 2024)"),
              ("JORDAN", "2025-06-01", "2025-06-01", "Iran–Israel 12-day war, regional airspace closures 13–24 Jun 2025"),
              ("EGYPT", "2024-03-01", "2024-07-01", "Egyptian pound floated 6 Mar 2024 (≈ −40%), seats kept growing"),
              ("CHINA", "2025-01-01", "2025-01-01", "Lunar New Year moved INTO January (29 Jan 2025 vs 10 Feb 2024)"),
              ("CHINA", "2025-02-01", "2025-02-01", "…and OUT of February")]
    rows = []
    for c, a, b, why in events:
        sel = (full.nationality_key == c) & (full.month >= a) & (full.month <= b)
        x, rr = full[sel], r[sel]
        fest_adj = fe["festival_day"] * rr.d_festival_days.mean() + fe["ramadan_day"] * rr.d_ramadan_days.mean()
        rows.append({"country": c, "months": f"{a[:7]}…{b[:7]}", "event": why,
                     "seat_change_vs_other_countries": float(np.expm1(rr.d_seats.mean())),
                     "inside_trusted_range": bool(x.d_seats.between(lo, hi_).all()),
                     "model_prediction_low": float(np.expm1(eps_lo * rr.d_seats.mean() + fest_adj)),
                     "model_prediction_high": float(np.expm1(eps_hi * rr.d_seats.mean() + fest_adj)),
                     "festival_part": float(np.expm1(fest_adj)),
                     "actual_checkin_change_vs_other_countries": float(np.expm1(rr.d_checkins.mean())),
                     "implied_response": float((rr.d_checkins.mean() - fest_adj) / rr.d_seats.mean())})
    sc = pd.DataFrame(rows); sc.to_csv(out / "real_world_showcases.csv", index=False)

    # ---------------- (4) why did the test fail? non-overlapping half-years (final window, trusted range)
    pf_all = pf
    drift = {}
    for a, b in (("2024-01-01", "2024-06-01"), ("2024-07-01", "2024-12-01"), ("2025-01-01", "2025-07-01")):
        q = pf_all[(pf_all.month >= a) & (pf_all.month <= b)]
        e = sa.fe_elasticity(q, "d_checkins", ["d_seats"] + FEST, n_boot=0); e1 = sa.fe_elasticity(q, "d_p2p", ["d_seats"], n_boot=0)
        sh = sa.fe_elasticity(q.assign(s=q.d_p2p - q.d_pax), "s", ["d_seats"], n_boot=0)
        drift[f"{a[:7]}…{b[:7]}"] = {"eps_total": e["estimate"], "eps_total_ci": e["ci_cluster"], "eps1": e1["estimate"],
                                     "eps1_ci": e1["ci_cluster"], "local_share_response": sh["estimate"], "n_obs": e["n_obs"]}
    eps_default = float(el.loc[("final", "eps_total_default"), "estimate"])
    decision = {"default_eps_total": eps_default,
                "simulator_band": [float(primary["best_eps_on_test"]), float(el.loc[("final", "eps_total_plain"), "estimate"])],
                "why": "older-window values (≥0.5) over-predict 2025 by ~2x: the lever weakened in 2025. The latest-window default "
                       "sits between the two held-out optima (0.25, 0.35). Caveat (audit): 0.31 and 0.44 were estimated on data that "
                       "include most of these held-out rows, so their scores here are partly in-sample, and the band's low end (0.25) "
                       "was selected on these months; the genuinely out-of-sample check is the official test window (Step 2.6)",
                "step_2_6_preregistered": "apply the default ε to official test-window relative growth (Aug 2025–Feb 2026 vs a year "
                                          "earlier, final trusted range, festival controls): pass if the slope's 95% interval contains 1 "
                                          "and skill vs 'all countries grow alike' is above 0; the naive one-for-one must score lower"}
    j = {"festival_rule": "festivals enter ε only if they shift it by > 1 cluster SE (final window); pass to 2.4 if Wald p < 0.05",
         "festivals": fest_res, "festivals_joint_wald_p": joint_p, "festivals_adopted_into_eps": adopt, "festivals_to_step_2_4": to_24,
         "festival_effect_by_group": {k: [v, by_fest["other_ci_cluster"][k]] for k, v in by_fest["other_coefs"].items()},
         "lever_test_rule": "fold B ε on Jan–Jul 2025 (trusted range, festival controls): slope 95% CI contains 1 AND skill > 0",
         "lever_test_passed": passed, "primary": primary, "secondary_foldA": second, "sensitivity_all_rows": allrows,
         "trusted_range_final": [float(np.expm1(lo)), float(np.expm1(hi_))], "drift_by_half_year": drift, "decision": decision}
    json.dump(j, open(out / "step23_lever_test.json", "w", encoding="utf-8"), indent=1, default=float)

    # ---------------- print
    print("FESTIVALS (per extra day in the month, vs same month last year)")
    for k in ("A/plain", "B/plain", "final/plain", "final/lead"):
        v = fest_res[k]
        print(f"  {k:12s} ε {v['eps']:.3f} → {v['eps_with_festivals']:.3f} with festivals (shift {v['shift_in_se']:+.2f} SE) | "
              f"festival day {v['festival_day']:+.4f} {np.round(v['festival_day_ci'], 4)} | Ramadan day {v['ramadan_day']:+.4f} {np.round(v['ramadan_day_ci'], 4)}")
    v = fest_res["final/seats_on_festivals"]
    print(f"  seats vs festivals (final): festival day {v['festival_day']:+.4f} {np.round(v['festival_day_ci'], 4)}, Ramadan day {v['ramadan_day']:+.4f} {np.round(v['ramadan_day_ci'], 4)}")
    print(f"  by group: " + ", ".join(f"{k.replace('d_fest_', '')} {vv:+.4f} {np.round(by_fest['other_ci_cluster'][k], 4)}" for k, vv in by_fest["other_coefs"].items()))
    print(f"  joint Wald p {joint_p:.1e} → festivals adopted into ε: {adopt}; passed to Step 2.4: {to_24}")
    for lab, R_ in (("PRIMARY  fold B → 2025", primary), ("SECOND   fold A → Aug24–Jul25", second), ("ALL ROWS fold B → 2025", allrows)):
        s = R_["calibration_slope"]
        print(f"{lab}: n={R_['n_obs']} ({R_['n_countries']} countries, {R_['test_months'][0][:7]}…{R_['test_months'][1][:7]}), ε used {R_['eps_used']:.3f}")
        print(f"  calibration slope {_fmt(s)} (cluster [{s['ci_cluster'][0]:.2f}, {s['ci_cluster'][1]:.2f}]); without festival controls {R_['calibration_slope_no_festivals']['estimate']:.3f}")
        print("  skill vs 'all countries grow alike': " + " | ".join(f"{k} {v:+.3f} [{R_['skill_ci'][k][0]:+.3f}, {R_['skill_ci'][k][1]:+.3f}]" for k, v in R_["skill"].items()))
        print("  volume-weighted skill: " + " | ".join(f"{k} {v:+.3f}" for k, v in R_["skill_volume_weighted"].items()))
        print(f"  honest − naive skill [{R_['honest_minus_naive_ci'][0]:+.3f}, {R_['honest_minus_naive_ci'][1]:+.3f}]; best ε on test {R_['best_eps_on_test']:.2f}")
    print(f"LEVER TEST PASSED: {passed}")
    print("DRIFT (non-overlapping half-years, festival controls)")
    for k, v in drift.items():
        print(f"  {k}: ε_total {v['eps_total']:.3f} {np.round(v['eps_total_ci'], 2)} | ε₁ {v['eps1']:.3f} {np.round(v['eps1_ci'], 2)} | "
              f"local-share response {v['local_share_response']:+.3f} | n={v['n_obs']}")
    print(f"DECISION: default ε_total {decision['default_eps_total']:.3f}, simulator band {np.round(decision['simulator_band'], 2)}")
    print("REAL-WORLD SHOCKS (changes relative to other countries in the same month)")
    for r_ in rows:
        print(f"  {r_['country']:8s} {r_['months']:16s} seats {r_['seat_change_vs_other_countries']:+5.0%} | in range {str(r_['inside_trusted_range']):5s} | "
              f"model {r_['model_prediction_low']:+4.0%}…{r_['model_prediction_high']:+4.0%} (festivals {r_['festival_part']:+.0%}) | actual {r_['actual_checkin_change_vs_other_countries']:+5.0%} "
              f"| implied response {r_['implied_response']:+.2f}  {r_['event']}")

# ============================================================================ Step 2.4
RULE_GAIN, RULE_TOL = 0.10, 0.05             # project keep/drop rule (fold A gain, fold B tolerance), in WMAPE points


def _series(cfg, fd, hi):
    """All 46 daily series (45 nationalities + domestic) with seats_7d where flight data exist."""
    P = cfg["paths"]["processed"]
    dom = pd.read_parquet(P / "hotel_dom_daily.parquet")
    s7 = fd[["nationality_key", "date", "seats_7d"]].rename(columns={"nationality_key": "key"})
    i = hi[["nationality", "date", "arrivals"]].rename(columns={"nationality": "key", "arrivals": "y"}).merge(s7, on=["key", "date"], how="left")
    d = dom[["date", "arrivals"]].rename(columns={"arrivals": "y"}).assign(key="DOMESTIC", seats_7d=np.nan)
    s = pd.concat([i, d], ignore_index=True)
    s = s[(s.date >= "2023-01-07") & (s.date <= cfg["dates"]["test_end"])].reset_index(drop=True)
    bad = set(fd.loc[fd.is_low_coverage, "date"])
    return s.assign(exclude=s.date.isin(bad))


def step_forecast():
    import hashlib
    from . import forecaster as fc
    cfg = load_config(); out = _out(cfg); fd, fm, hi = _load(cfg)
    RD, P, F = cfg["paths"]["reference"], cfg["paths"]["processed"], cfg["folds"]
    cal = pd.read_parquet(P / "calendar.parquet")
    off = pd.read_csv(RD / "calendar_events_official.csv", parse_dates=["start_date", "end_date"])
    fest = sa.festival_dates(RD, off); grp = pd.read_csv(RD / "festival_groups.csv")
    ev = pd.read_csv(RD / "external_events.csv", parse_dates=["start_date", "end_date"])
    S = _series(cfg, fd, hi)
    hh = pd.read_csv(RD / "home_holidays.csv", parse_dates=["start", "end"])
    D = fc.build_frame(S, cal, fc.uae_breaks(cal), fc.festival_daily(grp, fest), fc.event_daily(ev), fc.home_holiday_daily(hh))
    keys = sorted(D.key.unique()); intl = set(hi.nationality); flight = set(fd.nationality_key)
    el = pd.read_csv(out / "elasticities.csv").set_index(["fold", "parameter"])
    eps_fold = {f: float(el.loc[(f, "eps_total_default"), "estimate"]) for f in ("A", "B")}
    eps_final = float(el.loc[("final", "eps_total_default"), "estimate"])
    start = pd.Timestamp("2023-01-07")
    cache = {}

    def run(spec, fold, eps=None):
        spec = frozenset(spec); eps = eps_fold[fold] if eps is None else eps
        k = (spec, fold, round(eps, 4))
        if k not in cache:
            te, info = [], {}
            gr = fc.growth_rates(D, F[fold]["train_end"], eps, "C5p" in spec, intl) if "C5" in spec else {}
            for key in keys:
                p_, _, inf = fc.fit_predict(D[D.key == key], F[fold]["train_end"], start, F[fold]["holdout_end"], set(spec), eps,
                                            growth=gr.get(key, 0.0))
                te.append(p_); info[key] = inf
            pr = pd.concat(te, ignore_index=True)
            cache[k] = (pr, fc.metrics(pr, intl, flight), info)
        return cache[k]

    def score(spec, metric):
        return {f: run(spec, f)[1][metric] for f in ("A", "B")}

    def passes(new, cur, primary="A"):
        other = "B" if primary == "A" else "A"
        return cur[primary] - new[primary] >= RULE_GAIN and new[other] - cur[other] <= RULE_TOL

    DAILY, MONTHLY = "intl_daily_nat", "intl_monthly_nat"
    # ---------------- stage 1: level (judged on monthly × nationality, amendment A3); variants are mutually exclusive
    LEVEL = {"C4a": {"C4a"}, "C4b": {"C4b"}, "C5p": {"C4b", "C5", "C5p"}, "C5s": {"C4b", "C5", "C5s"}}
    base_m, base_d = score(set(), MONTHLY), score(set(), DAILY)
    level_res = {c: {"monthly": score(sp, MONTHLY), "daily": score(sp, DAILY), "passes": passes(score(sp, MONTHLY), base_m)}
                 for c, sp in LEVEL.items()}
    ok = [c for c, v in level_res.items() if v["passes"]]
    # mutually exclusive variants: among those passing, the smaller WORST-fold error (robust tie-break; amendment A3)
    level = min(ok, key=lambda c: max(level_res[c]["monthly"].values())) if ok else None
    chosen = set(LEVEL[level]) if level else set()
    # ---------------- stage 2: shape terms (judged on daily × nationality; C1 on fold B, amendment A1), forward
    PRIMARY = {"C1": "B"}
    shape0 = score(chosen, DAILY)
    single = {c: score(chosen | {c}, DAILY) for c in ("C1", "C2", "C3", "C6")}
    gain = lambda c: shape0[PRIMARY.get(c, "A")] - single[c][PRIMARY.get(c, "A")]
    cur, steps = shape0, []
    for c in sorted(single, key=gain, reverse=True):
        new = score(chosen | {c}, DAILY); kept = passes(new, cur, PRIMARY.get(c, "A"))
        steps.append({"candidate": c, "judged_on_fold": PRIMARY.get(c, "A"), "fold_A": new["A"], "fold_B": new["B"],
                      "gain_A": cur["A"] - new["A"], "gain_B": cur["B"] - new["B"], "kept": kept})
        if kept:
            chosen.add(c); cur = new
    # A4: split the UAE main break into Eid (Gulf-wide) and National Day (UAE residents); judged on fold B
    split = score(chosen | {"S2"}, DAILY); split_ok = passes(split, cur, "B")
    # what the split does on the days it targets (fold A holds the National Day break, fold B the Eid breaks)
    brk = fc.uae_breaks(cal); EXPAT = ["PHILIPPINES", "INDIA", "PAKISTAN", "EGYPT"]
    days = {"national_day": set(brk.date[brk.nd_break == 1]), "eid": set(brk.date[brk.eid_break == 1])}
    def _bias(pr, dd):
        x = pr[pr.key.isin(EXPAT) & pr.date.isin(dd)]
        return float((x.pred.sum() / x.y.sum() - 1) * 100) if len(x) else None
    split_diag = {"expat_series": EXPAT, "metric": "forecast / actual − 1, %"}
    for lab, sp in (("one_main_break", chosen), ("split", chosen | {"S2"})):
        split_diag[lab] = {"expat_bias_national_day_days_A": _bias(run(sp, "A")[0], days["national_day"]),
                           "expat_bias_eid_days_B": _bias(run(sp, "B")[0], days["eid"]),
                           "intl_monthly_nat": score(sp, MONTHLY)}
    split_diag["national_day_days_A"] = int(run(chosen, "A")[0].date.isin(days["national_day"]).groupby(run(chosen, "A")[0].date).any().sum())
    split_diag["eid_days_B"] = int(run(chosen, "B")[0].date.isin(days["eid"]).groupby(run(chosen, "B")[0].date).any().sum())
    if split_ok:
        chosen.add("S2"); cur = split
    swap = score(chosen | {"S1"}, DAILY) if not split_ok else split
    swap_ok = (not split_ok) and passes(swap, cur)
    if swap_ok:
        chosen.add("S1"); cur = swap
    # A4: C7 UAE school breaks reach only the DOMESTIC series (resident families) → judged on domestic daily error
    DOM = "dom_daily"
    dom_cur, dom_new = score(chosen, DOM), score(chosen | {"C7", "C7ws"}, DOM)      # as declared: winter + summer
    school_ok = passes(dom_new, dom_cur, "A")
    assert score(chosen | {"C7", "C7ws"}, DAILY) == cur and score(chosen | {"C7"}, DAILY) == cur   # international untouched
    spring = {"before": dom_new, "after": score(chosen | {"C7"}, DOM), "kept": False}
    if school_ok:
        # A5: add the spring break to the same term; its days fall in fold B only → fold B primary, fold A not worse
        spring["kept"] = passes(spring["after"], spring["before"], "B")
        chosen |= {"C7"} if spring["kept"] else {"C7", "C7ws"}
    spec = frozenset(chosen)

    # ---------------- benchmarks (same rows as the model)
    full = D.set_index(["key", "date"]).y
    s7 = D.set_index(["key", "date"]).seats_7d
    bench = {}
    for f in ("A", "B"):
        pr = run(spec, f)[0]
        ly_idx = pd.MultiIndex.from_arrays([pr.key, pr.date - pd.Timedelta(days=364)])
        snly = full.reindex(ly_idx).to_numpy()
        ratio = (s7.reindex(pd.MultiIndex.from_frame(pr[["key", "date"]])).to_numpy() / s7.reindex(ly_idx).to_numpy())
        seat_adj = np.where(np.isfinite(ratio) & (ratio > 0), ratio ** eps_fold[f], 1.0)
        te_ = F[f]["train_end"]
        rec = D[(D.date > te_ - pd.Timedelta(days=LEVEL_DAYS_BENCH)) & (D.date <= te_)].groupby("key").y.sum()
        prv = D[(D.date > te_ - pd.Timedelta(days=LEVEL_DAYS_BENCH + 364)) & (D.date <= te_ - pd.Timedelta(days=364))].groupby("key").y.sum()
        growth = (rec / prv).reindex(pr.key).to_numpy()
        for name, pred in (("same day last year", snly), ("last year × seat change", snly * seat_adj),
                           ("last year × recent growth", snly * growth)):
            bench[(name, f)] = fc.metrics(pr.assign(pred=pred), intl, flight)

    # ---------------- diagnostics: noise floor, oracle, bias, median effect, horizon, per market, ε sensitivity
    floor, horizon, per_mkt, oracle, bias, median_check = {}, {}, [], {}, {}, {}
    ospec = set(spec) - {"C4a", "C4b", "C5", "C5p", "C5s"}
    for f in ("A", "B"):
        orows = [fc.fit_predict(D[D.key == key], F[f]["holdout_end"], start, F[f]["holdout_end"], ospec, eps_fold[f],
                                pred_start=F[f]["holdout_start"])[0] for key in keys]
        oracle[f] = fc.metrics(pd.concat(orows, ignore_index=True), intl, flight)
        pr = run(spec, f)[0]; i = pr[pr.key.isin(intl)]; dm = pr[pr.key == "DOMESTIC"]
        bias[f] = {"intl": float((i.pred.sum() / i.y.sum() - 1) * 100), "domestic": float((dm.pred.sum() / dm.y.sum() - 1) * 100)}
        r = i.y / i.pred
        median_check[f] = {"daily_wmape_if_scaled": {str(c): fc.wmape(i.y, c * i.pred) for c in (0.85, 0.9, 0.95, 1.0, 1.05)},
                           "actual_over_forecast_mean": float(r.mean()), "actual_over_forecast_median": float(r.median())}
        floor[f] = float(fc.poisson_mad(i.pred.to_numpy()).sum() / i.pred.sum() * 100)
        ma = ((i.date.dt.year - F[f]["train_end"].year) * 12 + i.date.dt.month - F[f]["train_end"].month)
        horizon[f] = {int(h): fc.wmape(g.y, g.pred) for h, g in i.groupby(ma)}
        for key, g in i.groupby("key"):
            gm = g.assign(m=g.date.dt.to_period("M")).groupby("m")[["y", "pred"]].sum()
            per_mkt.append({"fold": f, "nationality": key, "has_flight_data": key in flight, "checkins": float(g.y.sum()),
                            "wmape_daily": fc.wmape(g.y, g.pred), "wmape_monthly": fc.wmape(gm.y, gm.pred),
                            "bias_pct": float((g.pred.sum() / g.y.sum() - 1) * 100)})
    pd.DataFrame(per_mkt).to_csv(out / "stagea_per_market.csv", index=False)
    eps_grid = {e: {m: score_e for m, score_e in (("daily", {f: run(spec, f, e)[1][DAILY] for f in ("A", "B")}),
                                                  ("monthly", {f: run(spec, f, e)[1][MONTHLY] for f in ("A", "B")}))}
                for e in (0.0, 0.25, eps_final, 0.44)}
    eps_switch = [e for e, v in eps_grid.items() if e != eps_final
                  and passes(v["daily"], eps_grid[eps_final]["daily"]) and passes(v["monthly"], eps_grid[eps_final]["monthly"])]

    # ---------------- validation table + fold predictions for Step 2.6
    rows = [{"model": "base", "spec": "base", "fold": f, **run(set(), f)[1]} for f in ("A", "B")]
    rows += [{"model": f"base + level {c}", "spec": "+".join(sorted(sp)), "fold": f, **run(sp, f)[1]} for c, sp in LEVEL.items() for f in ("A", "B")]
    rows += [{"model": f"level + shape {c}", "spec": c, "fold": f, **run((set(LEVEL[level]) if level else set()) | {c}, f)[1]}
             for c in ("C1", "C2", "C3", "C6", "C7") for f in ("A", "B")]
    rows += [{"model": "chosen", "spec": "+".join(sorted(spec)), "fold": f, **run(spec, f)[1]} for f in ("A", "B")]
    rows += [{"model": f"benchmark: {n}", "spec": "", "fold": f, **v} for (n, f), v in bench.items()]
    pd.DataFrame(rows).to_csv(out / "stagea_validation_metrics.csv", index=False)
    for f in ("A", "B"):
        run(spec, f)[0].to_parquet(out / f"stagea_pred_{f}.parquet", index=False)

    # ---------------- final model: train to Jul 2025, forecast the official test window ONCE (not scored here)
    test_end = cfg["dates"]["test_end"]; tr_end = F["final"]["train_end"]
    Dm = D.assign(y=D.y.where(D.date <= tr_end))                                   # test-window check-ins masked
    gr_final = fc.growth_rates(Dm, tr_end, eps_final, "C5p" in spec, intl) if "C5" in spec else {}   # audit fix: latent
    fc_rows, params, finfo = [], [], {}
    for key in keys:
        p_, beta, inf = fc.fit_predict(Dm[Dm.key == key], tr_end, start, test_end, set(spec), eps_final,
                                       growth=gr_final.get(key, 0.0))
        assert p_.y.isna().all()                                                    # never sees test-window check-ins
        fc_rows.append(p_.drop(columns="y")); finfo[key] = inf
        params += [{"series": key, "term": t, "coef": float(b), "multiplier": float(np.exp(b))} for t, b in beta.items()]
    tf = pd.concat(fc_rows, ignore_index=True).rename(columns={"key": "series", "pred": "checkins_forecast"})
    tf.to_csv(out / "stagea_test_forecast.csv", index=False)
    pd.DataFrame(params).to_csv(out / "stagea_final_params.csv", index=False)
    for stale in ["stagea_test_forecast_growth.csv", "stagea_pred_A_growth.parquet", "stagea_pred_B_growth.parquet"]:
        (out / stale).unlink(missing_ok=True)                                       # drafts from the rejected growth finalist
    fingerprint = hashlib.sha256((out / "stagea_test_forecast.csv").read_bytes()).hexdigest()

    nonconv = [k for (sp, f, e), (_, _, inf) in cache.items() for k, v in inf.items() if not v["converged"]] + \
              [k for k, v in finfo.items() if not v["converged"]]
    j = {"rule": f"keep if the judged fold improves ≥ {RULE_GAIN} pts and the other fold worsens ≤ {RULE_TOL} pts",
         "metric_for_level_terms": MONTHLY, "metric_for_shape_terms": DAILY,
         "base": {"daily": base_d, "monthly": base_m}, "level_candidates": level_res, "level_chosen": level,
         "shape_steps": steps, "split_eid_national_day": {"fold_A": split["A"], "fold_B": split["B"], "kept": split_ok,
                                                          "on_target_days": split_diag},
         "swap_all_holidays": {"fold_A": swap["A"], "fold_B": swap["B"], "kept": swap_ok},
         "school_breaks_domestic_only": {"metric": DOM, "before": dom_cur, "after": dom_new, "kept": school_ok},
         "school_spring_break": {"metric": DOM, "judged_on_fold": "B", **spring},
         "chosen_spec": sorted(spec), "chosen_metrics": {f: run(spec, f)[1] for f in ("A", "B")},
         "benchmarks": {f"{n} | {f}": v for (n, f), v in bench.items()},
         "poisson_noise_floor_daily_nat": floor, "oracle_in_sample": oracle, "bias_pct": bias, "median_check": median_check,
         "wmape_by_months_ahead": horizon, "eps_used_in_folds": eps_fold,
         "eps_sensitivity": {f"{k:.6f}": v for k, v in eps_grid.items()}, "eps_default": eps_final, "eps_switch_candidates": eps_switch,
         "final_model": {"train_end": str(tr_end.date()), "eps": eps_final,
                         "forecast_window": [str(cfg["dates"]["test_start"].date()), str(test_end.date())],
                         "rows": int(len(tf)), "sha256": fingerprint,
                         "recent_level_multiplier": {k: v["level"] for k, v in finfo.items()}},
         "non_converged_fits": sorted(set(nonconv))}
    json.dump(j, open(out / "step24_forecaster.json", "w", encoding="utf-8"), indent=1, default=float)

    # ---------------- print
    print(f"rule: judged fold improves ≥ {RULE_GAIN} pts, other fold worsens ≤ {RULE_TOL} pts (international WMAPE, %)")
    print(f"STAGE 1 — level terms, judged on monthly × nationality (base A {base_m['A']:.2f}, B {base_m['B']:.2f})")
    for c, v in level_res.items():
        print(f"  {c:4s} monthly×nat A {v['monthly']['A']:.2f} B {v['monthly']['B']:.2f} | daily×nat A {v['daily']['A']:.2f} B {v['daily']['B']:.2f} | passes {v['passes']}")
    print(f"  → level chosen (passing variant with the smaller worst-fold error): {level}")
    print(f"STAGE 2 — shape terms, judged on daily × nationality (start A {shape0['A']:.2f}, B {shape0['B']:.2f})")
    for st in steps:
        print(f"  add {st['candidate']} (judged on {st['judged_on_fold']}) → A {st['fold_A']:.2f} B {st['fold_B']:.2f} "
              f"(gain A {st['gain_A']:+.2f}, gain B {st['gain_B']:+.2f}) kept={st['kept']}")
    print(f"  split Eid / National Day (judged on B) → A {split['A']:.2f} B {split['B']:.2f} kept={split_ok}")
    sd = split_diag
    print(f"    expat bias on National Day (A, {sd['national_day_days_A']} days): one break {sd['one_main_break']['expat_bias_national_day_days_A']:+.1f}% "
          f"→ split {sd['split']['expat_bias_national_day_days_A']:+.1f}% | on Eid (B, {sd['eid_days_B']} days): "
          f"{sd['one_main_break']['expat_bias_eid_days_B']:+.1f}% → {sd['split']['expat_bias_eid_days_B']:+.1f}%")
    print(f"  swap to ALL UAE holidays → A {swap['A']:.2f} B {swap['B']:.2f} kept={swap_ok}")
    print(f"  add C7 UAE school breaks, domestic only (judged on domestic daily, A) → A {dom_cur['A']:.2f}→{dom_new['A']:.2f} "
          f"B {dom_cur['B']:.2f}→{dom_new['B']:.2f} kept={school_ok}")
    print(f"  A5 + spring break in the same term (judged on domestic daily, B) → A {spring['before']['A']:.2f}→{spring['after']['A']:.2f} "
          f"B {spring['before']['B']:.2f}→{spring['after']['B']:.2f} kept={spring['kept']}")
    print(f"CHOSEN: base + {sorted(spec)}")
    lv = ["intl_daily_nat", "intl_daily_total", "intl_monthly_nat", "intl_monthly_total", "flight_daily_nat", "noflight_daily_nat", "dom_daily", "dom_monthly"]
    print("  " + " ".join(f"{x:>17s}" for x in ["model / fold"] + lv))
    for f in ("A", "B"):
        print("  " + f"{'chosen ' + f:>17s} " + " ".join(f"{run(spec, f)[1][x]:17.2f}" for x in lv))
        for n in ("same day last year", "last year × seat change", "last year × recent growth"):
            print("  " + f"{n[:15] + ' ' + f:>17s} " + " ".join(f"{bench[(n, f)].get(x, float('nan')):17.2f}" for x in lv))
    print(f"  Poisson noise floor (daily×nat): A {floor['A']:.2f}  B {floor['B']:.2f} | oracle (same model fitted with hindsight): "
          + " | ".join(f"{f} {oracle[f]['intl_daily_nat']:.2f}" for f in ("A", "B")))
    print("  bias (forecast/actual − 1): " + " | ".join(f"{f}: intl {bias[f]['intl']:+.1f}%, domestic {bias[f]['domestic']:+.1f}%" for f in ("A", "B")))
    print("  median effect: actual/forecast median " + " | ".join(f"{f} {median_check[f]['actual_over_forecast_median']:.3f} (mean {median_check[f]['actual_over_forecast_mean']:.3f})" for f in ("A", "B")))
    for f in ("A", "B"):
        print(f"  by months ahead {f}: " + "  ".join(f"{h}:{v:.1f}" for h, v in horizon[f].items()))
    print("  ε sensitivity daily (A/B) | monthly (A/B): " + " ; ".join(
        f"{e:.2f}: {v['daily']['A']:.2f}/{v['daily']['B']:.2f} | {v['monthly']['A']:.2f}/{v['monthly']['B']:.2f}" for e, v in eps_grid.items())
          + f"  → switch default: {eps_switch or 'no'}")
    print(f"FINAL: {len(tf)} forecast rows for {j['final_model']['forecast_window']}, ε {eps_final:.3f}, sha256 {fingerprint[:16]}…; "
          f"non-converged fits: {j['non_converged_fits'] or 'none'}")



LEVEL_DAYS_BENCH = 182

# ============================================================================ Step 2.5
"""New-route method: a market with no seat history in the pack (the 12 nationalities without flight data, or a new
market). The log-seat lever cannot be used from zero seats, so the effect is built from a borrowed per-seat ratio.

Rules fixed before the run:
  * Ratio r_c = check-ins ÷ seats per flight country over the last 12 training months (Aug 2024–Jul 2025), flight-
    disruption days removed. Borrowed value = median over the 33 countries (robust to outliers).
  * Uncertainty: leave each country out, predict its ratio by the median of the other 32; error factor
    = exp(|log(actual / predicted)|). Band = ÷/× the 80th-percentile factor. A grouping (broad region; GCC vs rest)
    replaces the overall median only if it lowers the median log-error by ≥ 10%.
  * New-route check-ins per added seat = ε × borrowed ratio (extra seats bring hotel guests at the marginal, not the
    average, rate — Steps 2.2–2.3), never more than local passengers per seat (median load factor × local share).
  * Real-world test on the ~40 new city routes inside existing markets (2023–2025): split each country's YoY seat
    change into continuing cities, new cities and closed cities. If the 95% interval of (new-city response −
    continuing response) excludes 0, new routes get their own response; otherwise the same ε is used.
"""
BROAD = {"Western Europe": "Europe", "Northern Europe": "Europe", "Southern Europe": "Europe", "Eastern Europe": "Europe",
         "Western Asia": "Middle East & Africa", "Northern Africa": "Middle East & Africa", "Southern Africa": "Middle East & Africa",
         "Central Asia": "Central & South Asia", "Southern Asia": "Central & South Asia",
         "Eastern Asia": "East & South-east Asia", "South-eastern Asia": "East & South-east Asia",
         "Northern America": "Americas", "South America": "Americas", "Latin America and the Caribbean": "Americas",
         "Central America": "Americas", "Australia and New Zealand": "Oceania"}


def _loo(r: pd.Series, groups: pd.Series | None = None, min_group: int = 3) -> pd.DataFrame:
    """Leave-one-out prediction of each country's ratio by the median of the others (same group if ≥ min_group others)."""
    rows = []
    for c in r.index:
        others = r.drop(c)
        if groups is not None:
            same = others[groups.reindex(others.index) == groups[c]]
            pred = same.median() if len(same) >= min_group else others.median()
        else:
            pred = others.median()
        rows.append({"nationality": c, "ratio": r[c], "predicted": pred, "log_error": abs(np.log(r[c] / pred))})
    d = pd.DataFrame(rows); d["error_factor"] = np.exp(d.log_error)
    return d


def step_newroute():
    cfg = load_config(); out = _out(cfg); fd, fm, hi = _load(cfg)
    P, RD, F = cfg["paths"]["processed"], cfg["paths"]["reference"], cfg["folds"]
    cm = pd.read_csv(RD / "country_map.csv")
    tr_end = F["final"]["train_end"]; w0 = tr_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    bad = set(fd.loc[fd.is_low_coverage, "date"])
    cl = sa.closure_days(fd, w0, tr_end)                                    # a country's own airspace-closure days
    fk = pd.MultiIndex.from_frame(fd[["nationality_key", "date"]]).isin(cl)
    hk = pd.MultiIndex.from_frame(hi[["nationality", "date"]]).isin(cl)
    f12 = fd[~fk & (fd.date >= w0) & (fd.date <= tr_end) & ~fd.date.isin(bad)]
    h12 = hi[~hk & (hi.date >= w0) & (hi.date <= tr_end) & ~hi.date.isin(bad)]
    seats = f12.groupby("nationality_key")[["seats", "p2p", "pax"]].sum()
    chk = h12.groupby("nationality").arrivals.sum()
    r = (chk.reindex(seats.index) / seats.seats).rename("checkins_per_seat")
    r_med = float(r.median())

    # ---------------- (1) leave-one-out: how wrong is a borrowed ratio?
    meta = cm.set_index("nationality")
    region = meta.m49_subregion.map(BROAD).where(~meta.is_gcc, "GCC")
    gcc = meta.is_gcc.map({True: "GCC", False: "rest"})
    loo = {"overall median": _loo(r), "broad region": _loo(r, region), "GCC vs rest": _loo(r, gcc)}
    summ = {k: {"median_error_factor": float(np.exp(v.log_error.median())), "p80_error_factor": float(np.exp(v.log_error.quantile(.8))),
                "median_log_error": float(v.log_error.median())} for k, v in loo.items()}
    base_le = summ["overall median"]["median_log_error"]
    adopt = [k for k in ("broad region", "GCC vs rest") if summ[k]["median_log_error"] <= 0.9 * base_le]
    method = min(adopt, key=lambda k: summ[k]["median_log_error"]) if adopt else "overall median"
    loo[method].merge(region.rename("broad_region"), left_on="nationality", right_index=True) \
        .to_csv(out / "new_route_ratios.csv", index=False)

    # ---------------- (2) real-world test: new city routes inside existing markets
    rows = pd.read_parquet(P / "flights_rows.parquet")
    rows = rows[(rows.granularity == "daily") & ~rows.date.isin(bad)]
    name = cm.dropna(subset=["flight_country_name"]).set_index("flight_country_name").nationality
    cityM = rows.assign(nationality_key=rows.country.map(name), month=rows.date.dt.to_period("M").dt.to_timestamp()) \
        .groupby(["nationality_key", "city", "month"]).seats.sum().reset_index()
    prev = cityM.assign(month=cityM.month + pd.DateOffset(years=1)).rename(columns={"seats": "seats_prev"})
    cc = cityM.merge(prev, on=["nationality_key", "city", "month"], how="outer").fillna({"seats": 0, "seats_prev": 0})
    cc = cc[(cc.month >= "2024-01-01") & (cc.month <= tr_end)]
    g = cc.assign(cont_now=np.where((cc.seats > 0) & (cc.seats_prev > 0), cc.seats, 0),
                  cont_prev=np.where((cc.seats > 0) & (cc.seats_prev > 0), cc.seats_prev, 0),
                  new=np.where(cc.seats_prev == 0, cc.seats, 0), closed=np.where(cc.seats == 0, cc.seats_prev, 0)) \
        .groupby(["nationality_key", "month"])[["cont_now", "cont_prev", "new", "closed"]].sum().reset_index()
    g = g[(g.cont_now > 0) & (g.cont_prev > 0)]
    g = g.assign(g_cont=np.log(g.cont_now / g.cont_prev), g_new=np.log1p(g.new / g.cont_now), g_closed=-np.log1p(g.closed / g.cont_prev))
    m = sa.monthly_panel(fd, hi)
    full, rng_ = _window(m, tr_end)
    t = full.merge(g, on=["nationality_key", "month"], how="inner")
    ident = float(np.abs(t.d_seats - (t.g_cont + t.g_new + t.g_closed)).max())       # monthly-panel seats vs city sums
    tt = sa.trim(t, rng_)
    # reparametrised: y = a·d_seats + (b−a)·g_new + (c−a)·g_closed, so the extra coefficients ARE the differences
    nat = {lab: sa.fe_elasticity(d_, "d_checkins", ["d_seats", "g_new", "g_closed"], n_boot=0)
           for lab, d_ in (("trusted range (primary)", tt), ("all rows", t))}
    exposure = {"country_months_with_new_city_seats_over_2pct": int((tt.g_new > np.log1p(0.02)).sum()),
                "country_months_with_closures_over_2pct": int((tt.g_closed < -np.log1p(0.02)).sum()),
                "rows": int(len(tt)), "max_seat_identity_gap": ident}
    d_new = nat["trusted range (primary)"]["other_ci_cluster"]["g_new"]
    own_response = bool(d_new[0] > 0 or d_new[1] < 0)

    # ---------------- (3) the new-route rule
    el = pd.read_csv(out / "elasticities.csv").set_index(["fold", "parameter"])
    eps = float(el.loc[("final", "eps_total_default"), "estimate"])
    resp = eps + (nat["trusted range (primary)"]["other_coefs"]["g_new"] if own_response else 0.0)
    marginal = resp * r_med
    local_per_seat = float((seats.p2p / seats.seats).median())
    f80 = summ[method]["p80_error_factor"]
    band = [marginal / f80, min(marginal * f80, local_per_seat)]
    noflight = sorted(set(hi.nationality) - set(fd.nationality_key))
    cur = h12[h12.nationality.isin(noflight)].groupby("nationality").arrivals.sum().sort_values(ascending=False)
    ex_seats_month = 7 * 300 * 365.25 / 12 / 7                                  # 7 weekly flights × 300 seats
    example = {"route": "7 weekly flights × 300 seats (≈ 9,130 seats a month)", "for": cur.index[0],
               "current_checkins_per_month": float(cur.iloc[0] / 12), "added_checkins_per_month": marginal * ex_seats_month,
               "band": [band[0] * ex_seats_month, band[1] * ex_seats_month]}
    j = {"window": [str(w0.date()), str(tr_end.date())], "ratio_median": r_med, "ratio_pooled": float(chk.reindex(seats.index).sum() / seats.seats.sum()),
         "ratio_range": [float(r.min()), r.idxmin(), float(r.max()), r.idxmax()], "loo": summ, "grouping_rule": "≥10% lower median log-error",
         "method": method, "new_city_test": {k: {kk: vv for kk, vv in v.items() if kk not in ("boot_draws", "countries")} for k, v in nat.items()},
         "new_city_exposure": exposure, "new_routes_get_own_response": own_response, "eps": eps,
         "checkins_per_added_seat": marginal, "band": band, "cap_local_pax_per_seat": local_per_seat,
         "no_flight_markets_current_checkins_per_month": (cur / 12).round(0).to_dict(), "example": example}
    json.dump(j, open(out / "step25_new_route.json", "w", encoding="utf-8"), indent=1, default=float)

    print(f"check-ins per seat, {w0.date()}…{tr_end.date()}: median {r_med:.3f} | pooled {j['ratio_pooled']:.3f} | "
          f"range {r.min():.3f} ({r.idxmin()}) … {r.max():.3f} ({r.idxmax()})")
    for k, v in summ.items():
        print(f"  leave-one-out, {k:15s}: typical error ×{v['median_error_factor']:.2f}, 80th pct ×{v['p80_error_factor']:.2f}")
    print(f"  → method: {method} (grouping needs ≥10% lower typical error)")
    for k, v in nat.items():
        print(f"  new-city test ({k}): continuing-seat response {v['estimate']:.3f} {np.round(v['ci_cluster'], 3)}; "
              f"new-city minus continuing {v['other_coefs']['g_new']:+.3f} {np.round(v['other_ci_cluster']['g_new'], 3)}; "
              f"closed minus continuing {v['other_coefs']['g_closed']:+.3f} {np.round(v['other_ci_cluster']['g_closed'], 3)}; n={v['n_obs']}")
    print(f"  exposure: {exposure}")
    print(f"  → new routes get their own response: {own_response}")
    print(f"NEW-ROUTE RULE: {resp:.3f} × {r_med:.3f} = {marginal:.4f} check-ins per added seat, band {band[0]:.4f} … {band[1]:.4f} "
          f"(cap: local passengers per seat {local_per_seat:.3f})")
    print(f"  example — {example['for']}: now {example['current_checkins_per_month']:,.0f} check-ins/month; {example['route']} → "
          f"+{example['added_checkins_per_month']:,.0f} (range {example['band'][0]:,.0f} … {example['band'][1]:,.0f})")
    print(f"  12 no-flight markets, current check-ins per month: " + ", ".join(f"{k.title()} {v:,.0f}" for k, v in (cur / 12).items()))

# ============================================================================ Step 2.6
"""End-to-end test: seats → Stage A check-ins → Stage B guests, plus the one-time official test-window scoring.

Fixed before the run:
  * Folds A/B: Stage B = the Phase 1 selected engines (international 'base+market_step', domestic 'base+market_drift'), rebuilt from
    the saved fold parameters (reproduction verified to 0.002 WMAPE points). Held-out check-ins are replaced by the
    Stage A fold forecasts; check-ins before the fold cut-off stay observed (they are known on the forecast date).
  * Criteria: (a) end-to-end guests beat same-day-last-year guests on international daily × nationality AND monthly
    total in both folds; (b) 'Stage A cost' = end-to-end minus Phase 1 with actual check-ins, reported per level.
  * Official test window, ONE time: verify the frozen forecast's fingerprint, then score Stage A check-ins
    (criterion c: beats all three Stage A benchmarks on international daily × nationality and monthly total), then the
    Step 2.3 lever check exactly as pre-registered (slope 95% interval contains 1, skill > 0, naive one-for-one lower).
    Guests in the test window are hidden (they are what the organisers score), so they are not evaluated here.
"""


def _stageb_fold(cfg, hi, hd, fold):
    """Phase 1 selected Stage B engines for one fold → function(arrivals tables) → guests per market."""
    from . import stageb as sb, phase1 as p1
    _, _, cal, _, idx = p1.load(cfg)
    F = cfg["folds"][fold]
    S = json.load(open(cfg["paths"]["processed"] / "phase1" / "final_summary.json", encoding="utf-8"))["selection"]
    ci, cd = S["chosen"]["intl"], S["chosen"]["dom"]
    assert not ci["grand_prix"] and ci["long_break"] and not cd["long_break"] and not cd["arrival_dow"], "Phase 1 selection changed: update _stageb_fold"
    P = json.load(open(cfg["paths"]["processed"] / "phase1" / "validation_params.json", encoding="utf-8"))[f"{fold}|{S['intl_model']}"]
    X, names = sb.design(cal, idx, F["train_end"], {"long_break": True})
    e = np.exp(X @ np.array([P["intl_beta"][n] for n in names]))
    tau_i = sb.make_tau(idx, F["train_end"], ci.get("drift_shape", "trend")) if ci["drift"] else None
    ok = (idx >= F["train_start"]) & (idx <= F["train_end"])
    Xd, _ = sb.design(cal, idx, F["train_end"], {"long_break": False})
    rd = sb.fit_group({"DOMESTIC": sb.build_markets(hi, hd, idx)["DOMESTIC"]}, Xd, ok, drift=cd["drift"], idx=idx,
                      drift_shape=cd.get("drift_shape", "trend"))
    dm = rd.markets["DOMESTIC"]; ed = np.exp(Xd @ rd.beta)

    def guests(hi_, hd_):
        mk = sb.build_markets(hi_, hd_, idx)
        for n, m in mk.items():
            if n != "DOMESTIC":
                m.tau = tau_i
        out = {n: sb.predict(m, np.array(P["theta"][n]), e) for n, m in mk.items() if n != "DOMESTIC"}
        md = mk["DOMESTIC"]; md.tau = dm.tau
        out["DOMESTIC"] = sb.predict(md, dm.theta, ed)
        return out
    return guests, idx


def _lever_check_test(cfg, fd, hi, out):
    """Step 2.3 pre-registered lever check on the official test window (Aug 2025–Feb 2026 vs a year earlier)."""
    m, _, _ = _panel_with_festivals(cfg, fd, hi)
    F = cfg["folds"]; tr_end, te_end = F["final"]["train_end"], cfg["dates"]["test_end"]
    p_tr, rng_ = _primary(m, tr_end)
    g = sa.fe_elasticity(p_tr, "d_checkins", ["d_seats"] + FEST, n_boot=0)          # festival effects from training
    te = _test_rows(m, tr_end, te_end, rng_).dropna(subset=["d_seats", "d_checkins"])
    el = pd.read_csv(out / "elasticities.csv").set_index(["fold", "parameter"])
    eps = float(el.loc[("final", "eps_total_default"), "estimate"])
    te = te.assign(pred=eps * te.d_seats,
                   d_checkins_adj=te.d_checkins - g["other_coefs"]["d_festival_days"] * te.d_festival_days
                   - g["other_coefs"]["d_ramadan_days"] * te.d_ramadan_days)
    slope = sa.fe_elasticity(te, "d_checkins", ["pred"] + FEST, n_boot=N_BOOT)
    cands = {"default 0.31": eps, "upper 0.44": float(el.loc[("final", "eps_total_plain"), "estimate"]),
             "2025 held-out optimum 0.25": 0.25, "naive one-for-one (1.0)": 1.0}
    sk = sa.lever_skill(te.assign(d_checkins=te.d_checkins_adj), cands, n_boot=N_BOOT)
    prof = sa.lever_skill(te.assign(d_checkins=te.d_checkins_adj), {str(e_): float(e_) for e_ in np.round(np.arange(0, 1.01, 0.05), 2)}, n_boot=0)["skill"]
    passed = bool(slope["ci_boot"][0] <= 1 <= slope["ci_boot"][1] and sk["skill"]["default 0.31"] > 0
                  and sk["skill"]["naive one-for-one (1.0)"] < sk["skill"]["default 0.31"])
    return {"test_months": [str(te.month.min().date()), str(te.month.max().date())], "n_obs": int(len(te)),
            "n_countries": int(te.nationality_key.nunique()), "eps": eps, "calibration_slope": _slim(slope),
            "skill": sk["skill"], "skill_ci": sk["ci"], "best_eps_on_test": float(max(prof, key=prof.get)), "passed": passed}


def step_endtoend():
    import hashlib
    from . import forecaster as fc, phase1 as p1
    cfg = load_config(); out = _out(cfg); fd, fm, hi = _load(cfg)
    P, F = cfg["paths"]["processed"], cfg["folds"]
    hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    intl = set(hi.nationality); flight = set(fd.nationality_key)

    # ---------------- (1) folds A and B: end-to-end guests
    res, rows_mkt, horizon = {}, [], {}
    for fold in ("A", "B"):
        guests, idx = _stageb_fold(cfg, hi, hd, fold)
        ho = (idx >= F[fold]["holdout_start"]) & (idx <= F[fold]["holdout_end"])
        sa_pred = pd.read_parquet(out / f"stagea_pred_{fold}.parquet")
        assert sa_pred.date.min() == F[fold]["holdout_start"] and sa_pred.date.max() == F[fold]["holdout_end"]
        a_i = sa_pred[sa_pred.key != "DOMESTIC"].set_index(["key", "date"]).pred
        a_d = sa_pred[sa_pred.key == "DOMESTIC"].set_index("date").pred
        hi2 = hi.copy(); k = pd.MultiIndex.from_frame(hi2[["nationality", "date"]])
        hi2["arrivals"] = np.where(a_i.reindex(k).notna(), a_i.reindex(k), hi2.arrivals)
        hd2 = hd.copy(); hd2["arrivals"] = np.where(hd2.date.isin(a_d.index), a_d.reindex(hd2.date).to_numpy(), hd2.arrivals)
        from . import stageb as sb
        mk = sb.build_markets(hi, hd, idx)                                            # targets = observed guests
        g_obs, g_e2e = guests(hi, hd), guests(hi2, hd2)
        df_obs, df_e2e = p1.frame(mk, g_obs, idx, ho), p1.frame(mk, g_e2e, idx, ho)
        naive = pd.read_parquet(P / "phase1" / f"pred_{fold}_seasonal_naive.parquet")
        res[fold] = {"end_to_end": p1.metrics(df_e2e), "phase1_actual_checkins": p1.metrics(df_obs),
                     "same_day_last_year_guests": p1.metrics(naive)}
        I = df_e2e[df_e2e.market != "DOMESTIC"]
        res[fold]["bias_pct"] = {"intl": float((I.p.sum() / I.y.sum() - 1) * 100),
                                 "domestic": float((lambda d: d.p.sum() / d.y.sum() - 1)(df_e2e[df_e2e.market == "DOMESTIC"]) * 100)}
        ma = (I.date.dt.year - F[fold]["train_end"].year) * 12 + I.date.dt.month - F[fold]["train_end"].month
        horizon[fold] = {int(h): float(np.abs(g.y - g.p).sum() / g.y.sum() * 100) for h, g in I.groupby(ma)}
        for n, g in df_e2e.assign(m=df_e2e.date.dt.to_period("M")).groupby("market"):
            gm = g.groupby("m")[["y", "p"]].sum(); go = df_obs[df_obs.market == n]
            rows_mkt.append({"fold": fold, "market": n, "has_flight_data": n in flight, "guest_nights": float(g.y.sum()),
                             "e2e_wmape_daily": float(np.abs(g.y - g.p).sum() / g.y.sum() * 100),
                             "e2e_wmape_monthly": float(np.abs(gm.y - gm.p).sum() / gm.y.sum() * 100),
                             "phase1_wmape_daily": float(np.abs(go.y - go.p).sum() / go.y.sum() * 100),
                             "e2e_bias_pct": float((g.p.sum() / g.y.sum() - 1) * 100)})
        df_e2e.assign(fold=fold).to_parquet(out / f"endtoend_pred_{fold}.parquet", index=False)
        # is the daily × nationality gap to same-day-last-year meaningful? resample markets (audit addition)
        nv = naive[naive.market != "DOMESTIC"][["market", "date", "p"]].rename(columns={"p": "pn"})
        dd = I.merge(nv, on=["market", "date"])
        g = dd.assign(ae=(dd.y - dd.p).abs(), an=(dd.y - dd.pn).abs()).groupby("market")[["y", "ae", "an"]].sum()
        rng = np.random.default_rng(0); draws = []
        for _ in range(N_BOOT):
            s_ = g.iloc[rng.integers(0, len(g), len(g))]; draws.append((s_.an.sum() - s_.ae.sum()) / s_.y.sum() * 100)
        res[fold]["gap_vs_same_day_last_year_daily_nat"] = {"points": float((g.an.sum() - g.ae.sum()) / g.y.sum() * 100),
                                                            "ci95_market_bootstrap": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    pd.DataFrame(rows_mkt).to_csv(out / "endtoend_per_market.csv", index=False)
    crit_a = all(res[f]["end_to_end"][k] < res[f]["same_day_last_year_guests"][k]
                 for f in ("A", "B") for k in ("intl_daily_x_nationality", "intl_monthly_total"))
    tab = pd.DataFrame([{"fold": f, "model": mdl, **v} for f in res for mdl, v in res[f].items()
                        if mdl not in ("bias_pct", "gap_vs_same_day_last_year_daily_nat")])
    tab.to_csv(out / "endtoend_metrics.csv", index=False)

    # ---------------- (2) official test window, ONE time
    j24 = json.load(open(out / "step24_forecaster.json", encoding="utf-8"))
    # the test window is scored ONLY with the forecast that was scored once on 2 Oct 2026 (forecaster v1), kept as an
    # immutable record; later forecaster versions are chosen on folds A/B and are never scored on the test window
    ref = json.load(open(cfg["paths"]["reference"] / "frozen_test_forecast.json", encoding="utf-8"))
    fpath = cfg["paths"]["reference"].parent.parent / ref["file"]
    sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
    assert sha == ref["sha256"], "the scored test-window forecast record was modified"
    tf = pd.read_csv(fpath, parse_dates=["date"]).rename(columns={"series": "key", "checkins_forecast": "pred"})
    S = _series(cfg, fd, hi)
    act = S.set_index(["key", "date"]).y
    tt = tf.assign(y=act.reindex(pd.MultiIndex.from_frame(tf[["key", "date"]])).to_numpy())
    assert tt.y.notna().all()
    test_scores = {"model": fc.metrics(tt, intl, flight)}
    s7 = S.set_index(["key", "date"]).seats_7d
    ly = pd.MultiIndex.from_arrays([tt.key, tt.date - pd.Timedelta(days=364)])
    snly = act.reindex(ly).to_numpy()
    ratio = s7.reindex(pd.MultiIndex.from_frame(tt[["key", "date"]])).to_numpy() / s7.reindex(ly).to_numpy()
    eps = j24["eps_default"]; tr_end = F["final"]["train_end"]
    rec = S[(S.date > tr_end - pd.Timedelta(days=LEVEL_DAYS_BENCH)) & (S.date <= tr_end)].groupby("key").y.sum()
    prv = S[(S.date > tr_end - pd.Timedelta(days=LEVEL_DAYS_BENCH + 364)) & (S.date <= tr_end - pd.Timedelta(days=364))].groupby("key").y.sum()
    for name, pred in (("same day last year", snly),
                       ("last year × seat change", snly * np.where(np.isfinite(ratio) & (ratio > 0), ratio ** eps, 1.0)),
                       ("last year × recent growth", snly * (rec / prv).reindex(tt.key).to_numpy())):
        test_scores[f"benchmark: {name}"] = fc.metrics(tt.assign(pred=pred), intl, flight)
    ti = tt[tt.key.isin(intl)]
    test_bias = {"intl": float((ti.pred.sum() / ti.y.sum() - 1) * 100),
                 "by_month": {str(k): float(v) for k, v in (ti.groupby(ti.date.dt.to_period("M")).apply(lambda g: g.pred.sum() / g.y.sum() - 1) * 100).items()},
                 "domestic": float((lambda d: d.pred.sum() / d.y.sum() - 1)(tt[tt.key == "DOMESTIC"]) * 100)}
    crit_c = all(test_scores["model"][k] < test_scores[b][k] for b in test_scores if b != "model"
                 for k in ("intl_daily_nat", "intl_monthly_total"))
    pd.DataFrame([{"model": k, **v} for k, v in test_scores.items()]).to_csv(out / "test_window_stagea_scores.csv", index=False)
    lever = _lever_check_test(cfg, fd, hi, out)

    j = {"criteria": {"a_end_to_end_beats_same_day_last_year": crit_a, "c_test_stagea_beats_benchmarks": crit_c,
                      "d_lever_check_passed": lever["passed"]},
         "folds": res, "e2e_wmape_by_months_ahead": horizon, "test_window": {"frozen_forecast_sha256_verified": sha,
         "scored_forecaster": "v1 (as scored on 2 Oct 2026)", "current_forecaster_sha256_unscored": j24["final_model"]["sha256"],
         "stagea_scores": test_scores, "bias_pct": test_bias, "lever_check": lever}}
    json.dump(j, open(out / "step26_endtoend.json", "w", encoding="utf-8"), indent=1, default=float)

    # ---------------- print
    lv = ["intl_daily_x_nationality", "intl_daily_total", "intl_monthly_x_nationality", "intl_monthly_total", "domestic_daily", "domestic_monthly"]
    print("END-TO-END GUESTS (WMAPE %): seats → Stage A → Stage B")
    print("  " + " ".join(f"{x:>27s}" for x in ["fold / model"] + lv))
    for f in ("A", "B"):
        for mdl in ("end_to_end", "phase1_actual_checkins", "same_day_last_year_guests"):
            print("  " + f"{f + ' ' + mdl:>27s} " + " ".join(f"{res[f][mdl][x]:27.2f}" for x in lv))
        print(f"  {f} end-to-end bias: intl {res[f]['bias_pct']['intl']:+.1f}%, domestic {res[f]['bias_pct']['domestic']:+.1f}% | "
              f"by months ahead: " + "  ".join(f"{h}:{v:.1f}" for h, v in horizon[f].items()))
    print(f"  criterion (a) end-to-end beats same-day-last-year guests: {crit_a} | daily×nat gap (market bootstrap): "
          + " | ".join(f"{f} {res[f]['gap_vs_same_day_last_year_daily_nat']['points']:+.2f} pts "
                       f"{np.round(res[f]['gap_vs_same_day_last_year_daily_nat']['ci95_market_bootstrap'], 2)}" for f in ("A", "B")))
    print(f"TEST WINDOW (one-time; frozen forecast sha256 verified {sha[:16]}…) — Stage A check-ins")
    lv2 = ["intl_daily_nat", "intl_daily_total", "intl_monthly_nat", "intl_monthly_total", "flight_daily_nat", "noflight_daily_nat", "dom_daily", "dom_monthly"]
    print("  " + " ".join(f"{x:>18s}" for x in ["model"] + lv2))
    for k, v in test_scores.items():
        print("  " + f"{k.replace('benchmark: ', '')[:18]:>18s} " + " ".join(f"{v[x]:18.2f}" for x in lv2))
    print(f"  bias: intl {test_bias['intl']:+.1f}%, domestic {test_bias['domestic']:+.1f}% | by month " + ", ".join(f"{k} {v:+.1f}%" for k, v in test_bias["by_month"].items()))
    print(f"  criterion (c) beats all benchmarks (daily×nat, monthly total): {crit_c}")
    s_ = lever["calibration_slope"]
    print(f"LEVER CHECK on test window (pre-registered in 2.3): n={lever['n_obs']} ({lever['n_countries']} countries, {lever['test_months'][0][:7]}…{lever['test_months'][1][:7]}), ε {lever['eps']:.3f}")
    print(f"  calibration slope {_fmt(s_)} | skill: " + " | ".join(f"{k} {v:+.3f} [{lever['skill_ci'][k][0]:+.3f}, {lever['skill_ci'][k][1]:+.3f}]" for k, v in lever["skill"].items()))
    print(f"  best ε on test {lever['best_eps_on_test']:.2f} → PASSED: {lever['passed']}")

# ============================================================================ Step 2.7
"""Guests per seat (the executive view) and the generated Phase 2 report.

Guest nights per seat over the last 12 training months (disruption days removed):
    average  G/S = (G/A) · (A/S)                     (stay length × check-ins per seat)
    marginal  = min(ε · A/S, local passengers per seat) · G/A   — the seat lever's per-seat effect, in guest nights
Band: the simulator ε band from Step 2.3. Markets without seat data use the Step 2.5 new-route rate.
"""


def step_guests():
    cfg = load_config(); out = _out(cfg); fd, fm, hi = _load(cfg)
    F = cfg["folds"]; tr_end = F["final"]["train_end"]; w0 = tr_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    bad = set(fd.loc[fd.is_low_coverage, "date"])
    cl = sa.closure_days(fd, w0, tr_end)                                    # a country's own airspace-closure days
    fk = pd.MultiIndex.from_frame(fd[["nationality_key", "date"]]).isin(cl)
    hk = pd.MultiIndex.from_frame(hi[["nationality", "date"]]).isin(cl)
    f = fd[~fk & (fd.date >= w0) & (fd.date <= tr_end) & ~fd.date.isin(bad)].groupby("nationality_key")[["seats", "p2p", "n_flights"]].sum()
    h = hi[~hk & (hi.date >= w0) & (hi.date <= tr_end) & ~hi.date.isin(bad)]
    obs = h[h.is_target]                                               # rows whose guests are observed (stay length)
    alos = obs.groupby("nationality").guests.sum() / obs.groupby("nationality").arrivals.sum()
    A = h.groupby("nationality").arrivals.sum()
    j23 = json.load(open(out / "step23_lever_test.json", encoding="utf-8")); j25 = json.load(open(out / "step25_new_route.json", encoding="utf-8"))
    eps = j23["decision"]["default_eps_total"]
    lo, hi_ = sa.lever_interval(json.load(open(out / "step26_endtoend.json", encoding="utf-8")))      # out-of-sample 80% interval of ε
    mp = pd.read_csv(cfg["paths"]["processed"] / "phase1" / "market_parameters.csv", index_col=0)
    rows = []
    for n in sorted(A.index):
        has = n in f.index
        a_s = A[n] / f.seats[n] if has else np.nan
        lps = f.p2p[n] / f.seats[n] if has else j25["cap_local_pax_per_seat"]
        spf = f.seats[n] / f.n_flights[n] if has else np.nan
        per = (lambda e: min(e * a_s, lps)) if has else (lambda e: min(j25["checkins_per_added_seat"] * e / eps, lps))
        rows.append({"nationality": n, "has_flight_data": has, "stay_nights": alos[n], "stay_nights_model": mp.avg_stay_nights_model.get(n, np.nan),
                     "checkins_per_seat": a_s, "local_pax_per_seat": lps, "guest_nights_per_seat": a_s * alos[n] if has else np.nan,
                     "marginal_guest_nights_per_1000_added_seats": 1000 * per(eps) * alos[n],
                     "band_low": 1000 * per(lo) * alos[n], "band_high": 1000 * per(hi_) * alos[n],
                     "cap_binds": bool(has and eps * a_s > lps), "seats_per_flight": spf,
                     "one_extra_daily_flight_guest_nights_per_month": per(eps) * alos[n] * (spf if has else np.nan) * 365.25 / 12,
                     "checkins_12m": float(A[n])})
    t = pd.DataFrame(rows); t.to_csv(out / "guests_per_seat.csv", index=False)
    fl = t[t.has_flight_data]
    tot = {"pooled_guest_nights_per_seat": float((fl.guest_nights_per_seat * f.seats.reindex(fl.nationality).to_numpy()).sum() / f.seats.sum()),
           "median_guest_nights_per_seat": float(fl.guest_nights_per_seat.median()),
           "median_marginal_per_1000": float(fl.marginal_guest_nights_per_1000_added_seats.median()),
           "median_band_per_1000": [float(fl.band_low.median()), float(fl.band_high.median())],
           "stay_data_vs_model_median_ratio": float((t.stay_nights / t.stay_nights_model).median()),
           "cap_binds_for": fl.loc[fl.cap_binds, "nationality"].tolist(), "eps": eps, "band": [lo, hi_],
           "window": [str(w0.date()), str(tr_end.date())]}
    json.dump(tot, open(out / "step27_guests_per_seat.json", "w", encoding="utf-8"), indent=1, default=float)
    print(f"guest nights per seat ({w0.date()}…{tr_end.date()}): pooled {tot['pooled_guest_nights_per_seat']:.3f}, median {tot['median_guest_nights_per_seat']:.3f} | "
          f"marginal per 1,000 added seats: median {tot['median_marginal_per_1000']:.0f} (band {tot['median_band_per_1000'][0]:.0f}–{tot['median_band_per_1000'][1]:.0f}) | "
          f"stay length data/model median ratio {tot['stay_data_vs_model_median_ratio']:.3f} | cap binds for {tot['cap_binds_for']}")
    show = t.sort_values("checkins_12m", ascending=False).head(10)
    print(show[["nationality", "stay_nights", "checkins_per_seat", "guest_nights_per_seat", "marginal_guest_nights_per_1000_added_seats",
                "band_low", "band_high", "seats_per_flight", "one_extra_daily_flight_guest_nights_per_month"]].round(2).to_string(index=False))


def _limitations(cfg, fd, hi, out):
    """Recompute the verified limitations (critique check of 2 Oct 2026) so the report never carries a hand-typed number."""
    P, F = cfg["paths"]["processed"], cfg["folds"]
    sel1 = json.load(open(P / "phase1" / "final_summary.json", encoding="utf-8"))["selection"]["intl_model"]          # Phase 1's chosen model
    p = pd.read_parquet(P / "phase1" / f"pred_B_{sel1}.parquet"); i = p[p.market != "DOMESTIC"].groupby("date")[["y", "p"]].sum()
    b = (i.p / i.y - 1) * 100
    jump = (b.rolling(5).mean().shift(-4) - b.rolling(14).mean().shift(2)).dropna()
    d_j = float(jump.loc["2025-06-25"])
    tr = hi[(hi.date >= "2023-01-07") & (hi.date <= F["final"]["train_end"])]
    filled = tr.arrivals_status != "observed"
    te = hi[hi.split == "test"]; hid = te.arrivals_status != "observed"
    w0 = F["final"]["train_end"] - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    w = hi[(hi.date >= w0) & (hi.date <= F["final"]["train_end"])]
    nf = ~w.nationality.isin(set(fd.nationality_key))
    hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    dj = hd[hd.date.dt.month <= 7].groupby(hd.date.dt.year).arrivals.sum()
    mon = lambda s: s.groupby(s.date.dt.to_period("M")).arrivals.sum()
    pk = mon(hi[hi.nationality == "PAKISTAN"]); al = mon(hi)
    yoy = lambda s: (s / s.shift(12) - 1) * 100                   # monthly series without gaps (complete grid)
    per = lambda s_: float((s_.loc["2024-09":"2025-04"].sum() / s_.loc["2023-09":"2024-04"].sum() - 1) * 100)
    pk_y, al_y = yoy(pk).loc["2024-09":"2025-04"], yoy(al).loc["2024-09":"2025-04"]
    j24 = json.load(open(out / "step24_forecaster.json", encoding="utf-8")); j26 = json.load(open(out / "step26_endtoend.json", encoding="utf-8"))
    eg = pd.read_csv(out / "real_world_showcases.csv"); eg = eg[eg.country == "EGYPT"].iloc[0]
    j25 = json.load(open(out / "step25_new_route.json", encoding="utf-8"))
    return {"disruption_jump_pts": d_j, "disruption_jump_percentile": float((jump < d_j).mean() * 100),
            "largest_jump_date": str(jump.idxmax().date()), "largest_jump_pts": float(jump.max()),
            "china_checkins_per_seat": j25["ratio_range"][2] if j25["ratio_range"][3] == "CHINA" else None,
            "noflight_share_pct": float(w[nf].arrivals.sum() / w.arrivals.sum() * 100),
            "filled_rows_pct": float(filled.mean() * 100), "filled_volume_pct": float(tr[filled].arrivals.sum() / tr.arrivals.sum() * 100),
            "test_hidden_rows": int(hid.sum()), "test_hidden_volume_bound_pct": float(hid.sum() * 9 / te.arrivals.sum() * 100),
            "domestic_bias_pct": {"A": j24["bias_pct"]["A"]["domestic"], "B": j24["bias_pct"]["B"]["domestic"],
                                  "test": j26["test_window"]["bias_pct"]["domestic"]},
            "domestic_jan_jul_checkins": {int(k): int(v) for k, v in dj.items() if k <= 2025},
            "pakistan_yoy_range_pct": [float(pk_y.min()), float(pk_y.max())], "market_yoy_range_pct": [float(al_y.min()), float(al_y.max())],
            "pakistan_period_change_pct": per(pk), "market_period_change_pct": per(al),
            "egypt_rel_seats_pct": float(eg.seat_change_vs_other_countries * 100), "egypt_rel_checkins_pct": float(eg.actual_checkin_change_vs_other_countries * 100),
            "resident_evidence": _resident_evidence(cfg, hi), "closure_days": _closure_summary(cfg, fd),
            "home_holiday_checks": _home_holiday_checks(cfg, hi, out),
            **_lever_diagnostics(cfg, fd, hi, out)}


def _closure_summary(cfg, fd):
    """Airspace-closure days (seats < 5% of the country's mean) inside the airline-defaults window and the training window."""
    tr_end = cfg["folds"]["final"]["train_end"]
    cl = sa.closure_days(fd, pd.Timestamp("2023-01-07"), cfg["dates"]["test_end"]).to_frame(index=False)
    cl = cl[cl.date <= cfg["dates"]["test_end"]]
    runs = []
    for k, g in cl.sort_values("date").groupby("nationality_key"):
        r = (g.date.diff().dt.days != 1).cumsum()
        runs += [{"country": k, "start": str(x.date.min().date()), "end": str(x.date.max().date()), "days": len(x)} for _, x in g.groupby(r)]
    return {"runs": runs, "in_training_window": int((cl.date <= tr_end).sum())}


def _lever_diagnostics(cfg, fd, hi, out):
    """Audit checks of the lever: season, the 2025 'connecting passengers' explanation, functional form on real cuts."""
    m, _, _ = _panel_with_festivals(cfg, fd, hi)
    p, _ = _primary(m, cfg["folds"]["final"]["train_end"])
    w = p.assign(d_seats_winter=p.d_seats * p.month.dt.month.isin([10, 11, 12, 1, 2, 3]))
    se = sa.fe_elasticity(w, "d_checkins", ["d_seats", "d_seats_winter"] + FEST, n_boot=0)
    q = p[p.month >= "2025-01-01"]
    sh = sa.fe_elasticity(q.assign(s=q.d_p2p - q.d_pax), "s", ["d_seats"], n_boot=0)
    j23 = json.load(open(out / "step23_lever_test.json", encoding="utf-8")); eps = j23["decision"]["default_eps_total"]
    lo, hi_ = j23["trusted_range_final"]
    sc = pd.read_csv(out / "real_world_showcases.csv")
    rows = []
    for r in sc.itertuples():
        sch = r.seat_change_vs_other_countries
        rows.append({"event": f"{r.country.title()} {r.months}", "seat_change": sch,
                     "actual_without_festivals": (1 + r.actual_checkin_change_vs_other_countries) / (1 + r.festival_part) - 1,
                     "log_form": (1 + sch) ** eps - 1, "straight_line": eps * sch,
                     "adopted": float(sa.seat_lever(1.0, 1.0, sch, eps, 10.0, (lo, hi_))[0])})
    return {"summer_eps": se["estimate"], "summer_eps_ci": se["ci_cluster"],
            "winter_minus_summer": se["other_coefs"]["d_seats_winter"], "winter_minus_summer_ci": se["other_ci_cluster"]["d_seats_winter"],
            "local_share_response_2025": sh["estimate"], "local_share_response_2025_ci": sh["ci_cluster"],
            "lever_shape_evidence": rows, "full_cancellation_keeps": (1 - eps) * (1 + lo) ** eps}


RESIDENT_TEST = ["DOMESTIC", "PHILIPPINES", "INDIA", "PAKISTAN", "EGYPT", "LEBANON", "SAUDI ARABIA", "UNITED KINGDOM", "GERMANY"]


def _resident_evidence(cfg, hi):
    """Who is 'domestic'? Check-ins on UAE break days ÷ the same weekdays 8–28 days either side (no breaks, no Ramadan),
    training window only. National Day is a UAE-residents-only holiday; Eid is Gulf-wide. If UAE-resident expats were
    recorded under their nationality, expat series would rise on National Day; if they are in DOMESTIC, only it rises."""
    from . import forecaster as fc
    P = cfg["paths"]["processed"]; cal = pd.read_parquet(P / "calendar.parquet")
    b = fc.uae_breaks(cal).merge(cal[["date", "is_ramadan"]], on="date").set_index("date")
    hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    y = pd.concat([hi[["nationality", "date", "arrivals"]], hd[["date", "arrivals"]].assign(nationality="DOMESTIC")])
    y = y[(y.date >= "2023-01-07") & (y.date <= cfg["folds"]["final"]["train_end"])]
    Y = y.pivot_table(index="date", columns="nationality", values="arrivals", aggfunc="sum")[RESIDENT_TEST]
    b = b.reindex(Y.index); res = {}
    for flag in ("nd_break", "eid_break"):
        on = b[flag] == 1; run = (on != on.shift()).cumsum().where(on).dropna(); num = den = 0
        for _, g in run.groupby(run):
            d = g.index
            c = pd.date_range(d.min() - pd.Timedelta(days=28), d.max() + pd.Timedelta(days=28))
            c = c[((c < d.min() - pd.Timedelta(days=7)) | (c > d.max() + pd.Timedelta(days=7))) & c.isin(Y.index)]
            c = c[(b.loc[c, "all_break"] == 0).to_numpy() & ~b.loc[c, "is_ramadan"].astype(bool).to_numpy() & c.dayofweek.isin(d.dayofweek)]
            num = num + Y.loc[d].mean(); den = den + Y.loc[c].mean()
        res[flag] = {"breaks": int(run.nunique()), "ratio": {k: float(v) for k, v in (num / den).items()}}
    w0 = cfg["folds"]["final"]["train_end"] - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    w = lambda s: s[(s.date >= w0) & (s.date <= cfg["folds"]["final"]["train_end"])].arrivals.sum()
    res["domestic_share_of_checkins_12m_pct"] = float(w(hd) / (w(hd) + w(hi)) * 100)
    return res


def _home_holiday_checks(cfg, hi, out):
    """(1) Stability: each home holiday's uplift per year, training window only (holiday days ÷ the same weekdays 8–28 days
    either side, no UAE breaks, no Ramadan). (2) Exposure of the current forecaster (v2) vs the scored v1, from the two
    forecasts alone (no test-window actuals): total difference, the largest possible change in the daily × nationality
    score (triangle inequality), and home-holiday days that fall inside Ramadan (a combination training never saw)."""
    from . import forecaster as fc
    P, te = cfg["paths"]["processed"], cfg["folds"]["final"]["train_end"]
    cal = pd.read_parquet(P / "calendar.parquet")
    b = fc.uae_breaks(cal).merge(cal[["date", "is_ramadan"]], on="date").set_index("date")
    Y = hi.pivot_table(index="date", columns="nationality", values="arrivals", aggfunc="sum")
    hh = pd.read_csv(cfg["paths"]["reference"] / "home_holidays.csv", parse_dates=["start", "end"])
    stab = []
    for r in hh.itertuples():
        d = pd.date_range(r.start - pd.Timedelta(days=1), r.end)
        if d.max() > te or d.min() < pd.Timestamp("2023-01-07"):
            continue
        c = pd.date_range(d.min() - pd.Timedelta(days=28), d.max() + pd.Timedelta(days=28))
        c = c[((c < d.min() - pd.Timedelta(days=7)) | (c > d.max() + pd.Timedelta(days=7))) & c.isin(Y.index) & (c <= te)]
        c = c[(b.loc[c, "all_break"] == 0).to_numpy() & ~b.loc[c, "is_ramadan"].astype(bool).to_numpy() & c.dayofweek.isin(d.dayofweek)]
        stab.append({"nationality": r.nationality, "holiday": r.holiday, "year": int(r.start.year),
                     "uplift": float(Y.loc[d, r.nationality].mean() / Y.loc[c, r.nationality].mean())})
    ref = json.load(open(cfg["paths"]["reference"] / "frozen_test_forecast.json", encoding="utf-8"))
    v1 = pd.read_csv(cfg["paths"]["reference"].parent.parent / ref["file"], parse_dates=["date"])
    v2 = pd.read_csv(out / "stagea_test_forecast.csv", parse_dates=["date"])
    m = v1.merge(v2, on=["series", "date"], suffixes=("_v1", "_v2")); m = m[m.series != "DOMESTIC"]
    dv = m.checkins_forecast_v2 - m.checkins_forecast_v1
    hd = fc.home_holiday_daily(hh); rd = set(b.index[b.is_ramadan.astype(bool)])
    inr = hd[hd.date.isin(rd) & hd.date.between(cfg["dates"]["test_start"], cfg["dates"]["test_end"])]
    x = m.merge(inr.rename(columns={"nationality": "series"}), on=["series", "date"])
    by = dv.abs().groupby(m.series).sum()
    return {"stability": stab, "stability_share_above_1": float(np.mean([s["uplift"] > 1 for s in stab])), "n_occurrences": len(stab),
            "v2_vs_v1_intl_total_pct": float((m.checkins_forecast_v2.sum() / m.checkins_forecast_v1.sum() - 1) * 100),
            "max_change_daily_nat_pts": float(dv.abs().sum() / m.checkins_forecast_v1.sum() * 100),
            "largest_share_of_difference": [str(by.idxmax()), float(by.max() / by.sum() * 100)],
            "ramadan_overlap": {"markets": sorted(set(x.series)), "days": int(len(x)),
                                "dates": [str(d.date()) for d in sorted(set(x.date))],
                                "extra_checkins_v2_vs_v1": float((x.checkins_forecast_v2 - x.checkins_forecast_v1).sum()),
                                "share_of_intl_checkins_pct": float((x.checkins_forecast_v2 - x.checkins_forecast_v1).sum() / m.checkins_forecast_v1.sum() * 100)}}


def _cancellation_keeps(cfg, out, eps, trusted):
    """Share of check-ins a market keeps if ALL its Abu Dhabi seats are cancelled (seat lever with the cap)."""
    g = pd.read_csv(out / "guests_per_seat.csv"); g = g[g.has_flight_data & (g.checkins_per_seat > 0)]
    A = g.checkins_12m.to_numpy(); S = A / g.checkins_per_seat.to_numpy()
    dA, _ = sa.seat_lever(A, S, -S, eps, g.local_pax_per_seat.to_numpy(), tuple(trusted))
    return pd.Series(1 + dA / A, index=g.nationality.to_numpy()).sort_values()


def _ordinal(x) -> str:
    n = int(round(x)); suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _pretty(n: str) -> str:
    return {"UNITED STATES OF AMERICA": "United States", "RUSSIAN FEDERATION": "Russia"}.get(n, n.title())


def step_report():
    cfg = load_config(); out = _out(cfg); fd, fm, hi = _load(cfg)
    J = {k: json.load(open(out / f, encoding="utf-8")) for k, f in (("21", "step21_airline.json"), ("22", "step22_checkins.json"), ("23", "step23_lever_test.json"),
                                                 ("24", "step24_forecaster.json"), ("25", "step25_new_route.json"), ("26", "step26_endtoend.json"),
                                                 ("27", "step27_guests_per_seat.json"))}
    L_ = _limitations(cfg, fd, hi, out); json.dump(L_, open(out / "step27_limitations.json", "w", encoding="utf-8"), indent=1, default=float)
    v1 = pd.read_csv(cfg["paths"]["processed"] / "phase1" / "validation_metrics.csv", index_col=[0, 1])
    m1 = json.load(open(cfg["paths"]["processed"] / "phase1" / "final_summary.json", encoding="utf-8"))["selection"]["intl_model"]
    gps = pd.read_csv(out / "guests_per_seat.csv")
    e21, e22, e23, e24, e25, e26, e27 = (J[k] for k in ("21", "22", "23", "24", "25", "26", "27"))
    fin = e22["folds"]["final"]; d23 = e23["decision"]; s26 = e26["test_window"]; lv = s26["lever_check"]
    eps_iv = sa.lever_interval(e26); nc = e25["new_city_test"]["trusted range (primary)"]; na = e25["new_city_test"]["all rows"]
    f1 = lambda x: f"{x:.1f}"; pct = lambda x: f"{x:+.0f}%"
    L = ["# Phase 2 report — Stage A: flights → hotel check-ins", "",
         "_Generated by `make phase2` from the saved outputs in `data/processed/phase2/`; do not edit by hand._", "",
         "## What Stage A does", "",
         "It turns flight supply into hotel check-ins by nationality. A **seat lever** (ε) says how check-ins respond when seats from "
         "a country change; a **daily forecaster** gives the baseline check-ins from the calendar, festivals, events and seats; Stage B "
         "(Phase 1) then turns check-ins into guests in-house.", "",
         "## Results by step", "",
         f"**2.1 Airline defaults and ε₁.** Defaults table for 33 countries × 12 months (latest 12 months of flights; closure days excluded "
         f"so a war month is not the 'normal' schedule: "
         + ", ".join(f"{_pretty(r['country'])} {r['start'][5:]}…{r['end'][5:]} {r['start'][:4]}" for r in L_["closure_days"]["runs"])
         + f"). Local passengers "
         f"move about one-for-one with seats: ε₁ = {fin['eps1']['estimate']:.2f} [{fin['eps1']['ci_boot'][0]:.2f}, {fin['eps1']['ci_boot'][1]:.2f}] "
         f"(on the same rows as the default ε_total).", "",
         f"**2.2 Check-ins vs seats (ε_total).** Year-on-year changes with month effects, rows inside the trusted range of seat changes "
         f"({e23['trusted_range_final'][0] * 100:+.0f}% … {e23['trusted_range_final'][1] * 100:+.0f}%). Default ε = {fin['eps_total_default']['estimate']:.2f} "
         f"(lead-controlled), upper {fin['eps_total_plain']['estimate']:.2f} (plain); ε₂ = ε_total/ε₁ = {fin['eps2']['estimate']:.2f}. "
         f"No tourist/resident split (difference interval includes 0).", "",
         f"**2.3 Out-of-sample lever test.** The pre-registered test FAILED for the older estimate (fold B ε {e23['primary']['eps_used']:.2f} → "
         f"slope {e23['primary']['calibration_slope']['estimate']:.2f} [{e23['primary']['calibration_slope']['ci_boot'][0]:.2f}, "
         f"{e23['primary']['calibration_slope']['ci_boot'][1]:.2f}]): the link weakened in 2025 (ε_total by half-year "
         + ", ".join(f"{k} {v['eps_total']:.2f}" for k, v in e23["drift_by_half_year"].items())
         + f"). Decision: default {d23['default_eps_total']:.2f}, provisional band {d23['simulator_band'][0]:.2f}–{d23['simulator_band'][1]:.2f}, replaced from "
         f"Step 2.7 on (and in all of Phase 3) by ε's 80% interval from the unseen test window, {eps_iv[0]:.2f}–{eps_iv[1]:.2f} (2.6). "
         f"Main moving festivals (Ramadan/Eids, Easter, Lunar New Year, Diwali) move check-ins strongly but do not distort ε "
         f"(shift < 1 SE); fixed-date holidays cancel in year-on-year data. _Audit caveat:_ 0.31 and 0.44 were estimated on data that include "
         f"most of these held-out months, so their scores here are partly in-sample, and the band's low end (0.25) was picked on these months; "
         f"the genuinely out-of-sample confirmation is the official test window (2.6). Why the link weakened is not established: the local-share "
         f"response turned negative in 2025 ({L_['local_share_response_2025']:+.2f} [{L_['local_share_response_2025_ci'][0]:+.2f}, "
         f"{L_['local_share_response_2025_ci'][1]:+.2f}]) but not significantly.", "",
         f"**2.4 Daily forecaster.** Per series Poisson: calendar + main UAE breaks + own festival + year-end + Grand Prix/ADIPEC + recent "
         f"level + ε·log seats" + (" + home-country holidays (Golden Week, Chuseok, Gulf national days)" if "C6" in e24["chosen_spec"] else "")
         + (" + UAE school breaks (domestic only" + ("" if "C7ws" in e24["chosen_spec"] else "; winter, spring and summer") + ")" if "C7" in e24["chosen_spec"] else "")
         + f". Level terms judged on monthly error, timing terms on daily error. Held-out international "
         f"daily×nationality WMAPE {f1(e24['chosen_metrics']['A']['intl_daily_nat'])}% (A) / {f1(e24['chosen_metrics']['B']['intl_daily_nat'])}% (B), "
         f"monthly total {f1(e24['chosen_metrics']['A']['intl_monthly_total'])}% / {f1(e24['chosen_metrics']['B']['intl_monthly_total'])}%, bias "
         f"{pct(e24['bias_pct']['A']['intl'])} / {pct(e24['bias_pct']['B']['intl'])}. Growth extrapolation and all-holidays variants rejected"
         + f"; splitting the main break into Eid and National Day {'kept' if e24['split_eid_national_day']['kept'] else 'not kept'} "
         f"(A {e24['split_eid_national_day']['fold_A']:.2f}, B {e24['split_eid_national_day']['fold_B']:.2f}; judged on B as pre-registered).", "",
         f"**2.5 New routes.** Markets without seat data: {e25['checkins_per_added_seat']:.3f} check-ins per added seat "
         f"(band {e25['band'][0]:.3f}–{e25['band'][1]:.3f}) = ε × median check-ins per seat ({e25['ratio_median']:.3f}); leave-one-out error "
         f"×{e25['loo']['overall median']['median_error_factor']:.2f} typical. New city routes respond like added capacity "
         f"(difference {nc['other_coefs']['g_new']:+.2f} [{nc['other_ci_cluster']['g_new'][0]:+.2f}, {nc['other_ci_cluster']['g_new'][1]:+.2f}]). "
         f"Seats lost when a city route closes looked stronger in the trusted-range rows ({nc['other_coefs']['g_closed']:+.2f} "
         f"[{nc['other_ci_cluster']['g_closed'][0]:+.2f}, {nc['other_ci_cluster']['g_closed'][1]:+.2f}]) but not on all rows "
         f"({na['other_coefs']['g_closed']:+.2f} [{na['other_ci_cluster']['g_closed'][0]:+.2f}, {na['other_ci_cluster']['g_closed'][1]:+.2f}]): not robust, so one "
         "lever is kept; Phase 3's replay of the Wizz Air exit (Step 3.7) points the same way and its range for deep cuts covers it.", "",
         "**2.6 End-to-end and the official test window.**", "",
         "| | Intl daily × nat | Intl monthly total | Domestic daily |", "|---|---|---|---|"]
    for f in ("A", "B"):
        for mdl, lab in (("end_to_end", "end-to-end (flights → guests)"), ("phase1_actual_checkins", "Phase 1 (actual check-ins)"),
                         ("same_day_last_year_guests", "same day last year")):
            m = e26["folds"][f][mdl]
            L.append(f"| Fold {f}, {lab} | {f1(m['intl_daily_x_nationality'])}% | {f1(m['intl_monthly_total'])}% | {f1(m['domestic_daily'])}% |")
    sm = s26["stagea_scores"]["model"]
    gp = {f: e26["folds"][f]["gap_vs_same_day_last_year_daily_nat"] for f in ("A", "B")}
    L += ["", "At daily × nationality the gain over same-day-last-year is a statistical tie (market bootstrap: "
          + "; ".join(f"fold {f} {g['points']:+.2f} pts [{g['ci95_market_bootstrap'][0]:+.1f}, {g['ci95_market_bootstrap'][1]:+.1f}]" for f, g in gp.items())
          + "); the gain is at totals (daily and monthly)."]
    L += ["", f"Official test window (scored once, on 2 Oct 2026, with forecaster v1 — archived as "
          f"`data/reference/stagea_test_forecast_v1_scored.csv`, fingerprint verified on every run; later forecaster versions are "
          f"chosen on folds A/B only and are never scored on the test window): Stage A check-ins daily×nationality "
          f"{f1(sm['intl_daily_nat'])}%, monthly total {f1(sm['intl_monthly_total'])}%, bias {pct(s26['bias_pct']['intl'])}; beats all benchmarks "
          f"on these two pre-registered measures: {e26['criteria']['c_test_stagea_beats_benchmarks']}. "
          f"UAE residents are the exception: daily {f1(sm['dom_daily'])}% and monthly {f1(sm['dom_monthly'])}% against "
          f"{f1(s26['stagea_scores']['benchmark: same day last year']['dom_daily'])}% and "
          f"{f1(s26['stagea_scores']['benchmark: same day last year']['dom_monthly'])}% for the same day last year (bias "
          f"{pct(s26['bias_pct']['domestic'])}), so residents are forecast worse than repeating last year on this window. "
          f"Lever check on unseen months: slope {lv['calibration_slope']['estimate']:.2f} "
          f"[{lv['calibration_slope']['ci_boot'][0]:.2f}, {lv['calibration_slope']['ci_boot'][1]:.2f}], skill {lv['skill']['default 0.31'] * 100:+.1f}% "
          f"(naive one-for-one {lv['skill']['naive one-for-one (1.0)'] * 100:+.0f}%) → passed: {lv['passed']}.", "",
          f"**2.7 Scenario levers.** Inside the trusted range a seat change r gives ΔA = A·[(1+r)^ε − 1], the log form ε was estimated "
          f"and validated in; outside it the effect continues in a straight line from the edge (full cancellation keeps "
          f"{L_['full_cancellation_keeps'] * 100:.0f}% of check-ins); never more than the local passengers the seats carry. "
          f"(Audit fix: the first draft used a straight line everywhere, which overstated a +109% increase by about a third.) "
          f"Real seat cuts outside the range, with festival effects removed:", "",
          "| Event | Seats vs other countries | Actual check-ins | Log form | Straight line | Adopted rule |", "|---|---|---|---|---|---|"]
    for r in L_["lever_shape_evidence"]:
        L.append(f"| {r['event']} | {r['seat_change'] * 100:+.0f}% | {r['actual_without_festivals'] * 100:+.0f}% | {r['log_form'] * 100:+.0f}% | "
                 f"{r['straight_line'] * 100:+.0f}% | {r['adopted'] * 100:+.0f}% |")
    L += ["", "Israel's months are war-driven demand collapses and Egypt's a currency shock: no seat rule can match them.", "",
          f"**2.7 Guests per seat.** Pooled {e27['pooled_guest_nights_per_seat']:.2f} guest nights per seat. Each 1,000 added seats ≈ "
          f"{e27['median_marginal_per_1000']:.0f} extra guest nights for the median market (band {e27['median_band_per_1000'][0]:.0f}–"
          f"{e27['median_band_per_1000'][1]:.0f}; bands use ε {eps_iv[0]:.2f}–{eps_iv[1]:.2f}). Cap (extra local passengers) binds for: {', '.join(_pretty(x) for x in e27['cap_binds_for']) or 'none'}.", "",
          "| Market | Stay (nights) | Check-ins per seat | Extra guest nights per 1,000 added seats (band) | One extra daily flight → guest nights / month |",
          "|---|---|---|---|---|"]
    r3 = lambda x: f"{round(x, 2 - int(np.floor(np.log10(abs(x))))):,.0f}" if x else "0"      # model estimates: 3 significant figures
    for r in gps[gps.has_flight_data].sort_values("checkins_12m", ascending=False).head(12).itertuples():
        le = "≤ " if r.cap_binds else ""
        band = "cap" if r.cap_binds else f"{r3(r.band_low)}–{r3(r.band_high)}"
        L.append(f"| {_pretty(r.nationality)} | {r.stay_nights:.1f} | {r.checkins_per_seat:.2f} | {le}{r3(r.marginal_guest_nights_per_1000_added_seats)} "
                 f"({band}) | {le}{r3(r.one_extra_daily_flight_guest_nights_per_month)} |")
    L += ["", "“≤” = the cap binds: the effect is an upper bound (every extra local passenger becomes a hotel guest). Markets with high "
          "check-ins per seat (China, Russia) receive many guests through other gateways, so added Abu Dhabi seats may partly shift the "
          "gateway rather than add guests; their absolute effects are the least certain."]
    # ---------------- nationality vs departure country (external-factor audit, 2 Oct 2026)
    R = L_["resident_evidence"]; nd, ed = R["nd_break"]["ratio"], R["eid_break"]["ratio"]
    sd = e24["split_eid_national_day"]["on_target_days"]
    keeps = _cancellation_keeps(cfg, out, d23["default_eps_total"], e23["trusted_range_final"])
    capk = keeps[keeps > L_["full_cancellation_keeps"] + 0.005].sort_values(ascending=False)
    expat = ["PHILIPPINES", "INDIA", "PAKISTAN", "EGYPT"]
    L += ["", "## Nationality vs departure country", "",
          "Hotel check-ins are counted by guest **nationality**; flights by **departure country** (Abu Dhabi airport only). They are not "
          "the same people. A natural experiment shows who is counted where: National Day is a holiday for UAE residents only, Eid is "
          "a holiday across the Gulf.", "",
          "| Series | National Day break | Eid breaks |", "|---|---|---|"]
    for k in RESIDENT_TEST:
        L.append(f"| {'Domestic' if k == 'DOMESTIC' else _pretty(k)} | ×{nd[k]:.2f} | ×{ed[k]:.2f} |")
    L += ["", f"_Check-ins on break days ÷ the same weekdays 8–28 days either side (no breaks, no Ramadan); {R['nd_break']['breaks']} National "
          f"Day and {R['eid_break']['breaks']} Eid breaks in the training window._", "",
          f"- **Domestic = UAE residents of every nationality.** On National Day only Domestic rises (×{nd['DOMESTIC']:.2f}); the expat "
          f"nationalities fall ({', '.join(f'{_pretty(k)} ×{nd[k]:.2f}' for k in expat)}), so UAE-resident expats are not counted under "
          f"their passport. Domestic is {R['domestic_share_of_checkins_12m_pct']:.0f}% of check-ins (last 12 training months) and no "
          f"international seat lever moves it.",
          f"- **Gulf-based expats.** On Eid the same nationalities surge ({', '.join(f'{_pretty(k)} ×{ed[k]:.2f}' for k in expat)}; "
          f"Saudi Arabia ×{ed['SAUDI ARABIA']:.2f}) while Western ones do not (United Kingdom ×{ed['UNITED KINGDOM']:.2f}, Germany "
          f"×{ed['GERMANY']:.2f}). The most likely visitors are those nationalities living elsewhere in the Gulf, arriving by road or on Gulf "
          f"routes, not from their home country: another reason a home-route cancellation does not remove all of a nationality's check-ins.",
          f"- **What the seat lever means.** ε links a country's seats to the matching nationality's check-ins, so it already contains the "
          f"mix of passports on board and guests arriving by other routes: a market response, not a passenger count. A full cancellation "
          f"keeps {L_['full_cancellation_keeps'] * 100:.0f}% of check-ins; where the local-passenger cap binds, more ("
          + (", ".join(f"{_pretty(k)} {v * 100:.0f}%" for k, v in capk.items()) or "none") + ").",
          f"- **Calendar.** Main fixed-date home holidays now enter the forecaster for the markets where they move check-ins "
          f"(screened before fold A; China and Japan Golden Week, Russia's May holidays, Korea's Chuseok, Saudi, Kuwaiti and Bahraini national "
          f"days); UAE school breaks move Domestic only. One main-break effect per series still covers Eid and National Day together, a "
          f"trade-off: for the four expat series above it over-forecasts National Day ({sd['one_main_break']['expat_bias_national_day_days_A']:+.0f}% "
          f"on fold A's {sd['national_day_days_A']} days; split {sd['split']['expat_bias_national_day_days_A']:+.0f}%), while the split "
          f"over-forecasts the weaker 2025 Eids ({sd['split']['expat_bias_eid_days_B']:+.0f}% on fold B's {sd['eid_days_B']} Eid days vs "
          f"{sd['one_main_break']['expat_bias_eid_days_B']:+.0f}%) and is no better on monthly error (A {sd['split']['intl_monthly_nat']['A']:.2f} vs "
          f"{sd['one_main_break']['intl_monthly_nat']['A']:.2f}). Not kept, as pre-registered; Phase 3 daily views for these markets on "
          f"National Day should be read with this in mind.",
          "- **Not in the data:** passport mix by route, guests arriving through Dubai airport, residents vs visitors inside each nationality. "
          "Cross-route spill-overs are not estimated (far more unknowns than months).", ""]
    H = L_["home_holiday_checks"]; st = pd.DataFrame(H["stability"])
    yrs = sorted(st.year.unique())
    fp = pd.read_csv(out / "stagea_final_params.csv"); fc_mult = fp[fp.term == "home_holiday"].set_index("series").multiplier
    L += [f"**Are the home-holiday effects stable?** Uplift per year before the test window (holiday days ÷ the same weekdays 8–28 days "
          f"either side): above ×1 in {H['stability_share_above_1'] * 100:.0f}% of {H['n_occurrences']} occurrences. The forecaster's "
          f"multiplier pools the years, so a noisy market (Japan) gets a small one.", "",
          "| Market | Holiday | " + " | ".join(str(y) for y in yrs) + " | Forecaster multiplier |", "|---|---|" + "---|" * (len(yrs) + 1)]
    for (n, h), g in st.groupby(["nationality", "holiday"]):
        v = g.set_index("year").uplift
        L.append(f"| {_pretty(n)} | {h.replace('_', ' ')} | " + " | ".join(f"×{v[y]:.2f}" if y in v else "–" for y in yrs)
                 + f" | ×{fc_mult.get(n, float('nan')):.2f} |")
    L += ["", "Questions for DCT: (1) is Domestic = all UAE residents (as the data imply)? (2) is nationality-by-route or Dubai-airport arrival "
          "data available? Either would sharpen the lever for markets with many indirect guests."]
    L += ["", "## Limitations (verified in the data)", "",
          f"- **Airspace disruptions.** After 24 Jun 2025 the engine over-predicted guests for ~5 days (error jump {L_['disruption_jump_pts']:+.1f} pts, "
          f"{_ordinal(L_['disruption_jump_percentile'])} percentile of fold-B windows). Larger: the checkout wave after holiday breaks "
          f"({L_['largest_jump_date']}, {L_['largest_jump_pts']:+.1f} pts). From flights alone, Stage A cannot foresee the check-in spike of a disruption day.",
          f"- **Other gateways.** Flights cover Abu Dhabi airport only; China has {L_['china_checkins_per_seat']:.2f} check-ins per seat, so many guests arrive "
          f"by other routes. The 12 nationalities without flight data ({L_['noflight_share_pct']:.1f}% of check-ins) use calendar and recent level only.",
          f"- **Non-aviation shocks.** Seats are necessary but not sufficient. Pakistan (no flight data in the pack): check-ins Sep 2024–Apr 2025 "
          f"{L_['pakistan_period_change_pct']:+.0f}% vs a year earlier (worst month {L_['pakistan_yoy_range_pct'][0]:+.0f}%) while all international check-ins "
          f"moved {L_['market_period_change_pct']:+.0f}%. Reports of 28 Nov 2025 describe 70–80% rejections of first-time and single-entry Dubai visa "
          f"applications from Pakistan, and a Senate committee was told in January 2025 that some UAE visas were \"unofficially closed\" [1][2]; the drop in our data starts earlier (Sep 2024), so the cause is consistent but not proven; inflation does not explain it "
          f"(4.9% in Nov 2024, a 6.5-year low [3]). Egypt: seats {L_['egypt_rel_seats_pct']:+.0f}% vs other countries while check-ins moved "
          f"{L_['egypt_rel_checkins_pct']:+.0f}% (Mar–Jul 2024), after the Mar 2024 currency float (association, not proof).",
          f"- **Domestic level.** Bias {pct(L_['domestic_bias_pct']['A'])} (fold A), {pct(L_['domestic_bias_pct']['B'])} (fold B), "
          f"{pct(L_['domestic_bias_pct']['test'])} (test): the sign is not stable. Jan–Jul domestic check-ins: "
          + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in L_["domestic_jan_jul_checkins"].items()) + ".",
          f"- **Small markets.** Suppression is negligible for fitting ({L_['filled_rows_pct']:.1f}% of training rows, {L_['filled_volume_pct']:.3f}% of volume; "
          f"test hidden check-ins ≤ {L_['test_hidden_volume_bound_pct']:.2f}% of volume); tiny markets are simply noisy.",
          f"- **Lever drift.** The seat link weakened in 2025 (cause not established); confirmed back near 0.31 on Aug 2025–Feb 2026.",
          f"- **Seats partly follow demand.** Seats two months ahead predict today's check-ins (coefficient "
          f"{fin['eps_total_lead']['other_coefs']['d_seats_lead2']:+.2f} [{fin['eps_total_lead']['other_ci_cluster']['d_seats_lead2'][0]:+.2f}, "
          f"{fin['eps_total_lead']['other_ci_cluster']['d_seats_lead2'][1]:+.2f}]): airlines add capacity where demand is rising. The lead "
          f"control lowers ε for this reason, and the lever is validated for prediction on unseen months, but it is not a proven causal effect "
          f"and may still overstate what a pure capacity decision achieves.",
          f"- **Season.** ε may be lower in winter (Oct–Mar) than in summer: difference {L_['winter_minus_summer']:+.2f} "
          f"[{L_['winter_minus_summer_ci'][0]:+.2f}, {L_['winter_minus_summer_ci'][1]:+.2f}], not significant; one ε is used all year.",
          "- **Sustained changes only.** ε comes from monthly year-on-year changes, so it describes schedule changes that last weeks or months, "
          "not a single extra flight on one day.",
          f"- **Forecaster versions.** The test window was scored once with v1; the current v2 (home holidays, domestic school breaks) is "
          f"validated on folds A/B only. From the two forecasts alone (no test actuals): international totals differ by "
          f"{L_['home_holiday_checks']['v2_vs_v1_intl_total_pct']:+.2f}%, and v2's daily×nationality test score could differ from v1's by at most "
          f"±{L_['home_holiday_checks']['max_change_daily_nat_pts']:.1f} points ({_pretty(L_['home_holiday_checks']['largest_share_of_difference'][0])} "
          f"is {L_['home_holiday_checks']['largest_share_of_difference'][1]:.0f}% of the difference). One 7-month window cannot rank two models "
          f"this close, which is why v2 is never scored there.",
          f"- **Holidays inside Ramadan.** In 2026 Ramadan covers the home holidays of "
          + " and ".join(_pretty(k) for k in L_["home_holiday_checks"]["ramadan_overlap"]["markets"])
          + f" ({', '.join(str(int(d[8:])) for d in L_['home_holiday_checks']['ramadan_overlap']['dates'])} Feb 2026), a combination "
          f"training never saw. The model multiplies the two effects (holiday days above the surrounding Ramadan days, below pre-Ramadan "
          f"normal); whether a holiday lifts travel during Ramadan is untested. Exposure: "
          f"{L_['home_holiday_checks']['ramadan_overlap']['extra_checkins_v2_vs_v1']:+,.0f} check-ins "
          f"({L_['home_holiday_checks']['ramadan_overlap']['share_of_intl_checkins_pct']:+.3f}% of the window).",
          "- **Prices are not modelled.** No hotel rates (ADR) or airfares are in the pack; demand reacting to price is outside the model.",
          "- **Realised seats.** The test-window forecast used the seats actually flown (including cancellations on 28 Feb 2026); a forecast "
          "made in advance from published schedules would be slightly less accurate.",
          "- **After the data ends.** Airline defaults describe the schedule up to 28 Feb 2026, before the 2026 regional airspace disruption; "
          "scenarios for March–May 2026 set that disruption explicitly (Phase 3's March 2026 replay presets).",
          "- **Bands differ in meaning.** ε interval (plausible levers, out of sample), new-route band (80% leave-one-out), Stage B bands "
          "(calibrated ~80% coverage): Phase 3 labels them (effect range vs level range) and never mixes them.", "",
          "Sources: [1] [Dawn, 28 Nov 2025](https://www.dawn.com/news/1957715) · [2] [The New Arab, 28 Nov 2025](https://www.newarab.com/news/uae-quietly-blocking-visas-pakistani-citizens) "
          "· [3] [Express Tribune, Dec 2024](https://tribune.com.pk/story/2513267/november-cpi-inflation-drops-to-49-lowest-in-65-years)", "",
          "## Independent audit (2 Oct 2026)", "",
          "A separate reviewer re-read all Phase 2 code. No leakage: corrupting every check-in and seat after a fold's cut-off leaves its ε, "
          "trusted range and forecasts unchanged (now a permanent test). Fixed: seat-lever shape (above); weighted fixed-effects now use weighted "
          "month means (robustness rows only); ε₁ for ε₂ now on the same rows as ε_total; a latent bug that would have dropped a growth term "
          "from the final forecast; the scored test forecast's fingerprint is stored in `data/reference/frozen_test_forecast.json` so a changed "
          "forecaster can never be re-scored on the test window; weak tests strengthened.", "",
          "**External-factor audit (2 Oct 2026).** Every external input was re-checked against its source: nationality ↔ country mapping "
          "(45 nationalities, complete), festival groups (Israel's note corrected), official UAE holidays, school breaks, F1/ADIPEC, and "
          "home-country holiday dates (State Council and national calendars). Changes: closure days removed from airline defaults and from "
          "the per-seat ratios (2.5, 2.7); home holidays and domestic-only school breaks tested as forecaster terms under the same rule "
          f"(kept: {', '.join(c for c in ('C6', 'C7') if c in e24['chosen_spec']) or 'none'}); Eid/National Day split tested; the scored "
          "test forecast archived as v1 (the forecaster now in use is v2, validated on folds A/B only). ε and the lever are unchanged "
          "(estimated from monthly year-on-year changes, where fixed-date holidays cancel).", "",
          "## Carried forward", "",
          "- Phase 3: seat lever (`stagea.seat_lever`, log form inside the trusted range, straight line outside, capped), new-route lever (`stagea.new_route_lever`), trusted-range flag, ε band; "
          "a manual external-demand lever per nationality (visa, currency, other-gateway capacity).",
          "- Phase 5 audit (pre-registered rule): post-break checkout term in Stage B; recency weighting of ε.", "",
          "## Files (`data/processed/phase2/`)", "",
          "`airline_defaults.csv`, `elasticities.csv`, `step21…step27*.json`, `lever_test_rows.csv`, `real_world_showcases.csv`, "
          "`main_festival_days.csv`, `stagea_validation_metrics.csv`, `stagea_per_market.csv`, `stagea_final_params.csv`, `stagea_pred_{A,B}.parquet`, "
          "`stagea_test_forecast.csv` (current forecaster v2, unscored), `new_route_ratios.csv`, `endtoend_*`, `test_window_stagea_scores.csv`, "
          "`guests_per_seat.csv`. Reference (`data/reference/`): `home_holidays.csv`, `festival_groups.csv`, `source_market_festivals.csv`, "
          "`stagea_test_forecast_v1_scored.csv` + `frozen_test_forecast.json` (the scored forecast and its fingerprint). "
          "Reproduce: `make phase2` then `make test`.", ""]
    # ---------------- draft summary analysis (≤ 1,000 characters, numbers from the outputs)
    summ = (f"Strengths: two-stage, auditable simulator. Stage A turns seat changes into hotel check-ins with a lever estimated across "
            f"33 source countries (+10% seats ≈ +{d23['default_eps_total'] * 10:.0f}% check-ins), confirmed on unseen Aug 2025–Feb 2026 months "
            f"(calibration slope {lv['calibration_slope']['estimate']:.1f}); naive one-for-one scaling is rejected. Stage B converts check-ins into guests "
            f"in-house: {v1.loc[('A', m1), 'intl_daily_x_nationality']:.1f}% daily-by-nationality and "
            f"{v1.loc[('A', m1), 'intl_monthly_total']:.1f}% monthly WMAPE; from flights alone, monthly guest nights within "
            f"{e26['folds']['A']['end_to_end']['intl_monthly_total']:.0f}–{e26['folds']['B']['end_to_end']['intl_monthly_total']:.0f}%. Calendar covers "
            f"Ramadan, Eids, source-market festivals" + (", home holidays (Golden Week)" if "C6" in e24["chosen_spec"] else "")
            + (", school breaks" if "C7" in e24["chosen_spec"] else "") + f", F1 and ADIPEC. Limitations: flights cover Abu Dhabi airport only, so guests via other gateways "
            f"and 12 nationalities without flight data rely on calendar trends; visa or currency shocks (Pakistan, Egypt) break the seat link; "
            f"daily nationality forecasts without check-ins stay noisy (~{e26['folds']['A']['end_to_end']['intl_daily_x_nationality']:.0f}%); "
            f"stays cut short by airspace disruptions and post-holiday checkout waves are not modelled separately.")
    assert len(summ) <= 1000, len(summ)
    L += ["## Draft summary analysis (≤ 1,000 characters)", "", f"{summ}", "", f"_{len(summ)} characters._", ""]
    (cfg["paths"]["reports"] / "phase2_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote reports/phase2_report.md ({len(L)} lines); summary {len(summ)} characters")
    print(summ)


if __name__ == "__main__":
    {"airline": step_airline, "checkins": step_checkins, "lever": step_lever, "forecast": step_forecast,
     "newroute": step_newroute, "endtoend": step_endtoend, "guests": step_guests, "report": step_report}[sys.argv[1]]()
