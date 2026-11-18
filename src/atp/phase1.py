"""Phase 1 — Stage B validation: baselines, model variants (ablation), metrics on folds A and B.

Pre-registered acceptance rule for optional terms (fixed before results were seen):
    keep a term iff it LOWERS daily × nationality WMAPE on fold A (primary: same season as the official test)
    AND does not RAISE fold-B WMAPE by more than 0.05 points.

Run:  PYTHONPATH=src python -m atp.phase1 validate    (writes data/processed/phase1/*)
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from .config import load_config
from . import stageb as sb

INTL_VARIANTS = {                      # name: (design spec, arrival_dow)
    "base_no_holiday": ({"long_break": False}, False),
    "base": ({"long_break": True}, False),
    "base+grand_prix": ({"long_break": True, "grand_prix": True}, False),
}
DOM_VARIANTS = {
    "base_no_holiday": ({"long_break": False}, False),
    "base": ({"long_break": True}, False),
    "base+arrival_weekday": ({"long_break": True}, True),
}


def load(cfg):
    P = cfg["paths"]["processed"]
    hi = pd.read_parquet(P / "hotel_intl_daily.parquet")
    hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    cal = pd.read_parquet(P / "calendar.parquet")
    ext = pd.read_parquet(P / "external_events.parquet")
    gp = ext[ext.event.str.startswith("abu_dhabi_grand_prix")]
    days = pd.DatetimeIndex(np.concatenate([pd.date_range(r.start_date - pd.Timedelta(days=1), r.end_date) for r in gp.itertuples()]))
    gp_days = pd.Series(True, index=days)
    idx = pd.date_range(cfg["dates"]["hotel_start"], cfg["dates"]["test_end"])
    return hi, hd, cal, gp_days, idx


def baselines(markets, idx, ok, ho):
    """Three honest benchmarks a DCT analyst could build without our model."""
    out = {}
    month = idx.month.to_numpy()
    for n, m in markets.items():
        a = m.L[:, 0]; y = m.y
        tr = ok & (y > 0)
        ratio = pd.Series(y[tr]).groupby(month[tr]).sum() / pd.Series(a[tr]).groupby(month[tr]).sum()
        p_ratio = a * pd.Series(month).map(ratio).fillna(ratio.mean()).to_numpy()
        ly = np.r_[np.full(364, np.nan), y[:-364]]; ly[ly == 0] = np.nan
        p_naive = np.where(np.isnan(ly), p_ratio, ly)
        th = sb.init_theta(m, tr)
        p_curve = sb.predict(m, th, np.ones(len(idx)))
        out[n] = {"seasonal_naive": p_naive, "stay_ratio": p_ratio, "fixed_curve": p_curve}
    return out


def metrics(df: pd.DataFrame) -> dict:
    """df: market, date, y, p (target rows only). WMAPE at every level the scorer might use."""
    w = lambda y, p: float(np.abs(y - p).sum() / y.sum() * 100)
    I, D = df[df.market != "DOMESTIC"], df[df.market == "DOMESTIC"]
    mo = lambda x: x.assign(m=x.date.dt.to_period("M"))
    it = I.groupby("date")[["y", "p"]].sum(); at = df.groupby("date")[["y", "p"]].sum()
    im = mo(I).groupby(["market", "m"])[["y", "p"]].sum(); imt = mo(I).groupby("m")[["y", "p"]].sum()
    dm = mo(D).groupby("m")[["y", "p"]].sum()
    return {
        "intl_daily_x_nationality": w(I.y, I.p), "intl_daily_total": w(it.y, it.p),
        "intl_monthly_x_nationality": w(im.y, im.p), "intl_monthly_total": w(imt.y, imt.p),
        "domestic_daily": w(D.y, D.p) if len(D) else np.nan, "domestic_monthly": w(dm.y, dm.p) if len(D) else np.nan,
        "all_daily_total": w(at.y, at.p),
        "intl_worst_month_bias": float(((imt.p / imt.y - 1) * 100).abs().max()),
    }


def frame(markets, preds, idx, mask):
    rows = []
    for n, m in markets.items():
        sel = mask & (m.y > 0)
        rows.append(pd.DataFrame({"market": n, "date": idx[sel], "y": m.y[sel], "p": preds[n][sel]}))
    return pd.concat(rows, ignore_index=True)


def validate():
    cfg = load_config(); out_dir = cfg["paths"]["processed"] / "phase1"; out_dir.mkdir(exist_ok=True)
    hi, hd, cal, gp_days, idx = load(cfg)
    results, params = {}, {}
    for fold in ("A", "B"):
        f = cfg["folds"][fold]
        ok = (idx >= f["train_start"]) & (idx <= f["train_end"])
        ho = (idx >= f["holdout_start"]) & (idx <= f["holdout_end"])
        sb.assert_lag_windows(hi, hd, idx[ok | ho])
        mk = sb.build_markets(hi, hd, idx)
        intl = {n: m for n, m in mk.items() if n != "DOMESTIC"}; dom = {"DOMESTIC": mk["DOMESTIC"]}
        base = baselines(mk, idx, ok, ho)
        for bname in ("seasonal_naive", "stay_ratio", "fixed_curve"):
            df = frame(mk, {n: base[n][bname] for n in mk}, idx, ho)
            results[(fold, bname)] = metrics(df); df.assign(fold=fold, model=bname).to_parquet(out_dir / f"pred_{fold}_{bname}.parquet")
        for vname in INTL_VARIANTS:
            spec, adow = INTL_VARIANTS[vname]; dspec, ddow = DOM_VARIANTS.get(vname, DOM_VARIANTS["base"])
            if vname == "base+grand_prix":
                dspec, ddow = DOM_VARIANTS["base"]
            t0 = time.time()
            X, names = sb.design(cal, idx, f["train_end"], {**spec, "gp_days": gp_days})
            ri = sb.fit_group(intl, X, ok, adow)
            Xd, dnames = sb.design(cal, idx, f["train_end"], {**dspec, "gp_days": gp_days})
            rd = sb.fit_group(dom, Xd, ok, ddow)
            pred = {**sb.predict_all(ri, X), **sb.predict_all(rd, Xd)}
            df = frame(mk, pred, idx, ho)
            results[(fold, vname)] = metrics(df)
            df.assign(fold=fold, model=vname).to_parquet(out_dir / f"pred_{fold}_{vname}.parquet")
            params[(fold, vname)] = {"intl_beta": dict(zip(names, ri.beta.round(4).tolist())),
                                     "dom_beta": dict(zip(dnames, rd.beta.round(4).tolist())),
                                     "intl_grad_rel": ri.grad_rel, "dom_grad_rel": rd.grad_rel,
                                     "intl_status": int(ri.status), "dom_status": int(rd.status),
                                     "theta": {n: m.theta.round(5).tolist() for n, m in {**ri.markets, **rd.markets}.items()}}
            print(f"[{fold}] {vname}: {time.time() - t0:.0f}s  grad_rel intl {ri.grad_rel:.1e} dom {rd.grad_rel:.1e}  "
                  f"daily×nat {results[(fold, vname)]['intl_daily_x_nationality']:.2f}%", flush=True)
        # domestic arrival-weekday variant (domestic only; international kept at base)
        Xd, dnames = sb.design(cal, idx, f["train_end"], {"long_break": True})
        rd = sb.fit_group(dom, Xd, ok, True)
        pdm = sb.predict_all(rd, Xd)
        dfb = pd.read_parquet(out_dir / f"pred_{fold}_base.parquet")
        dfb = dfb[dfb.market != "DOMESTIC"].drop(columns=["fold", "model"])
        df = pd.concat([dfb, frame(dom, pdm, idx, ho)], ignore_index=True)
        results[(fold, "base+arrival_weekday")] = metrics(df)
        df.assign(fold=fold, model="base+arrival_weekday").to_parquet(out_dir / f"pred_{fold}_base+arrival_weekday.parquet")
        params[(fold, "base+arrival_weekday")] = {"dom_beta": dict(zip(dnames, rd.beta.round(4).tolist())),
                                                  "dom_gamma": rd.gamma.round(4).tolist(), "dom_grad_rel": rd.grad_rel}
    R = pd.DataFrame(results).T; R.index.names = ["fold", "model"]
    R.to_csv(out_dir / "validation_metrics.csv")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in params.items()}, open(out_dir / "validation_params.json", "w", encoding="utf-8"), indent=1)
    print(R.round(2).to_string())




# ============================================================================ selection, bands, final fit
RULE_TOL = 0.05
DRIFT_MIN_GAIN = 0.10   # pre-registered before the drift results were seen


def validate_drift():
    """Candidate added after diagnosing persistent per-market bias: per-market checkout-rate drift."""
    cfg = load_config(); out_dir = cfg["paths"]["processed"] / "phase1"
    hi, hd, cal, gp_days, idx = load(cfg)
    R = pd.read_csv(out_dir / "validation_metrics.csv", index_col=[0, 1])
    P = json.load(open(out_dir / "validation_params.json", encoding="utf-8"))
    for fold in ("A", "B"):
        f = cfg["folds"][fold]
        ok = (idx >= f["train_start"]) & (idx <= f["train_end"]); ho = (idx >= f["holdout_start"]) & (idx <= f["holdout_end"])
        mk = sb.build_markets(hi, hd, idx)
        intl = {n: m for n, m in mk.items() if n != "DOMESTIC"}; dom = {"DOMESTIC": mk["DOMESTIC"]}
        t0 = time.time()
        X, names = sb.design(cal, idx, f["train_end"], {"long_break": True})
        ri = sb.fit_group(intl, X, ok, drift=True, idx=idx)
        Xd, dnames = sb.design(cal, idx, f["train_end"], {"long_break": False})
        rd = sb.fit_group(dom, Xd, ok, drift=True, idx=idx)
        pred = {**sb.predict_all(ri, X), **sb.predict_all(rd, Xd)}
        df = frame(mk, pred, idx, ho)
        R.loc[(fold, "base+market_drift"), :] = pd.Series(metrics(df))
        df.assign(fold=fold, model="base+market_drift").to_parquet(out_dir / f"pred_{fold}_base+market_drift.parquet")
        P[f"{fold}|base+market_drift"] = {"intl_beta": dict(zip(names, ri.beta.round(4).tolist())),
                                          "dom_beta": dict(zip(dnames, rd.beta.round(4).tolist())),
                                          "intl_grad_rel": ri.grad_rel, "dom_grad_rel": rd.grad_rel,
                                          "drift_per_year": {n: round(float(m.theta[-1]), 4) for n, m in {**ri.markets, **rd.markets}.items()}}
        print(f"[{fold}] base+market_drift: {time.time() - t0:.0f}s grad_rel intl {ri.grad_rel:.1e} dom {rd.grad_rel:.1e} "
              f"daily×nat {R.loc[(fold, 'base+market_drift'), 'intl_daily_x_nationality']:.2f}% dom {R.loc[(fold, 'base+market_drift'), 'domestic_daily']:.2f}%", flush=True)
    R.to_csv(out_dir / "validation_metrics.csv")
    json.dump(P, open(out_dir / "validation_params.json", "w", encoding="utf-8"), indent=1)


def validate_step():
    """Audit candidate (4 Oct 2026), declared before it was run: a per-market RECENT STEP in the checkout rate over the
    last 182 training days, frozen after the train end — the Stage B counterpart of Stage A's C4b level term. Reason:
    holdout biases per market repeat across folds (actual ÷ predicted: China +19% / +16%, Egypt +18% / +27%, Germany
    −10% / −8%), i.e. recent levels the full-sample fit averages away; the linear trend (drift) fixed
    fold A but not fold B. Judged by the drift rule on daily × nationality WMAPE (fold A ≥ 0.10 pts better, fold B not
    worse by > 0.05) and it must not worsen monthly × nationality WMAPE on either fold. Domestic is evaluated too: an
    alternative level shape replaces the kept one only if it is better on both folds on daily and monthly WMAPE."""
    cfg = load_config(); out_dir = cfg["paths"]["processed"] / "phase1"
    hi, hd, cal, gp_days, idx = load(cfg)
    R = pd.read_csv(out_dir / "validation_metrics.csv", index_col=[0, 1])
    P = json.load(open(out_dir / "validation_params.json", encoding="utf-8"))
    for fold in ("A", "B"):
        f = cfg["folds"][fold]
        ok = (idx >= f["train_start"]) & (idx <= f["train_end"]); ho = (idx >= f["holdout_start"]) & (idx <= f["holdout_end"])
        mk = sb.build_markets(hi, hd, idx)
        intl = {n: m for n, m in mk.items() if n != "DOMESTIC"}; dom = {"DOMESTIC": mk["DOMESTIC"]}
        t0 = time.time()
        X, names = sb.design(cal, idx, f["train_end"], {"long_break": True})
        ri = sb.fit_group(intl, X, ok, drift=True, idx=idx, drift_shape="step")
        Xd, dnames = sb.design(cal, idx, f["train_end"], {"long_break": False})
        rd = sb.fit_group(dom, Xd, ok, drift=True, idx=idx, drift_shape="step")
        pred = {**sb.predict_all(ri, X), **sb.predict_all(rd, Xd)}
        df = frame(mk, pred, idx, ho)
        R.loc[(fold, "base+market_step"), :] = pd.Series(metrics(df))
        df.assign(fold=fold, model="base+market_step").to_parquet(out_dir / f"pred_{fold}_base+market_step.parquet")
        P[f"{fold}|base+market_step"] = {"intl_beta": dict(zip(names, ri.beta.round(4).tolist())),
                                          "dom_beta": dict(zip(dnames, rd.beta.round(4).tolist())),
                                          "intl_grad_rel": ri.grad_rel, "dom_grad_rel": rd.grad_rel,
                                          "theta": {n: m.theta.round(5).tolist() for n, m in {**ri.markets, **rd.markets}.items()},
                                          "recent_step": {n: round(float(m.theta[-1]), 4) for n, m in {**ri.markets, **rd.markets}.items()}}
        print(f"[{fold}] base+market_step: {time.time() - t0:.0f}s grad_rel intl {ri.grad_rel:.1e} dom {rd.grad_rel:.1e} "
              f"daily×nat {R.loc[(fold, 'base+market_step'), 'intl_daily_x_nationality']:.2f}% dom {R.loc[(fold, 'base+market_step'), 'domestic_daily']:.2f}%", flush=True)
    R.to_csv(out_dir / "validation_metrics.csv")
    json.dump(P, open(out_dir / "validation_params.json", "w", encoding="utf-8"), indent=1)


def step_checks():
    """Two checks on the kept recent step, from the saved fold fits (no refit): (1) carrying only part of the step into
    the holdout (share w of the recent change; w = 1 is the model) — would a shrunk step forecast better? (2) the band
    cost: calibrated 80% band width with and without the step (same rule as final())."""
    cfg = load_config(); out_dir = cfg["paths"]["processed"] / "phase1"
    hi, hd, cal, gp_days, idx = load(cfg)
    R = pd.read_csv(out_dir / "validation_metrics.csv", index_col=[0, 1]); sel = select(R)
    P = json.load(open(out_dir / "validation_params.json", encoding="utf-8"))
    shares, partial = (0.25, 0.5, 0.75, 1.0), {}
    for fold in ("A", "B"):
        f = cfg["folds"][fold]; ho = (idx >= f["holdout_start"]) & (idx <= f["holdout_end"]); te = pd.Timestamp(f["train_end"])
        intl = {n: m for n, m in sb.build_markets(hi, hd, idx).items() if n != "DOMESTIC"}
        X, names = sb.design(cal, idx, te, {"long_break": True})
        pp = P[f"{fold}|base+market_step"]; e = np.exp(X @ np.array([pp["intl_beta"][n] for n in names]))
        tau, after = sb.make_tau(idx, te, "step"), idx > te
        for w in shares:
            pred = {}
            for n, m in intl.items():
                m.tau = np.where(after, -(1 - w), tau)                      # hazard × exp(−(1−w)·δ) after the train end
                pred[n] = sb.predict(m, np.array(pp["theta"][n]), e)
            r = metrics(frame(intl, pred, idx, ho))
            partial[f"{fold}|{w}"] = {k: r[k] for k in ("intl_daily_x_nationality", "intl_monthly_x_nationality", "intl_monthly_total")}
        assert abs(partial[f"{fold}|1.0"]["intl_daily_x_nationality"] - R.loc[(fold, "base+market_step"), "intl_daily_x_nationality"]) < 0.01
    width = {}
    for lab, im in (("without_step", "base"), ("with_step", sel["intl_model"])):
        pr = {}
        for fold in ("A", "B"):
            a = pd.read_parquet(out_dir / f"pred_{fold}_{im}.parquet"); b = pd.read_parquet(out_dir / f"pred_{fold}_{sel['dom_model']}.parquet")
            pr[fold] = pd.concat([a[a.market != "DOMESTIC"], b[b.market == "DOMESTIC"]], ignore_index=True)
        q = next(q for q in np.arange(0.10, 0.009, -0.005)
                 if min(coverage(pr["B"], bands_from(pr["A"], q, 1 - q)), coverage(pr["A"], bands_from(pr["B"], q, 1 - q))) >= 0.80)
        b = bands_from(pd.concat([pr["A"], pr["B"]]), q, 1 - q)
        width[lab] = {"quantile": float(q), "median_log_width": float(np.log(b.q_hi / b.q_lo).median())}
    width["ratio_with_to_without"] = width["with_step"]["median_log_width"] / width["without_step"]["median_log_width"]
    n_step = sum(1 for k, v in P["A|base+market_step"]["recent_step"].items() if k != "DOMESTIC")
    best = all(partial[f"{f}|{w}"][k] > partial[f"{f}|1.0"][k] for f in "AB" for w in shares[:-1]
               for k in ("intl_daily_x_nationality", "intl_monthly_x_nationality"))
    j = {"partial_carry_forward": partial, "full_step_best_on_both_folds_and_metrics": best, "band_width": width, "n_step_parameters": n_step}
    json.dump(j, open(out_dir / "step_checks.json", "w", encoding="utf-8"), indent=1)
    for fold in ("A", "B"):
        print(f"[{fold}] daily × nat by share carried: " + ", ".join(f"{w:.0%} {partial[f'{fold}|{w}']['intl_daily_x_nationality']:.2f}%" for w in shares))
    print(f"band median log-width: without step {width['without_step']['median_log_width']:.3f} (q {width['without_step']['quantile']:.3f}), "
          f"with {width['with_step']['median_log_width']:.3f} (q {width['with_step']['quantile']:.3f}) → ×{width['ratio_with_to_without']:.2f}; {n_step} step parameters")


def select(R: pd.DataFrame) -> dict:
    """Apply the pre-registered rule; returns chosen spec + an audit trail."""
    g = lambda fold, model, col: float(R.loc[(fold, model), col])
    audit, chosen = [], {"intl": {"long_break": False, "grand_prix": False}, "dom": {"long_break": False, "arrival_dow": False}}

    def rule(label, off, on, col):
        dA = g("A", on, col) - g("A", off, col); dB = g("B", on, col) - g("B", off, col)
        keep = dA < 0 and dB <= RULE_TOL
        audit.append(f"{label}: fold A {dA:+.2f} pts, fold B {dB:+.2f} pts → {'KEEP' if keep else 'DROP'}")
        return keep
    chosen["intl"]["long_break"] = rule("International · long-break term", "base_no_holiday", "base", "intl_daily_x_nationality")
    ref_i = "base" if chosen["intl"]["long_break"] else "base_no_holiday"
    gp = rule("International · Grand Prix term (tested on top of long-break)", "base", "base+grand_prix", "intl_daily_x_nationality")
    chosen["intl"]["grand_prix"] = gp and chosen["intl"]["long_break"]
    chosen["dom"]["long_break"] = rule("Domestic · long-break term", "base_no_holiday", "base", "domestic_daily")
    adow = rule("Domestic · arrival-weekday term (tested on top of long-break)", "base", "base+arrival_weekday", "domestic_daily")
    chosen["dom"]["arrival_dow"] = adow and chosen["dom"]["long_break"]
    if ("A", "base+market_drift") in R.index:
        ref = "base"   # drift was tested on top of the terms already kept (intl: long-break; domestic: none)
        dA_i = g("A", "base+market_drift", "intl_daily_x_nationality") - g("A", ref, "intl_daily_x_nationality")
        dB_i = g("B", "base+market_drift", "intl_daily_x_nationality") - g("B", ref, "intl_daily_x_nationality")
        dA_d = g("A", "base+market_drift", "domestic_daily") - g("A", "base_no_holiday", "domestic_daily")
        dB_d = g("B", "base+market_drift", "domestic_daily") - g("B", "base_no_holiday", "domestic_daily")
        ki = dA_i <= -DRIFT_MIN_GAIN and dB_i <= RULE_TOL; kd = dA_d <= -DRIFT_MIN_GAIN and dB_d <= RULE_TOL
        audit.append(f"International · per-market drift (needs ≥{DRIFT_MIN_GAIN} pt gain): fold A {dA_i:+.2f}, fold B {dB_i:+.2f} → {'KEEP' if ki else 'DROP'}")
        audit.append(f"Domestic · drift (needs ≥{DRIFT_MIN_GAIN} pt gain): fold A {dA_d:+.2f}, fold B {dB_d:+.2f} → {'KEEP' if kd else 'DROP'}")
        chosen["intl"]["drift"], chosen["dom"]["drift"] = ki, kd
    else:
        chosen["intl"]["drift"] = chosen["dom"]["drift"] = False
    chosen["intl"]["drift_shape"] = chosen["dom"]["drift_shape"] = "trend"
    if ("A", "base+market_step") in R.index:                                  # audit candidate, 4 Oct 2026 (validate_step)
        st = "base+market_step"
        d_ = {f: g(f, st, "intl_daily_x_nationality") - g(f, "base", "intl_daily_x_nationality") for f in "AB"}
        m_ = {f: g(f, st, "intl_monthly_x_nationality") - g(f, "base", "intl_monthly_x_nationality") for f in "AB"}
        ks = d_["A"] <= -DRIFT_MIN_GAIN and d_["B"] <= RULE_TOL and m_["A"] <= 0 and m_["B"] <= 0
        trend_ok = chosen["intl"]["drift"]
        worst = lambda mdl: max(g(f, mdl, "intl_daily_x_nationality") for f in "AB")
        use = ks and (not trend_ok or worst(st) < worst("base+market_drift"))
        audit.append(f"International · per-market recent step (182 days; audit 4 Oct 2026): daily × nationality fold A {d_['A']:+.2f}, "
                     f"fold B {d_['B']:+.2f}; monthly × nationality A {m_['A']:+.2f}, B {m_['B']:+.2f} → {'KEEP' if use else 'DROP'}")
        if use:
            chosen["intl"]["drift"], chosen["intl"]["drift_shape"] = True, "step"
        if chosen["dom"]["drift"]:
            dd = {(f, c): g(f, st, c) - g(f, "base+market_drift", c) for f in "AB" for c in ("domestic_daily", "domestic_monthly")}
            better = all(v < 0 for v in dd.values())
            audit.append(f"Domestic · recent step instead of the kept trend: daily A {dd[('A', 'domestic_daily')]:+.2f}, B {dd[('B', 'domestic_daily')]:+.2f}; "
                         f"monthly A {dd[('A', 'domestic_monthly')]:+.2f}, B {dd[('B', 'domestic_monthly')]:+.2f} → "
                         + ("REPLACE" if better else "trend kept (not better on both folds and both metrics)"))
            if better:
                chosen["dom"]["drift_shape"] = "step"
    intl_model = "base+grand_prix" if chosen["intl"]["grand_prix"] else ref_i
    if chosen["intl"]["drift"]:
        intl_model = "base+market_step" if chosen["intl"]["drift_shape"] == "step" else "base+market_drift"
    dom_model = ("base+arrival_weekday" if chosen["dom"]["arrival_dow"] else "base") if chosen["dom"]["long_break"] else "base_no_holiday"
    if chosen["dom"]["drift"]:
        dom_model = "base+market_step" if chosen["dom"]["drift_shape"] == "step" else "base+market_drift"
    return {"chosen": chosen, "intl_model": intl_model, "dom_model": dom_model, "audit": audit}


def chosen_preds(out_dir, fold, sel):
    a = pd.read_parquet(out_dir / f"pred_{fold}_{sel['intl_model']}.parquet")
    b = pd.read_parquet(out_dir / f"pred_{fold}_{sel['dom_model']}.parquet")
    return pd.concat([a[a.market != "DOMESTIC"], b[b.market == "DOMESTIC"]], ignore_index=True)


def bands_from(df: pd.DataFrame, lo=0.10, hi=0.90) -> pd.DataFrame:
    r = df.assign(ratio=df.y / df.p).groupby("market").ratio
    return pd.DataFrame({"q_lo": r.quantile(lo), "q_hi": r.quantile(hi)})


def coverage(df: pd.DataFrame, b: pd.DataFrame) -> float:
    d = df.join(b, on="market")
    return float(((d.y >= d.p * d.q_lo) & (d.y <= d.p * d.q_hi)).mean())


def final():
    cfg = load_config(); out_dir = cfg["paths"]["processed"] / "phase1"
    R = pd.read_csv(out_dir / "validation_metrics.csv", index_col=[0, 1])
    sel = select(R)
    pA, pB = chosen_preds(out_dir, "A", sel), chosen_preds(out_dir, "B", sel)
    # calibration: bands learned on one fold, coverage measured on the other (both directions)
    # calibration: narrowest band whose cross-fold coverage (learn on one fold, test on the other) is ≥ 80% both ways
    q = next(q for q in np.arange(0.10, 0.009, -0.005)
             if min(coverage(pB, bands_from(pA, q, 1 - q)), coverage(pA, bands_from(pB, q, 1 - q))) >= 0.80)
    cal_AB, cal_BA = coverage(pB, bands_from(pA, q, 1 - q)), coverage(pA, bands_from(pB, q, 1 - q))
    bands = bands_from(pd.concat([pA, pB]), q, 1 - q)
    hi, hd, cal, gp_days, idx = load(cfg)
    f = cfg["folds"]["final"]
    ok = (idx >= f["train_start"]) & (idx <= f["train_end"]); te = (idx >= f["holdout_start"]) & (idx <= f["holdout_end"])
    sb.assert_lag_windows(hi, hd, idx[ok | te])
    mk = sb.build_markets(hi, hd, idx)
    intl = {n: m for n, m in mk.items() if n != "DOMESTIC"}; dom = {"DOMESTIC": mk["DOMESTIC"]}
    ci, cd = sel["chosen"]["intl"], sel["chosen"]["dom"]
    X, names = sb.design(cal, idx, f["train_end"], {"long_break": ci["long_break"], "grand_prix": ci["grand_prix"], "gp_days": gp_days})
    Xd, dnames = sb.design(cal, idx, f["train_end"], {"long_break": cd["long_break"]})
    t0 = time.time()
    ri = sb.fit_group(intl, X, ok, drift=ci["drift"], idx=idx, drift_shape=ci.get("drift_shape", "trend"))
    rd = sb.fit_group(dom, Xd, ok, cd["arrival_dow"], drift=cd["drift"], idx=idx, max_nfev=3000, drift_shape=cd.get("drift_shape", "trend"))
    pred = {**sb.predict_all(ri, X), **sb.predict_all(rd, Xd)}
    print(f"final fit {time.time() - t0:.0f}s grad_rel intl {ri.grad_rel:.1e} dom {rd.grad_rel:.1e}", flush=True)
    # test predictions: every market, every day of the official test window
    dq = pd.read_parquet(cfg["paths"]["processed"] / "flights_day_quality.parquet").set_index("date").is_low_coverage
    status = hi.set_index(["nationality", "date"]).arrivals_status
    rows = []
    for n, m in mk.items():
        p = pred[n][te]; bq = bands.loc[n]
        st = status.loc[n].reindex(idx[te]).fillna("observed").to_numpy() if n != "DOMESTIC" else np.full(te.sum(), "observed")
        rows.append(pd.DataFrame({"date": idx[te], "market": n, "checkins": m.L[te, 0], "checkins_status": st,
                                  "pred_guests": p, "lo80": p * bq.q_lo, "hi80": p * bq.q_hi,
                                  "flight_disruption_day": dq.reindex(idx[te], fill_value=False).to_numpy()}))
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(out_dir / "test_predictions_daily.csv", index=False)
    T.assign(month=T.date.dt.to_period("M").astype(str)).groupby(["market", "month"])[["checkins", "pred_guests", "lo80", "hi80"]] \
        .sum().rename(columns={"pred_guests": "pred_guest_nights"}).to_csv(out_dir / "test_predictions_monthly.csv")
    # readable parameters
    hr = pd.DataFrame({"international": pd.Series(np.exp(ri.beta), index=names),
                       "domestic": pd.Series(np.exp(rd.beta), index=dnames)})
    # DCT definition: guest nights ÷ check-ins over the last 12 training months, at the checkout rate the model carries
    # forward (drift at its train-end value, τ = 0), i.e. the stay the forecasts and scenarios use
    last12 = ok & (idx > f["train_end"] - pd.Timedelta(days=365))
    stay = {}
    for res_, X_ in ((ri, X), (rd, Xd)):
        e_ = np.exp(X_ @ res_.beta)
        for n, m in res_.markets.items():
            tau_ = m.tau
            if tau_ is not None:
                m.tau = np.zeros(len(idx))
            stay[n] = float(sb.predict(m, m.theta, e_, res_.gamma)[last12].sum() / m.L[last12, 0].sum())
            m.tau = tau_
    shape = {n: (ci if n != "DOMESTIC" else cd).get("drift_shape", "trend") for n in mk}
    dlt = {n: (float(m.theta[-1]) if len(m.theta) == 1 + sb.NH + 1 else 0.0) for n, m in {**ri.markets, **rd.markets}.items()}
    s0 = {n: float(np.exp(m.theta[0])) for n, m in {**ri.markets, **rd.markets}.items()}
    mp = pd.DataFrame({"avg_stay_nights_model": stay,
                       "checkout_drift_per_year": {n: v if shape[n] == "trend" else 0.0 for n, v in dlt.items()},
                       "checkout_rate_change_last_182d": {n: float(np.expm1(v)) if shape[n] == "step" else 0.0 for n, v in dlt.items()},
                       "arrival_night_ratio_S0": s0, "band_lo_ratio": bands.q_lo, "band_hi_ratio": bands.q_hi})
    hr.to_csv(out_dir / "hazard_ratios.csv"); mp.to_csv(out_dir / "market_parameters.csv")
    summary = {"selection": sel, "calibration": {"bands_A_on_B": cal_AB, "bands_B_on_A": cal_BA, "quantile": float(q)},
               "final_grad_rel": {"intl": ri.grad_rel, "dom": rd.grad_rel},
               "dom_gamma": None if rd.gamma is None else np.exp(np.r_[0, rd.gamma]).round(3).tolist()}
    json.dump(summary, open(out_dir / "final_summary.json", "w", encoding="utf-8"), indent=1, default=str)
    report(cfg, R, sel, summary, hr, mp, T, pA)
    print("\n".join(sel["audit"])); print(summary["calibration"])


def _report_facts(cfg, sel, mp) -> dict:
    """Numbers quoted in the report's notes, computed (never typed): the 24 Jun 2025 disruption analogue and the
    same-day-visitor check on S0."""
    from scipy.stats import spearmanr
    P = cfg["paths"]["processed"]
    p = pd.read_parquet(P / "phase1" / f"pred_B_{sel['intl_model']}.parquet")
    i = p[p.market != "DOMESTIC"].groupby("date")[["y", "p"]].sum(); e = (i.p / i.y - 1) * 100
    jun = e.loc["2025-06-01":"2025-06-23"]; nxt = e.loc["2025-06-25":"2025-06-28"]
    hi = pd.read_parquet(P / "hotel_intl_daily.parquet"); tr = hi[hi.split == "train"]
    lo_ = tr.groupby("nationality").apply(lambda g: g.same_day_raw.sum() / g.arrivals.sum(), include_groups=False)
    ob = tr[tr.same_day_raw.notna()]
    hi_ = ob.groupby("nationality").apply(lambda g: g.same_day_raw.sum() / g.arrivals.sum(), include_groups=False)
    s0 = mp.arrival_night_ratio_S0.drop("DOMESTIC")
    bias = {}
    for f in "AB":                                                           # the earlier model's per-market holdout bias
        b_ = pd.read_parquet(P / "phase1" / f"pred_{f}_base.parquet"); b_ = b_[b_.market != "DOMESTIC"].groupby("market")[["y", "p"]].sum()
        bias[f] = b_.y / b_.p - 1; vol = b_.y / b_.y.sum()
    big = vol[vol >= 0.005].index
    return {"bias_corr_AB": float(np.corrcoef(bias["A"][big], bias["B"][big])[0, 1]), "shock_day": float(e.loc["2025-06-24"]), "june_mean": float(jun.mean()), "june_sd": float(jun.std()),
            "after_lo": float(nxt.min()), "after_hi": float(nxt.max()),
            "sd_lo": float(lo_.median()), "sd_hi": float(hi_.median()),
            "rho_lo": float(spearmanr(lo_.reindex(s0.index), s0).correlation), "rho_hi": float(spearmanr(hi_.reindex(s0.index), s0).correlation),
            "s0_china": float(s0["CHINA"]), "s0_lo": float(s0.min()), "s0_hi": float(s0.max()),
            "china_lowest_same_day": bool(lo_.idxmin() == "CHINA" and hi_.idxmin() == "CHINA")}


def report(cfg, R, sel, summary, hr, mp, T, pA):
    rows = [(("seasonal_naive",) * 2, "Same day last year"), (("stay_ratio",) * 2, "Check-ins × monthly stay ratio"),
            (("fixed_curve",) * 2, "Fixed stay curve (no calendar)"),
            (("base", "base_no_holiday"), "Our model before the domestic drift term"),
            (("base", "base+market_drift"), "Phase 1 before the 4 Oct 2026 audit (no international recent step)"),
            ((sel["intl_model"], sel["dom_model"]), f"**Our model, final** (international: `{sel['intl_model']}`, domestic: `{sel['dom_model']}`)")]
    cols = [("intl_daily_x_nationality", "Intl daily × nationality"), ("intl_daily_total", "Intl daily total"),
            ("intl_monthly_x_nationality", "Intl monthly × nationality"), ("intl_monthly_total", "Intl monthly total"),
            ("domestic_daily", "Domestic daily"), ("intl_worst_month_bias", "Worst month bias (intl)")]

    fx = _report_facts(cfg, sel, mp)

    def cell(fold, models, c):
        return f"{R.loc[(fold, models[1] if c.startswith('domestic') else models[0]), c]:.2f}%"
    L = ["# Phase 1 report — Stage B occupancy engine", "", "_Generated by `make phase1`; do not edit by hand._", "",
         "## 1. Accuracy on held-out periods (WMAPE, lower is better)", ""]
    for fold, lab in (("A", "Fold A — Aug 2024 → Feb 2025 (primary: same season as the official test)"), ("B", "Fold B — Jan → Jul 2025")):
        L += [f"**{lab}**", "", "| Model | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        L += [f"| {lab_} | " + " | ".join(cell(fold, m, c[0]) for c in cols) + " |" for m, lab_ in rows] + [""]
    L += ["## 2. Which optional terms were kept (rule fixed before results)", "",
          f"Keep a term only if it lowers daily × nationality WMAPE on fold A and does not raise fold B by more than {RULE_TOL} points.", ""]
    L += [f"- {a}" for a in sel["audit"]]
    if any("recent step" in a for a in sel["audit"]):
        L += ["", f"The recent step was added in the 4 Oct 2026 audit, after the per-market holdout biases of the earlier model were seen to "
                  f"repeat across the two folds (correlation {fx['bias_corr_AB']:.2f} over markets with ≥ 0.5% of guests): a recent level the "
                  "full-sample fit averages away. It mirrors Stage A's pre-registered recent-level term (C4b, last 182 training days) and "
                  "was judged by the drift rule above, plus no loss on monthly × nationality WMAPE in either fold."]
        sc_ = json.load(open(cfg["paths"]["processed"] / "phase1" / "step_checks.json", encoding="utf-8")); pc = sc_["partial_carry_forward"]
        L += ["", f"It adds {sc_['n_step_parameters']} parameters (one per international nationality), each set by the contrast between "
                  "the last 182 training days and the earlier ones. Would a shrunk step do better? Carrying only part of the recent change "
                  "forward (" + ", ".join(f"{w:.0%}" for w in (0.25, 0.5, 0.75)) + ") was less accurate on daily and monthly × nationality "
                  "WMAPE in both folds at every share (fold A daily × nationality: " + ", ".join(
                      f"{pc[f'A|{w}']['intl_daily_x_nationality']:.2f}%" for w in (0.25, 0.5, 0.75)) + f" vs {pc['A|1.0']['intl_daily_x_nationality']:.2f}%; "
                  "fold B: " + ", ".join(f"{pc[f'B|{w}']['intl_daily_x_nationality']:.2f}%" for w in (0.25, 0.5, 0.75))
                  + f" vs {pc['B|1.0']['intl_daily_x_nationality']:.2f}%), so the full step is kept."]
        assert sc_["full_step_best_on_both_folds_and_metrics"], "report text assumes the full step is best — revisit"
    c = summary["calibration"]
    L += ["", "## 3. Uncertainty bands", "",
          f"80% bands per market from held-out errors, widened by calibration: the narrowest width (error quantiles "
          f"{c['quantile']:.3f}–{1 - c['quantile']:.3f}) whose cross-fold coverage reaches 80%. Honesty check (learned on one "
          f"fold, tested on the other): {c['bands_A_on_B']:.0%} and {c['bands_B_on_A']:.0%} of actual values fell inside."
          + (f" The recent step costs width: its per-market errors carry over less from one fold to the other, so calibration needs "
             f"wider quantiles ({sc_['band_width']['with_step']['quantile']:.3f} vs {sc_['band_width']['without_step']['quantile']:.3f} "
             f"without it) and the median band is ×{sc_['band_width']['ratio_with_to_without']:.2f} as wide (in log terms). The bands stay "
             "honest (calibrated); the step was kept for the point accuracy above." if any("recent step" in a for a in sel["audit"]) else ""), "",
          "## 4. Calendar effects on the checkout rate (1.00 = normal day)", "", hr.round(3).to_markdown(), "",
          "## 5. Market parameters (final fit, training through Jul 2025)", "",
          "`avg_stay_nights_model` = predicted guest nights ÷ check-ins over the last 12 training months (DCT definition), at the "
          "checkout rate carried forward; `checkout_drift_per_year` (domestic trend) and `checkout_rate_change_last_182d` "
          "(international recent step: checkout rate in the last 182 training days vs before) > 0 mean stays got shorter, "
          "< 0 longer; both are frozen at the end of training.", "",
          mp.sort_values("avg_stay_nights_model", ascending=False).round(3).to_markdown(), "",
          "## 6. Official test predictions (Aug 2025 → Feb 2026)", "",
          f"- `test_predictions_daily.csv`: {len(T):,} rows = the complete grid (46 markets × {T.date.nunique()} days) with 80% bands. "
          "Includes the 338 nationality-days the organisers omit from their test file (< 10 check-ins).",
          "- `submission_international.csv` (9,202 rows) and `submission_domestic.csv` (212 rows) = 9,414 scored rows: the "
          "organisers' own test files, same rows and order, with `Guests` (+ 80% band) added — made by `make export` "
          "(inner join on Date × Nationality, every organiser row asserted to get exactly one prediction).",
          "- `test_predictions_monthly.csv`: guest nights per market per month",
          f"- Predicted international guest nights over the window: {T[T.market != 'DOMESTIC'].pred_guests.sum() / 1e6:.2f}M; "
          f"domestic: {T[T.market == 'DOMESTIC'].pred_guests.sum() / 1e6:.2f}M",
          f"- 28 Feb 2026 is a flight-disruption day (conflict onset, last test day). Evidence from the training-period "
          f"analogue (24 Jun 2025, fold-B holdout, chosen model): on the shock day itself the model's error was normal "
          f"({fx['shock_day']:+.1f}% vs {fx['june_mean']:+.1f}% ± {fx['june_sd']:.1f}% on 1–23 June) because the check-in surge is observed; "
          f"on the next four days it over-predicted by {fx['after_lo']:.1f}–{fx['after_hi']:.1f}% because stranded travellers stay 1–2 nights, "
          "not their nationality's usual stay. For 28 Feb 2026 those following days are outside the test window. No ad-hoc term is added.",
          f"- `S0` (arrival-night ratio, {fx['s0_lo']:.2f}–{fx['s0_hi']:.2f} across nationalities) is a per-market reporting ratio absorbing "
          f"definitional differences between check-ins and nightly counts. Same-day visitors cannot explain it: they are "
          f"{fx['sd_lo'] * 100:.1f}–{fx['sd_hi'] * 100:.1f}% of check-ins in the median market (lower bound: suppressed same-day cells counted "
          f"as 0; upper bound: only cells where it is reported), rank correlation with S0 {fx['rho_lo']:+.2f} / {fx['rho_hi']:+.2f}, and "
          f"China has the lowest same-day share yet S0 = {fx['s0_china']:.2f}. Cause to confirm with DCT.", "",
          "## 7. Convergence", "", f"Relative gradient at the final solution: international {summary['final_grad_rel']['intl']:.1e}, "
          f"domestic {summary['final_grad_rel']['dom']:.1e} (≈0 means the optimum was reached).", ""]
    (cfg["paths"]["reports"] / "phase1_report.md").write_text("\n".join(L), encoding="utf-8")




def report_only():
    cfg = load_config(); d = cfg["paths"]["processed"] / "phase1"
    R = pd.read_csv(d / "validation_metrics.csv", index_col=[0, 1]); sel = select(R)
    summary = json.load(open(d / "final_summary.json", encoding="utf-8"))
    hr = pd.read_csv(d / "hazard_ratios.csv", index_col=0); mp = pd.read_csv(d / "market_parameters.csv", index_col=0)
    T = pd.read_csv(d / "test_predictions_daily.csv", parse_dates=["date"])
    report(cfg, R, sel, summary, hr, mp, T, None)


# ============================================================================ submission export
def export_submission():
    """Join predictions onto the ORIGINAL test files (their rows, order and columns), one file per test file.
    The prediction grid is complete (46 markets × 212 days); the organisers' test files omit nationality-days with
    < 10 check-ins, so an inner join on (Date, Nationality) keeps exactly the rows they will score."""
    from .io import read_raw
    from .ledger import Ledger
    cfg = load_config(); d = cfg["paths"]["processed"] / "phase1"; led = Ledger()
    P = pd.read_csv(d / "test_predictions_daily.csv", parse_dates=["date"])
    out = {}
    for key, market_col in (("intl_test", "Nationality"), ("dom_test", None)):
        raw = read_raw(cfg, key, led).reset_index(drop=True)
        raw["_key_date"] = pd.to_datetime(raw["Date"])
        if market_col:
            raw["_key_mkt"] = raw[market_col]
        else:
            raw["_key_mkt"] = "DOMESTIC"
        pred = P.rename(columns={"date": "_key_date", "market": "_key_mkt"})[["_key_date", "_key_mkt", "pred_guests", "lo80", "hi80"]]
        assert not raw.duplicated(["_key_date", "_key_mkt"]).any(), f"{key}: duplicate keys in the organisers' file"
        m = raw.merge(pred, on=["_key_date", "_key_mkt"], how="left", validate="one_to_one")
        assert len(m) == len(raw) and (m.index == raw.index).all(), f"{key}: row count/order changed"
        assert m.pred_guests.notna().all(), f"{key}: {int(m.pred_guests.isna().sum())} organiser rows have no prediction"
        sub = raw.drop(columns=["_key_date", "_key_mkt"]).copy()
        sub["Guests"] = m.pred_guests.round(1).to_numpy()
        sub["Guests_lo80"] = m.lo80.round(1).to_numpy(); sub["Guests_hi80"] = m.hi80.round(1).to_numpy()
        out[key] = (sub, len(raw), int(len(pred[pred._key_mkt != "DOMESTIC"]) if market_col else len(pred[pred._key_mkt == "DOMESTIC"])))
    sub_i, n_i, grid_i = out["intl_test"]; sub_d, n_d, grid_d = out["dom_test"]
    sub_i.to_csv(d / "submission_international.csv", index=False); sub_d.to_csv(d / "submission_domestic.csv", index=False)
    print(f"international: {n_i} organiser rows ← {grid_i} predicted cells ({grid_i - n_i} not in the organisers' file: nationality-days with <10 check-ins)")
    print(f"domestic:      {n_d} organiser rows ← {grid_d} predicted cells")
    print(f"total scored rows: {n_i + n_d} (prediction grid {grid_i + grid_d})")
    assert n_i + n_d == 9414 and grid_i + grid_d == 9752


if __name__ == "__main__":
    {"validate": validate, "drift": validate_drift, "step": validate_step, "stepcheck": step_checks, "final": final, "report": report_only,
     "export": export_submission}[sys.argv[1]]()
