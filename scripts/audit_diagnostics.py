"""Diagnostics added in response to the independent maths audit (Phase 5). They need the real data, so the audit pack
could not run them. Nothing here changes a fitted value; each part only reports.

1. M2: plain and lead-controlled ε on the SAME complete-case sample (the saved plain estimate uses a larger sample, so
   the published gap mixes the effect of the lead controls with a sample change). Also ε + γ₁ + γ₂, the response to a
   persistent change if the leads were read as structural.
2. M5: Stage B identification at the production fit: for every international market, the singular values of its own
   parameter block of the Jacobian (S0, ten hazard bands, drift), columns scaled to unit length, on its training days.
   A condition number near 1e3 or below means each parameter moves the fit in its own direction; a very large one
   means some combination is barely determined by the stock series.
3. U1: retrospective interval score and conditional coverage of the market-month guest-night level band, cross-fold
   (band learned on one fold, scored on the other). The band width was chosen on these same folds, so this is a
   retrospective description, not an independent check. Interval score for a nominal 80% interval [l, u]:
   IS = (u − l) + 10·(l − y)·1[y < l] + 10·(y − u)·1[y > u]   (Gneiting & Raftery 2007).

Run: PYTHONPATH=src python scripts/audit_diagnostics.py   → data/processed/phase5/audit_diagnostics.json
"""
import json
import numpy as np
import pandas as pd
from atp.config import load_config
from atp import phase2 as p2, stagea as sa, bands as bd


def m2_common_sample(cfg) -> dict:
    fd, _, hi = p2._load(cfg)
    p, _ = p2._primary(sa.monthly_panel(fd, hi), cfg["folds"]["final"]["train_end"])
    d = p.dropna(subset=["d_checkins"] + p2.LEADS).reset_index(drop=True)
    t = d.month.astype(str).to_numpy()
    out = {"n_obs": int(len(d)), "months": int(len(np.unique(t))), "countries": int(d.nationality_key.nunique())}
    for name, cols in (("plain", ["d_seats"]), ("lead_controlled", p2.LEADS)):
        Z = sa._demean(d.d_checkins.to_numpy(), d[cols].to_numpy(), t)
        b = np.linalg.lstsq(Z[:, 1:], Z[:, 0], rcond=None)[0]
        out[name] = {c: float(v) for c, v in zip(cols, b)}
    lc = out["lead_controlled"]
    out["eps_plus_leads"] = float(sum(lc.values()))
    full = p.dropna(subset=["d_checkins", "d_seats"])
    tf = full.month.astype(str).to_numpy()
    Zf = sa._demean(full.d_checkins.to_numpy(), full[["d_seats"]].to_numpy(), tf)
    out["plain_on_its_own_larger_sample"] = {"eps": float(np.linalg.lstsq(Zf[:, 1:], Zf[:, 0], rcond=None)[0][0]),
                                             "n_obs": int(len(full)), "months": int(len(np.unique(tf)))}
    return out


def m5_identification(cfg) -> dict:
    from atp import stageb as sb, phase1 as p1
    hi, hd, cal, gp_days, idx = p1.load(cfg)
    f = cfg["folds"]["final"]; ok = (idx >= f["train_start"]) & (idx <= f["train_end"])
    sel = json.load(open(cfg["paths"]["processed"] / "phase1" / "final_summary.json", encoding="utf-8"))["selection"]["chosen"]["intl"]
    par = json.load(open(cfg["paths"]["processed"] / "phase3" / "stageb_params.json", encoding="utf-8"))
    mk = sb.build_markets(hi, hd, idx)
    X, _ = sb.design(cal, idx, f["train_end"], {"long_break": sel["long_break"], "grand_prix": sel["grand_prix"], "gp_days": gp_days})
    e = np.exp(X @ np.array(par["intl_beta"]))
    rows = {}
    for n, m in mk.items():
        if n == "DOMESTIC" or n not in par["theta"]:
            continue
        th = np.array(par["theta"][n])
        if len(th) == 1 + sb.NH + 1:
            m.tau = sb.make_tau(idx, f["train_end"], sel.get("drift_shape", "trend"))
        _, Jth, _, _ = sb.predict(m, th, e, None, jac=True, X=X)
        J = Jth[ok & (m.y > 0)]
        at_lb = np.r_[False, th[1:1 + sb.NH] <= sb.LB_TH[1:1 + sb.NH] + 1e-6, [False] * (len(th) - 1 - sb.NH)]
        Jf = J[:, ~at_lb]                                                       # parameters held at a bound are not estimated
        Jn = Jf / np.linalg.norm(Jf, axis=0)
        s = np.linalg.svd(Jn, compute_uv=False)
        rows[n] = {"free_params": int(Jf.shape[1]), "cond": float(s[0] / s[-1]), "smallest_sv": float(s[-1]),
                   "S0": float(np.exp(th[0]))}
    c = pd.Series({k: v["cond"] for k, v in rows.items()})
    return {"markets": len(rows), "condition_number_quantiles": {q: float(c.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)},
            "markets_cond_over_1e4": sorted(c[c > 1e4].index.tolist()), "by_market": rows,
            "note": "columns scaled to unit length; parameters held at their lower bound left out (not estimated)"}


def u1_interval_score(cfg) -> dict:
    tiers = bd.market_tiers(cfg)
    F = {f: bd._fold_monthly(cfg, f, "guests").assign(tier=lambda x: x.m.map(tiers), r=lambda x: x.y / x.pred) for f in "AB"}
    lv = pd.read_csv(cfg["paths"]["processed"] / "phase3" / "level_bands.csv")
    q = float(lv.query("kind == 'guests' and level == 'market-month'").q.iloc[0])
    out = {"q": q}
    for a, b in (("A", "B"), ("B", "A")):
        band = {t: bd._band(g.r, q) for t, g in F[a].groupby("tier")}
        T = F[b].assign(l=lambda x: x.pred * x.tier.map(lambda t: band[t][0]), u=lambda x: x.pred * x.tier.map(lambda t: band[t][1]))
        IS = (T.u - T.l) + 10 * (T.l - T.y).clip(lower=0) + 10 * (T.y - T.u).clip(lower=0)
        cov = (T.y >= T.l) & (T.y <= T.u)
        qtr = T.month.dt.quarter
        out[f"{a}_to_{b}"] = {"n": int(len(T)), "coverage": float(cov.mean()),
                              "interval_score_over_actual_pct": float(IS.sum() / T.y.sum() * 100),
                              "width_over_actual_pct": float((T.u - T.l).sum() / T.y.sum() * 100),
                              "miss_penalty_share_pct": float((IS - (T.u - T.l)).sum() / IS.sum() * 100),
                              "coverage_by_tier": {k: float(v) for k, v in cov.groupby(T.tier).mean().items()},
                              "coverage_by_calendar_quarter": {f"Q{k}": float(v) for k, v in cov.groupby(qtr).mean().items()},
                              "below_band_share": float((T.y < T.l).mean()), "above_band_share": float((T.y > T.u).mean())}
    out["note"] = ("retrospective: the band's quantile was chosen on these two folds, and the folds share Jan–Feb 2025 as "
                   "target months, so these are descriptions of the chosen band, not an independent coverage test")
    return out


def main():
    cfg = load_config()
    out = {"m2_common_sample": m2_common_sample(cfg), "u1_interval_score": u1_interval_score(cfg),
           "m5_stageb_identification": m5_identification(cfg)}
    path = cfg["paths"]["processed"] / "phase5"; path.mkdir(exist_ok=True)
    json.dump(out, open(path / "audit_diagnostics.json", "w", encoding="utf-8"), indent=1)
    s = dict(out); s["m5_stageb_identification"] = {k: v for k, v in out["m5_stageb_identification"].items() if k != "by_market"}
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
