"""Two checks quoted in the technical brief for the maths audit (Phase 5):

1. Small-cluster check of ε's interval (33 countries): wild cluster bootstrap-t next to the pairs cluster bootstrap.
2. Are markets' baseline errors independent (the reason level parts add in quadrature across markets)?

The project's interval for ε (Phase 2, Step 2.2) is a pairs cluster bootstrap: whole countries resampled with
replacement, the month effects re-estimated in each draw, percentile interval. With about 30 clusters the usual
recommendation is a wild cluster bootstrap (Cameron, Gelbach & Miller 2008; Webb 2023 for few clusters). This script
computes it on the same estimation sample and specification (lead-controlled, month fixed effects):

  WCU bootstrap-t: residuals û from the fitted model; for each draw b, every country g gets one Webb weight w_g
  (six-point: ±√(1/2), ±1, ±√(3/2), equal probability); y* = ŷ + w_g·û; re-estimate ε* and its cluster-robust SE*
  (CR1); t* = (ε* − ε̂) / SE*. 95% interval = [ε̂ − q₉₇.₅(t*)·SE, ε̂ − q₂.₅(t*)·SE]. In each draw y* is demeaned by
  month again (the weights break the residuals' zero month means), and CR1 counts the absorbed month effects (audit).
  Also the analytic CR1 interval with t(G − 1) critical values.

Check 2, on the back-test (data/processed/phase4/backtest_monthly.csv): each market's relative error σ_m = RMS over the
fold's months of actual/predicted − 1; for each month, the error of the all-market total predicted under independence
√Σ(p_m σ_m)² / Σp_m and under full correlation Σ p_m σ_m / Σ p_m (p_m = predicted guest nights), each as an RMS over
the months, against the observed RMS of the total's actual/predicted − 1.

Run: PYTHONPATH=src python scripts/audit_checks.py   → data/processed/phase5/audit_checks.json
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from atp.config import load_config
from atp import phase2 as p2, stagea as sa

B, SEED = 9999, 0


def cr1(Z, r, g, k_absorbed: int = 0):
    """OLS on demeaned data: coefficients' CR1 cluster-robust covariance, G/(G−1)·(N−1)/(N−K) with K counting the slope
    columns AND the absorbed month effects (k_absorbed = number of months: demeaning absorbs one effect per month and leaves no intercept; the month effects are not nested in the country
    clusters, so they are counted, as a regression with month dummies would)."""
    n, k = Z.shape; G = len(np.unique(g)); K = k + k_absorbed
    bread = np.linalg.inv(Z.T @ Z)
    meat = sum(np.outer(s, s) for s in (Z[g == c].T @ r[g == c] for c in np.unique(g)))
    return G / (G - 1) * (n - 1) / (n - K) * bread @ meat @ bread


def independence(cfg) -> dict:
    b = pd.read_csv(cfg["paths"]["processed"] / "phase4" / "backtest_monthly.csv")
    res = {}
    for f, d in b.groupby("fold"):
        d = d.assign(e=d.actual_guest_nights / d.rebuilt_guest_nights - 1)
        sig = d.groupby("market").e.apply(lambda x: np.sqrt((x ** 2).mean()))
        t = d.groupby("month")[["actual_guest_nights", "rebuilt_guest_nights"]].sum()
        ind, full = [], []
        for _, g in d.groupby("month"):
            s, p = sig.reindex(g.market).to_numpy(), g.rebuilt_guest_nights.to_numpy()
            ind.append(np.sqrt(((p * s) ** 2).sum()) / p.sum()); full.append((p * s).sum() / p.sum())
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))) * 100)
        res[f] = {"observed_total_error_rms_pct": rms(t.actual_guest_nights / t.rebuilt_guest_nights - 1),
                  "if_independent_pct": rms(ind), "if_fully_correlated_pct": rms(full)}
    return res


def main():
    cfg = load_config(); fd, _, hi = p2._load(cfg)
    p, _ = p2._primary(sa.monthly_panel(fd, hi), cfg["folds"]["final"]["train_end"])
    d = p.dropna(subset=["d_checkins"] + p2.LEADS).reset_index(drop=True)
    t = d.month.astype(str).to_numpy()
    Z = sa._demean(d.d_checkins.to_numpy(), d[p2.LEADS].to_numpy(), t)
    y, X, g = Z[:, 0], Z[:, 1:], d.nationality_key.to_numpy()
    ka = len(np.unique(t))                                                      # absorbed month effects (one per month: no separate intercept)
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    se = float(np.sqrt(cr1(X, r, g, ka)[0, 0])); se_slope_only = float(np.sqrt(cr1(X, r, g)[0, 0])); G = len(np.unique(g))
    countries, gi = np.unique(g, return_inverse=True)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    rng = np.random.default_rng(SEED); fit = X @ b; tstar = np.empty(B)
    for i in range(B):
        ys = fit + webb[rng.integers(0, 6, G)][gi] * r
        # the country weights break the zero month means of the residuals, so the month effects are re-estimated in every
        # draw (within transformation of y*) before the slope and its residuals: the full model is refitted, not just X
        ys = sa._demean(ys, np.zeros((len(ys), 0)), t)[:, 0]
        bs = np.linalg.lstsq(X, ys, rcond=None)[0]
        tstar[i] = (bs[0] - b[0]) / np.sqrt(cr1(X, ys - X @ bs, g, ka)[0, 0])
    q_lo, q_hi = np.percentile(tstar, [2.5, 97.5])
    tc = stats.t.ppf(0.975, G - 1)
    # the second seat lead: the same analytic interval with t(G−1) and the full-design count, next to the normal-based
    # interval statsmodels gives in Step 2.2 (the convention of the saved Phase 2 outputs)
    cov = cr1(X, r, g, ka); se2 = float(np.sqrt(cov[2, 2])) if X.shape[1] > 2 else None
    out = {"eps": float(b[0]), "n_obs": int(len(d)), "clusters": int(G), "months": int(ka),
           "se_cr1": se, "se_cr1_slope_columns_only": se_slope_only, "dof_factor_vs_slope_only": se / se_slope_only,
           "ci95_wild_cluster_bootstrap_t": [float(b[0] - q_hi * se), float(b[0] - q_lo * se)],
           "ci95_cr1_t_G_minus_1": [float(b[0] - tc * se), float(b[0] + tc * se)],
           "lead2": None if se2 is None else {"coef": float(b[2]), "se_cr1": se2,
                                              "ci95_cr1_t_G_minus_1": [float(b[2] - tc * se2), float(b[2] + tc * se2)]},
           "draws": B, "weights": "Webb six-point", "seed": SEED,
           "convention": "CR1 with K = slope columns + absorbed month effects; bootstrap outcome re-demeaned by month in each draw; t(G−1) critical values"}
    out = {"wild_cluster_bootstrap": out, "market_independence": independence(cfg)}
    path = cfg["paths"]["processed"] / "phase5"; path.mkdir(exist_ok=True)
    json.dump(out, open(path / "audit_checks.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
