"""Phase 5, Step 5.5 — a gradient-boosting benchmark for the daily check-in forecaster.

Pre-registered in docs/phase5_5_preregistration.md. One LightGBM model (Poisson objective) over all 46 series, given the
forecaster's own inputs — the calendar terms of its chosen specification and the 7-day seats — and nothing from the
hotel data it predicts. Same rows, same folds (A and B; the official test window is not used), same WMAPE functions.

Run:  python -m atp.phase5 benchmark
"""
import json
import numpy as np
import pandas as pd
from . import forecaster as fc

PREREG_BENCH = "docs/phase5_5_preregistration.md"
CAL = ["month", "dow", "ramadan", "main_break", "festival", "yearend", "f1", "adipec", "home_holiday", "school_break", "recent"]
SEATS = ["log_seats", "rel_seats"]
SPEC = {"C1", "C2", "C3", "C4b", "C6", "C7"}      # the chosen forecaster specification: main_break + festival, year-end, events, recent step, home holidays, school breaks
PRIMARY = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, reg_lambda=1.0)
SEEDS = (1, 2, 3, 4, 5)
GRID = [dict(PRIMARY, num_leaves=nl, n_estimators=ne, learning_rate=lr)
        for nl in (7, 15, 31) for ne, lr in ((200, 0.05), (600, 0.03))]
# Amendment A1 (after the first run, declared in the pre-registration): the first grid's accuracy rose with size, so four
# larger settings are added as a best-case check for the benchmark. They are chosen with hindsight and are not the conclusion.
GRID_EXT = [dict(PRIMARY, num_leaves=nl, n_estimators=ne, learning_rate=0.03, min_child_samples=mc)
            for nl, ne, mc in ((31, 1200, 20), (63, 1200, 20), (31, 2000, 10), (63, 2000, 10))]
SCALES = (0.9, 1.1, 1.5)
N_BOOT, BOOT_SEED = 2000, 7
START = pd.Timestamp("2023-01-07")


def _features(D: pd.DataFrame, train_end, spec: set) -> pd.DataFrame:
    """The forecaster's inputs for every row of every series, as numbers a tree can split on. The seat floor and the
    series' training mean are the forecaster's (5% of the mean of the 7-day seats over its training rows)."""
    assert spec == SPEC, f"the benchmark's inputs mirror the forecaster's chosen specification {sorted(SPEC)}, not {sorted(spec)}"
    out = []
    for key, d in D.groupby("key", sort=True):
        d = d.copy()
        tr = d[(d.date >= START) & (d.date <= train_end)]
        has = tr.seats_7d.notna().any()
        d["floor"] = fc.SEAT_FLOOR * tr.seats_7d.mean() if has else np.nan
        d["log_seats"] = np.log(np.maximum(d.seats_7d, d["floor"])) if has else np.nan
        d["log_mean"] = np.log(tr.seats_7d.mean()) if has else np.nan
        d["rel_seats"] = d.log_seats - d.log_mean
        d["recent"] = (d.date > train_end - pd.Timedelta(days=fc.LEVEL_DAYS)).astype(float)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def fit_predict_lgbm(D, train_end, horizon_end, spec: set, params: dict = PRIMARY, use_seats: bool = True,
                     seat_scale: float = 1.0, seeds=SEEDS) -> pd.DataFrame:
    """Train on the forecaster's estimation rows (7 Jan 2023 … train end, without disruption days and days below the seat
    floor) and predict every day after the training end up to `horizon_end`. Returns key, date, y, pred."""
    import lightgbm as lgb
    F = _features(D, train_end, spec)
    keys = sorted(F.key.unique())
    F["series"] = pd.Categorical(F.key, categories=keys)
    cols = ["series"] + CAL + (SEATS if use_seats else [])
    tr = F[(F.date >= START) & (F.date <= train_end)]
    ok = ~tr.exclude & (tr.floor.isna() | (tr.seats_7d >= tr.floor))
    fit = tr[ok]
    te = F[(F.date > train_end) & (F.date <= horizon_end)].copy()
    if seat_scale != 1.0:                                       # a scenario: every series' seats scaled, floor unchanged
        te["log_seats"] = np.log(np.maximum(te.seats_7d * seat_scale, te["floor"]))
        te["rel_seats"] = te.log_seats - te.log_mean
    mu = np.zeros(len(te))
    for s in seeds:
        m = lgb.LGBMRegressor(objective="poisson", random_state=s, n_jobs=1, deterministic=True, force_row_wise=True,
                              verbose=-1, **params)
        m.fit(fit[cols], fit.y.to_numpy())
        mu += m.predict(te[cols])
    return te[["key", "date", "y"]].assign(pred=mu / len(seeds)).reset_index(drop=True)


# ----------------------------------------------------------------------------- the comparison
MEASURES = [("intl_daily_nat", "International, daily × nationality (primary)"),
            ("intl_monthly_nat", "International, monthly × nationality"),
            ("intl_daily_total", "International total, daily"), ("intl_monthly_total", "International total, monthly"),
            ("dom_daily", "Domestic, daily"), ("dom_monthly", "Domestic, monthly")]


def _ours(D, F, spec, eps, keys):
    """Our forecaster refitted for every series (the production code), optionally at another ε."""
    return pd.concat([fc.fit_predict(D[D.key == k], F["train_end"], START, F["holdout_end"], spec, eps)[0] for k in keys],
                     ignore_index=True)


def month_block_interval(prod: pd.DataFrame, alt: pd.DataFrame, intl: set) -> list:
    """95% interval of WMAPE(alt) − WMAPE(prod) on international daily × nationality, resampling the holdout's months."""
    j = prod[prod.key.isin(intl)].merge(alt, on=["key", "date"], suffixes=("_p", "_a"))
    j["m"] = j.date.dt.to_period("M")
    g = j.assign(ep=(j.y_p - j.pred_p).abs(), ea=(j.y_p - j.pred_a).abs()).groupby("m")[["y_p", "ep", "ea"]].sum()
    rng = np.random.default_rng(BOOT_SEED); n = len(g); v = g.to_numpy()
    idx = rng.integers(0, n, size=(N_BOOT, n))
    s = v[idx].sum(axis=1)
    d = (s[:, 2] - s[:, 1]) / s[:, 0] * 100
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def implied_elasticity(base: pd.DataFrame, up: pd.DataFrame, flight: set, scale: float) -> float:
    b, u = base[base.key.isin(flight)], up[up.key.isin(flight)]
    return float(np.log(u.pred.sum() / b.pred.sum()) / np.log(scale))


def benchmark(cfg) -> dict:
    from .phase5 import _forecaster_frame, sha256
    P, Fo = cfg["paths"]["processed"], cfg["folds"]
    D, intl, flight = _forecaster_frame(cfg)
    j24 = json.load(open(P / "phase2" / "step24_forecaster.json", encoding="utf-8"))
    spec = set(j24["chosen_spec"]); eps = j24["eps_used_in_folds"]
    keys = sorted(D.key.unique())
    res = {"prereg_sha256": sha256(cfg["root"] / PREREG_BENCH), "spec": sorted(spec), "params": PRIMARY, "seeds": list(SEEDS),
           "features": ["series"] + CAL + SEATS}
    for f in ("A", "B"):
        F = Fo[f]
        prod = pd.read_parquet(P / "phase2" / f"stagea_pred_{f}.parquet")
        m_prod = fc.metrics(prod, intl, flight)
        assert all(abs(m_prod[k] - j24["chosen_metrics"][f][k]) < 1e-9 for k in m_prod), "our saved scores are not reproduced"
        re_ = _ours(D, F, spec, eps[f], keys)
        chk = prod.merge(re_, on=["key", "date"], suffixes=("", "_re"))
        assert len(chk) == len(prod) and np.allclose(chk.pred, chk.pred_re, rtol=1e-9), "our forecasts are not reproduced"
        lg = fit_predict_lgbm(D, F["train_end"], F["holdout_end"], spec)
        assert len(lg) == len(prod) and (lg.sort_values(["key", "date"]).y.to_numpy() == prod.sort_values(["key", "date"]).y.to_numpy()).all()
        m_lg = fc.metrics(lg, intl, flight)
        ns = fit_predict_lgbm(D, F["train_end"], F["holdout_end"], spec, use_seats=False)
        e0 = _ours(D, F, spec, 0.0, keys)
        grid, ext, preds = [], [], []
        for out, settings in ((grid, GRID), (ext, GRID_EXT)):
            for p in settings:
                g = fit_predict_lgbm(D, F["train_end"], F["holdout_end"], spec, params=p)
                out.append({"num_leaves": p["num_leaves"], "n_estimators": p["n_estimators"], "learning_rate": p["learning_rate"],
                            "min_child_samples": p["min_child_samples"],
                            **{k: v for k, v in fc.metrics(g, intl, flight).items() if k in dict(MEASURES)}})
                preds.append(g)
        i_best = min(range(len(preds)), key=lambda i: (grid + ext)[i]["intl_daily_nat"])     # with hindsight on the holdout
        best = {**(grid + ext)[i_best], "diff_interval_95": month_block_interval(prod, preds[i_best], intl)}
        el = {str(s): {"lightgbm": implied_elasticity(lg, fit_predict_lgbm(D, F["train_end"], F["holdout_end"], spec, seat_scale=s), flight, s)}
              for s in SCALES}
        res[f] = {"ours": m_prod, "lightgbm": m_lg, "lightgbm_no_seats": fc.metrics(ns, intl, flight),
                  "ours_eps0": fc.metrics(e0, intl, flight), "grid": grid, "grid_extended": ext, "best_of_ten": best, "eps": eps[f], "implied_elasticity": el,
                  "diff_interval_95": month_block_interval(prod, lg, intl),
                  "diff_interval_95_no_seats": month_block_interval(prod, ns, intl),
                  "n_rows": int(len(prod))}
        print(f"[{f}] intl daily×nat: ours {m_prod['intl_daily_nat']:.2f}  LightGBM {m_lg['intl_daily_nat']:.2f}  "
              f"no seats {res[f]['lightgbm_no_seats']['intl_daily_nat']:.2f}  ours ε=0 {res[f]['ours_eps0']['intl_daily_nat']:.2f}", flush=True)
    res["verdict"] = verdict(res)
    return res


def verdict(r: dict) -> dict:
    """The pre-registered reading: LightGBM is 'more accurate' only if it wins the primary measure on both folds, and a
    fold whose month-block interval of the difference contains 0 is a tie."""
    k = "intl_daily_nat"
    per = {}
    for f in ("A", "B"):
        d = r[f]["lightgbm"][k] - r[f]["ours"][k]; lo, hi = r[f]["diff_interval_95"]
        per[f] = {"difference": d, "interval": [lo, hi], "lightgbm_lower": d < 0, "clear": bool(hi < 0 or lo > 0)}
    wins = all(per[f]["lightgbm_lower"] for f in per)
    ours = all(not per[f]["lightgbm_lower"] for f in per)
    clear = [f for f in per if per[f]["clear"]]
    label = ("LightGBM more accurate" if wins else
             ("ours lower on both folds, clearly so on " + (" and ".join(clear) if clear else "neither") + " (month-block interval excludes 0)") if ours
             else "mixed: one fold each")
    return {"per_fold": per, "lightgbm_more_accurate": bool(wins), "ours_lower_on_both_folds": bool(ours), "label": label}


# ----------------------------------------------------------------------------- the report
def _best_word(r: dict, f: str) -> str:
    """How the best-of-ten benchmark setting compares with ours on fold f, with its month-block interval."""
    b = r[f]["best_of_ten"]; lo, hi = b["diff_interval_95"]; d = b["intl_daily_nat"] - r[f]["ours"]["intl_daily_nat"]
    clear = lo > 0 or hi < 0
    return (f"the benchmark is {abs(d):.2f} points {'worse' if d > 0 else 'better'}, "
            + ("clearly (95% month-block interval excludes 0)" if clear else "within noise (interval includes 0), a tie"))


def _best_sentence(r: dict, f: str) -> str:
    b = r[f]["best_of_ten"]; lo, hi = b["diff_interval_95"]
    return (f"Fold {f}: the best is {b['num_leaves']} leaves and {b['n_estimators']} trees, gap to ours "
            f"{b['intl_daily_nat'] - r[f]['ours']['intl_daily_nat']:+.2f} points (95% month-block interval {lo:+.2f} to {hi:+.2f}): "
            + ("clearly worse than ours." if lo > 0 else "clearly better than ours." if hi < 0 else "within noise, a tie."))


def _helps(r: dict, without: str, with_: str) -> str:
    k = "intl_daily_nat"; d = {f: r[f][without][k] - r[f][with_][k] for f in "AB"}
    word = lambda x: "help" if x > 0 else "hurt"
    return (f"{word(d['A'])} on fold A and {word(d['B'])} on fold B" if (d["A"] > 0) != (d["B"] > 0) else f"{word(d['A'])} on both folds")


def report(r: dict) -> str:
    v = r["verdict"]; k = "intl_daily_nat"
    pf = v["per_fold"]
    allg = {f: r[f]["grid"] + r[f]["grid_extended"] for f in "AB"}
    best = {f: min(g[k] for g in allg[f]) for f in "AB"}
    L = ["# Phase 5, Step 5.5 — gradient-boosting benchmark", "",
         "A LightGBM model is given the same inputs as our daily check-in forecaster (the calendar terms of its chosen "
         "specification and the 7-day seats; never a hotel figure from the period it predicts), trained on the same rows and scored "
         "on the same days with the same WMAPE. Folds A and B only; the official test window is not used. Rules and settings: "
         "`docs/phase5_5_preregistration.md` (sha256 recorded in the results). Reported whatever the outcome.", "",
         "## Result", "",
         f"**{v['label'][0].upper() + v['label'][1:]}.** On the primary measure (international check-ins, daily × nationality) ours scores "
         f"{r['A']['ours'][k]:.2f}% on fold A and {r['B']['ours'][k]:.2f}% on fold B; the benchmark scores {r['A']['lightgbm'][k]:.2f}% and "
         f"{r['B']['lightgbm'][k]:.2f}%. The gap is {pf['A']['difference']:+.2f} points on A (95% month-block interval "
         f"{pf['A']['interval'][0]:+.2f} to {pf['A']['interval'][1]:+.2f}) and {pf['B']['difference']:+.2f} on B "
         f"({pf['B']['interval'][0]:+.2f} to {pf['B']['interval'][1]:+.2f}). Lower is better; a positive gap means the benchmark is worse.", "",
         "| Measure | Fold | Ours | LightGBM | Gap (LightGBM − ours) |", "|---|---|---|---|---|"]
    for m, lab in MEASURES:
        for f in "AB":
            a, b = r[f]["ours"][m], r[f]["lightgbm"][m]
            L.append(f"| {lab} | {f} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
    wins = [(lab, f) for m, lab in MEASURES for f in "AB" if r[f]["lightgbm"][m] < r[f]["ours"][m]]
    L += ["", f"The benchmark is lower on {len(wins)} of the 12 fold-measure pairs"
          + (": " + "; ".join(f"{lab.split(' (')[0].lower()} on fold {f}" for lab, f in wins) + "." if wins else ".")
          + (f" All of them are on fold {wins[0][1]} and none is the primary measure." if wins and len({f for _, f in wins}) == 1
             and not any(lab.startswith("International, daily × nationality") for lab, _ in wins) else ""), "",
          "## Is the benchmark too small? A best case for it", "",
          "Ten settings were run (the six of the pre-registered grid and four larger ones added after the first run, Amendment A1). "
          "None was used to choose the primary setting. The table gives the primary measure on each.", "",
          "| Leaves | Trees | Learning rate | Min rows/leaf | Fold A | Fold B |", "|---|---|---|---|---|---|"]
    rows = lambda f: r[f]["grid"] + r[f]["grid_extended"]
    for ga, gb in zip(rows("A"), rows("B")):
        L.append(f"| {ga['num_leaves']} | {ga['n_estimators']} | {ga['learning_rate']} | {ga['min_child_samples']} | {ga[k]:.2f} | {gb[k]:.2f} |")
    L += ["", f"Best of the ten, picked with hindsight on the holdout: {best['A']:.2f}% on A (ours {r['A']['ours'][k]:.2f}%) and "
          f"{best['B']:.2f}% on B (ours {r['B']['ours'][k]:.2f}%). "
          + " ".join(_best_sentence(r, f) for f in "AB"), "",
          "## How much does each model owe to seats?", "",
          "| Primary measure | Fold A | Fold B |", "|---|---|---|",
          f"| Ours | {r['A']['ours'][k]:.2f} | {r['B']['ours'][k]:.2f} |",
          f"| Ours with ε = 0 (no seat link) | {r['A']['ours_eps0'][k]:.2f} | {r['B']['ours_eps0'][k]:.2f} |",
          f"| LightGBM | {r['A']['lightgbm'][k]:.2f} | {r['B']['lightgbm'][k]:.2f} |",
          f"| LightGBM without seat inputs | {r['A']['lightgbm_no_seats'][k]:.2f} | {r['B']['lightgbm_no_seats'][k]:.2f} |", "",
          "Taking the seat link out changes our score by "
          f"{r['A']['ours_eps0'][k] - r['A']['ours'][k]:+.2f} (A) and {r['B']['ours_eps0'][k] - r['B']['ours'][k]:+.2f} (B) points, "
          f"the picture of the Step 2.4 ε sensitivity: seats {_helps(r, 'ours_eps0', 'ours')}. For the benchmark the "
          f"change is {r['A']['lightgbm_no_seats'][k] - r['A']['lightgbm'][k]:+.2f} (A) and {r['B']['lightgbm_no_seats'][k] - r['B']['lightgbm'][k]:+.2f} (B): "
          f"seats {_helps(r, 'lightgbm_no_seats', 'lightgbm')}."
          + (" Neither model has a seat signal that helps on both folds." if not any(
              all(r[f][a][k] > r[f][b][k] for f in "AB") for a, b in (("ours_eps0", "ours"), ("lightgbm_no_seats", "lightgbm"))) else ""), "",
          "## What the benchmark says about a seat change", "",
          "Seats of every series with flights are multiplied on the holdout days and the benchmark is asked again; the implied "
          "elasticity is log(predicted check-ins ratio) ÷ log(seat ratio). Ours is a fixed ε by construction.", "",
          "| Seat change | Fold A | Fold B |", "|---|---|---|"]
    for s in SCALES:
        L.append(f"| × {s} | {r['A']['implied_elasticity'][str(s)]['lightgbm']:.2f} | {r['B']['implied_elasticity'][str(s)]['lightgbm']:.2f} |")
    L += [f"| **Ours (ε)** | {r['A']['eps']:.2f} | {r['B']['eps']:.2f} |", "",
          f"The benchmark's response to seats averages {np.mean([r[f]['implied_elasticity'][str(s_)]['lightgbm'] for f in 'AB' for s_ in SCALES]):.2f} "
          f"over the six cells, against ε of {r['A']['eps']:.2f} and {r['B']['eps']:.2f}, but it is uneven: a 10% cut implies "
          f"{min(r[f]['implied_elasticity']['0.9']['lightgbm'] for f in 'AB'):.2f}–{max(r[f]['implied_elasticity']['0.9']['lightgbm'] for f in 'AB'):.2f}, "
          f"a 10% rise {min(r[f]['implied_elasticity']['1.1']['lightgbm'] for f in 'AB'):.2f}–{max(r[f]['implied_elasticity']['1.1']['lightgbm'] for f in 'AB'):.2f}. "
          "It is not a number a planner can set, check or put a range on, and it cannot be carried on to guest-nights (it stops at "
          "check-ins). In ours the same step is one stated number with an interval, tested out of sample (Step 2.3).", "",
          "## What follows", "",
          ("Accuracy is not a reason to switch. " if v["ours_lower_on_both_folds"] else "")
          + "On the primary measure, with the benchmark's pre-registered setting: " + v["label"] + ". Against the best of ten settings chosen "
          "with hindsight: " + "; ".join(f"fold {f} {_best_word(r, f)}" for f in "AB") + ". "
          + (f"The benchmark is lower on {len(wins)} secondary fold-measure pair{'s' if len(wins) > 1 else ''} ("
             + "; ".join(f"{lab.split(' (')[0].lower()}, fold {f}" for lab, f in wins) + "); we have not isolated why. " if wins else "")
          + "It would in any case replace only the check-in step: it has no hotel-stay side (Stage B), no seat lever with a range, and "
          "nothing a planner can audit. **No change to the model.**", ""]
    return "\n".join(L)
