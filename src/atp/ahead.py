"""Phase 5, Step 5.7 — accuracy by months ahead.

Pre-registered in docs/phase5_7_preregistration.md. Months ahead h = months since the training end (1…7 in each of the two
held-out periods). Monthly guest nights (flights → check-ins → guests, Step 2.6) and check-ins (Stage A) from the saved
back-test predictions; the international total and by nationality; coverage of the 80% bands by h; a test of whether the
error grows with h.

Run:  python -m atp.phase5 ahead
"""
import json
import numpy as np
import pandas as pd

PREREG_AHEAD = "docs/phase5_7_preregistration.md"
LEADS = list(range(1, 8))
N_BOOT, SEED = 2000, 11
CONSISTENT = 1e-6


def _monthly(cfg, fold: str) -> dict:
    """Monthly actual and forecast by market for one fold: guest nights and check-ins, with months ahead."""
    P, F = cfg["paths"]["processed"], cfg["folds"][fold]
    g = pd.read_parquet(P / "phase2" / f"endtoend_pred_{fold}.parquet").rename(columns={"p": "pred"})
    c = pd.read_parquet(P / "phase2" / f"stagea_pred_{fold}.parquet").rename(columns={"key": "market"})
    out = {}
    for name, d in (("guests", g), ("checkins", c)):
        m = d.assign(month=d.date.dt.to_period("M")).groupby(["market", "month"])[["y", "pred"]].sum().reset_index()
        te = pd.Timestamp(F["train_end"])
        m["h"] = (m.month.dt.year - te.year) * 12 + m.month.dt.month - te.month
        out[name] = m.assign(fold=fold)
    return out


def _check_consistent(cfg, M: dict) -> float:
    """The monthly guest nights used here are the ones Tab 3 shows (phase4/backtest_monthly.csv)."""
    bm = pd.read_csv(cfg["paths"]["processed"] / "phase4" / "backtest_monthly.csv")
    m = pd.concat([M[f]["guests"] for f in "AB"]); m["month"] = m.month.astype(str)
    j = m.merge(bm, left_on=["fold", "market", "month"], right_on=["fold", "market", "month"])
    assert len(j) == len(m) == len(bm), "months differ from backtest_monthly.csv"
    err = max(float((j.y - j.actual_guest_nights).abs().max()), float((j.pred - j.rebuilt_guest_nights).abs().max()))
    assert err <= CONSISTENT * float(j.y.max()), f"monthly guest nights differ from backtest_monthly.csv by {err}"
    return err


def wmape(d: pd.DataFrame) -> float:
    return float((d.y - d.pred).abs().sum() / d.y.sum() * 100)


def _by_lead(d: pd.DataFrame) -> dict:
    return {int(h): wmape(g) for h, g in d.groupby("h")}


def _slope(v: dict) -> float:
    h = np.array(sorted(v)); y = np.array([v[k] for k in h])
    return float(np.polyfit(h, y, 1)[0])


def _boot_by_lead(d: pd.DataFrame, n=N_BOOT, seed=SEED) -> dict:
    """Nationality bootstrap of the nationality-month WMAPE at each h and of its slope on h (the same resampled
    nationalities in every h)."""
    names = np.array(sorted(d.market.unique())); rng = np.random.default_rng(seed)
    piv = {h: g.set_index("market")[["y", "pred"]] for h, g in d.groupby("h")}
    absr = np.array([[(piv[h].loc[m].y - piv[h].loc[m].pred).__abs__() if m in piv[h].index else 0.0 for m in names] for h in LEADS])
    act = np.array([[piv[h].loc[m].y if m in piv[h].index else 0.0 for m in names] for h in LEADS])
    idx = rng.integers(0, len(names), size=(n, len(names)))
    w = np.stack([absr[:, i].sum(1) / act[:, i].sum(1) * 100 for i in idx])               # n × 7
    sl = np.array([np.polyfit(LEADS, x, 1)[0] for x in w])
    ct = w[:, 1:].mean(1) - w[:, 0]                                                      # Amendment A1: month 1 vs months 2–7
    return {"contrast_ci": [float(np.percentile(ct, 2.5)), float(np.percentile(ct, 97.5))], "wmape_ci": {h: [float(np.percentile(w[:, k], 2.5)), float(np.percentile(w[:, k], 97.5))] for k, h in enumerate(LEADS)},
            "slope_ci": [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]}


def _total(d: pd.DataFrame) -> dict:
    t = d.groupby("h")[["y", "pred"]].sum()
    return {int(h): {"actual": float(r.y), "forecast": float(r.pred), "abs_error_pct": float(abs(r.pred - r.y) / r.y * 100),
                     "signed_error_pct": float((r.pred / r.y - 1) * 100)} for h, r in t.iterrows()}


def _coverage(cfg, fold: str) -> tuple[dict, float]:
    """Share of international nationality-months inside their size tier's 80% guest-night band, the band LEARNED ON THE
    OTHER FOLD (Step 3.5's cross-fold rule, same quantile q). Also returns the all-market coverage, which must equal
    Step 3.5's own cross-fold figure."""
    from . import bands as bd
    tiers = bd.market_tiers(cfg)
    q = float(pd.read_csv(cfg["paths"]["processed"] / "phase3" / "level_bands.csv").query("kind == 'guests' and level == 'market-month'").q.iloc[0])
    other = "B" if fold == "A" else "A"
    learn = bd._fold_monthly(cfg, other, "guests").assign(tier=lambda x: x.m.map(tiers), r=lambda x: x.y / x.pred)
    test = bd._fold_monthly(cfg, fold, "guests").assign(tier=lambda x: x.m.map(tiers), r=lambda x: x.y / x.pred)
    band = {t: bd._band(g.r, q) for t, g in learn.groupby("tier")}
    ok = (test.r >= test.tier.map(lambda t: band[t][0])) & (test.r <= test.tier.map(lambda t: band[t][1]))
    te = pd.Timestamp(cfg["folds"][fold]["train_end"])
    h = (test.month.dt.year - te.year) * 12 + test.month.dt.month - te.month
    intl = test.m != "DOMESTIC"
    return {int(k): float(v) for k, v in ok[intl].groupby(h[intl]).mean().items()}, float(ok.mean())


def run(cfg) -> dict:
    from .phase5 import sha256
    M = {f: _monthly(cfg, f) for f in "AB"}
    res = {"prereg_sha256": sha256(cfg["root"] / PREREG_AHEAD), "consistency_max_abs_diff": _check_consistent(cfg, M),
           "leads": LEADS, "n_boot": N_BOOT,
           "months": {f: [str(m) for m in sorted(M[f]["guests"].month.unique())] for f in "AB"}}
    tiers = pd.read_csv(cfg["paths"]["processed"] / "phase3" / "market_tiers.csv").set_index("market").tier
    for kind in ("guests", "checkins"):
        R = {}
        for f in ("A", "B", "pooled"):
            d = pd.concat([M[x][kind] for x in "AB"]) if f == "pooled" else M[f][kind]
            d = d[d.market != "DOMESTIC"]
            assert sorted(d.h.unique()) == LEADS
            R[f] = {"total": _total(d) if f != "pooled" else None, "by_nationality": _by_lead(d),
                    "by_tier": {t: _by_lead(d[d.market.map(tiers) == t]) for t in ("top 10", "11–25", "26–46")}}
            if f != "pooled":
                b = _boot_by_lead(d)
                v = R[f]["by_nationality"]
                R[f] |= {"by_nationality_ci": b["wmape_ci"], "slope": _slope(v), "slope_ci": b["slope_ci"],
                         "first_vs_later": {"estimate": float(np.mean([v[h] for h in LEADS[1:]]) - v[1]), "ci": b["contrast_ci"]}}
                if kind == "guests":
                    R[f]["coverage"], allm = _coverage(cfg, f)
                    s35 = json.load(open(cfg["paths"]["processed"] / "phase3" / "step35_bands.json", encoding="utf-8"))["level_diagnostics"]["guests"]
                    want = s35["coverage_B_to_A" if f == "A" else "coverage_A_to_B"]           # learned on the other fold, tested on f
                    assert abs(allm - want) < 1e-12, f"cross-fold coverage {allm} is not Step 3.5's {want}"
                    R[f]["coverage_all_markets"] = allm
                    R[f]["n_nationality_months"] = int(len(d))
        res[kind] = R
    res["pooled_total_abs_error_pct"] = {h: float(np.mean([res["guests"][f]["total"][h]["abs_error_pct"] for f in "AB"])) for h in LEADS}
    res["grows_with_lead"] = grows(res["guests"])
    return res


def grows(g: dict) -> dict:
    """The pre-registered rule: the error grows with lead if the slope's 95% interval is above 0 on both folds."""
    per = {f: {"slope": g[f]["slope"], "ci": g[f]["slope_ci"], "above_zero": bool(g[f]["slope_ci"][0] > 0)} for f in "AB"}
    yes = all(per[f]["above_zero"] for f in "AB")
    return {"per_fold": per, "grows": bool(yes), "label": "the error grows with lead" if yes else "no detectable growth with lead in months 1–7"}


# ----------------------------------------------------------------------------- Tab 3 and the report
def _pc(x, d=0) -> str:
    return f"{x:.{d}f}%"


def _minus(t: str) -> str:
    """Typographic minus for negative numbers in text shown to the user."""
    import re
    return re.sub(r"(?<![\w.])-(?=\d)", "−", t)


def tab3(r: dict, folds: list) -> dict:
    """The Tab 3 block. folds: [{"id", "months", "label"}] as the tab already carries them (months by lead)."""
    out = _tab3(r, folds)
    out["points"] = [_minus(x) for x in out["points"]]; out["method"] = _minus(out["method"])
    return out


def _tab3(r: dict, folds: list) -> dict:
    g = r["guests"]; P = g["pooled"]["by_nationality"]; gr = r["grows_with_lead"]; L = LEADS
    first, rest = P[1], [P[h] for h in L[1:]]
    chart = {"leads": L,
             "country": {f: [round(g[f]["by_nationality"][h], 1) for h in L] for f in "AB"},
             "country_ci": {f: [[round(x, 1) for x in g[f]["by_nationality_ci"][h]] for h in L] for f in "AB"},
             "total": {f: [round(g[f]["total"][h]["abs_error_pct"], 1) for h in L] for f in "AB"},
             "total_signed": {f: [round(g[f]["total"][h]["signed_error_pct"], 1) for h in L] for f in "AB"},
             "coverage": {f: [round(g[f]["coverage"][h] * 100) for h in L] for f in "AB"},
             "months": {f["id"]: [m for m in f["months"]] for f in folds}}
    sl = {f: gr["per_fold"][f] for f in "AB"}; fl = {f: g[f]["first_vs_later"] for f in "AB"}
    hi = {f: max(L, key=lambda h: g[f]["by_nationality"][h]) for f in "AB"}
    return {
        "lead": (f"One country at a time, a month's guest nights is off by {_pc(first)} in the first month after the data and by {_pc(min(rest))} "
                 f"to {_pc(max(rest))} in months 2 to 7 (both periods together). Together, the first month is the most accurate, but the error does not "
                 f"keep growing with distance: it peaks around month 5 and is lower again at months 6 and 7."),
        "chart": chart,
        "points": [
            f"No detectable growth in months 1 to 7. The error rises by {sl['A']['slope']:+.1f} points a month in the first period "
            f"(95% range {sl['A']['ci'][0]:+.1f} to {sl['A']['ci'][1]:+.1f}) and {sl['B']['slope']:+.1f} in the second ({sl['B']['ci'][0]:+.1f} to "
            f"{sl['B']['ci'][1]:+.1f}); both ranges include zero, so a steady rise with distance is not shown in either period.",
            f"Months 2 to 7 average {fl['A']['estimate']:+.1f} (first period) and {fl['B']['estimate']:+.1f} (second) points above month 1; "
            f"{'the ranges exclude zero' if fl['A']['ci'][0] > 0 and fl['B']['ci'][0] > 0 else 'the ranges include zero, so even this step up from month 1 is within the noise'} "
            f"({fl['A']['ci'][0]:+.1f} to {fl['A']['ci'][1]:+.1f} and {fl['B']['ci'][0]:+.1f} to {fl['B']['ci'][1]:+.1f}). "
            + (f"The peak at month {hi['A']} in both periods" if hi["A"] == hi["B"] else f"The peaks at month {hi['A']} (first period) and month {hi['B']} (second)")
            + " may be distance or just which months fell there: two periods cannot tell.",
            "Every test month was forecast from the flights that actually operated, so this is how far the level and the calendar can be carried, "
            "not how well a future schedule is known. Months 8 to 12 are untested: results there are flagged."],
        "method": (f"Monthly guest nights by country, rebuilt against reported, in the two held-out periods; months ahead = months since the model's data ended. "
                   f"The international total has one month per distance in each period, too few to test, so it is shown. "
                   f"Check-ins alone show no growth either (slopes {r['checkins']['A']['slope']:+.1f} and {r['checkins']['B']['slope']:+.1f} points a month, both ranges include zero).")}


def report(r: dict) -> str:
    g, c, gr = r["guests"], r["checkins"], r["grows_with_lead"]
    pk = {f: max(LEADS, key=lambda h: g[f]["by_nationality"][h]) for f in "AB"}
    row = lambda name, d: f"| {name} | " + " | ".join(f"{d[h]:.1f}" for h in LEADS) + " |"
    head = "| | " + " | ".join(f"{h}" for h in LEADS) + " |\n|---|" + "---|" * len(LEADS)
    L = ["# Phase 5, Step 5.7 — accuracy by months ahead", "",
         "Months ahead = months since the training end: months 1–7 of each held-out period (fold A: Aug 2024–Feb 2025; fold B: Jan–Jul 2025). "
         "Rules: `docs/phase5_7_preregistration.md` (sha256 recorded in the results). The monthly guest nights used are the ones Tab 3 shows "
         f"(`backtest_monthly.csv`; largest difference {r['consistency_max_abs_diff']:.0e}). As in every back-test, the forecast is given the seats that actually flew.", "",
         "## By nationality: monthly guest nights (WMAPE %, 45 nationalities)", "", head,
         row("Fold A", g["A"]["by_nationality"]), row("Fold B", g["B"]["by_nationality"]), row("**Both**", g["pooled"]["by_nationality"]), "",
         "95% interval of each fold's value (nationality bootstrap, 2,000 draws): "
         + "; ".join(f"{f} month {h}: {g[f]['by_nationality_ci'][h][0]:.0f}–{g[f]['by_nationality_ci'][h][1]:.0f}" for f in "AB" for h in (1, 4, 7)) + " (months 1, 4, 7 shown; all in the results file).", "",
         "By market-size tier, both periods together:", "", head] + [row(t, g["pooled"]["by_tier"][t]) for t in ("top 10", "11–25", "26–46")] + [
         "", "## International total: monthly guest nights (absolute error %, one month per cell)", "", head,
         row("Fold A", {h: g["A"]["total"][h]["abs_error_pct"] for h in LEADS}), row("Fold B", {h: g["B"]["total"][h]["abs_error_pct"] for h in LEADS}),
         row("Mean of both", r["pooled_total_abs_error_pct"]), "",
         "Signed error (forecast ÷ actual − 1, %):", "", head,
         row("Fold A", {h: g["A"]["total"][h]["signed_error_pct"] for h in LEADS}), row("Fold B", {h: g["B"]["total"][h]["signed_error_pct"] for h in LEADS}), "",
         "One month per distance in each period: shown, not tested. The largest total error is "
         + (lambda f, h: f"fold {f}, month {h} ({r['months'][f][h - 1]}): {g[f]['total'][h]['signed_error_pct']:+.0f}%"
            + (", not at the longest distance." if h < max(LEADS) else ", at the longest distance."))(
             *max(((f, h) for f in "AB" for h in LEADS), key=lambda x: g[x[0]]["total"][x[1]]["abs_error_pct"])), "",
         "## Check-ins alone (Stage A, WMAPE %, 45 nationalities)", "", head,
         row("Fold A", c["A"]["by_nationality"]), row("Fold B", c["B"]["by_nationality"]), row("**Both**", c["pooled"]["by_nationality"]), "",
         "## Does the error grow with lead? (pre-registered rule)", "",
         "| Fold | Slope (points per month ahead) | 95% interval | Above 0? |", "|---|---|---|---|"] + [
         f"| {f} | {gr['per_fold'][f]['slope']:+.2f} | {gr['per_fold'][f]['ci'][0]:+.2f} to {gr['per_fold'][f]['ci'][1]:+.2f} | {'yes' if gr['per_fold'][f]['above_zero'] else 'no'} |" for f in "AB"] + [
         "", f"**{gr['label'][0].upper() + gr['label'][1:]}.** The rule needed both intervals above 0; both include it, and fold B's slope is negative. "
         "Check-ins alone: slopes " + ", ".join(f"{f} {c[f]['slope']:+.2f} ({c[f]['slope_ci'][0]:+.2f} to {c[f]['slope_ci'][1]:+.2f})" for f in "AB") + ".", "",
         "Not a steady rise: pooled, the error is lowest in month 1 "
         f"({g['pooled']['by_nationality'][1]:.1f}%), highest at month {max(LEADS, key=lambda h: g['pooled']['by_nationality'][h])} "
         f"({max(g['pooled']['by_nationality'].values()):.1f}%) and {g['pooled']['by_nationality'][7]:.1f}% at month 7. "
         + (f"Both folds peak at month {pk['A']}, in different calendar months ({r['months']['A'][pk['A'] - 1]} and {r['months']['B'][pk['B'] - 1]}), and "
            if pk["A"] == pk["B"] else
            f"The folds peak at month {pk['A']} (fold A, {r['months']['A'][pk['A'] - 1]}) and month {pk['B']} (fold B, {r['months']['B'][pk['B'] - 1]}), and ")
         + "each fold has one month of each distance, so distance and the particular months cannot be separated. "
         "This is a statement about 1–7 months only.", "",
         "**Added after the first run (Amendment A1; descriptive, not a decision rule):** month 1 is the lowest pooled, so the average of months 2–7 "
         "against month 1, with the same nationality bootstrap: "
         + "; ".join(f"fold {f} {g[f]['first_vs_later']['estimate']:+.1f} points ({g[f]['first_vs_later']['ci'][0]:+.1f} to {g[f]['first_vs_later']['ci'][1]:+.1f})" for f in "AB")
         + (". Both intervals include zero: even the step up from month 1 is within the noise of 45 nationalities."
            if all(g[f]['first_vs_later']['ci'][0] <= 0 for f in "AB") else ". At least one interval excludes zero: the first month is more accurate than later ones."), "",
         "## Coverage of the 80% range by months ahead (descriptive)", "", head,
         row("Fold A", {h: g["A"]["coverage"][h] * 100 for h in LEADS}), row("Fold B", {h: g["B"]["coverage"][h] * 100 for h in LEADS}), "",
         "Share of nationality-months inside their size tier's 80% range learned on the other fold (Step 3.5's cross-fold rule; 45 per cell, so each "
         f"cell moves 2 points per country). Over all markets it is {g['A']['coverage_all_markets'] * 100:.1f}% (fold A) and "
         f"{g['B']['coverage_all_markets'] * 100:.1f}% (fold B), Step 3.5's own figures. By distance: months 1–2 "
         f"{min(g[f]['coverage'][h] for f in 'AB' for h in (1, 2)) * 100:.0f}–{max(g[f]['coverage'][h] for f in 'AB' for h in (1, 2)) * 100:.0f}%, months 3–6 "
         f"{min(g[f]['coverage'][h] for f in 'AB' for h in (3, 4, 5, 6)) * 100:.0f}–{max(g[f]['coverage'][h] for f in 'AB' for h in (3, 4, 5, 6)) * 100:.0f}%, month 7 "
         f"{min(g[f]['coverage'][7] for f in 'AB') * 100:.0f}–{max(g[f]['coverage'][7] for f in 'AB') * 100:.0f}%: the nearest months are covered best, "
         "and the range is a little short in the middle months, not progressively worse.", "",
         "## What follows", "",
         "- The app's flag after 7 months stays where it is: the folds show no growth within 1–7 months, and nothing is known beyond.",
         "- Months 8–12 of the normal year (the last five of the 12-month horizon) remain an untested limitation (Step 5.9).",
         "- Tab 3 shows the chart and the numbers behind it.", "",
         "**Deck line:** " + deck_line(r), ""]
    return "\n".join(L)


def deck_line(r: dict) -> str:
    g = r["guests"]; P = g["pooled"]["by_nationality"]
    tot = {f: sum(abs(x["forecast"] - x["actual"]) for x in g[f]["total"].values()) / sum(x["actual"] for x in g[f]["total"].values()) * 100 for f in "AB"}
    worst = max(x["abs_error_pct"] for f in "AB" for x in g[f]["total"].values())
    return (f"By months ahead, one country at a time is off {_pc(P[1])} in month 1 and {_pc(min(P[h] for h in LEADS[1:]))}–"
            f"{_pc(max(P[h] for h in LEADS[1:]))} in months 2–7, peaking mid-range, with no detectable growth in distance; the international total "
            f"is off {_pc(tot['A'], 1)} and {_pc(tot['B'], 1)} on average in the two periods, at most {_pc(worst)} in any month. Months 8–12 are untested and flagged.")
