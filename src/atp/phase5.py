"""Phase 5 — validation audit. Step 5.1: the snapshot; Step 5.2: comparing a clean rebuild with it; Step 5.3: comparing an
independent reviewer's re-derivation with our numbers.

Everything later in Phase 5 (the clean rebuild, the independent review, the benchmarks) is compared against one fixed
record of what the project says today:

  numbers    every headline number with its source: the 57 facts the app shows (phase4 FACTS), the further numbers the
             reports, the technical brief and the deck quote (EXTRA, read from the saved Phase 1–3 outputs the same way),
             and the simulator's own answers for the cases the deck shows (presets at each check-in response, the seats
             lever, the reverse sum), computed live
  files      sha256 and size of every input, output and code file: raw data (also checked against config.yaml),
             reference and external data, every processed table, config, requirements, src/ and web/

Run:  python -m atp.phase5 snapshot    write data/processed/phase5/snapshot.json and reports/phase5_snapshot.md
      python -m atp.phase5 check       recompute everything and list what differs (exit 1 if a number or a data file did)
      python -m atp.phase5 compare DIR  Step 5.2: a rebuilt copy of the project (scripts/clean_rebuild.sh) against this snapshot
      python -m atp.phase5 review-inputs  Step 5.3: write review/inputs/ (raw files, dictionary, SCAD, forecasts without actuals)
      python -m atp.phase5 candidates   Step 5.4: three candidate improvements (reports/phase5_candidates.md)
      python -m atp.phase5 benchmark    Step 5.5: LightGBM benchmark against the check-in forecaster (reports/phase5_benchmark.md)
      python -m atp.phase5 limits       Step 5.9: the one limitations list (Tab 3, README block, reports/limitations.md)
      python -m atp.phase5 ahead        Step 5.7: accuracy by months ahead (reports/phase5_ahead.md; Tab 3 reads phase5/ahead.json)
      python -m atp.phase5 sobol        Step 5.6: Sobol indices on the engine (reports/phase5_sobol.md; Tab 3 reads phase5/sobol.json)
      python -m atp.phase5 report       Step 5.10: the Phase 5 report from the saved results (reports/phase5_report.md)
      python -m atp.phase5 review       Step 5.3: the independent reviewer's review/results.json against our outputs
                                        (reports/phase5_review.md); `make review` reruns the reviewer's own code first

A changed number, or a changed or missing data file, is a difference. Code that changed and files that were added are
listed but are not differences on their own: what matters is whether the numbers moved.
"""
import json
import math
import sys
from datetime import date
from pathlib import Path
from .config import load_config
from .phase4 import FACTS, read_fact, sha256, _j, _c

OUT = "phase5"
EL = "phase2/elasticities.csv"
P1 = "phase1/validation_metrics.csv"
E2E = "phase2/step26_endtoend.json"
SENS = "phase3/sens_levers.csv"
BEYOND = ("beyond_range_replay", "groups")


def _el(param, col, label):
    return _c(EL, {"fold": "final", "parameter": param}, col, label=label)


def _p1(fold, model, col, label):
    return _c(P1, {"fold": fold, "model": model}, col, label=f"{label}, period {fold}")


EXTRA = {
    # the seat lever, as estimated (Phase 2)
    "eps_lead_final": _el("eps_total_lead", "estimate", "ε, lead-controlled spec, final window (the default: same number as eps)"),
    "eps_plain_final": _el("eps_total_plain", "estimate", "ε, plain spec, final window"),
    "eps_ci95_lo": _el("eps_total_default", "ci_lo", "ε default, in-sample 95% interval, low (country bootstrap)"),
    "eps_ci95_hi": _el("eps_total_default", "ci_hi", "ε default, in-sample 95% interval, high"),
    "eps_n_obs": _el("eps_total_default", "n_obs", "country-months in the ε sample"),
    "eps_first_month": _el("eps_total_default", "first_month", "first year-on-year month in the ε sample"),
    "eps_last_month": _el("eps_total_default", "last_month", "last year-on-year month in the ε sample"),
    "eps1_final": _el("eps1", "estimate", "ε₁: seats → local passengers"),
    "eps_fold_A": _j("phase2/step24_forecaster.json", "eps_used_in_folds", "A", label="ε used in period A back-tests (that fold's default: plain spec)"),
    "eps_fold_B": _j("phase2/step24_forecaster.json", "eps_used_in_folds", "B", label="ε used in period B back-tests (that fold's default: lead-controlled spec)"),
    "eps_loo_min": _j("phase2/step22_checkins.json", "loo_min", 1, label="ε leaving one country out, lowest"),
    "eps_loo_max": _j("phase2/step22_checkins.json", "loo_max", 1, label="ε leaving one country out, highest"),
    "eps_increases": _j("phase2/step22_checkins.json", "asymmetry", "increases", label="ε on seat increases"),
    "eps_cuts": _j("phase2/step22_checkins.json", "asymmetry", "cuts", label="ε on seat cuts"),
    "split_diff_lo": _j("phase2/step22_checkins.json", "split_difference", "ci_boot", 0, label="ε, tourist-heavy minus resident-heavy countries, 95% low (stratified bootstrap)"),
    "split_diff_hi": _j("phase2/step22_checkins.json", "split_difference", "ci_boot", 1, label="ε, tourist-heavy minus resident-heavy countries, 95% high (stratified bootstrap)"),
    # the survival model (Phase 1), daily × nationality WMAPE % given actual check-ins
    **{f"stageb_{m}_{f}": _p1(f, model, "intl_daily_x_nationality", lab)
       for f in ("A", "B") for m, model, lab in [("final", "base+market_step", "survival model, guests given actual check-ins, daily × nationality WMAPE %"),
                                                 ("stay_ratio", "stay_ratio", "check-ins × monthly stay ratio, guests given actual check-ins, daily × nationality WMAPE %"),
                                                 ("fixed_curve", "fixed_curve", "fixed stay curve, guests given actual check-ins, daily × nationality WMAPE %"),
                                                 ("seasonal_naive", "seasonal_naive", "same day last year (guests), daily × nationality WMAPE %")]},
    "stageb_grad_intl": _j("phase1/final_summary.json", "final_grad_rel", "intl", label="survival model fit, relative gradient, international"),
    "stageb_grad_dom": _j("phase1/final_summary.json", "final_grad_rel", "dom", label="survival model fit, relative gradient, Domestic"),
    # end to end (Phase 2): worst month, the daily tie, error by months ahead, the official test window
    **{f"worst_month_bias_{f}": _j(E2E, "folds", f, "end_to_end", "intl_worst_month_bias", label=f"largest absolute monthly bias %, international total, end-to-end guests, period {f}")
       for f in ("A", "B")},
    **{f"daily_gap_vs_last_year_{f}": _j(E2E, "folds", f, "gap_vs_same_day_last_year_daily_nat", "points",
                                         label=f"daily × nationality gain over same day last year (points), period {f}") for f in ("A", "B")},
    **{f"e2e_lead_{f}_{m}": _j(E2E, "e2e_wmape_by_months_ahead", f, str(m), label=f"end-to-end daily × nationality WMAPE %, month {m} ahead, period {f}")
       for f in ("A", "B") for m in range(1, 8)},
    **{f"test_{k}": _j(E2E, "test_window", "stagea_scores", "model", k, label=f"official test window, check-ins WMAPE %, {lab}")
       for k, lab in [("intl_daily_nat", "daily × nationality"), ("intl_monthly_total", "monthly total"), ("dom_daily", "Domestic daily")]},
    **{f"test_last_year_{k}": _j(E2E, "test_window", "stagea_scores", "benchmark: same day last year", k,
                                 label=f"official test window, same day last year WMAPE %, {lab}")
       for k, lab in [("intl_daily_nat", "daily × nationality"), ("intl_monthly_total", "monthly total"), ("dom_daily", "Domestic daily")]},
    "test_bias_intl": _j(E2E, "test_window", "bias_pct", "intl", label="official test window, international check-ins bias % (forecast vs actual; negative = under)"),
    # ranges (Phase 3)
    "own_market_cov_A_to_B": _j("phase3/step35_bands.json", "level_diagnostics", "guests", "own_market_bands_coverage_A_to_B",
                                label="per-market band coverage, learned on A, tested on B"),
    "own_market_cov_B_to_A": _j("phase3/step35_bands.json", "level_diagnostics", "guests", "own_market_bands_coverage_B_to_A",
                                label="per-market band coverage, learned on B, tested on A"),
    "total_cov_A_to_B": _j("phase3/step35_bands.json", "level_diagnostics", "guests", "total_coverage_A_to_B",
                           label="all-market monthly total: share of period-B months inside period A's min–max range (no 80% band achievable)"),
    # sensitivity (Phase 3)
    **{f"elasticity_{k}": _c(SENS, {"lever": lever, "change_pct": 10.0}, "elasticity_small_change", label=f"elasticity at +1% (guest nights % per 1% lever change), {lab}")
       for k, lever, lab in [("seats", "seats", "seats"), ("load_factor", "load factor", "how full planes are"),
                             ("transfer", "transfer share", "connecting share"), ("stay", "stay length", "stay length"),
                             ("demand", "non-flight demand", "demand not caused by flights")]},
    "tornado_central": _j("phase3/step36_sensitivity.json", "uncertainty_tornado", "central", label="+10% network seats, Nov 2026–Feb 2027: extra guest nights (central)"),
    **{f"tornado_{k}_{e}": _j("phase3/step36_sensitivity.json", "uncertainty_tornado", key, i, label=f"+10% network seats, Nov 2026–Feb 2027: 80% {lab} range, {e}")
       for k, key, lab in [("lever", "lever_test_window_eps", "lever"), ("level", "baseline_level", "baseline level")]
       for i, e in enumerate(("lo", "hi"))},
    # validation (Phase 3)
    "deep_cuts_test_n": _j("phase3/step37_validation.json", *BEYOND, 0, "n", label="seat cuts beyond −29% in the test window, country-months"),
    "deep_cuts_test_actual": _j("phase3/step37_validation.json", *BEYOND, 0, "actual_mean_pct", label="seat cuts beyond −29%, test window: check-ins change % (YoY, relative to other countries, festival-adjusted)"),
    "deep_cuts_test_line": _j("phase3/step37_validation.json", *BEYOND, 0, "line", "pred_mean_pct", label="seat cuts beyond −29%, test window: straight-line prediction %"),
    "deep_cuts_test_log": _j("phase3/step37_validation.json", *BEYOND, 0, "log", "pred_mean_pct", label="seat cuts beyond −29%, test window: log-form prediction %"),
    "deep_cuts_train_actual": _j("phase3/step37_validation.json", *BEYOND, 4, "actual_mean_pct", label="seat cuts beyond −29%, training months without war: check-ins change % (relative, festival-adjusted)"),
    "march_seats_lo": _j("phase3/step37_validation.json", "march_2026", "seats_vs_normal", "low", label="March 2026 seats vs normal, low"),
    "march_seats_hi": _j("phase3/step37_validation.json", "march_2026", "seats_vs_normal", "high", label="March 2026 seats vs normal, high"),
    "march_share_lever_lo": _j("phase3/step37_validation.json", "march_2026", "total", "share_of_drop_explained_by_flights_lever_range", 0,
                               label="March 2026 share explained by flights, across ε's 80% range and both seat assumptions, low"),
    "march_share_lever_hi": _j("phase3/step37_validation.json", "march_2026", "total", "share_of_drop_explained_by_flights_lever_range", 1,
                               label="March 2026 share explained by flights, across ε's 80% range and both seat assumptions, high"),
    "engine_properties_passed": _j("phase3/step37_validation.json", "properties", "all_passed", label="200 random scenarios pass every engine rule"),
}


def simulator_numbers() -> dict:
    """The simulator's own answers for the cases the deck and the brief quote, computed live."""
    from .simulator import Simulator, RESPONSES
    S = Simulator(); out = {}
    for pid in ("etihad", "europe", "march", "new_route"):
        for resp in RESPONSES:
            h = S.run({**S.preset(pid), "response": resp})["headline"]
            for k in ("d_guest_nights", "lo", "hi", "d_checkins"):
                out[f"preset_{pid}_{resp}_{k}"] = {"value": h[k], "label": f"preset {pid}, {resp} check-in response: " + {"d_guest_nights": "extra guest nights", "lo": "guest nights, 80% range low", "hi": "guest nights, 80% range high", "d_checkins": "extra check-ins"}[k],
                                                   "source": {"kind": "simulator", "call": f"run(preset {pid!r}, response {resp!r})"}}
    cv = S.lever("UNITED KINGDOM", "winter")
    for name, i, what in (("cancel_all", 0, "all flights cancelled (beyond the tested range)"),
                          ("plus_5", cv["today_index"] + 5, "+5 flights a week"),
                          ("max", len(cv["steps"]) - 1, "+100 flights a week, about triple today (beyond the tested range)")):
        out[f"lever_uk_winter_{name}"] = {"value": cv["steps"][i]["headline"]["d_guest_nights"],
                                          "label": f"seats lever, United Kingdom, Nov 2026–Feb 2027, {what}: extra guest nights",
                                          "source": {"kind": "simulator", "call": "lever('UNITED KINGDOM', 'winter')"}}
    for slug, market, aircraft in (("uk", "UNITED KINGDOM", "today"), ("mexico", "MEXICO", "narrowbody")):
        r = S.reverse(market, "2026-11-01", "2027-02-28", 10000, aircraft)
        out[f"reverse_{slug}_{aircraft}"] = {
            "value": r["flights_per_week"], "label": f"flights a week for +10,000 guest nights from {market.title()}, winter, {aircraft}",
            "source": {"kind": "simulator", "call": f"reverse({market!r}, winter, 10000, {aircraft!r})"}}
    return out


def numbers(cfg) -> dict:
    P = cfg["paths"]["processed"]
    out = {k: {"value": read_fact(P, s), "label": s["label"], "source": s} for k, s in {**FACTS, **EXTRA}.items()}
    return {**out, **simulator_numbers()}


# what the hashes cover: data that must not change silently, and the code that produced it
DATA_DIRS = ("data/raw", "data/reference", "data/external", "data/processed")
CODE = ("config.yaml", "requirements.txt", "requirements-app.txt", "Dockerfile", "src", "web")


def _walk(root: Path, rel: str):
    p = root / rel
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
    return [f for f in files if "__pycache__" not in f.parts and not f.name.startswith(".")
            and f"data/processed/{OUT}" not in f.relative_to(root).as_posix()]


def files(cfg) -> dict:
    root = cfg["root"]; out = {}
    for group, rels in (("data", DATA_DIRS), ("code", CODE)):
        for rel in rels:
            for f in _walk(root, rel):
                out[f.relative_to(root).as_posix()] = {"group": group, "sha256": sha256(f), "bytes": f.stat().st_size}
    raw = cfg["raw_sha256"]                                                         # the raw files are the ones the config pins
    for name, want in raw.items():
        got = next((v["sha256"] for k, v in out.items() if k.startswith("data/raw/") and Path(k).name == Path(name).name), None)
        assert got == want, f"raw file {name}: sha256 {str(got)[:12]}… is not the pinned {want[:12]}…"
    return out


# ----------------------------------------------------------------------------- compare
def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def compare(old: dict, new: dict) -> dict:
    """What differs between two snapshots: numbers (changed, missing, added) and files by group."""
    on, nn = old["numbers"], new["numbers"]
    d = {"numbers_changed": {k: [on[k]["value"], nn[k]["value"]] for k in on.keys() & nn.keys() if not _same(on[k]["value"], nn[k]["value"])},
         "numbers_missing": sorted(on.keys() - nn.keys()), "numbers_added": sorted(nn.keys() - on.keys())}
    of, nf = old["files"], new["files"]
    for g in ("data", "code"):
        d[f"{g}_changed"] = sorted(k for k in of.keys() & nf.keys() if of[k]["group"] == g and of[k]["sha256"] != nf[k]["sha256"])
        d[f"{g}_missing"] = sorted(k for k in of.keys() - nf.keys() if of[k]["group"] == g)
        d[f"{g}_added"] = sorted(k for k in nf.keys() - of.keys() if nf[k]["group"] == g)
    d["ok"] = not (d["numbers_changed"] or d["numbers_missing"] or d["data_changed"] or d["data_missing"])
    return d


def take(cfg) -> dict:
    return {"note": "Phase 5.1 snapshot: every headline number with its source, and sha256 of every input, output and code "
                    "file. `python -m atp.phase5 check` recomputes and compares.", "date": date.today().isoformat(),
            "numbers": numbers(cfg), "files": files(cfg)}


def _fmt(v):
    if isinstance(v, bool) or not isinstance(v, float):
        return str(v)
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def report(snap: dict) -> str:
    n, f = snap["numbers"], snap["files"]
    src = lambda s: (s["call"] if s["kind"] == "simulator" else s["file"] + (f" → {' / '.join(map(str, s['path']))}" if s["kind"] == "json"
                     else f" → {s['column']}" + (f" where {s['where']}" if s.get("where") else "") + (f" ({s['agg']})" if s["kind"] == "agg" else "")))
    rows = [f"| `{k}` | {_fmt(v['value'])} | {v['label']} | {src(v['source'])} |" for k, v in n.items()]
    counts = {g: sum(1 for x in f.values() if x["group"] == g) for g in ("data", "code")}
    return "\n".join([
        "# Phase 5.1 — snapshot", "",
        f"_Generated by `python -m atp.phase5 snapshot` on {snap['date']}; do not edit by hand._", "",
        f"{len(n)} numbers ({len(FACTS)} the app shows, {len(EXTRA)} the reports and brief quote, "
        f"{len(n) - len(FACTS) - len(EXTRA)} computed live by the simulator) and the sha256 of {counts['data']} data files and "
        f"{counts['code']} code files, in `data/processed/{OUT}/snapshot.json`. `python -m atp.phase5 check` recomputes everything "
        "and lists what differs.", "",
        "| Key | Value | What it is | Source |", "|---|---|---|---|", *rows, ""])


# ----------------------------------------------------------------------------- 5.2: a rebuilt copy against the snapshot
REBUILD_TOL = 1e-6          # relative: a rebuilt number or table value within this is "reproduced" (floating-point noise)


def _rel(a, b) -> float:
    """Relative difference of two values (0 when equal; inf when they differ in type or one is missing)."""
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return 0.0 if a == b else math.inf
    if a == b or (a != a and b != b) or abs(a - b) <= 1e-12:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def _max_rel_json(a, b) -> float:
    if isinstance(a, dict) and isinstance(b, dict):                           # a file's sha256 is checked through the file itself
        return max([_max_rel_json(a[k], b[k]) for k in a.keys() & b.keys() if k != "sha256"]
                   + [math.inf if a.keys() != b.keys() else 0.0])
    if isinstance(a, list) and isinstance(b, list):
        return max([_max_rel_json(x, y) for x, y in zip(a, b)] + [math.inf if len(a) != len(b) else 0.0])
    return _rel(a, b)


def _max_rel_table(fa: Path, fb: Path) -> float:
    """Largest relative difference between two tables or arrays with the same layout (inf if the layout differs)."""
    import numpy as np, pandas as pd
    if fa.suffix in (".npy", ".npz"):
        A, B = np.load(fa, allow_pickle=False), np.load(fb, allow_pickle=False)
        pairs = [(A[k], B[k]) for k in A.files] if fa.suffix == ".npz" else [(A, B)]
        if fa.suffix == ".npz" and sorted(A.files) != sorted(B.files):
            return math.inf
    else:
        A, B = (pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in (fa, fb))
        if list(A.columns) != list(B.columns) or len(A) != len(B):
            return math.inf
        pairs = [(A[c].to_numpy(), B[c].to_numpy()) for c in A.columns]
    worst = 0.0
    for x, y in pairs:
        if x.shape != y.shape:
            return math.inf
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number) and x.dtype != bool:
            x, y = x.astype(float), y.astype(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                return math.inf
            ok = ~np.isnan(x)
            d = np.abs(x[ok] - y[ok]) / np.maximum(np.maximum(np.abs(x[ok]), np.abs(y[ok])), 1e-12)
            worst = max(worst, float(d.max()) if d.size else 0.0)
        elif not pd.Series(x).astype(str).equals(pd.Series(y).astype(str)):
            return math.inf
    return worst


HANDWRITTEN = {"audit_response.md"}          # reports written by hand, not by a pipeline step: nothing to rebuild


def compare_rebuild(src: Path, dest: Path) -> dict:
    """The rebuilt copy at `dest` against the project at `src`: every snapshot number, every data file, and the
    generated reports. Data files that differ in bytes are compared by content (tables, arrays, json)."""
    a = json.load(open(src / "data/processed" / OUT / "snapshot.json", encoding="utf-8"))
    b = json.load(open(dest / "data/processed" / OUT / "snapshot.json", encoding="utf-8"))
    nums = {k: _rel(a["numbers"][k]["value"], b["numbers"][k]["value"]) for k in a["numbers"].keys() & b["numbers"].keys()}
    out = {"numbers": len(nums), "numbers_identical": sum(v <= 1e-9 for v in nums.values()),
           "numbers_max_rel": max(nums.values(), default=0.0),
           "numbers_off": {k: [a["numbers"][k]["value"], b["numbers"][k]["value"]] for k, v in nums.items() if v > REBUILD_TOL},
           "numbers_missing": sorted(a["numbers"].keys() - b["numbers"].keys())}
    fa = {k: v for k, v in a["files"].items() if v["group"] == "data"}
    fb = {k: v for k, v in b["files"].items() if v["group"] == "data"}
    files = {"identical": [], "same_content": [], "off": {}, "missing": sorted(fa.keys() - fb.keys()), "extra": sorted(fb.keys() - fa.keys())}
    for k in sorted(fa.keys() & fb.keys()):
        if fa[k]["sha256"] == fb[k]["sha256"]:
            files["identical"].append(k); continue
        p, q = src / k, dest / k
        try:
            if p.suffix == ".json":
                r = _max_rel_json(json.load(open(p, encoding="utf-8")), json.load(open(q, encoding="utf-8")))
            elif p.suffix in (".parquet", ".csv", ".npy", ".npz"):
                r = _max_rel_table(p, q)
            else:
                r = math.inf if p.read_bytes() != q.read_bytes() else 0.0
        except Exception as e:                                                    # unreadable = different
            r = math.inf; print(f"  {k}: {e}")
        (files["same_content"].append(k) if r <= REBUILD_TOL else files["off"].__setitem__(k, r))
        if r <= REBUILD_TOL and r > 0:
            files.setdefault("max_rel", 0.0); files["max_rel"] = max(files["max_rel"], r)
    out["files"] = files
    reps = {}
    for r in sorted((src / "reports").glob("*.md")):
        if r.name.startswith("phase5") or r.name in HANDWRITTEN:
            continue
        q = dest / "reports" / r.name
        reps[r.name] = "missing" if not q.exists() else ("identical" if q.read_text(encoding="utf-8") == r.read_text(encoding="utf-8") else "differs")
    out["reports"] = reps
    out["ok"] = not (out["numbers_off"] or out["numbers_missing"] or files["off"] or files["missing"])
    return out


def rebuild_report(d: dict, dest: Path) -> str:
    f = d["files"]; g = lambda x: "∞" if x == math.inf else f"{x:.1e}"
    lines = ["# Phase 5.2 — clean rebuild", "",
             "_Generated by `scripts/clean_rebuild.sh` (`python -m atp.phase5 compare`); do not edit by hand._", "",
             f"The project was rebuilt from the five raw competition files and the reference/external inputs alone, in a fresh "
             f"folder with a fresh environment from the pinned `requirements.txt`, by running every phase and the validation "
             f"steps the app reads (`make data phase1 phase2 phase3 phase4data sobol ahead benchmark audit limits`). It was then compared with the 5.1 snapshot.", "",
             f"**Verdict: {'reproduced' if d['ok'] else 'NOT reproduced'}.** "
             f"A value counts as reproduced when it is within a relative {REBUILD_TOL:g} of the original (floating-point noise).", "",
             "| What | Result |", "|---|---|",
             f"| Headline numbers | {d['numbers_identical']} of {d['numbers']} identical; largest relative difference {g(d['numbers_max_rel'])}; "
             f"{len(d['numbers_off'])} beyond tolerance; {len(d['numbers_missing'])} missing |",
             f"| Data files | {len(f['identical'])} byte-identical; {len(f['same_content'])} same content (different bytes"
             + (f", largest relative difference {g(f['max_rel'])}" if f.get("max_rel") else "") + f"); {len(f['off'])} different; "
             f"{len(f['missing'])} missing; {len(f['extra'])} extra |",
             f"| Generated reports | " + "; ".join(f"`{k}` {v}" for k, v in d["reports"].items()) + " |", ""]
    if d["numbers_off"]:
        lines += ["## Numbers beyond tolerance", "", "| Key | Original | Rebuilt |", "|---|---|---|",
                  *[f"| `{k}` | {a} | {b} |" for k, (a, b) in d["numbers_off"].items()], ""]
    if f["off"] or f["missing"]:
        lines += ["## Data files that differ", "", *[f"- `{k}`: largest relative difference {g(v)}" for k, v in f["off"].items()],
                  *[f"- `{k}`: missing from the rebuild" for k in f["missing"]], ""]
    if f["same_content"]:
        lines += ["## Same content, different bytes", "", "These files hold the same values but are not byte-identical "
                  "(file metadata such as write order or library version stamps).", "",
                  *[f"- `{k}`" for k in f["same_content"]], ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- 5.3: the independent review
REVIEW = "review"           # the reviewer's brief (claims.md), code (work/), results.json and notes.md; inputs/ is rebuilt
APPROX = {"eps_ci95_lo": 0.03, "eps_ci95_hi": 0.03}         # bootstrap ends: other random draws (absolute tolerance)
APART = {k: "aircraft" for k in ("narrowbody_seats", "widebody_seats", "narrowbody_routes", "widebody_routes")}


def review_inputs(cfg, dest=None) -> Path:
    """The reviewer's inputs: the raw files, the data dictionary, the SCAD workbook, the scored test-window forecast and
    the back-test guest predictions WITHOUT their actuals (the reviewer rebuilds every actual from the raw files)."""
    import shutil
    import pandas as pd
    from .phase4 import SCAD_XLSX
    root, P = cfg["root"], cfg["paths"]["processed"]
    d = Path(dest or root / REVIEW / "inputs"); (d / "raw").mkdir(parents=True, exist_ok=True)
    for name in cfg["raw_files"].values():
        shutil.copy2(cfg["paths"]["raw"] / name, d / "raw" / name)
    shutil.copy2(root / "docs" / "Data_Dictionary.pdf", d)
    shutil.copy2(cfg["paths"]["external"] / SCAD_XLSX, d)
    ref = json.load(open(cfg["paths"]["reference"] / "frozen_test_forecast.json", encoding="utf-8"))
    shutil.copy2(root / ref["file"], d / "forecast_test_window.csv")
    intl = json.load(open(P / "phase1" / "final_summary.json", encoding="utf-8"))["selection"]["intl_model"]
    for f in "AB":
        e = pd.read_parquet(P / "phase2" / f"endtoend_pred_{f}.parquet")
        e[["market", "date", "p"]].rename(columns={"p": "pred_guests"}).to_csv(d / f"backtest_endtoend_{f}.csv", index=False)
        s = pd.read_parquet(P / "phase1" / f"pred_{f}_{intl}.parquet")
        s[s.market != "DOMESTIC"][["market", "date", "p"]].rename(columns={"p": "pred_guests"}).to_csv(d / f"backtest_stageb_{f}.csv", index=False)
    return d


def ours_for_review(cfg) -> dict:
    """Our value of every quantity in review/claims.md, defined as claims.md defines it, from the saved outputs."""
    import numpy as np
    import pandas as pd
    from . import stagea as sa
    P, F = cfg["paths"]["processed"], cfg["folds"]
    v = {k: read_fact(P, s) for k, s in {**FACTS, **EXTRA}.items()}
    hi, hd = pd.read_parquet(P / "hotel_intl_daily.parquet"), pd.read_parquet(P / "hotel_dom_daily.parquet")
    fd, dq = pd.read_parquet(P / "flights_country_daily.parquet"), pd.read_parquet(P / "flights_day_quality.parquet")
    st = hi.arrivals_status.value_counts()
    el = pd.read_csv(P / EL).set_index(["fold", "parameter"])
    j22, j25 = (json.load(open(P / "phase2" / f, encoding="utf-8")) for f in ("step22_checkins.json", "step25_new_route.json"))
    j35 = json.load(open(P / "phase3" / "step35_bands.json", encoding="utf-8"))["level_diagnostics"]["guests"]
    ac = pd.read_csv(P / "phase3" / "aircraft_seats.csv").set_index("aircraft_class")
    sm = pd.read_csv(P / "phase4" / "scad_monthly.csv").dct_over_scad.dropna()      # claims.md: every month with both
    cl = sa.closure_days(fd, "2024-08-01", cfg["dates"]["test_end"])
    cd = pd.Series(cl.get_level_values(1), index=cl.get_level_values(0))
    runs = {k: [str(g.min().date()), str(g.max().date())] for k, g in cd.groupby(level=0)}
    assert all(len(g) == (g.max() - g.min()).days + 1 for _, g in cd.groupby(level=0)), "one closure run per country"
    v |= {"fill_value": float(hi.loc[hi.arrivals_status.str.startswith("absent"), "arrivals"].iloc[0]),
          "n_observed": st["observed"], "n_blank_train": st["star_zero_train"], "n_absent_train": st["absent_train_lt10_guests"],
          "n_absent_test": st["absent_test_lt10_arrivals"], "n_blank_test": st["star_hidden_test"], "n_target_rows": hi.is_target.sum(),
          "n_noflight_nationalities": len(set(hi.nationality) - set(fd.nationality_key)), "n_flight_countries": fd.nationality_key.nunique(),
          "low_coverage_days": sorted(str(x.date()) for x in dq.date[dq.is_low_coverage]), "closure_days": runs,
          "eps_plain_n": el.loc[("final", "eps_total_plain"), "n_obs"],
          "trusted_lo_log": float(np.log1p(v["trusted_lo"])), "trusted_hi_log": float(np.log1p(v["trusted_hi"])),
          "eps_loo_min_country": j22["loo_min"][0], "eps_loo_max_country": j22["loo_max"][0],
          "ratio_median": j25["ratio_median"], "local_per_seat": j25["cap_local_pax_per_seat"],
          "narrowbody_routes": ac.routes["narrowbody"], "widebody_routes": ac.routes["widebody"],
          "q_market_month": j35["q_market_month"], "q_total": j35["q_total"], "total_cov_B_to_A": j35["total_coverage_B_to_A"],
          "rooms_month": pd.read_csv(P / "phase4" / "occupancy_basis.csv").rooms_source_month.iloc[0],
          "dct_over_scad_lo": sm.min(), "dct_over_scad_hi": sm.max(), "dct_over_scad_mean": sm.mean(), "dct_over_scad_n": len(sm)}
    for f in "AB":
        e = pd.read_parquet(P / "phase2" / f"endtoend_pred_{f}.parquet")
        h = F[f]; ho = lambda d: d[(d.date >= h["holdout_start"]) & (d.date <= h["holdout_end"])]
        targets = int(ho(hi).is_target.sum() + ho(hd).guests.notna().sum())
        v |= {f"eps_fold_{f}_spec": "plain" if el.loc[(f, "eps_total_default"), "spec"] == "eps_total_plain" else "lead",
              f"eps_fold_{f}_n": el.loc[(f, "eps_total_default"), "n_obs"],
              f"rows_matched_{f}": int(e.y.notna().sum()), f"rows_pred_without_actual_{f}": int(e.y.isna().sum()),
              f"rows_actual_without_pred_{f}": targets - int(e.y.notna().sum()),
              f"benchmark_{f}_daily_market": v[f"stageb_seasonal_naive_{f}"]}
    return {k: (x.item() if isinstance(x, np.generic) else x) for k, x in v.items()}


def _review_status(k, a, b):
    """'same', 'random draws' (bootstrap ends within APPROX), 'by definition' (APART: explained in the report) or 'DIFFERENT'."""
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    if num(a) and num(b):
        if k in APPROX:
            return "random draws" if abs(a - b) <= APPROX[k] else "DIFFERENT"
        same = _rel(float(a), float(b)) <= REBUILD_TOL
    elif isinstance(a, str) and isinstance(b, str) and a[:4].isdigit() and b[:4].isdigit():    # dates: 2024-01 = 2024-01-01
        n = min(len(a), len(b)); same = n >= 7 and a[:n] == b[:n]
    else:
        same = json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    return "same" if same else ("by definition" if k in APART else "DIFFERENT")


def compare_review(ours: dict, theirs: dict) -> dict:
    rows = {k: {"ours": ours.get(k), "reviewer": x, "status": _review_status(k, ours[k], x) if k in ours else "NOT OURS"}
            for k, x in theirs.items()}
    n = {s: sum(r["status"] == s for r in rows.values()) for s in ("same", "random draws", "by definition", "DIFFERENT", "NOT OURS")}
    return {"rows": rows, "counts": n, "ok": n["DIFFERENT"] == n["NOT OURS"] == 0}


def _aircraft_explained(cfg) -> dict:
    """Route by route: the reviewer's seats per flight (all routes, raw months) against ours (the baseline year's routes)."""
    import pandas as pd
    alt = cfg["root"] / REVIEW / "work" / "alt_results.json"
    theirs = pd.Series(json.load(open(alt, encoding="utf-8"))["q8_route_values"])
    rb = pd.read_csv(cfg["paths"]["processed"] / "phase3" / "route_baseline.csv")
    e = rb[rb.airline == "Etihad Airways"]
    g = e[e.n_flights > 0].groupby(["nationality_key", "city"])[["seats", "n_flights"]].sum()
    ours = pd.Series((g.seats / g.n_flights).to_numpy(), index=[f"{a}|{b}" for a, b in g.index])
    status = e.groupby(["nationality_key", "city"]).status.first(); status.index = [f"{a}|{b}" for a, b in status.index]
    d = (theirs.reindex(ours.index) - ours).abs()
    extra = sorted(set(theirs.index) - set(ours.index))
    return {"shared": len(ours), "identical": int((d <= 1e-6).sum()), "max_diff_other": float(d[d > 1e-6].max()),
            "other": {k: status[k] for k in d[d > 1e-6].index}, "only_reviewer": {k: [status.get(k), float(theirs[k])] for k in extra}}


def _questioned(cfg) -> dict:
    """The numbers behind the answers to what the reviewer questioned (all read or recomputed, none typed)."""
    import numpy as np
    from . import phase2 as p2, stagea as sa
    P, F = cfg["paths"]["processed"], cfg["folds"]
    fd, _, hi = p2._load(cfg)
    m, _, _ = p2._panel_with_festivals(cfg, fd, hi)
    tr, te_end = F["final"]["train_end"], cfg["dates"]["test_end"]
    p_tr, rng_ = p2._primary(m, tr)
    g = sa.fe_elasticity(p_tr, "d_checkins", ["d_seats"] + p2.FEST, n_boot=0)["other_coefs"]
    full, _ = p2._window(m, tr)
    untrim = {k: sa.fe_elasticity(full, "d_checkins", x, n_boot=0)["estimate"] for k, x in (("plain", ["d_seats"]), ("lead", p2.LEADS))}
    el = json.load(open(P / "phase3" / "effect_band.json", encoding="utf-8"))
    cand = {"default": el["eps"], "plain (upper)": read_fact(P, _el("eps_total_plain", "estimate", "")),
            "untrimmed lead": untrim["lead"], "untrimmed plain": untrim["plain"]}
    skill = {}
    for lab, rows in (("inside the trusted range", p2._test_rows(m, tr, te_end, rng_)), ("all rows", p2._test_rows(m, tr, te_end))):
        t = rows.dropna(subset=["d_seats", "d_checkins"])
        t = t.assign(d_checkins=t.d_checkins - g["d_festival_days"] * t.d_festival_days - g["d_ramadan_days"] * t.d_ramadan_days)
        prof = sa.lever_skill(t, {f"{e:.2f}": e for e in np.round(np.arange(0, 1.001, 0.01), 2)}, n_boot=0)["skill"]
        skill[lab] = {"n": len(t), "skill": sa.lever_skill(t, cand, n_boot=0)["skill"], "best": float(max(prof, key=prof.get))}
    j25 = json.load(open(P / "phase2" / "step25_new_route.json", encoding="utf-8"))
    f80 = j25["loo"][j25["method"]]["p80_error_factor"]; mrg = j25["checkins_per_added_seat"]
    lo_f, hi_f = np.log(el["eps"] / el["eps_lo"]), np.log(el["eps_hi"] / el["eps"])
    nr = [mrg * np.exp(-np.hypot(np.log(f80), lo_f)), min(mrg * np.exp(np.hypot(np.log(f80), hi_f)), j25["cap_local_pax_per_seat"])]
    s26 = json.load(open(P / "phase2" / "step26_endtoend.json", encoding="utf-8"))["test_window"]
    sm = __import__("pandas").read_csv(P / "phase4" / "scad_monthly.csv")
    r = sm.dropna(subset=["dct_over_scad"])
    drift = {k: v["eps_total"] for k, v in json.load(open(P / "phase2" / "step23_lever_test.json", encoding="utf-8"))["drift_by_half_year"].items()}
    stb = {k: read_fact(P, {**FACTS, **EXTRA}[f"stageb_{k}"]) for k in ("stay_ratio_A", "stay_ratio_B", "fixed_curve_A", "fixed_curve_B")}
    return {"eps": cand, "untrimmed": untrim, "stageb_benchmarks": stb, "drift": drift, "ratio_max": j25["ratio_range"][2:], "skill": skill, "eps_range": [el["eps_lo"], el["eps_hi"]],
            "new_route": {"band": j25["band"], "with_eps_range": [float(x) for x in nr], "f80": f80, "eps_factor": [float(np.exp(-lo_f)), float(np.exp(hi_f))]},
            "domestic_test": {"model": [s26["stagea_scores"]["model"][k] for k in ("dom_daily", "dom_monthly")],
                              "last_year": [s26["stagea_scores"]["benchmark: same day last year"][k] for k in ("dom_daily", "dom_monthly")],
                              "bias": s26["bias_pct"]["domestic"]},
            "scad": {"2023": [float(r[r.month < "2024-01"].dct_over_scad.min()), float(r[r.month < "2024-01"].dct_over_scad.max())],
                     "2024_on": [float(r[r.month >= "2024-01"].dct_over_scad.min()), float(r[r.month >= "2024-01"].dct_over_scad.max())],
                     "all_mean": float(r.dct_over_scad.mean()), "used_mean": float(r[r.in_reference].dct_over_scad.mean()),
                     "used": [str(r[r.in_reference].month.min()), str(r[r.in_reference].month.max())]}}


def review_report(d: dict, ac: dict, q: dict) -> str:
    c = d["counts"]; R = d["rows"]
    fv = lambda x: _fmt(float(x)) if isinstance(x, (int, float)) and not isinstance(x, bool) else (json.dumps(x, default=str) if isinstance(x, (dict, list)) else str(x))
    pc = lambda x: f"{x * 100:.1f}%"
    sk = q["skill"]; e = q["eps"]; nr = q["new_route"]; dt = q["domestic_test"]; sc = q["scad"]; sb = q["stageb_benchmarks"]
    keys = list(e)
    L = ["# Phase 5.3 — independent review", "",
         "_Generated by `python -m atp.phase5 review` (`make review`); do not edit by hand._", "",
         "A reviewer with no access to our code, reports or reasoning was given the five raw competition files, the data "
         "dictionary, the public SCAD workbook, our scored test-window forecast and our back-test predictions **without their "
         "actuals**, plus a list of quantities with exact definitions and no values (`review/claims.md`). It re-derived every "
         "quantity with its own code (`review/work/`, rerun by `make review`; a second code path, `check.py`, reproduces 31 of "
         "them) and wrote down everything it questioned (`review/notes.md`).", "",
         f"**Verdict: {'every difference explained' if d['ok'] else 'UNEXPLAINED DIFFERENCES'}.** Of {len(R)} quantities, "
         f"{c['same']} agree to a relative {REBUILD_TOL:g}, {c['random draws']} are bootstrap interval ends that differ only by "
         f"random draws, {c['by definition']} differ because claims.md described one of our steps imprecisely (aircraft, below), "
         f"and {c['DIFFERENT']} are unexplained.", "",
         "## Differences, explained", "",
         f"- **Bootstrap interval of ε** ({R['eps_ci95_lo']['ours']:.3f}–{R['eps_ci95_hi']['ours']:.3f} ours, "
         f"{R['eps_ci95_lo']['reviewer']:.3f}–{R['eps_ci95_hi']['reviewer']:.3f} reviewer): 2,000 country resamples with a different "
         "random seed (the reviewer's notes give three more seeds, moving each end by about 0.02). The simulator does not use this interval: its ε "
         f"range is the 80% interval from the unseen test months, {q['eps_range'][0]:.2f}–{q['eps_range'][1]:.2f}.",
         f"- **Seats per flight** (narrowbody {fv(R['narrowbody_seats']['ours'])} ours vs {fv(R['narrowbody_seats']['reviewer'])}; "
         f"widebody {fv(R['widebody_seats']['ours'])} vs {fv(R['widebody_seats']['reviewer'])}): claims.md said \"every Etihad route "
         "in the last 12 months\"; the project uses the baseline year's routes, the ones a planner can still fly. Route by route: "
         f"{ac['identical']} of our {ac['shared']} routes are identical in the reviewer's list; the other "
         f"{ac['shared'] - ac['identical']} differ by at most {ac['max_diff_other']:.2f} seats (months before a launch set to the route's "
         "current level, or a published seasonal schedule: " + ", ".join(f"{k.split('|')[1]} ({v})" for k, v in ac["other"].items())
         + "); the reviewer also counted " + ", ".join(f"{k.split('|')[1]} ({v[1]:.0f} seats)" for k, v in ac["only_reviewer"].items())
         + ", routes that ended (one, Finkenwerder, is Airbus's Hamburg delivery airfield). Kept: a planner adding a flight "
         "should get the size of planes still flying; the docstring now says exactly this.", "",
         "## What the reviewer questioned, and what we did", "",
         "| Point | Evidence | Decision |", "|---|---|---|",
         f"| ε depends on trimming the largest seat changes (untrimmed {q['untrimmed']['lead']:.2f} lead-controlled, "
         f"{q['untrimmed']['plain']:.2f} plain), and taking the smaller of two estimates pulls it down | On the unseen test months "
         f"(festival-adjusted, {sk['all rows']['n']} country-months) the share of country differences explained is "
         + ", ".join(f"{e[k]:.2f} → {pc(sk['all rows']['skill'][k])}" for k in keys)
         + f"; the best single value is {sk['all rows']['best']:.2f} (inside the trusted range only: "
         + ", ".join(f"{e[k]:.2f} → {pc(sk['inside the trusted range']['skill'][k])}" for k in keys)
         + f", best {sk['inside the trusted range']['best']:.2f}). The trimmed estimates predict unseen months better than the untrimmed ones. "
         f"| Kept ε = {e['default']:.2f} and its 80% range {q['eps_range'][0]:.2f}–{q['eps_range'][1]:.2f}, which includes the best "
         "value; the trimming sensitivity goes into the limitations list (5.9). |",
         f"| ε falls as 2025 data come in (period A {R['eps_fold_A']['ours']:.2f}, B {R['eps_fold_B']['ours']:.2f}, final "
         f"{e['default']:.2f}) | Already reported (Phase 2, 2.3, by half-year: " + " / ".join(f"{v:.2f}" for v in q["drift"].values())
         + "); each back-test used only its own period's ε. | No change; limitations list (5.9). |",
         f"| UAE residents: the test-window forecast is worse than repeating last year | Daily {dt['model'][0]:.1f}% vs "
         f"{dt['last_year'][0]:.1f}%, monthly {dt['model'][1]:.1f}% vs {dt['last_year'][1]:.1f}%, bias {dt['bias']:+.0f}%. The Phase 2 report "
         "said \"beats all benchmarks\", true only for the two pre-registered international measures. | **Fixed:** the Phase 2 report "
         "now states the residents' result; limitations list (5.9). |",
         f"| The 80% range's width was chosen so both periods reach 80% ({pc(R['coverage_A_to_B']['ours'])} / "
         f"{pc(R['coverage_B_to_A']['ours'])} at q = {R['q_market_month']['ours']:.3f}), one market-month above the next step | Correct: "
         f"with two periods there is no third to confirm it; the all-market monthly total stays inside the other period's range in "
         f"{pc(R['total_cov_A_to_B']['ours'])} / {pc(R['total_cov_B_to_A']['ours'])} of months, each market's own range in "
         f"{pc(R['own_market_cov_A_to_B']['ours'])} / {pc(R['own_market_cov_B_to_A']['ours'])}. | **Fixed:** the app's note now says how "
         "the width was set; limitations list (5.9). |",
         f"| The new-route rate is borrowed, and its range ignores ε's own uncertainty | The new-city-route test (Phase 2, 2.5) found new "
         f"routes respond like added seats. Adding ε's 80% range (×{nr['eps_factor'][0]:.2f} to ×{nr['eps_factor'][1]:.2f}) to the "
         f"country spread (×{nr['f80']:.2f}) in quadrature moves the range from {nr['band'][0]:.3f}–{nr['band'][1]:.3f} to "
         f"{nr['with_eps_range'][0]:.3f}–{nr['with_eps_range'][1]:.3f} check-ins per seat. | Kept (the country spread dominates); stated "
         "in the limitations list (5.9). |",
         "| Stage B is compared with a benchmark that does not use the actual check-ins | The Phase 1 report also gives two that do "
         f"(periods A / B): check-ins × monthly stay ratio {sb['stay_ratio_A']:.1f}% / {sb['stay_ratio_B']:.1f}%, fixed stay curve "
         f"{sb['fixed_curve_A']:.1f}% / {sb['fixed_curve_B']:.1f}%, against {R['stageb_final_A']['ours']:.1f}% / "
         f"{R['stageb_final_B']['ours']:.1f}% for the survival model. | No change. |",
         f"| DCT ÷ SCAD guest nights shifts level at Jan 2024 ({sc['2023'][0]:.2f}–{sc['2023'][1]:.2f} in 2023, "
         f"{sc['2024_on'][0]:.2f}–{sc['2024_on'][1]:.2f} after), so one average mixes two bases | SCAD's guests per occupied room "
         f"steps up at the same month. | **Fixed:** the occupancy conversion uses the latest 12 months with both ({sc['used'][0]} – "
         f"{sc['used'][1]}, mean {sc['used_mean']:.3f} instead of {sc['all_mean']:.3f}); occupancy points rise by "
         f"{(sc['all_mean'] / sc['used_mean'] - 1) * 100:.1f}%. |",
         f"| {q['ratio_max'][1].title()} has {q['ratio_max'][0]:.2f} check-ins per seat | The nationality vs departure-country catch, "
         "central to the deck. | No change. |",
         "| The data dictionary is wrong (flights daily from 2023; no Guests in the test file); the test file drops small rows "
         "instead of blanking them | Both already handled (data card rules F2, H1–H3). | No change. |",
         "| 28 Feb 2026 may be an incomplete extract | The data card ties it to the first day of the regional conflict: flights "
         "fell and check-ins rose (stranded passengers). | No change. |", "",
         "Also fixed: the closure-day docstring said Armenia 23–26 Jun 2025; the code (and the report) give 24–26.", "",
         *([f"**Snapshot.** Retaken after these fixes (`snapshot.json`; the 5.1 snapshot is kept as `snapshot_5_1.json`). Numbers "
            f"that changed: " + ", ".join(f"`{k}` {a:.3f} → {b:.3f}" for k, (a, b) in q["since_5_1"]["numbers_changed"].items())
            + f"; data files that changed: " + ", ".join(f"`{Path(k).name}`" for k in q["since_5_1"]["data_changed"]) + ".", ""]
           if q.get("since_5_1") else []),
         "## Every quantity", "", "| Key | Ours | Reviewer | Status |", "|---|---|---|---|",
         *[f"| `{k}` | {fv(r['ours'])} | {fv(r['reviewer'])} | {r['status']} |" for k, r in R.items()], ""]
    return "\n".join(L)

# ----------------------------------------------------------------------------- 5.4: three pre-registered candidates
PREREG = "docs/phase5_4_preregistration.md"
HALF_LIFE = 12           # C2: recency half-life in months (fixed in the pre-registration)
GAIN_C3 = 0.10           # C3: required fold-A gain (the 4 Oct audit rule)
RULE_TOL = 0.05          # the project rule's tolerance on the other fold (phase1.RULE_TOL; asserted equal)


def _stageb_fits(cfg, fold, post_break=False, weights=None):
    """The production survival model (international: long break + per-market recent step; Domestic: trend) on one fold,
    optionally with the post-break term (C1) or observation weights (C3). Returns predictions for every day and market."""
    from . import phase1 as p1, stageb as sb
    hi, hd, cal, _, idx = p1.load(cfg); f = cfg["folds"][fold]
    ok = (idx >= f["train_start"]) & (idx <= f["train_end"])
    mk = sb.build_markets(hi, hd, idx)
    intl = {n: m for n, m in mk.items() if n != "DOMESTIC"}; dom = {"DOMESTIC": mk["DOMESTIC"]}
    X, _ = sb.design(cal, idx, f["train_end"], {"long_break": True, "post_break": post_break})
    Xd, _ = sb.design(cal, idx, f["train_end"], {"long_break": False, "post_break": post_break})
    w = (lambda g: None if weights is None else {n: weights[n] for n in g})
    ri = sb.fit_group(intl, X, ok, drift=True, idx=idx, drift_shape="step", weights=w(intl))
    rd = sb.fit_group(dom, Xd, ok, drift=True, idx=idx, drift_shape="trend", weights=w(dom))
    pred = {**sb.predict_all(ri, X), **sb.predict_all(rd, Xd)}
    info = {"intl_grad_rel": ri.grad_rel, "dom_grad_rel": rd.grad_rel, "converged": bool(ri.converged and rd.converged),
            "post_break_beta": {"intl": float(ri.beta[-1]), "dom": float(rd.beta[-1])} if post_break else None}
    return mk, idx, pred, info


def _score(cfg, fold, mk, idx, pred) -> dict:
    from . import phase1 as p1
    f = cfg["folds"][fold]; ho = (idx >= f["holdout_start"]) & (idx <= f["holdout_end"])
    m = p1.metrics(p1.frame(mk, pred, idx, ho))
    return {k: m[k] for k in ("intl_daily_x_nationality", "intl_monthly_x_nationality", "intl_monthly_total", "domestic_daily", "domestic_monthly")}


def candidate_stageb(cfg) -> dict:
    import numpy as np
    out = {"production": {}, "C1_post_break": {}, "C3_poisson_weights": {}}
    for fold in ("A", "B"):
        mk, idx, pred, info = _stageb_fits(cfg, fold)
        out["production"][fold] = {**_score(cfg, fold, mk, idx, pred), **info}
        w = {n: 1.0 / np.maximum(p, 10.0) for n, p in pred.items()}                        # one IRLS step towards Poisson
        mk1, _, pred1, info1 = _stageb_fits(cfg, fold, post_break=True)
        out["C1_post_break"][fold] = {**_score(cfg, fold, mk1, idx, pred1), **info1}
        mk3, _, pred3, info3 = _stageb_fits(cfg, fold, weights=w)
        out["C3_poisson_weights"][fold] = {**_score(cfg, fold, mk3, idx, pred3), **info3}
        print(f"[{fold}] " + " | ".join(f"{k}: intl {v[fold]['intl_daily_x_nationality']:.3f} dom {v[fold]['domestic_daily']:.3f}" for k, v in out.items()), flush=True)
    return out


def candidate_eps(cfg) -> dict:
    """C2: recency-weighted ε, judged by the Step 2.3 lever test on folds B (primary) and A."""
    import numpy as np
    import pandas as pd
    from . import phase2 as p2, stagea as sa
    fd, _, hi = p2._load(cfg)
    m, _, _ = p2._panel_with_festivals(cfg, fd, hi)
    F = cfg["folds"]; test_end = F["final"]["train_end"]
    el = pd.read_csv(cfg["paths"]["processed"] / EL).set_index(["fold", "parameter"])
    res = {}
    for fold in ("B", "A"):
        tr_end = F[fold]["train_end"]
        p, rng_ = p2._primary(m, tr_end)
        last = p.month.max()
        p = p.assign(w=0.5 ** (((last.year - p.month.dt.year) * 12 + last.month - p.month.dt.month) / HALF_LIFE))
        est = {k: sa.fe_elasticity(p, "d_checkins", x, w="w", n_boot=0) for k, x in (("plain", ["d_seats"]), ("lead", p2.LEADS))}
        spec = min(est, key=lambda k: est[k]["estimate"]); eps_w = est[spec]["estimate"]
        eps_u = float(el.loc[(fold, "eps_total_default"), "estimate"])
        g = sa.fe_elasticity(p, "d_checkins", ["d_seats"] + p2.FEST, n_boot=0)["other_coefs"]   # fold festival effects (unweighted, as 2.3)
        te = p2._test_rows(m, tr_end, test_end, rng_).dropna(subset=["d_seats", "d_checkins"])
        adj = te.d_checkins - g["d_festival_days"] * te.d_festival_days - g["d_ramadan_days"] * te.d_ramadan_days
        slope = {k: sa.fe_elasticity(te.assign(pred=e * te.d_seats), "d_checkins", ["pred"] + p2.FEST, n_boot=p2.N_BOOT)
                 for k, e in (("weighted", eps_w), ("unweighted", eps_u))}
        sk = sa.lever_skill(te.assign(d_checkins=adj), {"weighted": eps_w, "unweighted": eps_u}, n_boot=0)["skill"]
        res[fold] = {"train_end": str(tr_end.date()), "test_months": [str(te.month.min().date()), str(te.month.max().date())],
                     "n_test": int(len(te)), "eps_weighted": eps_w, "spec_weighted": spec,
                     "eps_weighted_plain": est["plain"]["estimate"], "eps_weighted_lead": est["lead"]["estimate"], "eps_unweighted": eps_u,
                     "slope": {k: [v["estimate"], *v["ci_boot"]] for k, v in slope.items()}, "skill": sk}
        print(f"[{fold}] ε weighted {eps_w:.3f} ({spec}) vs {eps_u:.3f}; skill {sk['weighted']:.4f} vs {sk['unweighted']:.4f}; "
              f"slope {res[fold]['slope']['weighted']}", flush=True)
    return res


def decide(sb_: dict, eps: dict) -> dict:
    P, C1, C3 = sb_["production"], sb_["C1_post_break"], sb_["C3_poisson_weights"]
    d = lambda c, f, k: c[f][k] - P[f][k]
    c1 = {}
    for grp, dk, mk_ in (("international", "intl_daily_x_nationality", "intl_monthly_x_nationality"), ("domestic", "domestic_daily", "domestic_monthly")):
        c1[grp] = {"daily_change": {f: d(C1, f, dk) for f in "AB"}, "monthly_change": {f: d(C1, f, mk_) for f in "AB"}}
        c1[grp]["keep"] = bool(c1[grp]["daily_change"]["A"] < 0 and c1[grp]["daily_change"]["B"] <= RULE_TOL
                               and all(v <= 0 for v in c1[grp]["monthly_change"].values()))
    c3 = {k: {f: d(C3, f, k) for f in "AB"} for k in ("intl_daily_x_nationality", "intl_monthly_x_nationality", "domestic_daily", "domestic_monthly")}
    keep3 = bool(c3["intl_daily_x_nationality"]["A"] <= -GAIN_C3 and c3["intl_daily_x_nationality"]["B"] <= RULE_TOL
                 and all(v <= 0 for v in c3["intl_monthly_x_nationality"].values())
                 and all(v <= 0 for k in ("domestic_daily", "domestic_monthly") for v in c3[k].values()))
    B, A = eps["B"], eps["A"]
    c2 = {"passes_2_3_test": bool(B["slope"]["weighted"][1] <= 1 <= B["slope"]["weighted"][2] and B["skill"]["weighted"] > 0),
          "skill_gain_B": B["skill"]["weighted"] - B["skill"]["unweighted"], "skill_gain_A": A["skill"]["weighted"] - A["skill"]["unweighted"]}
    c2["keep"] = bool(c2["passes_2_3_test"] and c2["skill_gain_B"] >= 0.01 and c2["skill_gain_A"] >= -0.01)
    return {"C1_post_break": c1, "C2_recency_eps": c2, "C3_poisson_weights": {"changes": c3, "keep": keep3}}


def _bc_sections(bc) -> list:
    B_, C_ = bc["b_gcc_seasonal"], bc["c_resident_split"]
    v = lambda x: f"{x:.2f}%"
    L = ["## (b) GCC seasonal term", "",
         "Bahrain, Kuwait, Oman, Qatar and Saudi Arabia share one month-of-year profile (11 parameters, fitted jointly) instead of "
         "their own month dummies; every other forecaster term stays per series. Rule (Step 2.4, shape term): international daily × "
         f"nationality at least 0.10 points better on fold A, fold B not worse by more than {RULE_TOL}. Check-ins, each fold with its own ε:", "",
         "| | A intl daily × nat | A intl monthly × nat | A GCC daily | B intl daily × nat | B intl monthly × nat | B GCC daily |",
         "|---|---|---|---|---|---|---|",
         *[f"| {lab} | " + " | ".join(f"{v(B_[f][k]['intl_daily_nat'])} | {v(B_[f][k]['intl_monthly_nat'])} | {v(B_[f][g])}"
                                      for f in "AB") + " |"
           for lab, k, g in (("production", "production", "gcc_daily_production"), ("shared GCC profile", "candidate", "gcc_daily_candidate"))], "",
         (lambda g, w: f"Fold A {'better' if g > 0 else 'worse'} by {abs(g):.2f} points (the rule needs 0.10 better), fold B "
                       f"{'worse' if w > 0 else 'better'} by {abs(w):.2f}."
                       + (" Pooling made both folds worse: the five series' own month terms forecast better than a shared GCC profile."
                          if g < 0 and w > 0 else ""))(B_['decision']['gain_A'], B_['decision']['worse_B']), "",
         "## (c) Residents vs visitors split of ε", "",
         "Resident index per flight nationality: check-ins on National Day breaks (UAE-only holiday) ÷ the same weekdays 8–28 days "
         "either side, on each fold's training data; above the median = resident-heavy. ε estimated on each half with the fold's "
         "default specification; 95% interval of the difference from a stratified country bootstrap. Rule: the interval excludes 0 on "
         "both folds, with the same sign.", "",
         "| Fold | single ε | resident-heavy (countries) | visitor-heavy (countries) | difference [95%] |", "|---|---|---|---|---|",
         *[f"| {f} | {C_[f]['eps_single']:.2f} | {C_[f]['groups']['resident-heavy']['estimate']:.2f} ({C_[f]['groups']['resident-heavy']['n_countries']}) | "
           f"{C_[f]['groups']['visitor-heavy']['estimate']:.2f} ({C_[f]['groups']['visitor-heavy']['n_countries']}) | "
           f"{C_[f]['difference']['estimate']:+.2f} [{C_[f]['difference']['ci_boot'][0]:+.2f}, {C_[f]['difference']['ci_boot'][1]:+.2f}] |" for f in "AB"], "",
         ("Neither interval excludes 0" if not (C_["A"]["excludes_zero"] or C_["B"]["excludes_zero"]) else "The rule is not met")
         + (" and the sign flips between folds" if (C_["A"]["difference"]["estimate"] > 0) != (C_["B"]["difference"]["estimate"] > 0) else "")
         + ": no evidence that residents and visitors respond differently to seats, so one ε stays (as Step 2.2 found with a different split)."
         if not C_["decision"]["keep"] else "The split passes and is carried through in Step 5.8.", ""]
    return L


def candidates_report(r: dict) -> str:
    sb_, eps, dec = r["stageb"], r["eps"], r["decisions"]
    P, C1, C3 = sb_["production"], sb_["C1_post_break"], sb_["C3_poisson_weights"]
    v = lambda x: f"{x:.2f}%"; s = lambda x: f"{x:+.2f}"
    yes = lambda b: "**kept**" if b else "**dropped**"
    c1, c2, c3 = dec["C1_post_break"], dec["C2_recency_eps"], dec["C3_poisson_weights"]
    bc = r.get("bc"); B_, C_ = (bc["b_gcc_seasonal"], bc["c_resident_split"]) if bc else (None, None)
    any_kept = any([c1["international"]["keep"], c1["domestic"]["keep"], c2["keep"], c3["keep"]]
                   + ([B_["decision"]["keep"], C_["decision"]["keep"]] if bc else []))
    L = ["# Phase 5.4 — candidate improvements, pre-registered", "",
         "_Generated by `python -m atp.phase5 candidates-report` from `candidates.json` and `candidates_bc.json`; do not edit by hand._", "",
         "Every candidate's definition and decision rule was written down before it was fitted: (a) and two additional checks in "
         f"`{PREREG}` (sha256 `{r['prereg_sha256'][:16]}…`), (b) and (c) in `{PREREG_BC}`"
         + (f" (sha256 `{bc['prereg_sha256'][:16]}…`)" if bc else "") + ". The second file explains why: when the step was first run "
         "only (a) of the plan's three candidates was found in the project's records, so two substitutes were registered and run; "
         "(b) and (c) were registered and run once the plan was recovered. All are judged on the validation folds only (A: Aug 2024 "
         "– Feb 2025, B: Jan – Jul 2025, each trained on data before it); the official test window chooses nothing. Before each "
         "comparison the production model was refitted with the same code and reproduced its saved fold scores exactly"
         + ("." if r["production_matches_saved"] else " — EXCEPT the survival model, see below."), "",
         "| Candidate | Where | Decision |", "|---|---|---|",
         f"| (a) post-holiday checkout term, nationalities | survival model | {yes(c1['international']['keep'])} |",
         f"| (a) post-holiday checkout term, UAE residents | survival model | {yes(c1['domestic']['keep'])} |",
         *([f"| (b) GCC seasonal term (11 shared month parameters) | daily check-in forecaster | {yes(B_['decision']['keep'])} |",
            f"| (c) residents vs visitors split of ε | seat lever | {yes(C_['decision']['keep'])} |"] if bc else []),
         f"| additional: recency-weighted ε (half-life {HALF_LIFE} months) | seat lever | {yes(c2['keep'])} |",
         f"| additional: Poisson-type weighting of the survival fit | survival model | {yes(c3['keep'])} |", "",
         ("**Outcome: the production model is unchanged.** " if not any_kept else "**Outcome: the kept candidates are carried through in Step 5.8.** ")
         + "Each candidate's numbers are below, whatever the decision.", "",
         "## (a) Post-holiday checkout term", "",
         "A checkout-day dummy on the two days after each long break. Rule: fold A daily better, fold B daily not worse by more than "
         f"{RULE_TOL} points, monthly not worse on either fold, per group (the Phase 1 rule plus the monthly condition; under the "
         "Phase 1 rule alone the decision is the same, since fold-A daily error rose for both groups). WMAPE, change from production in brackets (guests given "
         "actual check-ins):", "",
         "| International | A daily × nat | A monthly × nat | B daily × nat | B monthly × nat |", "|---|---|---|---|---|",
         *[f"| {lab} | " + " | ".join(f"{v(c[f][k])} ({s(c[f][k] - P[f][k])})" if c is not P else v(c[f][k])
                                      for f in "AB" for k in ("intl_daily_x_nationality", "intl_monthly_x_nationality")) + " |"
           for lab, c in (("production", P), ("+ post-break", C1))], "",
         "| Domestic | A daily | A monthly | B daily | B monthly |", "|---|---|---|---|---|",
         *[f"| {lab} | " + " | ".join(f"{v(c[f][k])} ({s(c[f][k] - P[f][k])})" if c is not P else v(c[f][k])
                                      for f in "AB" for k in ("domestic_daily", "domestic_monthly")) + " |"
           for lab, c in (("production", P), ("+ post-break", C1))], "",
         f"Fitted post-break effect on the checkout hazard: international ×{__import__('math').exp(C1['A']['post_break_beta']['intl']):.2f} (A), "
         f"×{__import__('math').exp(C1['B']['post_break_beta']['intl']):.2f} (B); Domestic ×{__import__('math').exp(C1['A']['post_break_beta']['dom']):.2f} (A), "
         f"×{__import__('math').exp(C1['B']['post_break_beta']['dom']):.2f} (B)."
         + (" For nationalities the fitted effect is the opposite of the hypothesis (fewer checkouts after a break, not a wave), so the "
            "2 Apr 2025 error was not a general post-break pattern." if max(C1[f]['post_break_beta']['intl'] for f in "AB") < 0 else ""), "",
         *(_bc_sections(bc) if bc else []),
         "## Additional check: recency-weighted ε", "",
         f"Weighted least squares with weight 0.5^(months before the window's last month / {HALF_LIFE}); same trusted range, month "
         "effects and default rule. Judged by the Step 2.3 lever test (calibration slope's 95% interval contains 1, skill > 0) on the "
         "fold-B test rows, needing at least 0.01 more skill than the unweighted fold-B ε, and no more than 0.01 less on fold A.", "",
         "| Fold | ε unweighted | ε weighted (spec) | test months | slope, unweighted [95%] | slope, weighted [95%] | skill unweighted | skill weighted |",
         "|---|---|---|---|---|---|---|---|",
         *[f"| {f} | {e['eps_unweighted']:.3f} | {e['eps_weighted']:.3f} ({e['spec_weighted']}) | {e['test_months'][0][:7]} – {e['test_months'][1][:7]} ({e['n_test']}) | "
           f"{e['slope']['unweighted'][0]:.2f} [{e['slope']['unweighted'][1]:.2f}, {e['slope']['unweighted'][2]:.2f}] | "
           f"{e['slope']['weighted'][0]:.2f} [{e['slope']['weighted'][1]:.2f}, {e['slope']['weighted'][2]:.2f}] | "
           f"{e['skill']['unweighted']:+.3f} | {e['skill']['weighted']:+.3f} |" for f, e in (("B", eps["B"]), ("A", eps["A"]))], "",
         f"Passes the 2.3 test on fold B: {c2['passes_2_3_test']}; skill gain B {c2['skill_gain_B']:+.3f}, A {c2['skill_gain_A']:+.3f}.", "",
         "## Additional check: Poisson-type weighting of the survival fit", "",
         f"One IRLS step towards the Poisson quasi-likelihood (weights 1 / production prediction, floored at 10). Rule: international "
         f"daily × nationality at least {GAIN_C3} points better on fold A, fold B not worse by more than {RULE_TOL}, monthly × nationality "
         "not worse on either fold, Domestic not worse on either fold.", "",
         "| | A intl daily × nat | A intl monthly × nat | A dom daily | A dom monthly | B intl daily × nat | B intl monthly × nat | B dom daily | B dom monthly |",
         "|---|---|---|---|---|---|---|---|---|",
         *[f"| {lab} | " + " | ".join(f"{v(c[f][k])} ({s(c[f][k] - P[f][k])})" if c is not P else v(c[f][k])
                                      for f in "AB" for k in ("intl_daily_x_nationality", "intl_monthly_x_nationality", "domestic_daily", "domestic_monthly")) + " |"
           for lab, c in (("production", P), ("Poisson-weighted", C3))], "",
         "The weighting moves the folds in opposite directions (fold A worse, fold B better), so it is not a reliable improvement.", "",
         "Convergence (relative gradient): " + "; ".join(f"{k} {f}: intl {x[f]['intl_grad_rel']:.1e}, dom {x[f]['dom_grad_rel']:.1e}"
                                                         for k, x in sb_.items() for f in "AB") + ".", ""]
    return "\n".join(L)


def candidates(cfg) -> dict:
    import pandas as pd
    from .phase1 import RULE_TOL as _t
    assert _t == RULE_TOL
    sb_ = candidate_stageb(cfg); eps = candidate_eps(cfg)
    saved = pd.read_csv(cfg["paths"]["processed"] / P1.split("/")[0] / "validation_metrics.csv").set_index(["fold", "model"])
    match = all(abs(sb_["production"][f]["intl_daily_x_nationality"] - saved.loc[(f, "base+market_step"), "intl_daily_x_nationality"]) < 1e-6
                and abs(sb_["production"][f]["domestic_daily"] - saved.loc[(f, "base+market_drift"), "domestic_daily"]) < 1e-6 for f in "AB")
    return {"prereg_sha256": sha256(cfg["root"] / PREREG), "production_matches_saved": bool(match),
            "stageb": sb_, "eps": eps, "decisions": decide(sb_, eps)}



# ----------------------------------------------------------------------------- 5.4 (b) and (c), pre-registered separately
PREREG_BC = "docs/phase5_4_preregistration_b_c.md"
GCC = ["BAHRAIN", "KUWAIT", "OMAN", "QATAR", "SAUDI ARABIA"]
MONTHS = [f"m{m}" for m in range(2, 13)]


def _forecaster_frame(cfg):
    """The Step 2.4 daily frame, exactly as phase2.step_forecast builds it."""
    import pandas as pd
    from . import forecaster as fc, phase2 as p2, stagea as sa
    RD, P = cfg["paths"]["reference"], cfg["paths"]["processed"]
    fd, _, hi = p2._load(cfg); cal = pd.read_parquet(P / "calendar.parquet")
    off = pd.read_csv(RD / "calendar_events_official.csv", parse_dates=["start_date", "end_date"])
    fest = sa.festival_dates(RD, off); grp = pd.read_csv(RD / "festival_groups.csv")
    ev = pd.read_csv(RD / "external_events.csv", parse_dates=["start_date", "end_date"])
    hh = pd.read_csv(RD / "home_holidays.csv", parse_dates=["start", "end"])
    D = fc.build_frame(p2._series(cfg, fd, hi), cal, fc.uae_breaks(cal), fc.festival_daily(grp, fest), fc.event_daily(ev),
                       fc.home_holiday_daily(hh))
    return D, set(hi.nationality), set(fd.nationality_key)


def fit_predict_pooled_months(Dg, train_end, start, horizon_end, spec: set, eps: float):
    """(b): the GCC series fitted jointly — one shared month-of-year profile (11 parameters), every other production term
    per series — mirroring forecaster.fit_predict row for row (same estimation rows, seat floor, frozen recent step)."""
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from . import forecaster as fc
    own = [c for c in fc.columns(spec) if c not in MONTHS]
    fit_parts, pred_parts = [], []
    for key, d in Dg.groupby("key"):
        d = d.assign(const=1.0, recent=(d.date > train_end - pd.Timedelta(days=fc.LEVEL_DAYS)).astype(float))
        tr = d[(d.date >= start) & (d.date <= train_end)]
        floor = fc.SEAT_FLOOR * tr.seats_7d.mean()
        rows = tr[~tr.exclude & (tr.seats_7d >= floor)]
        cols = [c for c in own if c == "const" or rows[c].std() > 0]
        te = d[(d.date > train_end) & (d.date <= horizon_end)]
        mk = lambda x: pd.concat([x[MONTHS].reset_index(drop=True),
                                  x[cols].reset_index(drop=True).add_prefix(f"{key}|")], axis=1)
        fit_parts.append((mk(rows), rows.y.to_numpy(), fc._offset(rows, eps, floor)))
        pred_parts.append((key, te, mk(te), fc._offset(te, eps, floor)))
    X = pd.concat([f[0] for f in fit_parts], ignore_index=True).fillna(0.0)
    res = sm.GLM(np.concatenate([f[1] for f in fit_parts]), X.to_numpy(), family=sm.families.Poisson(),
                 offset=np.concatenate([f[2] for f in fit_parts])).fit()
    beta = pd.Series(res.params, index=X.columns)
    out = []
    for key, te, Xt, off in pred_parts:
        Xt = Xt.reindex(columns=X.columns, fill_value=0.0)
        out.append(te[["key", "date", "y"]].assign(pred=np.exp(Xt.to_numpy() @ beta.to_numpy() + off)))
    return pd.concat(out, ignore_index=True), bool(res.converged), beta[MONTHS]


def candidate_gcc(cfg) -> dict:
    import numpy as np
    import pandas as pd
    from . import forecaster as fc
    P, F = cfg["paths"]["processed"], cfg["folds"]
    D, intl, flight = _forecaster_frame(cfg)
    j24 = json.load(open(P / "phase2" / "step24_forecaster.json", encoding="utf-8"))
    spec = set(j24["chosen_spec"]); eps_fold = j24["eps_used_in_folds"]; start = pd.Timestamp("2023-01-07")
    assert not spec & {"C5", "C5p", "C5s", "C4a"}, "the pooled fit mirrors a production spec without growth or trend terms"
    res = {"spec": sorted(spec), "series": GCC}
    for f in ("A", "B"):
        prod = pd.read_parquet(P / "phase2" / f"stagea_pred_{f}.parquet")
        # the saved production forecasts are reproduced by the production code before anything is compared
        chk = pd.concat([fc.fit_predict(D[D.key == k], F[f]["train_end"], start, F[f]["holdout_end"], spec, eps_fold[f])[0] for k in GCC])
        m_ = prod.merge(chk, on=["key", "date"], suffixes=("", "_re"))
        assert len(m_) == len(chk) and np.allclose(m_.pred, m_.pred_re, rtol=1e-9), "production forecasts not reproduced"
        new, conv, months = fit_predict_pooled_months(D[D.key.isin(GCC)], F[f]["train_end"], start, F[f]["holdout_end"], spec, eps_fold[f])
        cand = pd.concat([prod[~prod.key.isin(GCC)], new], ignore_index=True)
        base, alt = fc.metrics(prod, intl, flight), fc.metrics(cand, intl, flight)
        g = lambda pr: fc.wmape(pr[pr.key.isin(GCC)].y, pr[pr.key.isin(GCC)].pred)
        res[f] = {"production": base, "candidate": alt, "gcc_daily_production": g(prod), "gcc_daily_candidate": g(cand),
                  "converged": conv, "shared_month_effects": {k: float(v) for k, v in months.items()}}
        assert abs(base["intl_daily_nat"] - j24["chosen_metrics"][f]["intl_daily_nat"]) < 1e-9
        print(f"[{f}] (b) intl daily×nat {base['intl_daily_nat']:.3f} → {alt['intl_daily_nat']:.3f}; GCC daily {res[f]['gcc_daily_production']:.2f} → "
              f"{res[f]['gcc_daily_candidate']:.2f}", flush=True)
    gA = res["A"]["production"]["intl_daily_nat"] - res["A"]["candidate"]["intl_daily_nat"]
    wB = res["B"]["candidate"]["intl_daily_nat"] - res["B"]["production"]["intl_daily_nat"]
    res["decision"] = {"gain_A": gA, "worse_B": wB, "keep": bool(gA >= 0.10 and wB <= RULE_TOL)}
    return res


def resident_index(cfg, hi, end) -> "pd.Series":
    """(c): National Day break check-ins ÷ the same weekdays 8–28 days either side (no breaks, no Ramadan), summed over
    the National Day breaks from 7 Jan 2023 to `end` (the Step 2.7 measure, on the fold's own training data)."""
    import pandas as pd
    from . import forecaster as fc
    cal = pd.read_parquet(cfg["paths"]["processed"] / "calendar.parquet")
    b = fc.uae_breaks(cal).merge(cal[["date", "is_ramadan"]], on="date").set_index("date")
    y = hi[(hi.date >= "2023-01-07") & (hi.date <= end)]
    Y = y.pivot_table(index="date", columns="nationality", values="arrivals", aggfunc="sum")
    b = b.reindex(Y.index)
    on = b.nd_break == 1; run = (on != on.shift()).cumsum().where(on).dropna(); num = den = 0
    for _, g in run.groupby(run):
        d = g.index
        if d.max() + pd.Timedelta(days=28) > pd.Timestamp(end):
            continue                                                          # its comparison days are not all in training
        c = pd.date_range(d.min() - pd.Timedelta(days=28), d.max() + pd.Timedelta(days=28))
        c = c[((c < d.min() - pd.Timedelta(days=7)) | (c > d.max() + pd.Timedelta(days=7))) & c.isin(Y.index)]
        c = c[(b.loc[c, "all_break"] == 0).to_numpy() & ~b.loc[c, "is_ramadan"].astype(bool).to_numpy() & c.dayofweek.isin(d.dayofweek)]
        num = num + Y.loc[d].mean(); den = den + Y.loc[c].mean()
    return (num / den).rename("national_day_ratio")


def candidate_split(cfg) -> dict:
    import pandas as pd
    from . import phase2 as p2, stagea as sa
    fd, _, hi = p2._load(cfg)
    m = sa.monthly_panel(fd, hi); F = cfg["folds"]; res = {}
    for f in ("A", "B"):
        p, _, R, dkey = p2._fold_estimates(m, F[f]["train_end"])
        xs = p2.LEADS if dkey == "eps_total_lead" else ["d_seats"]
        idx = resident_index(cfg, hi, F[f]["train_end"]).reindex(sorted(fd.nationality_key.unique()))
        assert idx.notna().all(), "every flight nationality needs a resident index"
        res_heavy = set(idx[idx > idx.median()].index)
        grp = {lab: sa.fe_elasticity(p[sel], "d_checkins", xs, n_boot=p2.N_BOOT, seed=s)
               for s, (lab, sel) in enumerate((("resident-heavy", p.nationality_key.isin(res_heavy)),
                                               ("visitor-heavy", ~p.nationality_key.isin(res_heavy))), 1)}
        diff = sa.difference(grp["resident-heavy"], grp["visitor-heavy"])
        res[f] = {"spec": dkey, "eps_single": R["eps_total_default"]["estimate"], "resident_heavy": sorted(res_heavy),
                  "national_day_ratio": {k: float(v) for k, v in idx.items()},
                  "groups": {k: {"estimate": v["estimate"], "ci_boot": v["ci_boot"], "n_obs": v["n_obs"], "n_countries": v["n_countries"]}
                             for k, v in grp.items()},
                  "difference": {"estimate": diff["estimate"], "ci_boot": diff["ci_boot"]},
                  "excludes_zero": bool(diff["ci_boot"][0] > 0 or diff["ci_boot"][1] < 0)}
        print(f"[{f}] (c) resident-heavy {grp['resident-heavy']['estimate']:.3f}, visitor-heavy {grp['visitor-heavy']['estimate']:.3f}, "
              f"difference {diff['estimate']:+.3f} {[round(x, 3) for x in diff['ci_boot']]}", flush=True)
    same_sign = (res["A"]["difference"]["estimate"] > 0) == (res["B"]["difference"]["estimate"] > 0)
    res["decision"] = {"keep": bool(res["A"]["excludes_zero"] and res["B"]["excludes_zero"] and same_sign)}
    return res


def candidates_bc(cfg) -> dict:
    return {"prereg_sha256": sha256(cfg["root"] / PREREG_BC), "b_gcc_seasonal": candidate_gcc(cfg), "c_resident_split": candidate_split(cfg)}


def main(argv):
    cfg = load_config(); out = cfg["paths"]["processed"] / OUT; path = out / "snapshot.json"
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "snapshot":
        out.mkdir(exist_ok=True)
        snap = take(cfg)
        json.dump(snap, open(path, "w", encoding="utf-8"), indent=1, default=str)
        (cfg["root"] / "reports" / "phase5_snapshot.md").write_text(report(snap), encoding="utf-8")
        g = {x: sum(1 for v in snap["files"].values() if v["group"] == x) for x in ("data", "code")}
        print(f"SNAPSHOT: {len(snap['numbers'])} numbers, {g['data']} data files, {g['code']} code files → {path.relative_to(cfg['root'])}")
        return 0
    if cmd == "check":
        d = compare(json.load(open(path, encoding="utf-8")), {"numbers": numbers(cfg), "files": files(cfg)})
        for k, v in d.items():
            if k != "ok" and v:
                print(f"{k}: {len(v)}"); [print(f"  {x}" + (f": {v[x][0]!r} → {v[x][1]!r}" if isinstance(v, dict) else "")) for x in list(v)[:25]]
        print("CHECK:", "no number or data file changed" if d["ok"] else "DIFFERENCES (see above)")
        return 0 if d["ok"] else 1
    if cmd == "compare" and len(argv) > 2:
        dest = Path(argv[2]).resolve()
        d = compare_rebuild(cfg["root"], dest)
        json.dump(d, open(out / "rebuild_compare.json", "w", encoding="utf-8"), indent=1, default=str)
        (cfg["root"] / "reports" / "phase5_rebuild.md").write_text(rebuild_report(d, dest), encoding="utf-8")
        f = d["files"]
        print(f"NUMBERS: {d['numbers_identical']}/{d['numbers']} identical, largest relative difference {d['numbers_max_rel']:.2g}, "
              f"{len(d['numbers_off'])} beyond {REBUILD_TOL:g}")
        print(f"FILES: {len(f['identical'])} identical, {len(f['same_content'])} same content, {len(f['off'])} different, "
              f"{len(f['missing'])} missing, {len(f['extra'])} extra")
        for k, v in list(d["numbers_off"].items())[:25]: print(f"  number {k}: {v[0]!r} → {v[1]!r}")
        for k, v in list(f["off"].items())[:40]: print(f"  file {k}: {v:.2g}")
        print("REPORTS:", d["reports"])
        print("REBUILD:", "reproduced" if d["ok"] else "NOT reproduced")
        return 0 if d["ok"] else 1
    if cmd in ("candidates", "candidates-report"):
        out.mkdir(exist_ok=True)
        r = candidates(cfg) if cmd == "candidates" else json.load(open(out / "candidates.json", encoding="utf-8"))
        if (out / "candidates_bc.json").exists():
            r = {**r, "bc": json.load(open(out / "candidates_bc.json", encoding="utf-8"))}
        if cmd == "candidates":
            json.dump({k: x for k, x in r.items() if k != "bc"}, open(out / "candidates.json", "w", encoding="utf-8"), indent=1, default=float)
        (cfg["root"] / "reports" / "phase5_candidates.md").write_text(candidates_report(r), encoding="utf-8")
        print("DECISIONS:", json.dumps({k: (v["keep"] if "keep" in v else {g: x["keep"] for g, x in v.items()}) for k, v in r["decisions"].items()}))
        return 0
    if cmd == "candidates-bc":
        out.mkdir(exist_ok=True)
        r = candidates_bc(cfg)
        json.dump(r, open(out / "candidates_bc.json", "w", encoding="utf-8"), indent=1, default=float)
        (cfg["root"] / "reports" / "phase5_candidates.md").write_text(candidates_report({**json.load(open(out / "candidates.json", encoding="utf-8")), "bc": r}), encoding="utf-8")
        print("DECISIONS:", {k: v["decision"]["keep"] for k, v in r.items() if isinstance(v, dict)})
        return 0
    if cmd == "benchmark":
        from . import benchmark as bm
        out.mkdir(exist_ok=True)
        r = bm.benchmark(cfg)
        json.dump(r, open(out / "benchmark.json", "w", encoding="utf-8"), indent=1, default=float)
        (cfg["root"] / "reports" / "phase5_benchmark.md").write_text(bm.report(r), encoding="utf-8")
        print("BENCHMARK:", r["verdict"]["label"])
        return 0
    if cmd == "benchmark-report":
        from . import benchmark as bm
        r = json.load(open(out / "benchmark.json", encoding="utf-8"))
        (cfg["root"] / "reports" / "phase5_benchmark.md").write_text(bm.report(r), encoding="utf-8")
        return 0
    if cmd == "limits":
        from . import limits as lm
        out.mkdir(exist_ok=True)
        items = lm.build(cfg)
        json.dump(items, open(out / "limitations.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        (cfg["root"] / "reports" / "limitations.md").write_text(lm.report(items), encoding="utf-8")
        lm.update_readme(cfg["root"], items)
        print(f"LIMITS: {len(items)} limitations, {sum(i['deck'] for i in items)} on the deck shortlist")
        return 0
    if cmd in ("ahead", "ahead-report"):
        from . import ahead as ah
        out.mkdir(exist_ok=True)
        if cmd == "ahead":
            json.dump(ah.run(cfg), open(out / "ahead.json", "w", encoding="utf-8"), indent=1, default=float)
        r = json.load(open(out / "ahead.json", encoding="utf-8")); r = json.loads(json.dumps(r), object_pairs_hook=lambda p: {(int(k) if k.isdigit() else k): v for k, v in p})
        (cfg["root"] / "reports" / "phase5_ahead.md").write_text(ah.report(r), encoding="utf-8")
        print("AHEAD:", r["grows_with_lead"]["label"])
        return 0
    if cmd in ("sobol", "sobol-report"):
        from . import sobol as so
        out.mkdir(exist_ok=True)
        if cmd == "sobol":
            r = so.run(cfg)
            json.dump(r, open(out / "sobol.json", "w", encoding="utf-8"), indent=1, default=float)
            if not r["gate"]["passed"]:
                print("SOBOL: gate failed, no index computed"); return 1
        r = json.load(open(out / "sobol.json", encoding="utf-8"))
        (cfg["root"] / "reports" / "phase5_sobol.md").write_text(so.report(r), encoding="utf-8")
        print("SOBOL:", "ranking", "confirmed" if r["ranking"]["confirmed"] else "corrected", "; interactions",
              "matter" if r["interactions"]["matter"] else "do not matter")
        return 0
    if cmd == "report":
        from . import phase5_report as pr
        pr.write(cfg)
        print("REPORT:", pr.OUT)
        return 0
    if cmd == "review-inputs":
        d = review_inputs(cfg)
        print(f"REVIEW INPUTS: {sum(1 for x in d.rglob('*') if x.is_file())} files → {d.relative_to(cfg['root'])}")
        return 0
    if cmd == "review":
        theirs = json.load(open(cfg["root"] / REVIEW / "results.json", encoding="utf-8"))
        d = compare_review(ours_for_review(cfg), theirs)
        out.mkdir(exist_ok=True)
        json.dump(d, open(out / "review_compare.json", "w", encoding="utf-8"), indent=1, default=str)
        q = _questioned(cfg); old = out / "snapshot_5_1.json"
        if old.exists() and path.exists():
            q["since_5_1"] = compare(json.load(open(old, encoding="utf-8")), json.load(open(path, encoding="utf-8")))
        (cfg["root"] / "reports" / "phase5_review.md").write_text(review_report(d, _aircraft_explained(cfg), q), encoding="utf-8")
        print("REVIEW:", ", ".join(f"{v} {k}" for k, v in d["counts"].items()))
        for k, r in d["rows"].items():
            if r["status"] not in ("same",):
                print(f"  {k}: ours {r['ours']!r}, reviewer {r['reviewer']!r} → {r['status']}")
        return 0 if d["ok"] else 1
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
