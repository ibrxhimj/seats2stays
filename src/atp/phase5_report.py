"""Phase 5, Step 5.10 — the Phase 5 report: what the validation audit found, in one page.

Every number is read from the saved result of the step it comes from (data/processed/phase5/*.json); the pre-registration
hashes are recomputed. Nothing is fitted here.

Run:  python -m atp.phase5 report      → reports/phase5_report.md
"""
import json
from pathlib import Path
from .phase4 import sha256

OUT = "reports/phase5_report.md"
PREREGS = [   # step, pre-registration, the result that records its sha256
    ("5.4 (a) and two additional checks", "docs/phase5_4_preregistration.md", "candidates.json"),
    ("5.4 (b) and (c)", "docs/phase5_4_preregistration_b_c.md", "candidates_bc.json"),
    ("5.5 LightGBM benchmark", "docs/phase5_5_preregistration.md", "benchmark.json"),
    ("5.6 Sobol check", "docs/phase5_6_preregistration.md", "sobol.json"),
    ("5.7 accuracy by months ahead", "docs/phase5_7_preregistration.md", "ahead.json")]
EPS = "ε (check-in response)"


def load(cfg) -> dict:
    P = cfg["paths"]["processed"] / "phase5"
    r = {k: json.load(open(P / f"{k}.json", encoding="utf-8")) for k in ("snapshot", "snapshot_5_1", "rebuild_compare", "review_compare", "candidates",
                                                       "candidates_bc", "benchmark", "sobol", "ahead")}
    r["limitations"] = json.load(open(P / "limitations.json", encoding="utf-8"))
    r["root"] = cfg["root"]
    return r


def preregs(r: dict) -> list:
    """Each pre-registration: does the file today have the hash recorded with its result, and how many amendments it declares."""
    out = []
    for step, doc, res in PREREGS:
        p = Path(r["root"]) / doc
        rec = r[res[:-5]]["prereg_sha256"]
        out.append({"step": step, "doc": doc, "sha": rec, "matches": sha256(p) == rec,
                    "amendments": p.read_text(encoding="utf-8").count("\n## Amendment")})
    return out


def decisions(r: dict) -> list:
    """The six pre-registered decisions of Step 5.4 (the post-holiday term is judged for nationalities and residents apart)."""
    d, bc = r["candidates"]["decisions"], r["candidates_bc"]
    return [("(a) post-holiday checkout term, nationalities", d["C1_post_break"]["international"]["keep"]),
            ("(a) post-holiday checkout term, UAE residents", d["C1_post_break"]["domestic"]["keep"]),
            ("(b) GCC seasonal term", bc["b_gcc_seasonal"]["decision"]["keep"]),
            ("(c) residents vs visitors split of ε", bc["c_resident_split"]["decision"]["keep"]),
            ("recency-weighted ε", d["C2_recency_eps"]["keep"]),
            ("Poisson-type weighting of the survival fit", d["C3_poisson_weights"]["keep"])]


def _pct(x, d=1):
    return f"{x:.{d}f}%"


def _rv(r, k):
    return r["review_compare"]["rows"][k]["ours"]


def accuracy(r: dict) -> dict:
    """The accuracy figures the deck quotes, all re-derived by the independent reviewer (status 'same')."""
    keys = [f"{m}_{f}_{x}" for m in ("backtest", "benchmark") for f in "AB" for x in ("monthly_total", "monthly_market", "daily_market")]
    keys += [f"backtest_{f}_domestic_monthly" for f in "AB"] + ["coverage_A_to_B", "coverage_B_to_A"]
    keys += [f"test_{x}" for x in ("intl_daily_nat", "intl_monthly_total", "dom_daily", "last_year_intl_daily_nat",
                                   "last_year_intl_monthly_total", "last_year_dom_daily")]
    rows = r["review_compare"]["rows"]
    return {"values": {k: rows[k]["ours"] for k in keys}, "all_reviewed": all(rows[k]["status"] == "same" for k in keys)}


def _benchmark_line(b: dict) -> str:
    k = "intl_daily_nat"
    best = {f: b[f]["best_of_ten"] for f in "AB"}
    tie = [f for f in "AB" if best[f]["diff_interval_95"][0] <= 0 <= best[f]["diff_interval_95"][1]]
    worse = [f for f in "AB" if best[f]["diff_interval_95"][0] > 0]
    return (f"Given the same inputs, a LightGBM model is less accurate country by country in both held-out periods "
            f"({_pct(b['A']['lightgbm'][k])} and {_pct(b['B']['lightgbm'][k])} daily against our {_pct(b['A']['ours'][k])} and "
            f"{_pct(b['B']['ours'][k])}); even the best of ten settings, picked with hindsight, "
            + (f"is clearly worse in period {' and '.join(worse)}" if worse else "")
            + (" and " if worse and tie else "") + (f"only ties in period {' and '.join(tie)}" if tie else "")
            + ". It also cannot run a scenario through the chain or give a range.")


def _rebuild_review_line(r: dict) -> str:
    rb, rv = r["rebuild_compare"], r["review_compare"]["counts"]
    n = sum(rv.values())
    same = f"all {rb['numbers']}" if rb["numbers_identical"] == rb["numbers"] else f"{rb['numbers_identical']} of {rb['numbers']}"
    return (f"Rebuilt from the raw files in a fresh environment, {same} headline numbers came out "
            f"identical; an independent reviewer that never saw our code re-derived {n} quantities with its own code and none is "
            f"unexplained ({rv['same']} identical, the rest bootstrap draws or a definition we had worded loosely).")


def build(r: dict) -> str:
    from . import sobol as so, ahead as ah
    snap, s51, rb, rv = r["snapshot"], r["snapshot_5_1"], r["rebuild_compare"], r["review_compare"]
    b, S, A, lim = r["benchmark"], r["sobol"], r["ahead"], r["limitations"]
    A = json.loads(json.dumps(A), object_pairs_hook=lambda p: {(int(k) if k.isdigit() else k): v for k, v in p})
    groups = {f: sum(1 for v in snap["files"].values() if v["group"] == f) for f in ("data", "code")}
    dec = decisions(r); kept = [n for n, k in dec if k]
    pr = preregs(r); acc = accuracy(r); a = acc["values"]
    k = "intl_daily_nat"; v = b["verdict"]
    gl = A["grows_with_lead"]["per_fold"]; P = A["guests"]["pooled"]["by_nationality"]
    st = S["1a"]["ST"]; e_lo, e_hi = S["R1_shared"]["S1"][EPS], S["R1"]["S1"][EPS]
    from .phase5 import compare
    since = compare(s51, snap)
    n_groups = len({i["group"] for i in lim})
    n_tests = sum(p.read_text(encoding="utf-8").count("\ndef test_") for p in (Path(r["root"]) / "tests").glob("test_phase5_*.py"))
    fw = lambda x: f"{x:.3f}"
    L = ["# Phase 5 — validation audit: report", "",
         "_Generated by `python -m atp.phase5 report` (Step 5.10) from the saved results of Steps 5.1–5.9; do not edit by hand._", "",
         "Phase 5 asked four things of the simulator before it is handed over: are its numbers what the code produces, can anyone "
         "rebuild them from the raw files, are they as good as this data allows, and what can it not do. The official test window "
         "(Aug 2025 – Feb 2026) chose nothing in this phase: every comparison was made on the two earlier held-out periods, "
         "A (Aug 2024 – Feb 2025) and B (Jan – Jul 2025), each forecast from the data before it.", "",
         "## Phase 5 at a glance", "",
         "| Step | Question | Answer | Details |", "|---|---|---|---|",
         f"| 5.1 Snapshot | What does the project say today? | {len(snap['numbers'])} headline numbers, each with its source, and the "
         f"sha256 of {groups['data']} data files and {groups['code']} code files, fixed as the reference for every later step | `phase5_snapshot.md` |",
         f"| 5.2 Clean rebuild | Can anyone rebuild it from the raw files? | **{'Yes' if rb['ok'] else 'No'}.** {rb['numbers_identical']} of {rb['numbers']} numbers identical "
         f"(largest relative difference {rb['numbers_max_rel']:.0g}); {len(rb['files']['identical'])} data files byte-identical, "
         f"{len(rb['files']['same_content'])} the same content, {len(rb['files']['off'])} different; "
         f"{sum(x == 'identical' for x in rb['reports'].values())} of {len(rb['reports'])} generated reports identical | `phase5_rebuild.md` |",
         f"| 5.3 Independent review | Does a reviewer who never saw our code get the same numbers? | **{'Yes' if rv['ok'] else 'No'}.** {sum(rv['counts'].values())} quantities: "
         f"{rv['counts']['same']} identical, {rv['counts']['random draws']} bootstrap ends that differ by random draws, "
         f"{rv['counts']['by definition']} by a definition (aircraft sizes), {rv['counts']['DIFFERENT']} unexplained | `phase5_review.md` |",
         f"| 5.4 Candidate improvements | Can the model be improved? | **{'Not with these candidates' if not kept else 'Yes'}.** {len({n.split(',')[0] for n, _ in dec})} candidates, "
         f"{len(dec)} pre-registered decisions, {len(kept)} passed; the production model is unchanged | `phase5_candidates.md` |",
         f"| 5.5 Gradient-boosting benchmark | Would machine learning forecast check-ins better? | **{'Yes' if v['lightgbm_more_accurate'] else 'No'}.** {v['label'][0].upper() + v['label'][1:]}: "
         f"{_pct(b['A']['ours'][k])} / {_pct(b['B']['ours'][k])} against {_pct(b['A']['lightgbm'][k])} / {_pct(b['B']['lightgbm'][k])} "
         f"(daily, country by country) | `phase5_benchmark.md` |",
         f"| 5.6 Sobol check | Is the lever ranking right, and what drives the range? | **Ranking {'confirmed' if S['ranking']['confirmed'] else 'corrected'}**; "
         f"interactions {'matter' if S['interactions']['matter'] else 'negligible'} ({S['interactions']['max_interaction_share'] * 100:.1f}%); "
         f"the check-in response ε is {e_lo * 100:.0f}–{e_hi * 100:.0f}% of a network flight result's uncertainty | `phase5_sobol.md` |",
         f"| 5.7 Accuracy by months ahead | Does the error grow the further ahead? | **{A['grows_with_lead']['label'][0].upper() + A['grows_with_lead']['label'][1:]}** "
         f"(slope {gl['A']['slope']:+.2f} and {gl['B']['slope']:+.2f} points a month, both intervals include 0); months 8–12 untested | `phase5_ahead.md` |",
         f"| 5.8 Carry changes through | Rebuild every phase with the accepted changes | **Not triggered:** {'no candidate passed in 5.4' if not kept else ', '.join(kept)} | — |",
         f"| 5.9 One limitations list | What can it not do? | {len(lim)} limitations in {n_groups} groups, every number read from the saved results; "
         f"{sum(i['deck'] for i in lim)} on the deck shortlist | `limitations.md` |", "",
         "## Accuracy at the end of Phase 5", "",
         "Phase 5 did not change the forecast, so these are the Phase 2 back-test figures; "
         + ("each was also re-derived by the independent reviewer with its own code." if acc["all_reviewed"] else
            "**not every one was re-derived identically by the reviewer (see `phase5_review.md`).**")
         + " WMAPE: the total gap divided by the total reported; lower is better.", "",
         "| Hotel guest nights | Period A | Period B | Same day last year (A / B) |", "|---|---|---|---|",
         f"| International total, by month | {_pct(a['backtest_A_monthly_total'])} | {_pct(a['backtest_B_monthly_total'])} | "
         f"{_pct(a['benchmark_A_monthly_total'])} / {_pct(a['benchmark_B_monthly_total'])} |",
         f"| One country at a time, by month | {_pct(a['backtest_A_monthly_market'])} | {_pct(a['backtest_B_monthly_market'])} | "
         f"{_pct(a['benchmark_A_monthly_market'])} / {_pct(a['benchmark_B_monthly_market'])} |",
         f"| One country at a time, by day | {_pct(a['backtest_A_daily_market'])} | {_pct(a['backtest_B_daily_market'])} | "
         f"{_pct(a['benchmark_A_daily_market'])} / {_pct(a['benchmark_B_daily_market'])} |",
         f"| UAE residents, by month | {_pct(a['backtest_A_domestic_monthly'])} | {_pct(a['backtest_B_domestic_monthly'])} | — |",
         f"| Country-months inside the 80% range (range learned on the other period) | {_pct(a['coverage_B_to_A'] * 100)} | "
         f"{_pct(a['coverage_A_to_B'] * 100)} | — |", "",
         "The official test window was scored once, on check-ins (its file has no guest nights), with the first version of the check-in "
         "forecast; against repeating last year: "
         f"international, daily country by country {_pct(a['test_intl_daily_nat'])} against {_pct(a['test_last_year_intl_daily_nat'])}; "
         f"international total by month {_pct(a['test_intl_monthly_total'])} against {_pct(a['test_last_year_intl_monthly_total'])}; "
         f"UAE residents daily {_pct(a['test_dom_daily'])} against {_pct(a['test_last_year_dom_daily'])} "
         + ("(worse: a stated limitation)." if a["test_dom_daily"] > a["test_last_year_dom_daily"] else "(better)."), "",
         "## How the audit was kept honest", "",
         "- **Rules before results.** Every candidate change and every benchmark (Steps 5.4–5.7) had its definitions and decision rule "
         "written down before it was run; the independent reviewer worked from written definitions without our values. The file's sha256 is stored with the result and checked again here:", "",
         "| Step | Pre-registration | sha256 | Matches the result | Amendments |", "|---|---|---|---|---|",
         *[f"| {p['step']} | `{p['doc']}` | `{p['sha'][:16]}…` | {'yes' if p['matches'] else '**NO**'} | {p['amendments']} |" for p in pr], "",
         "Each amendment was added after a first run, is declared in the file with its reason, and only adds a check: none changes a "
         "rule that decides.", "",
         "- **The model only changes through a passed rule.** None passed, so the production model is the one Phase 4 shipped.",
         f"- **Numbers are protected.** `python -m atp.phase5 check` recomputes all {len(snap['numbers'])} headline numbers and the hashes of "
         f"every data file and compares them with the snapshot; {n_tests} Phase 5 tests check the saved results against their rules "
         "and pre-registrations.",
         "- **Reports are computed, not typed.** Each step's report, this one included, is generated from its saved results, so a "
         "rerun that changes a number changes the report.",
         "- **The audit was audited.** Steps 5.3–5.7 were each re-checked line by line before the limitations list was written "
         "(`phase5_audit_5_3_to_5_7.md`): fixes in 5.5, 5.6 and 5.7, none of which moved a production number.", "",
         "## What Phase 5 changed", "",
         "Numbers that differ from the first snapshot (5.1), all from one fix the independent reviewer prompted: hotel guest nights "
         "are converted to SCAD's occupancy basis using the latest 12 months with both sources, since the ratio shifts level in "
         "January 2024 (" + "; ".join(f"`{x}` {o:.3f} → {n:.3f}" for x, (o, n) in sorted(since["numbers_changed"].items())) + "). "
         f"Data files changed: {', '.join('`' + Path(x).name + '`' for x in since['data_changed'])}. No forecast, range or lever moved.", "",
         "Other fixes:", "",
         "- **5.2:** a hotel-stay model fit that runs out of steps before converging now continues instead of stopping early.",
         "- **5.3:** the Phase 2 report states the UAE residents' test result; the app says how the 80% range width was set; two code "
         "comments corrected (aircraft sizes use the baseline year's routes; the Armenia closure days).",
         "- **Audit of 5.3–5.7:** the benchmark's best setting got its own interval; the Sobol gate was extended to the ±25% runs "
         "(30 cases) and a run with one shared level error added; months-ahead coverage now uses the range learned on the other period; "
         "typed sentences replaced by computed ones.",
         f"- **Added to the app (Tab 3, \"How we know\"):** \"All levers at once\" (5.6), \"Does the error grow the further ahead?\" (5.7) and "
         f"\"What this does not do\", now the full grouped list (5.9).", "",
         "## What it means for a planner", "",
         f"- **Trust totals more than single countries.** The international total is off {_pct(a['backtest_A_monthly_total'])} and "
         f"{_pct(a['backtest_B_monthly_total'])} by month; one country's month is off about {_pct(a['backtest_A_monthly_market'], 0)}, so read "
         "country results with their range.",
         f"- **A flight result is a range, and the range is mostly the check-in response.** {e_lo * 100:.0f}–{e_hi * 100:.0f}% of the "
         "uncertainty of a network-wide seat change is how strongly check-ins follow seats. Recording every route change and what "
         "hotels saw afterwards is what narrows it; a better baseline forecast would move it far less.",
         f"- **Demand and stay length move guest nights most.** With every lever moved ±10% together, demand and stay length explain "
         f"{st['non-flight demand'] * 100:.0f}% each of the spread, the connecting share {st['transfer share'] * 100:.0f}%, seats "
         f"{st['seats'] * 100:.0f}% and load factor {st['load factor'] * 100:.0f}%. Seats move guest nights less per percent because only part "
         "of each added seat becomes a hotel guest.",
         f"- **Within seven months the error does not grow with distance; beyond that, re-run as data arrive.** One country is off "
         f"{P[1]:.0f}% in month 1 and {min(P[h] for h in range(2, 8)):.0f}–{max(P[h] for h in range(2, 8)):.0f}% in months 2–7 with no "
         "steady rise; nothing is known past month 7, and the app flags those months.", "",
         "## Lines for the deck", "",
         f"- {_rebuild_review_line(r)}",
         f"- {_benchmark_line(b)}",
         f"- {so.deck_line(S)}",
         f"- {ah.deck_line(A)}", "",
         "## Reproduce", "",
         "| Step | Command | Result |", "|---|---|---|",
         "| 5.1 | `make snapshot`, `make snapshot-check` | `phase5/snapshot.json` |",
         "| 5.2 | `make rebuild` (about 2 hours on 2 cores) | `phase5/rebuild_compare.json` |",
         "| 5.3 | `make review` | `phase5/review_compare.json` |",
         "| 5.4 | `make candidates` | `phase5/candidates.json`, `candidates_bc.json` |",
         "| 5.5 | `make benchmark` | `phase5/benchmark.json` |",
         "| 5.6 | `make sobol` | `phase5/sobol.json` |",
         "| 5.7 | `make ahead` | `phase5/ahead.json` |",
         "| 5.9 | `make limits` | `phase5/limitations.json` |",
         "| 5.10 | `make phase5report` | this page |", ""]
    return "\n".join(L)


def write(cfg) -> str:
    text = build(load(cfg))
    (cfg["root"] / OUT).write_text(text, encoding="utf-8")
    return text
