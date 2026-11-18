"""Phase 5, Step 5.9 — one limitations list.

Every limitation of the simulator in one place, with its numbers read from the saved results (none typed), written once
and used everywhere:

  data/processed/phase5/limitations.json   Tab 3 ("What this does not do") reads it
  reports/limitations.md                    the full list, with where each number comes from, and the deck's shortlist
  docs/TECHNICAL.md                         the block between <!-- limitations:start --> and <!-- limitations:end -->

Run:  python -m atp.phase5 limits
"""
import json
import numpy as np
import pandas as pd

GROUPS = ["What the data cannot see", "The seat link", "How sure the numbers are", "Using the tool"]
START, END = "<!-- limitations:start -->", "<!-- limitations:end -->"


def _pct(x, d=0, signed=False) -> str:
    return f"{x:+.{d}f}%" if signed else f"{x:.{d}f}%"


def _minus(t: str) -> str:
    import re
    return re.sub(r"(?<![\w.])-(?=\d)", "−", t)


def stranded(cfg) -> dict:
    """The two disruption days in the data: check-ins against the same weekday in the three weeks before, and the
    excess as a share of the month (international check-ins; guest nights at the 1–2 nights stranded travellers stay)."""
    hi = pd.read_parquet(cfg["paths"]["processed"] / "hotel_intl_daily.parquet")
    d = hi.groupby("date")[["arrivals", "guests"]].sum()
    out = {}
    for day in ("2025-06-24", "2026-02-28"):
        D = pd.Timestamp(day); prev = [D - pd.Timedelta(days=7 * k) for k in (1, 2, 3)]
        normal = float(d.loc[prev].arrivals.mean()); actual = float(d.loc[D].arrivals); ex = actual - normal
        mon = d.loc[D.strftime("%Y-%m")]
        out[day] = {"checkins": actual, "normal": normal, "ratio": actual / normal, "excess": ex,
                    "share_of_month_checkins_pct": ex / float(mon.arrivals.sum()) * 100,
                    "share_of_month_guest_nights_pct": ([ex * n / float(mon.guests.sum()) * 100 for n in (1, 2)]
                                                        if mon.guests.sum() > 0 else None),       # the test file has no guest nights
                    "month_complete": bool(len(mon) == D.days_in_month)}
    return out


def after_disruption(cfg) -> dict:
    """Guests in-house after 24 Jun 2025, the survival model given the actual check-ins (fold B, chosen model): the error on
    the shock day, on 1–23 June, and on the next four days (Phase 1's own measure)."""
    P = cfg["paths"]["processed"]
    sel = json.load(open(P / "phase1" / "final_summary.json", encoding="utf-8"))["selection"]
    p = pd.read_parquet(P / "phase1" / f"pred_B_{sel['intl_model']}.parquet")
    i = p[p.market != "DOMESTIC"].groupby("date")[["y", "p"]].sum(); e = (i.p / i.y - 1) * 100
    jun = e.loc["2025-06-01":"2025-06-23"]; nxt = e.loc["2025-06-25":"2025-06-28"]
    return {"shock_day": float(e.loc["2025-06-24"]), "june_mean": float(jun.mean()), "june_sd": float(jun.std()),
            "after_lo": float(nxt.min()), "after_hi": float(nxt.max())}


def build(cfg) -> list:
    from .phase4 import FACTS, read_fact
    from . import phase5 as p5
    P = cfg["paths"]["processed"]
    F = lambda k: read_fact(P, FACTS[k])
    J = lambda rel: json.load(open(P / rel, encoding="utf-8"))
    L27, e22, sob, ah, bm = J("phase2/step27_limitations.json"), J("phase2/step22_checkins.json"), J("phase5/sobol.json"), J("phase5/ahead.json"), J("phase5/benchmark.json")
    va, j23, eb = J("phase3/step37_validation.json"), J("phase2/step23_lever_test.json"), J("phase3/effect_band.json")
    rv = {k: {"ours": v} for k, v in p5.ours_for_review(cfg).items()}                # our side of the 5.3 comparison, computed here
    q = p5._questioned(cfg)
    st, ad = stranded(cfg), after_disruption(cfg)
    lead = e22["folds"]["final"]["eps_total_lead"]
    lc, lci = lead["other_coefs"]["d_seats_lead2"], lead["other_ci_cluster"]["d_seats_lead2"]
    ac = P / "phase5" / "audit_checks.json"                  # the audit's convention (month effects counted, t with G − 1), when run
    if ac.exists():
        l2 = json.load(open(ac, encoding="utf-8"))["wild_cluster_bootstrap"]["lead2"]; lc, lci = l2["coef"], l2["ci95_cr1_t_G_minus_1"]
    s35 = J("phase3/step35_bands.json")["level_diagnostics"]["guests"]
    tier_min = min(v for d in s35["coverage_by_tier"].values() for v in d.values())
    tc = next(g for g in va["beyond_range_replay"]["groups"] if g["window"].startswith("test") and g["side"] == "cut" and not g["excluding_war_months"])
    g = ah["guests"]; P1 = g["pooled"]["by_nationality"]
    e = "ε (check-in response)"
    j24, j26 = st["2025-06-24"], st["2026-02-28"]
    best = {f: bm[f]["best_of_ten"] for f in "AB"}
    wins = [(lab, f) for m, lab in (("intl_daily_total", "international total by day"), ("intl_monthly_total", "international total by month"),
                                    ("dom_daily", "UAE residents by day"), ("dom_monthly", "UAE residents by month"))
            for f in "AB" if bm[f]["lightgbm"][m] < bm[f]["ours"][m]]
    m27 = va["march_2026"]["total"]["share_of_drop_explained_by_flights"]
    items = [
        # ------------------------------------------------------------ what the data cannot see
        dict(id="gateway", group=0, deck=True, step="2.7", source="phase2/step27_limitations.json",
             title="Abu Dhabi airport only.",
             text=f"Guests who arrive through Dubai or by road sit inside each country's response, not counted passenger by passenger: China has "
                  f"{L27['china_checkins_per_seat']:.2f} hotel check-ins per Abu Dhabi seat. {F('noflight_share_pct'):.1f}% of check-ins come from 12 "
                  f"nationalities with no flights in the data; they use the new-route response."),
        dict(id="demand", group=0, deck=True, step="3.7", source="phase3/step37_validation.json",
             title="Demand shocks are not flights.",
             text=f"War, travel advisories, visas and currency move guests directly. In March 2026 flights explain {m27[0]:.0%} to {m27[1]:.0%} of the fall "
                  f"in check-ins; the rest is the demand lever, which the user sets."),
        dict(id="stranded", group=0, deck=False, step="5.9", source="hotel_intl_daily.parquet (computed in limits.stranded)",
             title="Stranded travellers are not modelled.",
             text=f"When airspace closes, passengers who cannot fly on check in. On 24 Jun 2025 international check-ins were ×{j24['ratio']:.2f} the same "
                  f"weekday of the three weeks before, about {j24['excess']:,.0f} extra: {j24['share_of_month_checkins_pct']:.1f}% of June's check-ins "
                  f"and {j24['share_of_month_guest_nights_pct'][0]:.1f}–{j24['share_of_month_guest_nights_pct'][1]:.1f}% of its guest nights. On 28 Feb 2026, "
                  f"the first day of the 2026 conflict, ×{j26['ratio']:.2f}. Real but small, so no term is added."),
        dict(id="checkouts", group=0, deck=False, step="5.9", source="phase1/pred_B_*.parquet (Phase 1's disruption measure)",
             title="After a disruption, stays are shorter than usual.",
             text=f"Stranded travellers stay one or two nights, not their nationality's usual stay. After 24 Jun 2025 the stay model, given the actual "
                  f"check-ins, over-counted guests by {ad['after_lo']:.1f}–{ad['after_hi']:.1f}% on the next four days (normal June days: "
                  f"{0.0 if abs(ad['june_mean']) < 0.05 else ad['june_mean']:+.1f}% ± {ad['june_sd']:.1f}%). There is one such event in the data, too few to learn a rule from; "
                  f"28 Feb 2026 is the last day of the data."),
        dict(id="stays", group=0, deck=False, step="3.7", source="reports/phase3_report.md",
             title="Stays change in a shock.",
             text="In March 2026 guests stayed longer than usual. Stays are normal unless the user changes them with the stay lever."),
        dict(id="prices", group=0, deck=True, step="2.7", source="phase3/peak_guests.json",
             title="No prices and no room limit.",
             text=f"Hotel rates and airfares are not in the data, so demand reacting to price is outside the model. A scenario above the record "
                  f"{F('peak_guests'):,.0f} guests in-house is flagged, not capped."),
        # ------------------------------------------------------------ the seat link
        dict(id="follow", group=1, deck=True, step="2.7", source="phase2/step22_checkins.json",
             title="Seats partly follow demand.",
             text=f"Airlines add seats where demand is rising: seats two months ahead predict today's check-ins (coefficient {lc:+.2f}, 95% range "
                  f"{lci[0]:+.2f} to {lci[1]:+.2f}). The response is controlled for this and tested on unseen months, but it is a response seen in "
                  f"the data and used as an assumption, not a proven causal effect, so a pure capacity decision may achieve less."),
        dict(id="trim", group=1, deck=False, step="5.3", source="phase5 _questioned (independent review)",
             title="The response depends on leaving out the largest seat changes.",
             text=f"With every change kept it would be {q['untrimmed']['lead']:.2f} ({q['untrimmed']['plain']:.2f} without the control for seats following demand) instead of {q['eps']['default']:.2f}. "
                  f"On unseen months the trimmed value predicts better (best single value {q['skill']['all rows']['best']:.2f}, inside the 80% range "
                  f"{q['eps_range'][0]:.2f}–{q['eps_range'][1]:.2f})."),
        dict(id="drift", group=1, deck=False, step="5.3", source="phase2/step23_lever_test.json",
             title="The response moved over time.",
             text=f"Estimated on data to Jul 2024 it is {rv['eps_fold_A']['ours']:.2f}, to Dec 2024 {rv['eps_fold_B']['ours']:.2f}, to Jul 2025 "
                  f"{q['eps']['default']:.2f}; by half-year " + " / ".join(f"{v:.2f}" for v in q["drift"].values())
                  + ". The cause is not established; the final value was confirmed on Aug 2025–Feb 2026."),
        dict(id="sustained", group=1, deck=False, step="2.7", source="scenario.SHORT_DAYS",
             title="Sustained changes only.",
             text="The response comes from month-on-month changes a year apart, so it describes schedules that last weeks or months, not one "
                  "extra flight on one day; changes shorter than 28 days are flagged."),
        dict(id="deep", group=1, deck=False, step="3.7", source="phase3/step37_validation.json",
             title="Very large changes are extended, not estimated.",
             text=f"Below {F('trusted_lo') * 100:.0f}% the response is a straight line. When Wizz Air left Abu Dhabi, check-ins fell {-tc['actual_mean_pct']:.0f}% relative "
                  f"to other countries, against {-tc['line']['pred_mean_pct']:.0f}% predicted at the central value; the range for deep cuts covers it. "
                  f"Above +{F('trusted_hi') * 100:.0f}% the same line gives more than the fitted curve would, so very large additions may be overstated."),
        dict(id="newroute", group=1, deck=False, step="5.3", source="phase2/step25_new_route.json",
             title="Countries without flights borrow their response.",
             text=f"A new route from one of the 12 is worth {eb['per_seat']:.3f} check-ins per seat (80% range {q['new_route']['band'][0]:.3f}–"
                  f"{q['new_route']['band'][1]:.3f}), learned from new routes elsewhere. The range leaves out the response's own uncertainty; "
                  f"adding it widens it to {q['new_route']['with_eps_range'][0]:.3f}–{q['new_route']['with_eps_range'][1]:.3f}."),
        dict(id="season", group=1, deck=False, step="2.7", source="phase2/step27_limitations.json",
             title="One response all year.",
             text=f"It may be lower in winter than in summer (difference {L27['winter_minus_summer']:+.2f}, 95% range {L27['winter_minus_summer_ci'][0]:+.2f} "
                  f"to {L27['winter_minus_summer_ci'][1]:+.2f}), not significantly."),
        # ------------------------------------------------------------ how sure the numbers are
        dict(id="noisy", group=2, deck=True, step="2.6", source="phase4/facts.json",
             title="One country at a time is noisy.",
             text=f"A single country's month is off by about "
                  + (f"{F('backtest_A_monthly_market'):.0f}% in both held-out periods" if round(F('backtest_A_monthly_market')) == round(F('backtest_B_monthly_market'))
                     else f"{F('backtest_A_monthly_market'):.0f}% and {F('backtest_B_monthly_market'):.0f}% in the two held-out periods")
                  + f", against {F('backtest_A_monthly_total'):.1f}% and "
                  f"{F('backtest_B_monthly_total'):.1f}% for the international total, so country figures carry wider ranges."),
        dict(id="uncertainty", group=2, deck=True, step="5.6", source="phase5/sobol.json",
             title="Most of a flight result's range is the response itself.",
             text=f"For 10% more seats on every route over the winter, {sob['R1_shared']['S1'][e]:.0%}–{sob['R1']['S1'][e]:.0%} of the uncertainty "
                  f"is the check-in response and the rest the forecast level; for one country (India) {sob['R2']['S1'][e]:.0%}. A narrower response "
                  f"needs more seat changes with known outcomes."),
        dict(id="width", group=2, deck=False, step="5.3", source="phase5.ours_for_review (Step 5.3 quantities)",
             title="The 80% range was set on two periods, not confirmed on a third.",
             text=f"Its width was chosen so both held-out periods reach 80% ({rv['coverage_B_to_A']['ours']:.1%} and {rv['coverage_A_to_B']['ours']:.1%}), "
                  f"so those figures describe the choice rather than confirm it. The two periods share January and February 2025, and one size "
                  f"group of countries reached only {tier_min:.0%} in one direction. The international total's monthly range is "
                  f"the widest gap seen in 14 months and is not calibrated: in each period the other period's range would have held "
                  f"{rv['total_cov_A_to_B']['ours']:.0%} and {rv['total_cov_B_to_A']['ours']:.0%} of months. A single country's own range would hold "
                  f"{rv['own_market_cov_A_to_B']['ours']:.0%}–{rv['own_market_cov_B_to_A']['ours']:.0%}, which is why ranges are pooled by country size."),
        dict(id="testrange", group=2, deck=False, step="audit", source="phase3/effect_band.json",
             title="The response's range comes from one test window.",
             text=f"Its 80% range, {eb['eps_lo']:.2f} to {eb['eps_hi']:.2f}, is the sampling error of a single check on Aug 2025–Feb 2026, "
                  f"not how the response may drift over time, nor a causal range. A bootstrap on the training data gives a wider "
                  f"{eb['cross_checks']['bootstrap_80pct_train_window'][0]:.2f} to {eb['cross_checks']['bootstrap_80pct_train_window'][1]:.2f}."),
        dict(id="horizon", group=2, deck=True, step="5.7", source="phase5/ahead.json",
             title="Months 8 to 12 are untested.",
             text=f"In months 1 to 7 after the data the error does not grow with distance ({P1['1']:.0f}% in month 1, "
                  f"{min(P1[str(h)] for h in range(2, 8)):.0f}–{max(P1[str(h)] for h in range(2, 8)):.0f}% after, one country at a time); the 80% range "
                  f"held {min(min(g[f]['coverage'].values()) for f in 'AB'):.0%}–{max(max(g[f]['coverage'].values()) for f in 'AB'):.0%} of countries by "
                  f"month ahead, least in the middle months. Nothing is known beyond month 7: results from October 2026 are flagged."),
        dict(id="residents", group=2, deck=False, step="5.3", source="phase2/step26_endtoend.json",
             title="UAE residents are the hardest series.",
             text=f"On the official test window their forecast was worse than repeating last year (daily {q['domestic_test']['model'][0]:.1f}% against "
                  f"{q['domestic_test']['last_year'][0]:.1f}%, bias {q['domestic_test']['bias']:+.0f}%), and the bias changes sign between periods "
                  f"({L27['domestic_bias_pct']['A']:+.0f}%, {L27['domestic_bias_pct']['B']:+.0f}%). No flight lever moves them."),
        dict(id="benchmark", group=2, deck=False, step="5.5", source="phase5/benchmark.json",
             title="A machine-learning model does better on some totals.",
             text=f"Given the same inputs, LightGBM is less accurate country by country (in both held-out periods), but it beat our model on "
                  + "; ".join(f"{lab} in the {'first' if f == 'A' else 'second'} period" for lab, f in wins)
                  + ". It cannot run scenarios through the chain or give a range, so it is not used; the gap on totals is unexplained."),
        dict(id="realised", group=2, deck=False, step="2.7", source="reports/phase2_report.md",
             title="Tested on the flights that actually flew.",
             text="Every back-test was given the seats that operated, including cancellations; a forecast made from published schedules in advance "
                  "would be somewhat less accurate."),
        # ------------------------------------------------------------ using the tool
        dict(id="ramadan", group=3, deck=False, step="2.7", source="phase2/step27_limitations.json",
             title="Holidays inside Ramadan are untested.",
             text="In 2026 Ramadan covers the national holidays of Kuwait and Saudi Arabia, a combination the data never show; the model multiplies the two effects."),
        dict(id="versions", group=3, deck=False, step="2.7", source="reports/phase2_report.md",
             title="The test window was scored once.",
             text="It was scored with the first version of the check-in forecast; later versions were chosen on the two earlier periods only and "
                  "never scored on it."),
        dict(id="sameday", group=3, deck=False, step="audit", source="engine.Engine.run (independent audit, 12 Oct)",
             title="Scenarios act from the first day.",
             text="A scenario applies the response to each day's passengers at once, while the normal-year forecast smooths seats over 7 days. "
                  "Over weeks the two agree; in the first days of a sudden change the scenario moves faster than the forecast would. Changes "
                  "in how full planes are or in connecting passengers use the seat response, an assumption without separate evidence."),
        dict(id="screen", group=3, deck=False, step="5.9", source="Tab 3 at 390 px (audit of 5.3–5.7)",
             title="Built for a desktop screen.",
             text="The app works at laptop widths and above; on a phone the evidence tab scrolls sideways."),
    ]
    for it in items:
        it["group"] = GROUPS[it["group"]]; it["text"] = _minus(it["text"]); it["title"] = _minus(it["title"])
    return items


def tab3(items: list) -> list:
    return [{"group": gname, "items": [{"title": i["title"], "text": i["text"]} for i in items if i["group"] == gname]} for gname in GROUPS]


def markdown(items: list, with_source: bool = True) -> str:
    L = []
    for gname in GROUPS:
        L += [f"**{gname}**", ""]
        L += [f"- **{i['title']}** {i['text']}" + (f" _(Step {i['step']}; {i['source']})_" if with_source else "") for i in items if i["group"] == gname]
        L += [""]
    return "\n".join(L)


def report(items: list) -> str:
    deck = [i for i in items if i["deck"]]
    return "\n".join([
        "# Limitations — the one list", "",
        "_Generated by `python -m atp.phase5 limits` (Step 5.9); every number is read from the saved results named after it. Tab 3 "
        "(\"What this does not do\") and the README show this same list; do not edit by hand._", "",
        f"{len(items)} limitations in {len(GROUPS)} groups.", "",
        markdown(items),
        "## For the deck (shortlist)", "",
        *[f"- **{i['title']}** {i['text']}" for i in deck], "",
        "## Step 5.8", "",
        "Step 5.8 (carry accepted changes through every phase) was not triggered: no candidate in Step 5.4 passed its pre-registered rule, "
        "so the production model is unchanged.", ""])


def update_readme(root, items: list) -> bool:
    p = root / "docs" / "TECHNICAL.md"; s = p.read_text(encoding="utf-8")
    block = f"{START}\n_Generated from `reports/limitations.md` (Step 5.9)._\n\n{markdown(items, with_source=False)}{END}"
    if START in s:
        a, b = s.index(START), s.index(END) + len(END)
        new = s[:a] + block + s[b:]
    else:                                                   # first time: its own section, after the independent review
        at = s.index("## Reproduce Phase 0")
        new = s[:at] + "## Limitations\n\n" + block + "\n\n" + s[at:]
    changed = new != s
    p.write_text(new, encoding="utf-8")
    return changed
