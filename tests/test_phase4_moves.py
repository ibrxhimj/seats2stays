"""Step 4.6: "What moves guests". The tab's numbers are the Phase 3 sensitivity tables; the planner rules quote them;
a heat-map cell equals the full engine run of the change the page loads from it; the reverse sum's answer, loaded as a
change, delivers the target; and in a real browser the tab, the click-through, the keyboard and the reverse sentence
work. Browser tests are skipped without Playwright + Chromium."""
import math
import numpy as np
import pandas as pd
import pytest
from atp.config import load_config
from atp.simulator import Simulator, CELL_RULES, LEVERS, est, label, sig3

cfg = load_config(); P = cfg["paths"]["processed"]
if not (P / "phase4" / "facts.json").exists():
    pytest.skip("run `python -m atp.phase4 data` first", allow_module_level=True)
S = Simulator(cfg)
M = S.moves()
MM = pd.read_csv(P / "phase3" / "sens_market_month.csv")
RK = pd.read_csv(P / "phase3" / "sens_market_ranking.csv")
SE = pd.read_csv(P / "phase3" / "sens_season.csv").set_index("month").guest_nights_per_1000_seats
H0, H1 = str(S.w.h0.date()), str(S.w.h1.date())


def cell_change(market, ym):
    """The change the page builds when a heat-map cell is clicked (web/js/moves.js, newRouteSpec)."""
    p = pd.Period(ym)
    return {"type": "new_route", "market": market, "city": S.place[market]["place"], "airline": "generic", "start": str(p.start_time.date()),
            "end": str(p.end_time.date()), "flights_per_week": 7, "aircraft_class": "narrowbody"}


# ----------------------------------------------------------------------------- the levers
def test_lever_ranking_is_the_tables_and_equals_live_runs():
    rows = M["levers"]["rows"]
    assert [r["id"] for r in rows] == ["non-flight demand", "stay length", "transfer share", "seats", "load factor"]
    assert [r["elasticity"] for r in rows] == [1.0, 1.0, 0.44, 0.19, 0.19] and [r["opposite"] for r in rows] == [False, False, True, False, False]
    assert {r["id"] for r in rows} == {t[0] for t in LEVERS}
    by = {r["id"]: r for r in rows}
    assert by["transfer share"]["up"]["note"] and "66%" in by["transfer share"]["up"]["note"] and by["transfer share"]["down"]["note"] is None
    assert "7.3%" in by["load factor"]["up"]["note"] and by["seats"]["up"]["note"] is None
    for lever, kind, pct_ in [("seats", "route", 10), ("stay length", "stay_length", -10)]:
        ch = {"type": kind, "markets": "flight markets" if kind == "route" else "all", "start": H0, "end": H1, ("seats_pct" if kind == "route" else "pct"): pct_}
        h = S.run({"changes": [ch]})["headline"]; side = by[lever]["up" if pct_ > 0 else "down"]
        for k, hk in (("d_guest_nights", "d_guest_nights"), ("lo", "lo"), ("hi", "hi")):
            assert side[k] == pytest.approx(h[hk], rel=1e-5), (lever, k)
    assert M["levers"]["base_guest_nights"] == pytest.approx(15.08e6, rel=1e-3)
    assert "61%" in M["levers"]["scope"] and "35%" in M["levers"]["scope"]


# ----------------------------------------------------------------------------- the grid
def test_grid_is_the_tables():
    G = M["grid"]
    assert len(G["rows"]) == 45 and G["months"][0] == "2026-03" and G["months"][-1] == "2027-02" and len(G["month_labels"]) == 12
    flying = [r for r in G["rows"] if r["flights"]]
    assert len(flying) == 33 and all(r["flights"] for r in G["rows"][:33]) and not any(r["flights"] for r in G["rows"][33:])
    for grp in (G["rows"][:33], G["rows"][33:]):
        assert [r["year"][0] for r in grp] == sorted((r["year"][0] for r in grp), reverse=True)
    rk = RK.set_index("market")
    for r in G["rows"]:
        t = rk.loc[r["market"]]
        assert r["year"] == [pytest.approx(t.guest_nights_per_1000_seats, abs=0.06), pytest.approx(t.lo, abs=0.06), pytest.approx(t.hi, abs=0.06), CELL_RULES.index(t.rule)]
        assert r["label"] == label(r["market"]) and len(r["cells"]) == 12
        if not r["flights"]:
            assert all(c[4] == 4 for c in r["cells"])
        g = MM[MM.market == r["market"]].set_index("month").loc[G["months"]]
        for c, (mo, t) in zip(r["cells"], g.iterrows()):
            assert c[0] == pytest.approx(t.d_guest_nights / t.seats_added * 1000, abs=0.06) and c[3] == pytest.approx(t.d_guest_nights, abs=0.06)
            assert c[1] == pytest.approx(t.lo, abs=0.06) and c[2] == pytest.approx(t.hi, abs=0.06) and c[4] == CELL_RULES.index(t.rule)
    assert G["seats_per_flight"] == pytest.approx(S.w.aircraft["narrowbody"]) and len(G["rules"]) == 4
    # the "all markets with flights" row: guest nights over seats, markets whose response is borrowed left out
    ok = MM[MM.rule != CELL_RULES[4]].groupby("month").apply(lambda g: g.d_guest_nights.sum() / g.seats_added.sum() * 1000, include_groups=False)
    assert G["season"] == [pytest.approx(v, abs=0.06) for v in ok.loc[G["months"]]] and G["season"] == [pytest.approx(v, abs=0.06) for v in SE.loc[G["months"]]]


@pytest.mark.parametrize("market,i", [("GERMANY", 9), ("CHINA", 10), ("AUSTRALIA", 0), ("KAZAKHSTAN", 1), ("UZBEKISTAN", 5), ("QATAR", 8)])
def test_a_cell_equals_the_full_run_of_the_change_the_page_loads(market, i):
    G = M["grid"]; row = next(r for r in G["rows"] if r["market"] == market); c = row["cells"][i]
    r = S.run({"changes": [cell_change(market, G["months"][i])]}); h = r["headline"]
    assert h["d_guest_nights"] == pytest.approx(c[3], abs=0.06) and h["lo"] == pytest.approx(c[1], abs=0.06) and h["hi"] == pytest.approx(c[2], abs=0.06)
    assert r["chain"]["stages"][0]["value"] == pytest.approx(G["seats_added"][i], abs=0.06)
    assert h["d_guest_nights"] / r["chain"]["stages"][0]["value"] * 1000 == pytest.approx(c[0], abs=0.06)


# ----------------------------------------------------------------------------- the planner rules
def test_rules_quote_the_tables():
    rules = {r["id"]: r["evidence"] for r in M["rules"]}
    assert list(rules) == ["where", "when", "who", "beyond"]
    ok = RK[RK.rule == "seat lever"]; top, low = ok.iloc[0], ok.iloc[-1]
    assert f"from {label(top.market)} brings about {sig3(top.guest_nights_per_1000_seats):,.0f} guest nights per 1,000 seats" in rules["where"]
    assert f"from {label(low.market)}, {sig3(low.guest_nights_per_1000_seats):,.0f}." in rules["where"] and f"about {round(top.guest_nights_per_1000_seats / low.guest_nights_per_1000_seats)}-fold" in rules["where"]
    # the table puts Kazakhstan first; the rule says why China is cited, and names every † country that ranks above it
    above = RK[RK.rule.str.startswith("beyond") & (RK.guest_nights_per_1000_seats > top.guest_nights_per_1000_seats)]
    assert list(above.market) == ["KAZAKHSTAN"] and M["grid"]["rows"][0]["market"] == "KAZAKHSTAN" and M["grid"]["rows"][0]["year"][3] == 1
    assert "the most of any country within the model's tested range" in rules["where"]
    assert f"ranks {label('KAZAKHSTAN')} ({sig3(above.iloc[0].guest_nights_per_1000_seats):,.0f}) higher, marked †: a straight-line extension" in rules["where"]
    assert "Uzbekistan" not in rules["where"] and "Azerbaijan" not in rules["where"]                      # † countries ranked below it are not mentioned
    d = M["grid"]["dagger"]                                                                           # the legend's dagger, in the model's own numbers
    assert d.startswith("† Beyond the tested range") and "(−29% to +109%)" in d and "straight line, not estimated" in d and "Mostly" not in d
    hi, lo = SE.idxmax(), SE.idxmin()
    assert f"added in {pd.Period(hi).strftime('%B')} brings about {sig3(SE[hi]):,.0f}" in rules["when"] and f"in {pd.Period(lo).strftime('%B')}, {sig3(SE[lo]):,.0f} (" in rules["when"]
    assert f"({round((1 - SE[lo] / SE[hi]) * 100)}% less)" in rules["when"]
    el = pd.read_csv(P / "phase3" / "sens_levers.csv").groupby("lever").elasticity_small_change.apply(lambda x: x.abs().mean())
    assert f"{el['transfer share']:.2f}% more guest nights, {el['transfer share'] / el['seats']:.2g} times" in rules["who"] and f"({el['seats']:.2f}%)" in rules["who"]
    assert f"{S.meta()['network']['transfer_share'] * 100:.0f}% of passengers connect onwards" in rules["who"]
    assert f"{el['stay length']:.2f}% more guest nights, {el['stay length'] / el['seats']:.2g} times" in rules["beyond"]
    o = lambda k: f"{S.fact(k) * 100:.0f}%"
    assert f"about {o('offset_10')} more seats (80% range {o('offset_10_lo')} to {o('offset_10_hi')})" in rules["beyond"]
    # the offset is the engine's own rule: (1/(1-s))^(1/eps) - 1, with eps and its range
    for k, e in (("offset_10", S.fact("eps")), ("offset_10_lo", S.fact("eps_hi")), ("offset_10_hi", S.fact("eps_lo"))):
        assert S.fact(k) == pytest.approx((1 / 0.9) ** (1 / e) - 1, rel=1e-9)
    assert "0.44% more guest nights" in M["lead"] and "(0.19%)" in M["lead"] and "2.3 times" in M["lead"] and "one for one by definition" in M["lead"]
    assert f"about {o('offset_10')} more seats" in M["lead"]


# ----------------------------------------------------------------------------- the reverse sum, loaded as a change
@pytest.mark.parametrize("market,aircraft,window,target", [("UNITED KINGDOM", "today", ("2026-11-01", "2027-02-28"), 100_000),
                                                           ("UNITED KINGDOM", "narrowbody", ("2026-11-01", "2027-02-28"), 50_000),
                                                           ("MEXICO", "widebody", ("2026-11-01", "2027-02-28"), 20_000)])
def test_the_reverse_answer_loaded_as_a_change_delivers_the_target(market, aircraft, window, target):
    r = S.reverse(market, *window, target, aircraft)
    f, cur = r["flights_per_week"], r["current_flights_per_week"]
    if r["aircraft"] == "today":                                         # what moves.js builds: the lever's seats_pct
        ch = {"type": "route", "markets": market, "start": window[0], "end": window[1], "seats_pct": 100 * f / cur}
    else:
        ch = {"type": "new_route", "market": market, "city": S.place[market]["place"], "airline": "generic", "start": window[0], "end": window[1],
              "flights_per_week": f, "aircraft_class": r["aircraft"]}
    assert S.run({"changes": [ch]})["headline"]["d_guest_nights"] == pytest.approx(target, rel=2e-3)


def test_moves_endpoint():
    from fastapi.testclient import TestClient
    from atp import api
    api.state["sim"] = S
    c = TestClient(api.app)
    r = c.get("/api/moves"); assert r.status_code == 200
    j = r.json(); assert set(j) == {"span", "lead", "levers", "grid", "rules"} and j["grid"]["rows"][0]["cells"][0] and len(j["rules"]) == 4
    assert j["levers"]["rows"][0]["up"]["d_guest_nights"] == pytest.approx(M["levers"]["rows"][0]["up"]["d_guest_nights"], rel=1e-6)
    assert c.get("/api/moves").json() == j                                # cached, identical


# ----------------------------------------------------------------------------- in a browser
def to_moves(page):
    page.click(".tabs a[data-tab=moves]")
    page.wait_for_selector("#tor svg.tor-chart rect", timeout=20000)
    page.wait_for_function("document.getElementById('rv-n') && document.getElementById('rv-n').textContent !== '…'", timeout=20000)


def last_run(page):
    return [b for u, b in page.posts if u == "run"][-1]


def test_tab_two_shows_the_four_answers(page):
    to_moves(page)
    assert page.get_attribute(".tabs a[aria-current=page]", "data-tab") == "moves" and not page.is_visible(".rail") and not page.is_visible("#tab-simulator")
    assert page.inner_text("#mv-lead") == M["lead"]
    names = [t for t in page.locator(".tor-label b").all_inner_texts()]
    assert names == [r["name"] for r in M["levers"]["rows"]] and page.locator("#tor svg.tor-chart rect").count() == 10    # two bars a lever
    assert page.locator("#tor-notes li").count() == 2 and "†" in page.inner_text("#tor-notes") and "‡" in page.inner_text("#tor-notes")
    assert page.locator("#heat tbody tr[data-m]:visible").count() == 10 and page.is_visible("#heat tr.avg")                 # the top ten, then "Show all"
    assert page.inner_text("#heat-more") == "Show all 45 markets"
    page.click("#heat-more"); assert page.locator("#heat tbody tr[data-m]:visible").count() == 45 and page.is_visible("#heat tr.grp")
    assert page.inner_text("#heat-more") == "Show fewer markets"
    page.click("#heat-more"); assert page.locator("#heat tbody tr[data-m]:visible").count() == 10
    assert page.locator("#rules li").count() == 4 and page.inner_text("#rules li >> nth=0").startswith("Rank routes by hotel nights per seat")
    assert "ranks Kazakhstan (965) higher, marked †" in page.inner_text("#rules li >> nth=0")        # the rule explains the table's first row
    legend = page.inner_text("#heat-legend")
    assert "† Beyond the tested range" in legend and "(−29% to +109%)" in legend and "Mostly less certain" not in legend
    assert page.get_attribute("#heat abbr.dag >> nth=0", "title").startswith("Beyond the range the seat lever was tested on")
    assert not page.errors


def test_a_cell_loads_into_the_simulator_with_the_same_numbers(page):
    to_moves(page)
    ch = cell_change("CHINA", "2026-12")
    page.hover("#heat tr[data-m=CHINA] .cell[data-i='9']")
    read = page.inner_text("#heat-read"); c = next(r for r in M["grid"]["rows"] if r["market"] == "CHINA")["cells"][9]
    assert "China, December 2026" in read and "5,409 seats" in read and f"{round(sig3(c[3])):,}" in read and f"{int(c[0] + 0.5)} per 1,000 seats" in read
    page.click("#heat tr[data-m=CHINA] .cell[data-i='9']")
    page.wait_for_selector("#tab-simulator:not([hidden])")
    page.wait_for_function("document.getElementById('big').textContent === '+3,670'", timeout=20000)
    assert last_run(page) == {"changes": [ch], "response": "central"}                                                    # exactly one new change, nothing else
    assert page.inner_text("#big") == S.run({"changes": [ch]})["headline"]["text"] == "+" + f"{round(sig3(c[3])):,}"
    assert page.inner_text("#range-text") == f"80% range +{round(sig3(c[1])):,} to +{round(sig3(c[2])):,}"
    assert "New route: Shanghai (any airline)" in page.inner_text("#strips") and "7 flights a week, narrowbody, Dec 2026" in page.inner_text("#strips")
    assert page.locator("#months text").count() > 10                                              # the charts are drawn on return
    page.click("#strips .strip.form [data-edit]")                                                 # and it can be edited like any new route
    assert page.input_value("[name=market]") == "CHINA" and page.input_value("[name=airline]") == "generic" and page.input_value("[name=fpw]") == "7"
    assert not page.errors


def test_the_grid_reads_with_a_keyboard_and_one_tab_stop(page):
    to_moves(page)
    assert page.locator('#heat .cell[tabindex="0"]').count() == 1
    page.focus('#heat .cell[tabindex="0"]')
    assert "Kazakhstan, March 2026" in page.inner_text("#heat-read")
    page.keyboard.press("ArrowRight"); page.keyboard.press("ArrowDown")
    assert "China, April 2026" in page.inner_text("#heat-read") and page.locator('#heat .cell[tabindex="0"]').count() == 1
    page.keyboard.press("ArrowUp"); page.keyboard.press("ArrowLeft"); assert "Kazakhstan, March 2026" in page.inner_text("#heat-read")
    page.hover("#heat tr[data-m=KAZAKHSTAN] .cell[data-i='1']"); assert "straight line" in page.inner_text("#heat-read")      # why it is less certain
    page.hover("#heat tr[data-m=CHINA] td.yr"); assert "China, Mar 2026 to Feb 2027: 853 guest nights per 1,000 seats" in page.inner_text("#heat-read")
    assert not page.errors


def test_reverse_sentence_answers_and_loads(page):
    to_moves(page)
    r = page.api("/api/reverse", {"market": "UNITED KINGDOM", "start": "2026-11-01", "end": "2027-02-28", "target": 10000, "aircraft": "today"})
    assert page.inner_text("#rv-n") == str(round(r["flights_per_week"])) and f"{round(r['flights_per_week_range'][0])} to {round(r['flights_per_week_range'][1])}" in page.inner_text("#rv-detail")
    page.select_option("#rv-market", "MEXICO")                                                    # no flights: a new route is the only way
    page.wait_for_function("document.getElementById('rv-detail').textContent.includes('no flights')")
    assert page.input_value("#rv-air") == "narrowbody"
    assert page.evaluate("document.querySelector('#rv-air option[value=today]').disabled")
    page.select_option("#rv-market", "UNITED KINGDOM"); page.select_option("#rv-air", "narrowbody")
    page.fill("#rv-target", "abc"); page.wait_for_function("document.getElementById('rv-detail').textContent.startsWith('Enter a number')")
    assert not page.is_enabled("#rv-add")
    page.fill("#rv-target", "50000"); page.wait_for_function("/^Add \\d[\\d,]* flights? a week/.test(document.getElementById('rv-add').textContent)", timeout=10000)
    n = int(page.inner_text("#rv-n").replace(",", ""))
    assert page.inner_text("#rv-add") == f"Add {n} flights a week to the simulator"
    page.click("#rv-add"); page.wait_for_selector("#tab-simulator:not([hidden])")
    page.wait_for_function("document.getElementById('strips').innerText.includes('any airline')", timeout=20000)
    ch = last_run(page)["changes"][0]
    assert (ch["type"], ch["market"], ch["airline"], ch["aircraft_class"], ch["flights_per_week"], ch["start"], ch["end"]) == ("new_route", "UNITED KINGDOM", "generic", "narrowbody", n, "2026-11-01", "2027-02-28")
    got = float(page.inner_text("#big").replace(",", "").replace("+", ""))
    assert abs(got - 50000) / 50000 < 0.03                                                         # whole flights: within a few percent of the target
    page.click(".tabs a[data-tab=moves]"); page.select_option("#rv-air", "today"); page.wait_for_function("document.getElementById('rv-detail').textContent.includes('more seats')")
    page.click("#rv-add"); page.wait_for_function("document.getElementById('strips').innerText.includes('flights: +')", timeout=20000)
    assert last_run(page)["changes"][0]["type"] == "route" and "seats_pct" in last_run(page)["changes"][0] and not page.errors


def test_reverse_sentence_survives_extreme_targets(page):
    to_moves(page)
    asked = lambda: len([1 for u, _ in page.posts if u == "reverse"])
    n0 = asked()
    for bad in ("0", "-500", "abc", ""):                                                          # not a target: said so, no question sent
        page.fill("#rv-target", bad); page.wait_for_function("document.getElementById('rv-detail').textContent.startsWith('Enter a number')")
        assert page.get_attribute("#rv-target", "aria-invalid") == "true" and not page.is_enabled("#rv-add") and page.inner_text("#rv-n") == "…"
    page.wait_for_timeout(500); assert asked() == n0
    for huge in ("5000000", "1e300", "99999999999999999999"):                                     # more than the simulator can reach: the ceiling, not a number of flights
        page.fill("#rv-target", huge); page.wait_for_timeout(450)                                  # the 220 ms debounce, then the answer
        page.wait_for_function("document.getElementById('rv-detail').textContent.includes('out of reach') && !document.getElementById('rv').hasAttribute('aria-busy')", timeout=15000)
        assert page.get_attribute("#rv-target", "aria-invalid") == "false" and not page.is_enabled("#rv-add")
        assert page.inner_text("#rv-n") == "more than 100" and not page.is_visible("#rv-about") and "even 100 more flights a week would win only about" in page.inner_text("#rv-detail")
        assert len(page.inner_text("#rv-detail")) < 220
    page.fill("#rv-target", "1")                                                                  # a sliver of a flight: no error, one flight to add
    page.wait_for_function("document.getElementById('rv-add').textContent === 'Add 1 flight a week to the simulator'", timeout=15000)
    assert "needs under 1 more flight a week" in page.inner_text(".rv-sentence") and page.get_attribute("#rv-target", "aria-invalid") == "false"
    assert "under 1% more seats" in page.inner_text("#rv-detail")
    page.fill("#rv-target", "150000")                                                             # reachable, but past what the lever was tested on: said so
    page.wait_for_function("document.getElementById('rv-detail').textContent.includes('Beyond the range the model was tested on')", timeout=15000)
    assert "about 81" in page.inner_text(".rv-sentence") and "80% range 50 to more than 100" in page.inner_text("#rv-detail") and page.is_enabled("#rv-add")
    assert asked() > n0 and all(b["target"] > 0 for u, b in page.posts if u == "reverse") and not page.errors


def test_the_lever_follows_the_change_on_screen(page):
    assert page.eval_on_selector("#lever-market", "e => e.value") == "UNITED KINGDOM"
    to_moves(page)
    page.click("#heat tr[data-m=CHINA] .cell[data-i='9']")
    page.wait_for_function("document.getElementById('big').textContent === '+3,670'", timeout=20000)
    page.wait_for_function("!document.getElementById('lever-slider').disabled && document.getElementById('lever-market').value === 'CHINA'", timeout=20000)
    assert page.inner_text("#lever-season [aria-pressed=true]") == "Winter" and "18 flights a week" in page.inner_text("#lever-read")
    page.focus("#lever-slider"); page.keyboard.press("ArrowRight"); page.keyboard.press("ArrowRight")           # the drag an evaluator reaches for
    page.wait_for_function("document.getElementById('strips').innerText.includes('China flights, winter')", timeout=20000)
    strips = page.inner_text("#strips")
    assert "United Kingdom flights" not in strips and "New route: Shanghai" in strips                             # China's flights were changed; the clicked route stays
    page.select_option("#lever-market", "UNITED KINGDOM")                                                          # the lever wandered off ...
    page.wait_for_function("document.getElementById('lever-market').value === 'UNITED KINGDOM'")
    page.click("#strips .strip >> nth=1 >> .what")                                                                 # ... and selecting the Shanghai strip brings it back
    page.wait_for_function("document.getElementById('lever-market').value === 'CHINA'", timeout=20000)
    assert not page.errors


def test_picking_a_country_or_season_updates_the_screen_at_once(page):
    """Regression: choosing a country (list, map or drop-down) or a season changed only the drop-down; the title and the
    charts kept the previous result until the slider was dragged."""
    title = lambda: page.inner_text("#viewing").lower()
    y = page.evaluate("document.documentElement.scrollHeight")
    page.click("#list li >> nth=1")                                                # a country in the preset's list
    page.wait_for_function("document.getElementById('viewing').textContent.endsWith('flights, winter') && "
                           "document.getElementById('big').textContent === '0'", timeout=20000)
    assert page.evaluate("document.documentElement.scrollHeight") == y            # the page does not jump
    page.select_option("#lever-market", "ITALY")
    page.wait_for_function("document.getElementById('viewing').textContent === 'Italy flights, winter'", timeout=20000)
    assert page.inner_text("#big") == "0" and "Italy today" in page.inner_text("#lead")
    page.click('#lever-season [data-season="summer"]')
    page.wait_for_function("document.getElementById('viewing').textContent === 'Italy flights, summer'", timeout=20000)
    page.eval_on_selector("#lever-slider", "el => { el.value = String(+el.value + 4); el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change')); }")
    page.wait_for_function("document.getElementById('big').textContent !== '0'", timeout=20000)
    italy = page.inner_text("#big")
    page.select_option("#lever-market", "UNITED KINGDOM")                         # another country
    page.wait_for_function("document.getElementById('viewing').textContent === 'United Kingdom flights, summer'", timeout=20000)
    assert page.inner_text("#big") == "0"
    page.click('#lever-season [data-season="year"]')
    page.wait_for_function("document.getElementById('viewing').textContent.endsWith('flights, whole year')", timeout=20000)
    page.select_option("#lever-market", "ITALY"); page.click('#lever-season [data-season="summer"]')   # back to the change made: it is shown again
    page.wait_for_function(f"document.getElementById('big').textContent === {italy!r}", timeout=20000)
    assert title() == "italy flights, summer" and "Italy flights, summer" in page.inner_text("#strips .strip.active")
    assert not page.errors


def test_any_airline_in_the_new_route_form(page):
    page.click("#add-change"); page.click("#fd-kinds label:has-text('Open a new route')")
    assert page.evaluate("[...document.querySelectorAll('[name=airline] option')].at(-1).value") == "generic"
    page.select_option("[name=market]", "UNITED KINGDOM"); page.select_option("[name=airline]", "generic")
    assert "own" in page.inner_text("#fd-hint") and "UNITED KINGDOM" not in page.inner_text("#fd-hint")
    page.fill("[name=city]", "London"); page.click("#fd-form .fd-add"); page.wait_for_selector("#form-dialog", state="hidden", timeout=30000)
    ch = last_run(page)["changes"][0]
    assert ch["airline"] == "generic" and "New route: London (any airline)" in page.inner_text("#strips") and not page.errors


def test_tabs_swap_and_the_simulator_redraws_after_a_resize_elsewhere(page):
    to_moves(page)
    page.set_viewport_size({"width": 1300, "height": 1000}); page.wait_for_timeout(400)               # resized while the simulator is hidden
    page.click(".tabs a[data-tab=simulator]")
    assert page.is_visible("#tab-simulator") and not page.is_visible("#tab-moves") and page.is_visible(".rail")
    assert page.locator("#months text").count() > 10 and page.get_attribute(".tabs a[aria-current=page]", "data-tab") == "simulator"
    page.click(".tabs a[data-tab=know]"); assert not page.is_visible("#tab-simulator") and not page.is_visible("#tab-moves") and not page.is_visible(".rail")
    assert not page.errors


# ----------------------------------------------------------------------------- step 4.7: "How we know"
def test_know_numbers_are_the_frozen_tables():
    K = S.know(); P2 = P / "phase2"
    bm = pd.read_csv(P / "phase4" / "backtest_monthly.csv")
    f = S.fact
    assert K["stats"][0]["big"] == f"{f('backtest_A_monthly_total'):.1f}% · {f('backtest_B_monthly_total'):.1f}%"
    assert K["stats"][1]["big"] == f"{f('backtest_A_monthly_market'):.1f}% · {f('backtest_B_monthly_market'):.1f}%"
    tot = K["series"][0]
    for fold, fid in zip(K["folds"], "AB"):
        g = bm[(bm.fold == fid) & (bm.market != "DOMESTIC")].groupby("month")[["actual_guest_nights", "rebuilt_guest_nights"]].sum().sort_index()
        rows = tot["rows"][fid]
        assert [r[0] for r in rows] == pytest.approx(g.actual_guest_nights.round(1).tolist())
        assert [r[1] for r in rows] == pytest.approx(g.rebuilt_guest_nights.round(1).tolist())
        wm = (g.actual_guest_nights - g.rebuilt_guest_nights).abs().sum() / g.actual_guest_nights.sum() * 100       # the published WMAPE
        assert wm == pytest.approx(tot["error"][fid == "B"], abs=0.01)
    assert tot["error"] == pytest.approx([f("backtest_A_monthly_total"), f("backtest_B_monthly_total")], abs=1e-5)
    for s in K["series"]:                                                 # a range always contains the rebuilt value and is ordered
        for fid in "AB":
            for a, r, lo, hi in s["rows"][fid]:
                assert lo <= r <= hi and a >= 0
    assert [s["key"] for s in K["series"][1:]] == sorted([s["key"] for s in K["series"][1:]], key=label)
    assert K["series"][0]["label"] == "All international guests"


def test_know_march_and_catch_are_the_tables():
    K = S.know()
    mr = pd.read_csv(P / "phase3" / "march_2026_replay_by_region.csv")
    assert sum(r["normal_checkins"] for r in K["march"]["rows"]) == pytest.approx(mr.normal_checkins.sum(), rel=1e-4)
    assert len(K["march"]["rows"]) == len(mr)
    for r in K["march"]["rows"]:
        assert (r["flights"] is None) == (r["flight_markets"] == 0)
        if r["flights"]: assert r["flights"][0] <= r["flights"][1]
    pv = pd.read_csv(P / "phase4" / "passport_vs_flight.csv").set_index("market")
    got = {r["label"]: r for r in K["catch"]["rows"]}
    assert got["Saudi Arabia"]["eid"] == pytest.approx(pv.loc["SAUDI ARABIA", "eid"], rel=1e-5)
    assert got["UAE residents"]["national_day"] > 1 > got["India"]["national_day"]     # the evidence the text rests on


def test_know_assumptions_quote_the_engine():
    K = S.know(); a = {x["step"]: x for x in K["assumptions"]}
    assert f"{S.fact('new_route_per_seat'):.3f}" in a["Countries without flights"]["value"]
    assert f"{S.fact('rooms'):,.0f}" in a["Occupancy"]["value"]
    assert len(K["limits"]) == 4 and sum(len(g["items"]) for g in K["limits"]) >= 20 and "asks" not in K      # Step 5.9's list
    import json; assert "nan" not in json.dumps(K).lower()


def test_know_endpoint_and_bundle():
    from fastapi.testclient import TestClient
    from atp import api
    r = TestClient(api.app).get("/api/know"); assert r.status_code == 200
    assert r.json()["lead"] == S.know()["lead"]


def test_tab_three_shows_the_evidence(page):
    page.click('.tabs a[data-tab="know"]'); page.wait_for_selector("#bt svg path.actual")
    assert page.locator(".stat").count() == 4 and "3.4%" in page.inner_text("#kn-stats")
    first = page.inner_text("#bt-read")
    page.select_option("#bt-market", "UNITED KINGDOM"); page.click('#bt-fold button[data-f="B"]')
    assert "United Kingdom" in page.inner_text("#bt-read") and page.inner_text("#bt-read") != first
    assert page.locator("#bt circle.dot.actual").count() == 7
    assert page.locator("#assume tbody tr").count() == 9 and page.locator("#march [role=row]").count() >= 8
    assert page.locator("#sobol [role=row]").count() == 6 and "same order" in page.inner_text("#sobol")
    assert page.locator("#ah path.ah-line").count() == 4 and page.locator("#ah-table tbody tr").count() == 7
    assert page.locator("#limits h3").count() == 4 and page.locator("#limits li").count() >= 20
    page.set_viewport_size({"width": 1280, "height": 800}); page.wait_for_timeout(400)
    assert page.evaluate("document.documentElement.scrollWidth") <= 1280
    assert not page.errors
    page.click('.tabs a[data-tab="simulator"]'); assert page.locator("#tab-know").is_hidden()
