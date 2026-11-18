"""Step 4.9: the review fixes. (1) The check-in response — the chain's one estimated step — is adjustable: a result
can be shown at either end of its validated 80% interval, through the same engine, and the live lever still equals a
full run there. (2) The API refuses what it cannot honour with a plain 422, never a 500 or an absurd number.
(3) The wording of Tabs 2 and 3 claims no more than the evidence."""
import math
import pytest
from fastapi.testclient import TestClient
from atp import api
from atp.config import load_config
from atp.simulator import Simulator, RESPONSES

cfg = load_config()
if not (cfg["paths"]["processed"] / "phase4" / "facts.json").exists():
    pytest.skip("run `python -m atp.phase4 data` first", allow_module_level=True)
S = Simulator(cfg)
C = TestClient(api.app)
UK = {"type": "route", "markets": "UNITED KINGDOM", "start": "2026-11-01", "end": "2026-12-31", "seats_pct": 20}


# ----------------------------------------------------------------------------- (1) the check-in response
def test_responses_are_the_ends_of_the_validated_interval():
    e = S.B.effect
    assert S.response_params("central") == {"eps": e["eps"], "per_seat": e["per_seat"]}
    assert S.response_params("low") == {"eps": e["eps_lo"], "per_seat": e["per_seat_lo"]}
    assert S.response_params("high") == {"eps": e["eps_hi"], "per_seat": e["per_seat_hi"]}
    assert [r["id"] for r in S.meta()["responses"]] == list(RESPONSES)


@pytest.mark.parametrize("pid", ["etihad", "europe", "new_route"])
def test_the_result_moves_with_the_response_and_stays_inside_its_range(pid):
    sc_ = S.preset(pid)
    r = {k: S.run({**sc_, "response": k}) for k in RESPONSES}
    g = {k: v["headline"]["d_guest_nights"] for k, v in r.items()}
    assert abs(g["low"]) < abs(g["central"]) < abs(g["high"])                         # a gain or a loss: low responds least
    for k, v in r.items():
        h = v["headline"]
        assert h["lo"] <= h["d_guest_nights"] <= h["hi"] and v["response"]["id"] == k
        assert sum(m["d_guest_nights"] for m in v["by_market"]) == pytest.approx(h["d_guest_nights"], rel=1e-9)
    assert r["central"] == S.run(sc_) | {"name": r["central"]["name"]}                 # the default is the central estimate


def test_the_response_equals_the_engine_at_that_end():
    s = {"changes": [UK]}
    for k in ("low", "high"):
        m = S.E.run(s, **S.response_params(k))
        assert S.run({**s, "response": k})["headline"]["d_guest_nights"] == pytest.approx(m.summary["d_guest_nights"], rel=1e-12)


@pytest.mark.parametrize("response", ["low", "high"])
def test_the_live_lever_equals_a_full_run_at_every_response(response):
    cv = S.lever("FRANCE", "summer", response)
    assert cv["response"] == response
    for st in (cv["steps"][1], cv["steps"][-1], cv["steps"][len(cv["steps"]) // 3]):
        full = S.run({"changes": [{**cv["change"], "seats_pct": st["seats_pct"]}], "response": response})["headline"]
        for k in ("d_guest_nights", "lo", "hi", "d_checkins"):
            assert st["headline"][k] == pytest.approx(full[k], rel=1e-5, abs=0.05)          # the curve is sent to 7 significant figures


def test_assumption_levers_do_not_depend_on_the_response():
    s = {"changes": [{"type": "demand", "markets": "Europe", "start": "2026-11-01", "end": "2027-01-31", "pct": -10}]}
    v = {k: S.run({**s, "response": k})["headline"]["d_guest_nights"] for k in RESPONSES}
    assert v["low"] == pytest.approx(v["central"]) == pytest.approx(v["high"])


def test_export_and_api_carry_the_response():
    csv = S.export_csv({"changes": [UK], "response": "low"}).splitlines()
    assert csv[0].endswith(",check_in_response") and all(l.endswith(",low") for l in csv[1:])
    a = C.post("/api/run", json={"changes": [UK], "response": "high"}).json()
    assert a["response"]["id"] == "high" and a["headline"]["d_guest_nights"] == pytest.approx(S.run({"changes": [UK], "response": "high"})["headline"]["d_guest_nights"])
    assert C.get("/api/lever?market=FRANCE&season=summer&response=low").json()["response"] == "low"
    assert C.post("/api/run", json={"changes": [UK], "response": "pessimistic"}).status_code == 422
    assert C.get("/api/lever?market=FRANCE&season=summer&response=x").status_code == 422


# ----------------------------------------------------------------------------- (2) the API refuses, plainly
BAD = {
    "a text number": {**UK, "seats_pct": "nan"},
    "true": {**UK, "seats_pct": True},
    "a million percent": {**UK, "seats_pct": 1e6},
    "below -100%": {**UK, "seats_pct": -150},
    "no date": {**UK, "start": None},
    "a date that is not one": {**UK, "start": {"y": 2026}},
    "a nonsense date": {**UK, "start": "soon"},
    "a load factor above full": {"type": "new_route", "market": "DENMARK", "city": "Copenhagen", "airline": "x", "start": "2026-12-01",
                                 "end": "2027-02-28", "flights_per_week": 2, "aircraft_class": "narrowbody", "load_factor": 1.5},
    "a zero local share": {"type": "new_route", "market": "DENMARK", "city": "Copenhagen", "airline": "x", "start": "2026-12-01",
                           "end": "2027-02-28", "flights_per_week": 2, "aircraft_class": "narrowbody", "local_share": 0},
    "demand +5000%": {"type": "demand", "markets": "Europe", "start": "2026-11-01", "end": "2027-01-31", "pct": 5000},
    "stay as text": {"type": "stay_length", "markets": "CHINA", "start": "2026-07-01", "end": "2026-08-31", "pct": "10"},
}


@pytest.mark.parametrize("what", list(BAD))
def test_bad_numbers_and_dates_are_refused_with_a_message(what):
    r = C.post("/api/run", json={"changes": [BAD[what]]})
    assert r.status_code == 422 and r.json()["error"]


def test_a_raw_nan_or_infinity_is_refused():
    import json
    for v in ("NaN", "Infinity", "-Infinity"):
        body = json.dumps({"changes": [{**UK, "seats_pct": 0}]}).replace('"seats_pct": 0', f'"seats_pct": {v}')
        r = C.post("/api/run", content=body, headers={"content-type": "application/json"})
        assert r.status_code == 422, (v, r.status_code, r.text[:200])


def test_a_change_must_be_an_object():
    for body in ({"changes": ["route"]}, {"changes": [1]}, {"changes": "route"}):
        r = C.post("/api/run", json=body)
        assert r.status_code == 422 and r.json()["error"]


def test_every_preset_is_inside_the_limits():
    for p in ("etihad", "europe", "march", "new_route"):
        assert C.post("/api/run", json=S.preset(p)).status_code == 200


# ----------------------------------------------------------------------------- (3) the words
def test_tab_three_claims_no_more_than_the_back_test():
    K = S.know()
    assert "alone" not in K["lead"] and "never the hotel figures it was predicting" in K["lead"].replace("\n", "")
    note = next(s for s in K["stats"] if s["id"] == "lever")["note"]
    assert "removes 14% of the squared error" in note and "adds 31%" in note and "scores" not in note


def test_tab_two_leads_with_the_flight_finding():
    lead = S.moves()["lead"]
    assert lead.startswith("Of the flight levers, connecting passengers matter most")
    rows = {r["id"]: r for r in S.moves()["levers"]["rows"]}
    assert "one-for-one by definition" in rows["non-flight demand"]["what"] and "one-for-one by definition" in rows["stay length"]["what"]


# ----------------------------------------------------------------------------- in the browser
def test_the_response_switch_reruns_everything_on_screen(page):
    central = page.inner_text("#big")
    page.click("#pin")                                                            # A = the preset at the central response
    page.click('#response [data-resp="low"]')
    page.wait_for_function(f"document.getElementById('big').textContent !== {central!r}")
    low = S.run({**S.preset("etihad"), "response": "low"})["headline"]
    assert page.inner_text("#big") == low["text"] and "at the low check-in response" in page.inner_text("#range-text")
    assert page.posts[-1][1]["response"] == "low"
    assert "(low response)" in page.inner_text("#compare") and "Etihad" in page.inner_text("#compare")
    page.wait_for_function("!document.getElementById('lever-slider').disabled")
    assert "+1.9%" in page.inner_text("#resp-note")
    # the lever now drags at the low response: one step equals the full run at low
    page.select_option("#lever-market", "FRANCE"); page.click('#lever-season [data-season="summer"]')
    page.wait_for_function("!document.getElementById('lever-slider').disabled")
    page.eval_on_selector("#lever-slider", "el => { el.value = String(+el.max); el.dispatchEvent(new Event('input')); }")
    page.wait_for_timeout(300)
    cv = S.lever("FRANCE", "summer", "low")
    assert page.inner_text("#big") == cv["steps"][-1]["headline"]["text"]
    page.click('#response [data-resp="central"]')
    page.wait_for_function("document.querySelector('#response [data-resp=central]').getAttribute('aria-pressed') === 'true'")
    assert not page.errors


def test_a_click_never_scrolls_the_page_or_is_swallowed(page):
    """Regression: the page wrapper carries data-tab too, and the tab handler once caught every click on the page —
    scrolling to the top and cancelling the click (the notes and "Show the numbers" could not open)."""
    page.evaluate("scrollTo(0, 400)"); page.wait_for_timeout(150); y = page.evaluate("scrollY"); assert y > 0
    page.click(".split .method"); page.wait_for_timeout(200)
    assert page.evaluate("scrollY") == y
    page.click("#notes-summary"); assert page.evaluate("document.getElementById('notes-box').open")
    page.click("#list li >> nth=1")                                                 # picks that country: the screen follows it ...
    page.wait_for_function("document.getElementById('big').textContent === '0'", timeout=20000)
    assert page.evaluate("scrollY") == y                                            # ... without moving the page
    for t, sel in (("moves", "#heat-read"), ("know", "#bt-read")):
        page.click(f'.tabs a[data-tab="{t}"]'); page.wait_for_selector(sel + ":not(:empty)", timeout=20000); page.wait_for_timeout(300)
        assert page.evaluate("scrollY") == 0                                      # a new tab opens at its top
        page.evaluate("scrollTo(0, 700)"); page.wait_for_timeout(150); y = page.evaluate("scrollY")
        page.click(sel); page.wait_for_timeout(200)
        assert page.evaluate("scrollY") == y and page.evaluate("location.hash") == f"#{t}"
    page.click("details.tbl summary"); assert page.evaluate("document.querySelector('details.tbl').open")
    assert not page.errors


# ----------------------------------------------------------------------------- step 4.10: projector and first-glance checks
def test_every_text_on_every_tab_meets_wcag_aa(page):
    """WCAG AA for every visible text element (SVG labels included), on each tab: 4.5:1, or 3:1 for large text.
    Computed from the rendered colours over the actual background, so a style change cannot quietly fade text."""
    from pathlib import Path
    js = (Path(__file__).parent / "assets" / "contrast.js").read_text()
    bad = []
    for t in ("simulator", "moves", "know"):
        if t != "simulator":
            page.click(f'.tabs a[data-tab="{t}"]'); page.wait_for_timeout(900)
        bad += [(t, r["cls"], r["text"], r["cr"]) for r in page.evaluate(js) if r["cr"] < (3.0 if r["large"] else 4.5)]
    assert not bad, bad[:10]


def test_the_result_says_what_it_is_and_can_be_pinned_beside_it(page):
    assert page.inner_text("#viewing") == "ETIHAD'S 2026 LAUNCHES"                     # shown in capitals by CSS
    assert page.text_content("#viewing") == "Etihad's 2026 launches"
    box_pin, box_lead = page.locator("#pin").bounding_box(), page.locator("#lead").bounding_box()
    assert box_pin["y"] < box_lead["y"] < box_pin["y"] + 80 and box_pin["y"] + box_pin["height"] < 900   # beside the result, above the fold
    page.click('.preset[data-preset="europe"]'); page.wait_for_function("document.getElementById('viewing').textContent === 'Fewer European flights'")
    page.click('#response [data-resp="high"]'); page.wait_for_function("document.getElementById('viewing').textContent.includes('(high response)')")
    page.click("#pin"); assert page.inner_text("#pin") == "Replace A with this"
    assert not page.errors
