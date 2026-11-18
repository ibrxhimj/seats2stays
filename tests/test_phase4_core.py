"""Step 4.2: the simulator core and its API — what the app shows is the engine's own result (presets reproduce the
Phase 3 report), the formatting rules, occupancy points, plain notes for every engine flag, the fast reverse calculator
against the exact solver, speed, the bundle, and every endpoint."""
import json
import math
import time
import numpy as np
import pandas as pd
import pytest
from atp import phase4 as p4, scenario as sc, sensitivity as se, validation as va
from atp.config import load_config
from atp.simulator import Simulator, est, exact, pct, flights, sig3, label

cfg = load_config(); P = cfg["paths"]["processed"]
if not (P / "phase4" / "facts.json").exists():
    pytest.skip("run `python -m atp.phase4 data` first", allow_module_level=True)
S = Simulator(cfg)
S38 = json.load(open(P / "phase3" / "step38_showcase.json"))
ALL_KINDS = {"name": "every change type", "changes": [
    {"type": "route", "markets": "UNITED KINGDOM", "start": "2026-11-01", "end": "2026-12-31", "seats_pct": -20},
    {"type": "new_route", "market": "DENMARK", "city": "Copenhagen", "airline": "Etihad Airways", "start": "2026-12-01",
     "end": "2027-02-28", "flights_per_week": 2, "aircraft_class": "narrowbody"},
    {"type": "load_factor", "markets": "INDIA", "start": "2026-06-01", "end": "2026-08-31", "pct": 5},
    {"type": "transfer_share", "markets": "INDIA", "start": "2026-10-01", "end": "2026-12-31", "points": -10},
    {"type": "demand", "markets": "Europe", "start": "2026-11-01", "end": "2027-01-31", "pct": -10},
    {"type": "stay_length", "markets": "CHINA", "start": "2026-07-01", "end": "2026-08-31", "pct": 10}]}


def text(r):
    return "".join(x["text"] for x in r["sentence"])


# ----------------------------------------------------------------------------- formatting rules
def test_formatting_rules():
    assert est(37117.3) == "37,100" and est(37117.3, True) == "+37,100" and est(-78573, True) == "−78,600"
    assert est(2.8543) == "2.85" and est(0.04776) == "0.0478" and est(0) == "0" and sig3(13002) == 13000
    assert exact(196746.4, True) == "+196,746" and exact(-79888) == "−79,888"
    assert pct(0.996) == "99.6%" and pct(0.0001) == "0.01%" and pct(1.0) == "100%" and pct(0.23) == "23%"
    assert flights(37.2) == "37" and flights(0.3) == "under 1"
    assert label("UNITED STATES OF AMERICA") == "United States" and label("DOMESTIC") == "UAE residents" and label("SOUTH KOREA") == "South Korea"


# ----------------------------------------------------------------------------- what the app shows = the engine
def test_payload_equals_engine():
    s = S.preset("etihad"); r = S.run(s); R = S.B.run(S.E, s); Sm = R.summary
    h = r["headline"]
    assert h["d_guest_nights"] == Sm["d_guest_nights"] and [h["lo"], h["hi"]] == R.effect_band["summary"]["d_guest_nights"]
    v = {x["key"]: x["value"] for x in r["chain"]["stages"]}
    assert (v["seats"], v["passengers"], v["local"], v["checkins"], v["guest_nights"]) == \
           (Sm["d_seats"], Sm["d_passengers"], Sm["d_local_passengers"], Sm["d_checkins"], Sm["d_guest_nights"])
    assert np.isclose(sum(m["d_guest_nights"] for m in r["by_month"]) + h["after_horizon"], Sm["d_guest_nights"])
    assert np.isclose(sum(m["d_guest_nights"] for m in r["by_market"]), Sm["d_guest_nights"])
    t = R.daily.groupby("date")[["guests_base", "guests_scen"]].sum()
    assert np.allclose(r["daily"]["baseline"], t.guests_base, atol=0.06) and np.allclose(r["daily"]["scenario"], t.guests_scen, atol=0.06)
    st = r["chain"]["steps"]                                              # exits: what leaves between stages, exactly
    assert np.isclose(st[0]["value"], v["seats"] - v["passengers"]) and np.isclose(st[1]["value"], v["passengers"] - v["local"])
    assert np.isclose(st[2]["value"], v["local"] - Sm["d_checkins_flights"])
    json.dumps(r, allow_nan=False)                                        # JSON-safe as returned


def test_presets_reproduce_phase3_report():
    e = S.run(S.preset("etihad")); t = S38["etihad_launches"]["total"]
    assert e["headline"]["d_guest_nights"] == pytest.approx(t["guest_nights"], rel=1e-9)
    assert [e["headline"]["lo"], e["headline"]["hi"]] == pytest.approx(t["guest_nights_band"], rel=1e-9)
    assert e["headline"]["text"] == "+37,100" and e["headline"]["range_text"] == "+21,800 to +54,700"
    assert e["by_market"][0]["label"] == "United States" and e["by_market"][0]["text"] == "+13,700"
    eu = S.run(S.preset("europe")); x = S38["european_return"]["stay_away"]
    assert eu["headline"]["d_guest_nights"] == pytest.approx(x["guest_nights"], rel=1e-9)
    assert [eu["headline"]["lo"], eu["headline"]["hi"]] == pytest.approx(x["guest_nights_band"], rel=1e-9)
    assert eu["headline"]["text"] == "−78,600" and eu["note"].startswith("Those routes are worth the opposite: about +78,600")
    assert {m["label"]: m["text"] for m in eu["by_market"]} == {"United Kingdom": "−37,200", "Poland": "−26,100", "Germany": "−15,200"}
    ma = S.run(S.preset("march")); mar = next(m for m in ma["by_month"] if m["month"] == "2026-03")
    assert mar["d_guest_nights"] / mar["baseline"] == pytest.approx(-0.32, abs=0.01)   # fitted to SCAD's March guest nights
    assert "26%–36%" in ma["note"] and text(ma).startswith("Losing 865,783 seats")
    nr = S.run(S.preset("new_route"))
    assert nr["by_market"][0]["label"] == "Mexico" and any(n["id"] == "no_flight_data" for n in nr["notes"])


def test_sentence_and_chain_read_right():
    e = S.run(S.preset("etihad"))
    assert text(e) == "196,746 extra seats bring about 13,000 more hotel check-ins: +37,100 guest nights (80% range +21,800 to +54,700)."
    assert [x["label"] for x in e["chain"]["stages"]] == \
           ["Seats", "Arriving passengers", "Inbound passengers", "Hotel check-ins", "Guest nights"]   # the brief's chain
    assert [x["ratio_text"] for x in e["chain"]["stages"][1:]] == \
           ["87% of seats filled", "23% end their trip here", "0.32 per inbound passenger", "2.85 nights per check-in"]
    assert e["chain"]["per_100_seats"]["text"].startswith("For every 100 extra seats: 87 passengers, 21 end their trip")
    eu = S.run(S.preset("europe"))
    assert eu["chain"]["direction"] == "loss" and eu["chain"]["stages"][2]["ratio_text"] == "99.6% end their trip here"
    assert "fewer guest nights" in eu["chain"]["per_100_seats"]["text"]
    empty = S.run({"changes": []})
    assert empty["empty"] and text(empty).startswith("No changes yet") and empty["headline"]["occupancy"] is None
    stay = S.run({"changes": [{"type": "stay_length", "markets": "DOMESTIC", "start": "2026-07-01", "end": "2026-08-31", "pct": 10}]})
    assert text(stay).startswith("Guests checking in from 1 Jul to 31 Aug 2026 stay 10% longer")


def test_form_data_and_direction_without_flights():
    """What the add-a-change forms need from meta(), and the colour of a change that has no flights in it."""
    m = S.meta()
    air = {a["name"]: a for a in m["airlines"]}
    assert 0.8 < air["Etihad Airways"]["load_factor"] < 0.95 and air["Etihad Airways"]["local_share"] < 0.4 and air["IndiGo"]["local_share"] > 0.95
    assert all(a["load_factor"] > 0 and 0 < a["local_share"] <= 1 for a in air.values()) and "Private" not in air      # only airlines a new route can copy
    assert 0.8 < m["network"]["load_factor"] < 0.95 and 0.3 < m["network"]["transfer_share"] < 0.7
    win = {"start": "2026-11-01", "end": "2027-02-28"}
    cut = S.run({"changes": [{"type": "demand", "markets": "all", "pct": -10, **win}]})
    assert cut["chain"]["direction"] == "loss" and not cut["chain"]["has_flight_change"] and cut["headline"]["d_guest_nights"] < 0
    assert S.run({"changes": [{"type": "demand", "markets": "all", "pct": 10, **win}]})["chain"]["direction"] == "gain"
    stay = S.run({"changes": [{"type": "stay_length", "markets": "Europe", "pct": -10, **win}]})
    assert stay["chain"]["direction"] == "loss"                                       # check-ins unchanged, nights fewer


def test_occupancy_points():
    e = S.run(S.preset("etihad"))
    for m in e["by_month"]:
        pt = S.occ.loc[m["month"], "dct_guest_nights_per_occupancy_point"]
        assert m["occupancy_points"] == pytest.approx(m["d_guest_nights"] / pt)
    o = e["headline"]["occupancy"]
    assert o["month"] == max(e["by_month"], key=lambda m: abs(m["occupancy_points"]))["month"] and o["text"].startswith("≈ +0.")
    # March 2026 check against SCAD's published occupancy (54.27%): the normal year on SCAD's basis plus the replay's
    # change lands within 5 points (in the war fewer guests shared a room: 1.59 vs 1.65, which the normal-year ratio
    # cannot know)
    ma = S.run(S.preset("march")); mar = next(m for m in ma["by_month"] if m["month"] == "2026-03")
    normal = mar["baseline"] / S.occ.loc["2026-03", "dct_guest_nights_per_occupancy_point"]
    assert 65 < normal < 80 and abs(normal + mar["occupancy_points"] - 54.27) < 5


def test_every_engine_flag_has_a_plain_note():
    rng = np.random.default_rng(3)
    scen = [S.preset(p) for p in ("etihad", "europe", "march", "new_route")] + [ALL_KINDS]
    scen += [va.random_scenario(S.w, rng) for _ in range(25)]
    seen = set()
    for s in scen:
        try:
            r = S.run(s)
        except sc.ScenarioError:
            continue
        for n in r["notes"]:
            assert not n["id"].startswith("other:"), n["details"][0]
            assert "_" not in n["text"] and "…" not in n["text"].split(":")[0]
            seen.add(n["id"])
    assert {"beyond_validation", "short", "no_flight_data", "assumption", "cap"} <= seen
    cap = S.run({"changes": [{"type": "demand", "markets": "all", "start": "2026-03-15", "end": "2026-03-31", "pct": 20}]})
    w = [n for n in cap["notes"] if n["level"] == "warning"]
    assert cap["daily"]["days_above_peak"] > 0 and w[0]["id"] == "capacity" and w[0]["text"].strip() != ""


# ----------------------------------------------------------------------------- reverse calculator
def test_reverse_equals_stored_exact_solver():
    for x in json.load(open(P / "phase3" / "step36_sensitivity.json"))["seats_needed"]:
        f = S.reverse(x["market"], *x["window"], x["target_guest_nights"], seats_per_flight=x["seats_per_flight"])
        assert f["flights_per_week"] == pytest.approx(x["flights_per_week"], abs=0.02)
        assert f["flights_per_week_range"] == pytest.approx(x["flights_per_week_range"], abs=0.02)
        assert f["flights_per_week_range_lever_only"] == pytest.approx(x["flights_per_week_range_lever_only"], abs=0.02)


@pytest.mark.parametrize("market,start,end,target", [("AUSTRALIA", "2026-12-01", "2026-12-31", 2000),     # no flight data
                                                     ("UZBEKISTAN", "2026-08-01", "2026-08-31", 3000)])   # no baseline seats
def test_reverse_equals_live_exact_solver(market, start, end, target):
    spf = S.w.aircraft["narrowbody"]
    f = S.reverse(market, start, end, target, seats_per_flight=spf); x = se.seats_needed(S.E, S.B, market, start, end, target, spf)
    for k in ("flights_per_week", "flights_per_week_range", "flights_per_week_range_lever_only"):
        assert f[k] == pytest.approx(x[k], abs=0.02)


def test_reverse_text_and_limits():
    f = S.reverse("UNITED KINGDOM", "2026-11-01", "2026-11-30", 10000, aircraft="narrowbody")
    assert f["text"] == ("To win 10,000 more guest nights from United Kingdom, Abu Dhabi needs about 37 more narrowbody flights "
                         "a week (174 seats; 80% range 25 to 76).")
    au = S.reverse("AUSTRALIA", "2026-12-01", "2026-12-31", 2000)
    assert "on a new narrowbody route" in au["text"] and au["new_route"]
    far = S.reverse("QATAR", "2026-11-01", "2026-11-07", 5e6)
    assert far["flights_per_week"] is None and "out of reach" in far["text"]
    with pytest.raises(sc.ScenarioError):
        S.reverse("DOMESTIC", "2026-11-01", "2026-11-30", 1000)
    with pytest.raises(sc.ScenarioError):
        S.reverse("UNITED KINGDOM", "2026-11-01", "2026-11-30", 1000, aircraft="jumbo")


WIN = ("2026-11-01", "2027-02-28")


def test_reverse_refuses_what_is_not_a_target_and_never_crashes():
    for bad in (0, -500, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(sc.ScenarioError):
            S.reverse("UNITED KINGDOM", *WIN, bad)
    with pytest.raises(sc.ScenarioError):
        S.reverse("UNITED KINGDOM", "not-a-date", WIN[1], 1000)
    for t in (0.001, 0.5, 1, 2):                                           # a sliver of a flight is already enough: no solver error
        r = S.reverse("UNITED KINGDOM", *WIN, t)
        a, b = r["flights_per_week_range"]
        assert 0 < r["flights_per_week"] < 0.5 and a <= r["flights_per_week"] <= b and "under 1" in r["text"]


def test_reverse_answers_stay_inside_what_the_simulator_accepts():
    """On today's routes: up to double the country's flights (the slider). On an added route: 100 a week or double the country's
    flights, whichever is more (the form). Beyond that the answer is "out of reach", with what the ceiling would win."""
    uk = S.reverse("UNITED KINGDOM", *WIN, 10_000)
    cur = uk["current_flights_per_week"]
    assert uk["flights_per_week_max"] == 100 == math.floor(2 * cur) and not uk["outside_tested_range"]
    far = S.reverse("UNITED KINGDOM", *WIN, 5_000_000)
    assert far["flights_per_week"] is None and "out of reach" in far["text"] and "even 100 more flights a week" in far["text"] and len(far["text"]) < 200
    ch = {"type": "route", "markets": "UNITED KINGDOM", "start": WIN[0], "end": WIN[1], "seats_pct": 100 * 100 / cur}
    assert far["guest_nights_at_max"] == pytest.approx(S.run({"changes": [ch]})["headline"]["d_guest_nights"], rel=2e-3)    # the figure quoted is the engine's
    assert S.reverse("UNITED KINGDOM", *WIN, 1e300)["flights_per_week"] is None                  # no giant number in the text either
    assert len(S.reverse("UNITED KINGDOM", *WIN, 1e300)["text"]) < 200
    se_ = S.reverse("SWEDEN", *WIN, 5_000_000, "narrowbody")                                       # no flights today: an added route, 100 at most
    assert se_["flights_per_week"] is None and se_["flights_per_week_max"] == 100 and se_["new_route"]
    nr = {"type": "new_route", "market": "SWEDEN", "city": "Stockholm", "airline": "generic", "start": WIN[0], "end": WIN[1],
          "flights_per_week": 100, "aircraft_class": "narrowbody"}
    assert se_["guest_nights_at_max"] == pytest.approx(S.run({"changes": [nr]})["headline"]["d_guest_nights"], rel=2e-3)
    big = S.reverse("INDIA", "2026-12-01", "2026-12-31", 10_000, "narrowbody")                     # a big market: the Phase 3 answer (128) stands
    assert big["flights_per_week"] == pytest.approx(127.8, abs=0.1) and big["flights_per_week_max"] == max(100, math.floor(2 * big["current_flights_per_week"])) > 100
    beyond = S.reverse("UNITED KINGDOM", *WIN, 150_000)                                            # reachable, but past what the lever was tested on
    assert beyond["flights_per_week"] > beyond["tested_max_flights"] and beyond["outside_tested_range"] and beyond["flights_per_week"] <= beyond["flights_per_week_max"]
    ch["seats_pct"] = 100 * beyond["flights_per_week"] / cur
    assert S.run({"changes": [ch]})["headline"]["d_guest_nights"] == pytest.approx(150_000, rel=2e-3)
    for t in (1_000, 10_000, 50_000, 100_000):                                                     # every answer is within its ceiling
        for ac in ("today", "narrowbody", "widebody"):
            r = S.reverse("UNITED KINGDOM", *WIN, t, ac)
            assert r["flights_per_week"] is None or 0 < r["flights_per_week"] <= r["flights_per_week_max"]


# ----------------------------------------------------------------------------- speed
def test_speed():
    for s in (S.preset("etihad"), S.preset("europe"), ALL_KINDS):
        S._cache.clear(); S.E._resolved.clear()
        t = time.time(); S.run(s); assert time.time() - t < 1.5, s.get("name")
    S._cache.clear(); S.E._resolved.clear()
    t = time.time(); S.run(S.preset("march")); assert time.time() - t < 6       # 325 changes, cold (presets are warmed)
    t = time.time(); S.run(S.preset("march")); assert time.time() - t < 0.05    # computed once
    t = time.time(); S.reverse("INDIA", "2026-12-01", "2026-12-31", 10000, aircraft="narrowbody"); assert time.time() - t < 3


# ----------------------------------------------------------------------------- the bundle and the API
def test_simulator_on_bundle(tmp_path):
    b = tmp_path / "bundle"; p4.build_bundle(b, cfg)
    Sb = Simulator(load_config(root=b))
    for pid in ("etihad", "march"):
        r0, r1 = S.run(S.preset(pid)), Sb.run(Sb.preset(pid))
        assert json.dumps(r0, sort_keys=True) == json.dumps(r1, sort_keys=True)
    for name in ("levers", "market_month", "market_ranking", "season", "march_regions", "backtest_metrics",
                 "backtest_markets", "backtest_monthly", "occupancy"):
        assert Sb.data(name) == S.data(name)
    assert Sb.meta() == S.meta()
    assert json.dumps(Sb.moves(), sort_keys=True) == json.dumps(S.moves(), sort_keys=True)          # "What moves guests" runs on the bundle too


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from atp import api
    api.state["sim"] = S                                                   # reuse the loaded simulator
    c = TestClient(api.app)
    assert c.get("/api/health").json()["ok"]
    m = c.get("/api/meta").json()
    assert len(m["markets"]) == 46 and [p["id"] for p in m["presets"]] == ["etihad", "europe", "march", "new_route"]
    assert m["peak_guests"] > 0 and m["horizon"] == ["2026-03-01", "2027-02-28"]
    s = c.get("/api/presets/etihad").json()
    r = c.post("/api/run", json=s); assert r.status_code == 200 and r.json()["headline"]["text"] == "+37,100"
    bad = c.post("/api/run", json={"changes": [{"type": "route", "market": "NARNIA", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": 5}]})
    assert bad.status_code == 422 and "NARNIA" in bad.json()["error"]
    assert c.post("/api/run", json={"changes": "x"}).status_code == 422
    assert c.get("/api/presets/nope").status_code == 422
    rv = c.post("/api/reverse", json={"market": "UNITED KINGDOM", "start": "2026-11-01", "end": "2026-11-30", "target": 10000,
                                      "aircraft_class": "narrowbody"})
    assert rv.status_code == 200 and "about 37 more narrowbody flights a week" in rv.json()["text"]
    assert c.post("/api/reverse", json={"market": "UNITED KINGDOM"}).status_code == 422
    one = {"market": "UNITED KINGDOM", "start": "2026-11-01", "end": "2026-11-30"}
    for bad in ({"target": 0}, {"target": -500}, {"target": "5,000"}, {"target": "1e999"}, {"target": [1]}, {"target": 1000, "start": "nope"}):
        r_ = c.post("/api/reverse", json={**one, **bad}); assert r_.status_code == 422 and r_.json()["error"], bad     # a plain message, never a 500
    assert c.post("/api/reverse", json={**one, "target": 1}).status_code == 200
    far = c.post("/api/reverse", json={**one, "target": 5e6}); assert far.status_code == 200 and far.json()["flights_per_week"] is None
    ex = c.post("/api/export", json=s)
    assert ex.status_code == 200 and ex.text.startswith("market,month,guest_nights_normal_year") and "attachment" in ex.headers["content-disposition"]
    d = pd.DataFrame(pd.read_csv(pd.io.common.StringIO(ex.text)))
    assert d.change_guest_nights.sum() + r.json()["headline"]["after_horizon"] == pytest.approx(r.json()["headline"]["d_guest_nights"], abs=1)
    assert c.get("/api/data/levers").status_code == 200 and c.get("/api/data/nope").status_code == 404
