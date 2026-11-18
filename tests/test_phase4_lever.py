"""Step 4.4: the live seats lever. Every slider position must be the full engine's answer for the market-wide seats
change (closed form, no interpolation); the reverse calculator's "today's routes" answer is the same lever inverted;
curves build fast and are served from memory."""
import json
import time
import numpy as np
import pytest
from scipy.optimize import brentq
from atp import scenario as sc
from atp.config import load_config
from atp.simulator import Simulator, _Lever, SEASONS

cfg = load_config(); P = cfg["paths"]["processed"]
if not (P / "phase4" / "facts.json").exists():
    pytest.skip("run `python -m atp.phase4 data` first", allow_module_level=True)
S = Simulator(cfg)


def nearest(cv, pct):
    return min(cv["steps"], key=lambda st: abs(st["seats_pct"] - pct))


def close(a, b, rel=2e-5):                      # curve numbers are rounded to 6 significant figures for transport
    return abs(a - b) <= rel * max(1.0, abs(b))


@pytest.mark.parametrize("market,season,picks", [
    ("UNITED KINGDOM", "winter", [-100, -40, -20, 14, 150]),     # cancel all, beyond-range cut, cut, gain, beyond-range gain
    ("UZBEKISTAN", "year", [-45, 45]),                          # months without seats inside the window
    ("CHINA", "summer", [30]),                                  # the cap on check-ins binds
])
def test_every_step_equals_a_full_engine_run(market, season, picks):
    cv = S.lever(market, season)
    for pct_ in picks:
        st = nearest(cv, pct_)
        full = S.run({"changes": [{**cv["change"], "seats_pct": st["seats_pct"]}]})
        h, H = st["headline"], full["headline"]
        for k in ("d_guest_nights", "lo", "hi", "d_checkins", "checkins_lo", "checkins_hi", "after_horizon", "pct_guest_nights"):
            assert close(h[k], H[k]), (k, h[k], H[k])
        for a, b in zip(st["chain"]["stages"], full["chain"]["stages"]):
            assert close(a["value"], b["value"]) and a["text"] == b["text"] and a.get("ratio_text") == b.get("ratio_text")
        for a, b in zip(st["by_month"], full["by_month"]):
            for k in ("d_guest_nights", "lo", "hi", "occupancy_points"):
                assert close(a[k], b[k]), (k, a[k], b[k])
        assert h["text"] == H["text"] and h["range_text"] == H["range_text"]
        assert h["occupancy"]["text"] == H["occupancy"]["text"] and close(h["occupancy"]["points"], H["occupancy"]["points"])
        assert st["sentence"] == full["sentence"]
        assert [x and x["text"] for x in st["chain"]["steps"]] == [x and x["text"] for x in full["chain"]["steps"]]
        assert st["chain"]["per_100_seats"]["text"] == full["chain"]["per_100_seats"]["text"]
        beyond = any(n["id"] == "beyond_range" for n in full["notes"])
        assert st["outside_tested_range"] == beyond


def test_curve_shape():
    cv = S.lever("UNITED KINGDOM", "winter")
    st = cv["steps"]; i0 = cv["today_index"]
    assert st[i0]["flights_change"] == 0 and st[i0]["headline"]["d_guest_nights"] == 0 and st[i0]["because"] is None
    assert st[0]["seats_pct"] == -100 and len(st) <= 202
    pcts = [x["seats_pct"] for x in st]; gn = [x["headline"]["d_guest_nights"] for x in st]
    assert pcts == sorted(pcts) and all(b >= a for a, b in zip(gn, gn[1:]))      # more flights, more guest nights
    assert all(float(x["flights_change"]).is_integer() for x in st[1:])         # whole weekly flights (after cancel-all)
    lo, hi = cv["tested_range_flights"]
    assert lo < 0 < hi and cv["current_flights_per_week"] > 40
    assert all(x["outside_tested_range"] == (not lo <= x["flights_change"] <= hi) for x in st if x["flights_change"] != 0)
    gain = next(x for x in st if x["flights_change"] == 7)
    assert "each extra inbound passenger adds" in gain["because"] and "United Kingdom" in gain["because"]
    cut = next(x for x in st if x["flights_change"] == -7)
    assert "switch to other routes" in cut["because"]


def test_lever_is_refused_where_the_model_has_no_seats():
    with pytest.raises(sc.ScenarioError):
        S.lever("AUSTRALIA", "winter")                                           # no flight data: new route instead
    with pytest.raises(sc.ScenarioError):
        _Lever(S, "UZBEKISTAN", "2026-07-01", "2026-08-31")                      # its only route pauses in summer
    with pytest.raises(sc.ScenarioError):
        S.lever("UNITED KINGDOM", "spring")


def test_reverse_today_is_the_lever_inverted():
    r = S.reverse("UNITED KINGDOM", "2026-11-01", "2026-11-30", 10000)
    assert r["aircraft"] == "today" and "on today's routes" in r["text"]
    lv = _Lever(S, "UNITED KINGDOM", "2026-11-01", "2026-11-30")
    change = {"type": "route", "markets": "UNITED KINGDOM", "start": "2026-11-01", "end": "2026-11-30"}

    def full(f, k="d_guest_nights"):
        return S.B.run(S.E, {"changes": [{**change, "seats_pct": 100 * f / lv.current_fpw}]})

    exact = brentq(lambda f: full(f).summary["d_guest_nights"] - 10000, 1, 200, xtol=0.01)
    assert r["flights_per_week"] == pytest.approx(exact, abs=0.02)
    a, b = r["flights_per_week_range"]                                           # the 80% band reaches the target there
    assert full(a).effect_band["summary"]["d_guest_nights"][1] == pytest.approx(10000, rel=2e-3)
    assert full(b).effect_band["summary"]["d_guest_nights"][0] == pytest.approx(10000, rel=2e-3)
    nb = S.reverse("UNITED KINGDOM", "2026-11-01", "2026-11-30", 10000, aircraft="narrowbody")
    assert nb["seats_added"] == pytest.approx(r["seats_added"], rel=0.05)        # same seats, different aircraft size


def test_lever_latency():
    S._levers.pop(("UNITED KINGDOM", "winter"), None)
    t = time.time(); S.lever("UNITED KINGDOM", "winter"); cold = time.time() - t
    t = time.time(); S.lever("UNITED KINGDOM", "winter"); warm = time.time() - t
    assert cold < 0.8 and warm < 0.005


def test_lever_api():
    from fastapi.testclient import TestClient
    from atp import api
    api.state["sim"] = S
    c = TestClient(api.app)
    r = c.get("/api/lever", params={"market": "UNITED KINGDOM", "season": "winter"}, headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip" and r.json()["label"] == "United Kingdom"
    assert c.get("/api/lever", params={"market": "AUSTRALIA"}).status_code == 422
    assert [x["id"] for x in c.get("/api/meta").json()["seasons"]] == list(SEASONS)


def test_a_preset_sent_back_by_the_browser_keeps_its_note_and_cache():
    """A browser turns 1.0 into 1: the scenario key must not depend on that, or the preset loses its note and its
    cached result."""
    ch = S.preset("march")["changes"]
    S.run({"changes": ch})
    js = lambda o: (int(o) if isinstance(o, float) and o.is_integer() else {k: js(v) for k, v in o.items()} if isinstance(o, dict)
                    else [js(v) for v in o] if isinstance(o, list) else o)
    back = js(ch)                                                               # what JSON.stringify sends
    assert json.dumps(back) != json.dumps(ch)
    t = time.time(); r = S.run({"changes": back}); dt = time.time() - t
    assert r["preset"] == "march" and r["note"] and dt < 0.05
    assert S._key([{"seats_pct": 10.0}]) == S._key([{"seats_pct": 10}])
