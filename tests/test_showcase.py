"""Step 3.8: showcase scenarios are internally consistent and reproduce what they claim."""
import json
import numpy as np
import pandas as pd
import pytest
from atp import engine as en, scenario as sc
from atp.config import load_config

cfg = load_config(); O = cfg["paths"]["processed"] / "phase3"
if not (O / "step38_showcase.json").exists():
    pytest.skip("run `make phase3` first", allow_module_level=True)
J = json.load(open(O / "step38_showcase.json"))
E = en.Engine(cfg)


def test_march_replay():
    m = J["march_2026"]; v = json.load(open(O / "step37_validation.json"))["march_2026"]
    obs = {r["region"]: r["observed_vs_normal"] for r in v["by_region"]}
    for k in ("low", "high"):
        for r in m["bounds"][k]["by_region"]:
            assert abs(r["checkins_replay_vs_normal"] - obs[r["region"]]) < 1e-4          # SCAD guests by region, by construction
        t = m["bounds"][k]["total"]
        assert t["checkins_replay_vs_normal"] < t["checkins_flights_only_vs_normal"] < 1
        assert 0 < t["guest_nights_lost_flights"] < t["guest_nights_lost"]
        # with normal stays the replay misses SCAD's nights; the fitted stay change reproduces them
        assert t["guest_nights_replay_vs_normal"] < t["guest_nights_scad_vs_normal"]
        assert abs(t["guest_nights_with_fitted_stays_vs_normal"] - t["guest_nights_scad_vs_normal"]) < 1e-3 and 0 < t["stay_change_fitted_pct"] < 60
        assert 1 < t["nights_per_checkin_vs_normal_replay_normal_stays"] < t["nights_per_checkin_vs_normal_scad"]   # mechanical part < observed
    f = m["feb28_spike"]
    assert f["checkins_actual"] > f["checkins_normal"] and 0 < f["points_of_normal_march_guest_nights"] < 0.05
    w = sc.World(cfg)
    for k in ("low", "high"):
        p = json.load(open(O / "presets" / f"march_2026_replay_{k}.json"))
        assert not any("counted twice" in f for f in sc.validate(p, w))


def test_etihad_launches_route_by_route():
    L = J["etihad_launches"]; t = L["total"]
    assert abs(L["sum_of_routes_guest_nights"] - t["guest_nights"]) < 1e-6 * t["guest_nights"]     # markets and windows add up
    for x in L["routes"] + [t]:
        assert x["guest_nights_band"][0] < x["guest_nights"] < x["guest_nights_band"][1]
        assert np.isclose(x["guest_nights"], x["checkins"] * x["nights_per_checkin"])
        assert x["checkins_per_local_passenger"] <= 1 + 1e-9                                      # the cap


def test_new_route_listed_twice_is_refused():
    w = sc.World(cfg)
    L = sc.load(cfg["paths"]["reference"] / "scenarios" / "etihad_2026_launches.json")
    M = json.load(open(O / "presets" / "march_2026_replay_low.json"))
    with pytest.raises(sc.ScenarioError, match="added twice"):
        sc.validate({"changes": M["changes"] + L["changes"]}, w)                # both contain Charlotte from 20 Mar
    sc.validate(L, w)                                                            # consecutive windows (Salalah, Red Sea) are fine


def test_european_return():
    e = J["european_return"]
    assert e["stay_away"]["guest_nights"] < e["one_month_slip"]["guest_nights"] < 0
    assert np.isclose(sum(v["guest_nights"] for v in e["by_carrier"].values()), e["stay_away"]["guest_nights"])


def test_recovery_planner_offsets_the_shortfall():
    rp = J["recovery_planner"]; rule = rp["offset_rule"]["10%"]["central"]
    a, b = rp["assumption"]["window"]
    for x in rp["rows"][:3]:
        s = {"changes": [{"type": "demand", "markets": x["market"], "start": a, "end": b, "pct": -rp["assumption"]["shortfall_pct"]},
                         {"type": "new_route", "market": x["market"], "city": "added daily flight", "airline": "generic", "start": a, "end": b,
                          "flights_per_week": x["flights_per_week_needed"], "seats_per_flight": rp["seats_per_flight"]}]}
        assert abs(E.run(s).summary["d_guest_nights"]) < 0.01 * abs(x["guest_nights_lost"])
        assert abs(x["extra_seats_pct"] / 100 - rule) < 0.03                                      # engine = the closed-form rule
        assert x["range"][0] < x["flights_per_week_needed"] < x["range"][1]


def test_report():
    r = open(cfg["paths"]["processed"].parent.parent / "reports" / "phase3_report.md").read()
    assert "nan" not in r.lower().replace("finance", "") and "None" not in r
    s = r.split("## Draft summary (≤ 1,000 characters)")[1].split("\n\n")[1]
    assert len(s) <= 1000
