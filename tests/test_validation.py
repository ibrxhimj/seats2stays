"""Step 3.7: validation — beyond-range replay, engine properties, March 2026 replay."""
import json
import numpy as np
import pandas as pd
import pytest
from atp import bands as bd, engine as en, scenario as sc
from atp.config import load_config

cfg = load_config(); O = cfg["paths"]["processed"] / "phase3"
if not (O / "step37_validation.json").exists():
    pytest.skip("run `make phase3` first", allow_module_level=True)
J = json.load(open(O / "step37_validation.json"))
E = en.Engine(cfg); B = bd.Bands(cfg)


def test_alternative_cut_rule_only_changes_deep_cuts():
    p = E.params; fl = B.effect["cut_floor"]
    x = np.array([0.0, 0.01, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    line, log = en.lever_G(x, p.eps, p.lo, p.hi), en.lever_G(x, p.eps, p.lo, p.hi, "log", fl)
    inside = (x - 1 >= p.lo)
    assert np.allclose(line[inside], log[inside]) and (log[~inside] < line[~inside]).all()
    assert np.isclose(log[0], fl ** p.eps) and np.isclose(log[1], fl ** p.eps)          # flat below the deepest training cut
    assert 0 < fl < 0.1
    s = {"changes": [{"type": "route", "market": "INDIA", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": -10}]}
    assert np.allclose(E.run(s).daily.d_checkins, E.run(s, cuts="log", floor=fl).daily.d_checkins)
    assert not B.run(E, s).effect_band["rule_uncertainty"]
    deep = {"changes": [{"type": "route", "market": "KUWAIT", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": -80}]}
    r = B.run(E, deep)
    assert r.effect_band["rule_uncertainty"] and r.effect_band["summary"]["d_checkins"][0] < E.run(deep, eps=B.effect["eps_hi"]).summary["d_checkins"]


def test_beyond_range_replay():
    c = J["beyond_range_checks"]
    assert c["test_cuts_log_closer_than_line"] and c["test_cuts_mean_inside_widened_band"]
    assert not c["test_cuts_mean_inside_line_only_band"]                 # why the band was widened
    assert c["training_non_war_cuts_mean_inside_band"]
    t = {x["nationality_key"]: x for x in J["beyond_range_replay"]["test_window_cuts_by_country"]}
    for m in ("KAZAKHSTAN", "UZBEKISTAN", "AZERBAIJAN"):                 # Wizz Air exit markets
        assert t[m]["actual"] < t[m]["line"] and t[m]["inside_lever_band"]
    s = J["beyond_range_replay"]["inside_range_test_slope_local_passengers"]
    assert abs(s["slope"] - 1) < 1.96 * s["slope_se"]                    # inside the range the lever stays calibrated


def test_engine_properties_on_random_scenarios():
    p = J["properties"]
    assert p["all_passed"] and p["scenarios"] >= 200 and not p["failures"]
    assert len(p["change_types"]) == 6                                   # every change type exercised


def test_march_2026_replay():
    m = J["march_2026"]
    assert 0.2 < m["seats_vs_normal"]["low"] < m["seats_vs_normal"]["high"] < 0.5
    t = m["total"]
    assert t["observed_vs_normal"] < t["flights_only_low_seats"] < t["flights_only_high_seats"] < 1
    assert 0 < t["share_of_drop_explained_by_flights"][0] < t["share_of_drop_explained_by_flights"][1] < 1
    lr = t["share_of_drop_explained_by_flights_lever_range"]
    assert lr[0] <= t["share_of_drop_explained_by_flights"][0] and t["share_of_drop_explained_by_flights"][1] <= lr[1] < 1
    c = m["checks"]
    assert c["flights_plus_net_demand_reproduce_scad"] and c["no_double_count_flag"] and c["flights_never_over_explain_beyond_lever_band"]
    reg = {r["region"]: r for r in m["by_region"]}
    assert reg["UAE residents (Domestic)"]["flights_only_low_seats"] == 1     # no flights, all demand
    w = sc.World(cfg)
    for k in ("low", "high"):
        f = json.load(open(O / "presets" / f"march_2026_flights_{k}.json")); d = json.load(open(O / "presets" / f"march_2026_demand_net_of_flights_{k}.json"))
        sc.validate(f, w); sc.validate(d, w)
        assert all("scad_region" not in x for x in d["changes"]) and len(d["changes"]) == len(w.markets)
