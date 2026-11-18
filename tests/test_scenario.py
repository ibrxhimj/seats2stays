"""Step 3.3: scenario inputs — every lever moves exactly what it should, and nothing else; bad inputs are refused."""
import json
import numpy as np
import pandas as pd
import pytest
from atp import scenario as sc
from atp.config import load_config

cfg = load_config(); O = cfg["paths"]["processed"] / "phase3"
if not (O / "aircraft_seats.csv").exists():
    pytest.skip("run `make phase3` first", allow_module_level=True)
W = sc.World(cfg)
BASE, _ = sc.resolve({"changes": []}, W)


def run(*changes):
    r, flags = sc.resolve({"changes": list(changes)}, W)
    return r.set_index(["market", "date"]), flags


def window(r, a, b):
    d = r.index.get_level_values("date")
    return (d >= a) & (d <= b)


def test_null_scenario_is_the_baseline():
    j = json.load(open(O / "step33_inputs.json"))["null_scenario"]
    assert j["scenario_equals_base"] and j["max_seat_gap_vs_baseline"] < 1e-9 and j["markets"] == 46
    assert (BASE.demand_mult == 1).all() and (BASE.stay_mult == 1).all() and BASE.checkins_base.notna().all()


def test_cancel_market_for_one_month():
    r, flags = run({"type": "route", "market": "UNITED KINGDOM", "start": "2026-07-01", "end": "2026-07-31", "seats_pct": -100})
    uk = r.xs("UNITED KINGDOM", level="market")
    jul = uk.index.month == 7
    assert (uk.seats_scen[jul] == 0).all() and np.allclose(uk.seats_scen[~jul], uk.seats_base[~jul])
    others = r.drop("UNITED KINGDOM", level="market")
    assert np.allclose(others.seats_scen, others.seats_base) and not flags


def test_route_seats_frequency_and_aircraft():
    rb = W.routes.set_index(["nationality_key", "city", "airline", "moy"])
    lon = rb.loc[("UNITED KINGDOM", "London", "Etihad Airways", 8)]
    r, _ = run({"type": "route", "market": "UNITED KINGDOM", "city": "London", "airline": "Etihad Airways",
                "start": "2026-08-01", "end": "2026-08-31", "seats_pct": 50})
    d = r.xs("UNITED KINGDOM", level="market").loc["2026-08-10"]
    assert np.isclose(d.seats_scen - d.seats_base, 0.5 * lon.seats)
    r, _ = run({"type": "route", "market": "UNITED KINGDOM", "city": "London", "airline": "Etihad Airways",
                "start": "2026-08-01", "end": "2026-08-31", "flights_per_week": 14})
    d = r.xs("UNITED KINGDOM", level="market").loc["2026-08-10"]
    assert np.isclose(d.seats_scen - (d.seats_base - lon.seats), 2 * lon.seats / lon.n_flights)
    r, _ = run({"type": "route", "market": "UNITED KINGDOM", "city": "London", "airline": "Etihad Airways",
                "start": "2026-08-01", "end": "2026-08-31", "seats_per_flight": 500})
    d = r.xs("UNITED KINGDOM", level="market").loc["2026-08-10"]
    assert np.isclose(d.seats_scen - (d.seats_base - lon.seats), 500 * lon.n_flights)
    assert np.isclose(d.pax_scen / d.seats_scen, d.pax_base / d.seats_base, rtol=0.05)   # passengers move with seats


def test_load_factor_and_transfer_share():
    r, _ = run({"type": "load_factor", "markets": "INDIA", "start": "2026-09-01", "end": "2026-09-30", "pct": 5})
    i = r.xs("INDIA", level="market").loc["2026-09"]
    assert (i.pax_scen <= 1.05 * i.pax_base + 1e-6).all() and (i.pax_scen > i.pax_base).all() and np.allclose(i.seats_scen, i.seats_base)
    assert ((i.p2p_scen / i.p2p_base).between(1.0, 1.05 + 1e-9)).all()                        # each route keeps its local share
    r2, flags = run({"type": "load_factor", "markets": "INDIA", "start": "2026-09-01", "end": "2026-09-30", "pct": 100})
    i2 = r2.xs("INDIA", level="market").loc["2026-09"]
    assert any("capped at full planes" in f for f in flags) and (i2.pax_scen <= 1.03 * i2.seats_scen).all()   # full planes (+ lap infants)
    r, _ = run({"type": "transfer_share", "markets": "GERMANY", "start": "2026-10-01", "end": "2026-10-31", "value": 0.2})
    g = r.xs("GERMANY", level="market").loc["2026-10"]
    assert np.allclose(g.p2p_scen / g.pax_scen, 0.8)                                          # market share = value
    r, _ = run({"type": "transfer_share", "markets": "INDIA", "start": "2026-10-01", "end": "2026-10-31", "points": -10})
    i = r.xs("INDIA", level="market").loc["2026-10"]
    assert np.allclose((1 - i.p2p_scen / i.pax_scen) - (1 - i.p2p_base / i.pax_base), -0.10)       # −10 points, market level


def test_demand_by_region_and_stay():
    r, flags = run({"type": "demand", "markets": "Europe", "start": "2026-11-01", "end": "2026-11-30", "pct": -20},
                   {"type": "stay_length", "markets": "DOMESTIC", "start": "2026-12-01", "end": "2026-12-31", "pct": 10})
    eu = set(W.regions["Europe"])
    nov = window(r, pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-30"))
    in_eu = r.index.get_level_values("market").isin(eu)
    assert np.allclose(r.demand_mult[nov & in_eu], 0.8) and (r.demand_mult[~(nov & in_eu)] == 1).all()
    assert np.allclose(r.xs("DOMESTIC", level="market").loc["2026-12"].stay_mult, 1.1)
    assert sum("assumption lever" in f for f in flags) == 2


def test_new_route_fills_a_zero_seat_market():
    r, flags = run({"type": "new_route", "market": "UZBEKISTAN", "city": "Tashkent", "airline": "Etihad Airways",
                    "start": "2026-08-10", "end": "2026-12-31", "flights_per_week": 7, "aircraft_class": "narrowbody"})
    uz = r.xs("UZBEKISTAN", level="market")
    nb = pd.read_csv(O / "aircraft_seats.csv").set_index("aircraft_class").seats_per_flight["narrowbody"]
    assert (uz.loc["2026-08-01":"2026-08-09"].seats_scen == 0).all()
    assert np.allclose(uz.loc["2026-08-10":"2026-08-31"].new_route_seats, nb)
    assert (uz.loc["2026-08-10":"2026-08-31"].seats_base == 0).all()


def test_groups_keep_their_flight_markets():
    r, flags = run({"type": "load_factor", "markets": "Europe", "start": "2026-11-01", "end": "2026-11-30", "pct": 3})
    eu = set(W.regions["Europe"]); fl = eu & set(W.flight)
    nov = window(r, pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-30"))
    m = r.index.get_level_values("market")
    assert (r.pax_scen[nov & m.isin(fl)] > r.pax_base[nov & m.isin(fl)]).all()
    assert np.allclose(r.pax_scen[~(nov & m.isin(fl))], r.pax_base[~(nov & m.isin(fl))])
    assert any("no flight data and are left unchanged" in f for f in flags)


def test_new_route_defaults_come_from_the_airline():
    r, flags = run({"type": "new_route", "market": "UZBEKISTAN", "city": "Tashkent", "airline": "Etihad Airways",
                    "start": "2026-09-01", "end": "2026-09-30", "flights_per_week": 7, "aircraft_class": "narrowbody"})
    assert any("Etihad Airways's network at AUH" in f for f in flags)          # Etihad does not fly Uzbekistan yet
    e = W.routes[W.routes.airline == "Etihad Airways"].groupby("moy")[["pax", "p2p"]].sum().loc[9]
    uz = r.xs("UZBEKISTAN", level="market").loc["2026-09-10"]
    added = uz.p2p_scen - uz.p2p_base
    assert np.isclose(added / (uz.pax_scen - uz.pax_base), e.p2p / e.pax)        # Etihad's own local share, not Air Arabia's


def test_bad_inputs_are_refused():
    bad = [({"type": "demand", "markets": "ATLANTIS", "start": "2026-05-01", "end": "2026-05-31", "pct": 5}, "unknown market"),
           ({"type": "demand", "markets": "INDIA", "start": "2027-03-01", "end": "2027-03-31", "pct": 5}, "outside the horizon"),
           ({"type": "demand", "markets": "INDIA", "start": "2026-06-01", "end": "2026-05-01", "pct": 5}, "after end"),
           ({"type": "route", "market": "DOMESTIC", "start": "2026-05-01", "end": "2026-05-31", "seats_pct": 10}, "no flight data"),
           ({"type": "route", "market": "KAZAKHSTAN", "airline": "Wizz Air Abu Dhabi", "start": "2026-05-01", "end": "2026-05-31",
             "seats_pct": 10}, "new_route"),
           ({"type": "new_route", "market": "UNITED KINGDOM", "city": "London", "airline": "Etihad Airways", "start": "2026-05-01",
             "end": "2026-05-31", "flights_per_week": 7, "aircraft_class": "widebody"}, "already flies"),
           ({"type": "route", "market": "INDIA", "start": "2026-05-01", "end": "2026-05-31", "seats_pct": 10, "flights_per_week": 3}, "exactly one"),
           ({"type": "teleport", "start": "2026-05-01", "end": "2026-05-31"}, "unknown type"),
           ({"type": "route", "market": "INDIA", "start": "2026-05-01", "end": "2026-05-31", "flights_per_week": 7}, "one route"),
           ({"type": "route", "market": "SPAIN", "city": "Malaga", "airline": "Etihad Airways", "start": "2026-12-01",
             "end": "2026-12-31", "seats_pct": 10}, "in this window"),
           ({"type": "load_factor", "markets": "ROMANIA", "start": "2026-05-01", "end": "2026-05-31", "pct": 5}, "no flight data")]
    for c, msg in bad:
        with pytest.raises(sc.ScenarioError, match=msg):
            run(c)


def test_presets():
    j = json.load(open(O / "step33_inputs.json"))
    m = j["march_2026_observed"]
    assert abs(m["pack_total_mar26_over_mar25"] - m["scad_ratio_by_region"]["Total"]) < 1e-4      # matches SCAD total (ratio stored to 4 dp)
    p = j["presets"]
    assert abs(p["march_2026_observed"]["march_checkins_ratio_to_normal"] - m["all_causes_vs_normal_total"]) < 1e-4   # preset pct stored to 2 dp
    e = p["etihad_2026_launches"]
    assert e["uzbekistan_aug_seats_base_vs_scen"][0] == 0 < e["uzbekistan_aug_seats_base_vs_scen"][1]
    assert any("Romania has no flight data" in f for f in e["flags"])
    ac = pd.read_csv(O / "aircraft_seats.csv").set_index("aircraft_class")
    assert 150 < ac.seats_per_flight["narrowbody"] < 200 < 280 < ac.seats_per_flight["widebody"] < 330


def test_observed_level_plus_flights_is_flagged():
    mar = json.load(open(O / "presets" / "march_2026_observed.json"))
    s = {"changes": mar["changes"] + [{"type": "route", "market": "INDIA", "start": "2026-03-10", "end": "2026-03-31", "seats_pct": -30}]}
    _, flags = sc.resolve(s, W)
    assert any("counted twice" in f for f in flags)
    _, flags = sc.resolve({"changes": mar["changes"]}, W)
    assert not any("counted twice" in f for f in flags)
