"""Step 3.4: the engine reproduces the validated pieces exactly and behaves as its rules say."""
import json
import numpy as np
import pytest
from atp import engine as en, stagea as sa
from atp.config import load_config

cfg = load_config(); O = cfg["paths"]["processed"] / "phase3"
if not (O / "stageb_memory.parquet").exists():
    pytest.skip("run `make phase3` first", allow_module_level=True)
E = en.Engine(cfg); P = E.params


def one(change):
    return E.run({"changes": [change]})


def test_step_checks():
    j = json.load(open(O / "step34_engine.json"))["checks"]
    assert j["stageb_reproduces_baseline_max_abs"] < 1e-9 and j["null_max_abs_change"] < 1e-9
    assert j["lever_equals_phase2_seat_lever_max_abs"] < 1e-9 and j["additivity_max_abs"] < 1e-6
    assert abs(j["stay_plus_10pct_realised_pct"] - 10) < 1e-6 and abs(j["demand_minus_20pct_ratio"] - 0.8) < 1e-12
    assert j["ten_day_change"]["checkin_days"] == 10


def test_lever_shape():
    assert np.isclose(en.lever_G(1.0, P.eps, P.lo, P.hi), 1.0)
    x = np.linspace(0, 4, 401); g = en.lever_G(x, P.eps, P.lo, P.hi)
    assert (np.diff(g) > 0).all()                                                  # more local passengers, more check-ins
    inside = (x - 1 >= P.lo) & (x - 1 <= P.hi)
    assert np.allclose(g[inside], x[inside] ** P.eps)                              # log form inside the trusted range
    ref = sa.seat_lever(np.ones_like(x), np.ones_like(x), x - 1, P.eps, np.full_like(x, 1e12), (P.lo, P.hi))[0] + 1
    assert np.allclose(g, ref)                                                     # same function as Phase 2


def test_cap_and_cancellation():
    r = one({"type": "route", "market": "UNITED KINGDOM", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": -100})
    d = r.daily[(r.daily.market == "UNITED KINGDOM") & (r.daily.date.dt.month == 11)]
    assert np.isclose(d.checkins_scen.sum() / d.checkins_base.sum(), en.lever_G(0.0, P.eps, P.lo, P.hi))
    assert (np.abs(r.daily.d_checkins_flights) <= np.abs(r.daily.p2p_scen - r.daily.p2p_base) + 1e-9).all()   # cap
    assert d.outside_trusted_range.all()


def test_zero_seat_and_no_flight_markets():
    r = one({"type": "new_route", "market": "UZBEKISTAN", "city": "Tashkent", "airline": "Etihad Airways",
             "start": "2026-08-10", "end": "2026-08-31", "flights_per_week": 7, "aircraft_class": "narrowbody"})
    d = r.daily[r.daily.market == "UZBEKISTAN"]
    on = d.date.between("2026-08-10", "2026-08-31")
    assert d[on].zero_baseline_rule.all() and (d[on].d_checkins_flights > 0).all() and (d[~on].d_checkins_flights == 0).all()
    r = one({"type": "new_route", "market": "ROMANIA", "city": "Bucharest", "airline": "Etihad Airways",
             "start": "2026-12-17", "end": "2027-02-28", "flights_per_week": 4, "aircraft_class": "narrowbody"})
    d = r.daily[(r.daily.market == "ROMANIA") & r.daily.new_route_lever]
    assert np.allclose(d.d_checkins_flights, P.per_seat * d.new_route_seats)


def test_guests_follow_checkins_linearly():
    r = one({"type": "route", "market": "INDIA", "start": "2026-06-01", "end": "2026-06-30", "seats_pct": 30})
    i = r.daily[r.daily.market == "INDIA"].sort_values("date")
    extra = E.stageb.guests("INDIA", i.d_checkins.to_numpy(), memory=False)
    assert np.allclose(extra[:E.stageb.n_h], i.d_guest_nights) and np.isclose(extra.sum(), r.summary["d_guest_nights"])
    assert r.summary["d_guest_nights_after_horizon"] == 0                          # June stays end long before Feb 2027


def test_eps_parameter_moves_the_effect():
    c = {"type": "route", "market": "GERMANY", "start": "2026-10-01", "end": "2026-12-31", "seats_pct": 25}
    lo, mid, hi = (E.run({"changes": [c]}, eps=e).summary["d_checkins"] for e in (P.eps_band[0], P.eps, P.eps_band[1]))
    assert 0 < lo < mid < hi


def test_demand_and_flights_decompose():
    r = E.run({"changes": [{"type": "demand", "markets": "INDIA", "start": "2026-11-01", "end": "2026-11-30", "pct": -40},
                           {"type": "route", "market": "INDIA", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": 20}]})
    f = E.run({"changes": [{"type": "route", "market": "INDIA", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": 20}]})
    s = r.summary
    assert np.isclose(s["d_checkins_flights"], 0.6 * f.summary["d_checkins_flights"])     # flights act on the reduced demand
    assert np.isclose(s["d_checkins_flights"] + s["d_checkins_demand"], s["d_checkins"])


def test_stay_change_beyond_the_model_is_refused():
    from atp import scenario as sc
    with pytest.raises(sc.ScenarioError, match="outside what the stay model can represent"):
        E.run({"changes": [{"type": "stay_length", "markets": "INDIA", "start": "2026-11-01", "end": "2026-11-30", "pct": -80}]})
