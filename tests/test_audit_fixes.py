"""Contracts from the independent maths audit (12 Oct 2026): each test pins one fixed behaviour.

E3  a scenario that adds and cuts at once: the range spans the result's extremes over the whole ε interval, not just
    its ends;  E4  the passenger cap holds in the scenario's demand conditions;  E5  a market connecting-share target is
    reached even where single routes clip;  E6  the reverse query returns 0 when no flight is needed and brackets from 0;
E8  convergence is decided by the projected gradient alone (unit test in tests/test_stageb.py)."""
import numpy as np
import pandas as pd
import pytest
from atp import engine as en, bands as bd, scenario as sc, sensitivity as se
from atp.config import load_config

cfg = load_config(); O = cfg["paths"]["processed"] / "phase3"
if not (O / "stageb_memory.parquet").exists():
    pytest.skip("run `make phase3` first", allow_module_level=True)
E = en.Engine(cfg); B = bd.Bands(cfg)


def test_E4_cap_holds_after_demand_scaling():
    # a large seat increase (cap binds) together with doubled demand in the same market and window
    s = {"changes": [{"type": "route", "market": "INDIA", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": 400},
                     {"type": "demand", "markets": "INDIA", "start": "2026-11-01", "end": "2026-11-30", "pct": 100}]}
    d = E.run(s).daily
    dL = np.abs(d.p2p_scen - d.p2p_base)
    assert (np.abs(d.d_checkins_flights) <= dL + 1e-9).all()                     # the advertised bound, demand or not
    assert np.allclose(d.d_checkins, d.d_checkins_flights + d.d_checkins_demand)  # the decomposition still adds up
    assert np.allclose(d.checkins_scen, d.checkins_base * d.demand_mult + d.d_checkins_flights)


def test_E4_no_demand_change_is_unchanged():
    s = {"changes": [{"type": "route", "market": "INDIA", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": 400}]}
    d = E.run(s).daily
    raw = d.checkins_base * (en.lever_G(np.divide(d.p2p_scen, d.p2p_base, out=np.ones(len(d)), where=d.p2p_base > 0),
                                        E.params.eps, E.params.lo, E.params.hi) - 1)
    std = d.has_flight_data & (d.p2p_base > 0)
    want = np.sign(raw) * np.minimum(np.abs(raw), np.abs(d.p2p_scen - d.p2p_base))
    assert np.allclose(d.d_checkins_flights[std], want[std])


def test_E5_scale_shares_reaches_the_target_where_routes_clip():
    # the auditor's example: two equal routes at 90% and 10% connecting (market 50%), asked for 80%
    new, got = sc.scale_shares(np.array([0.9, 0.1]), np.array([1.0, 1.0]), np.array(["m", "m"]), np.array([0.8, 0.8]))
    assert np.isclose(got[0], 0.8, atol=1e-9) and new[0] <= sc.TRANSFER_ROUTE_CAP + 1e-12 and new[1] > 0.6
    assert new[0] > new[1]                                                      # routes keep their order
    # nothing clips: exactly proportional, as before
    new, got = sc.scale_shares(np.array([0.4, 0.2]), np.array([1.0, 3.0]), np.array(["m", "m"]), np.array([0.3, 0.3]))
    assert np.allclose(new, np.array([0.4, 0.2]) * 0.3 / 0.25) and np.allclose(got, 0.3)
    # a route with no connecting passengers cannot connect: the group stops at its maximum and says so
    new, got = sc.scale_shares(np.array([0.0, 0.9]), np.array([1.0, 1.0]), np.array(["m", "m"]), np.array([0.9, 0.9]))
    assert np.isclose(got[0], sc.TRANSFER_ROUTE_CAP / 2) and new[0] == 0.0


def test_E5_market_share_target_is_reached_in_the_engine():
    s = {"changes": [{"type": "transfer_share", "markets": "flight markets", "start": "2026-03-01", "end": "2027-02-28", "pct": 10}]}
    d, flags = E.resolve(s)
    k = d.has_flight_data & (d.pax_base > 0)
    t0 = 1 - d.p2p_base[k] / d.pax_base[k]; t1 = 1 - d.p2p_scen[k] / d.pax_scen[k]
    want = np.minimum(t0 * 1.10, 0.99)
    assert (t1 <= want + 1e-9).all()                                            # never beyond the target
    hit = np.isclose(t1, want, atol=1e-6)
    assert hit.mean() > 0.99 and (hit.all() or any("not reached" in f for f in flags))
    assert any("capped just below 100% connecting" in f for f in flags)        # some routes do clip at +10%


def test_E3_mixed_scenario_range_spans_interior_extremes():
    # adds in one market, cuts in another: the total need not be monotone in ε
    s = {"changes": [{"type": "route", "market": "CHINA", "start": "2026-11-01", "end": "2027-02-28", "seats_pct": 60},
                     {"type": "route", "market": "INDIA", "start": "2026-11-01", "end": "2027-02-28", "seats_pct": -25}]}
    r = B.run(E, s)
    lo, hi = r.effect_band["summary"]["d_guest_nights"]
    e = B.effect
    for x in np.linspace(e["eps_lo"], e["eps_hi"], 21):
        v = E.run(s, eps=x).summary["d_guest_nights"]
        assert lo - 1e-6 <= v <= hi + 1e-6
    assert r.effect_band["interior_runs"]


class _FakeRun:
    def __init__(self, v, daily):
        self.summary = {"d_guest_nights": v}; self.daily = daily


class _FakeEngine:
    """The auditor's two-market example as an engine: weights 100,000 and 293,829.209 on multipliers 2.0 and 0.71, so the
    total W1(2^ε − 1) + W2(0.71^ε − 1) has an interior minimum inside ε's interval."""
    def __init__(self):
        self.params = type("P", (), {"per_seat": 0.05})()
        self.daily = pd.DataFrame({"outside_trusted_range": [False, False], "p2p_scen": [2.0, 0.71], "p2p_base": [1.0, 1.0],
                                   "zero_baseline_rule": [False, False], "new_route_lever": [False, False],
                                   "d_checkins_flights": [1.0, -1.0]})

    def f(self, eps):
        return 100000 * (2.0 ** eps - 1) + 293829.209 * (0.71 ** eps - 1)

    def run(self, scenario, eps=None, per_seat=None, cuts="line", floor=0.0):
        return _FakeRun(self.f(0.3 if eps is None else eps), self.daily)


def test_E3_interior_minimum_is_found():
    fe = _FakeEngine(); e = B.effect
    lo, hi = e["eps_lo"], e["eps_hi"]
    xs = np.linspace(lo, hi, 2001); true_min = fe.f(xs).min(); true_max = fe.f(xs).max()
    assert true_min < min(fe.f(lo), fe.f(hi)) - 1                              # the example really is non-monotone here
    runs = B.end_runs(fe, {}, fe.run({}))
    got = [r.summary["d_guest_nights"] for r in runs]
    assert min(got) <= true_min + 1e-3 * abs(true_min) and max(got) >= true_max - 1e-6


def test_E3_same_sign_scenario_keeps_two_end_runs():
    s = {"changes": [{"type": "route", "market": "CHINA", "start": "2026-11-01", "end": "2027-02-28", "seats_pct": 30}]}
    mid = E.run(s)
    assert len(B.end_runs(E, s, mid)) == 2


def test_E6_reverse_query_zero_target_and_bracket():
    out = se.seats_needed(E, B, "UNITED KINGDOM", "2026-11-01", "2027-02-28", 0.0, 174.0)
    assert out["flights_per_week"] == 0.0
    out = se.seats_needed(E, B, "UNITED KINGDOM", "2026-11-01", "2027-02-28", 5000.0, 174.0)
    f = out["flights_per_week"]; assert f and 0 < f < se.TOP_FPW
    lo, hi = out["flights_per_week_range"]
    assert lo is None or lo <= f
    assert hi is None or hi >= f


def test_net_demand_is_exact_when_the_cap_binds():
    """E4 follow-up: the March replay's demand multiplier is solved under 'cap after demand', so flights + demand reproduce
    the target exactly even where the cap binds (observed ÷ flights-only would not)."""
    import pandas as pd
    from atp.validation import _net_demand
    d = pd.DataFrame({"market": ["X", "X", "Y"], "checkins_base": [100.0, 80.0, 50.0],
                      "flight_raw": [-60.0, -10.0, 5.0], "flight_cap": [20.0, 40.0, np.inf]})
    target = 120.0
    m = _net_demand(d, ["X", "Y"], target)
    got = (d.checkins_base * m + np.sign(d.flight_raw) * np.minimum(np.abs(d.flight_raw) * m, d.flight_cap)).sum()
    assert abs(got - target) < 1e-9
    naive = target / (d.checkins_base + np.sign(d.flight_raw) * np.minimum(np.abs(d.flight_raw), d.flight_cap)).sum()
    assert abs(m - naive) > 1e-3                                         # the cap binds, so the old ratio would miss
