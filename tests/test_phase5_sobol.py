"""Step 5.6: the Sobol check. The fast form must be the engine, and the saved result must be the one the
pre-registration (as it stands) produces, read with its own rules."""
import json
import numpy as np
import pytest
from atp import engine as en, sobol as so, phase5 as p5
from atp.config import load_config


@pytest.fixture(scope="module")
def fast():
    E = en.Engine(load_config()); return so.FastEngine(E)


def test_fast_form_is_the_engine(fast):
    cases = [("mixed", {"seats": 7.0, "lf": -4.0, "ts": 6.0, "demand": -3.0, "stay": 5.0}, so.HORIZON, None, None),
             ("winter, one market, high ε", {"seats": 10}, so.WINTER, fast.E.params.eps_band[1], "INDIA"),
             ("deep transfer change", {"ts": 25.0}, so.HORIZON, None, None)]
    g = so.gate(fast, cases)
    assert g["passed"], g


def test_guest_night_weights_reproduce_stage_b(fast):
    """W · check-ins = Stage B's guest nights (to h1 + 44) for any check-ins, market by market."""
    rng = np.random.default_rng(3)
    for m in ("INDIA", "DOMESTIC", "UZBEKISTAN"):
        a = rng.uniform(0, 500, len(fast.days)); i = fast.mix[m]
        assert np.isclose((fast.W[i] * a).sum(), fast.E.stageb.guests(m, a, memory=False).sum(), rtol=1e-12)


def test_rules():
    r = {"ST": dict(zip(so.TAB2_ORDER, (0.44, 0.44, 0.09, 0.02, 0.01))), "ST_conf": dict.fromkeys(so.TAB2_ORDER, 0.005)}
    assert so.ranking_check(r)["confirmed"]
    r["ST"]["load factor"] = 0.05                                       # clearly above seats: Tab 2 would be corrected
    c = so.ranking_check(r)
    assert not c["confirmed"] and ("seats", "load factor") in c["contradictions"]
    one = lambda s1, st: {"S1": {"x": s1}, "ST": {"x": st}, "interaction_share": 1 - s1}
    assert not so.interactions(one(0.99, 0.99))["matter"] and so.interactions(one(0.9, 0.97))["matter"]


def test_saved_result_matches_its_preregistration_and_rules():
    cfg = load_config()
    r = json.load(open(cfg["paths"]["processed"] / "phase5" / "sobol.json"))
    assert r["prereg_sha256"] == p5.sha256(cfg["root"] / so.PREREG_SOBOL)
    assert r["gate"]["passed"] and r["gate"]["max_rel"] <= so.GATE and len(r["gate"]["cases"]) == 30      # incl. the ±25% runs (A1)
    assert r["ranking"] == so.ranking_check(r["1a"]) and r["interactions"] == so.interactions(r["1a"], r["1b"])
    for k in ("1a", "1b", "1c"):                                        # converged: half the runs give the same shares
        assert all(abs(r[k]["ST"][n] - r[k + "_check"]["ST"][n]) < 0.01 for n in r[k]["ST"])
        assert r[k]["n_runs"] == so.N_BASE * (2 * len(r[k]["ST"]) + 2)
    e = "ε (check-in response)"
    for k in ("R1", "R1_shared", "R2"):
        assert r[k]["dominant"] == next((g for g, v in r[k]["ST"].items() if v > 0.5), None)
    # the app's split lies between the two ends of how market errors relate (independent, shared) for the network result
    assert r["R1"]["ST"]["baseline level"] <= 1 - r["R1"]["app_lever_share"] <= r["R1_shared"]["ST"]["baseline level"]
    assert abs(r["R2"]["ST"][e] - r["R2"]["app_lever_share"]) < 0.05      # one market: no floor, the split matches


def test_tab_three_block():
    from atp.simulator import Simulator, LEVERS
    K = Simulator().know()["sobol"]
    assert [x["id"] for x in K["rows"]] == so.TAB2_ORDER                # ranking confirmed: same order, ties as Tab 2
    names = {k: n for k, n, _, _ in LEVERS}
    assert names["transfer share"].lower() in K["lead"] and "same order" in K["lead"]
    assert len(K["flight"]) == 3 and len(K["points"]) == 3
