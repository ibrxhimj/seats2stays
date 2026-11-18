"""Step 5.4: the three pre-registered candidates. The decision rules are applied exactly as written in
docs/phase5_4_preregistration.md, and the saved result was produced under that file as it stands."""
import json
from atp import phase5 as p5
from atp.config import load_config


def _sb(prod, c1, c3):
    keys = ("intl_daily_x_nationality", "intl_monthly_x_nationality", "domestic_daily", "domestic_monthly")
    mk = lambda v: {f: dict(zip(keys, v[f])) for f in "AB"}
    return {"production": mk(prod), "C1_post_break": mk(c1), "C3_poisson_weights": mk(c3)}


def _eps(slope_lo, slope_hi, sk_w_B, sk_u_B, sk_w_A, sk_u_A):
    return {"B": {"slope": {"weighted": [0.9, slope_lo, slope_hi]}, "skill": {"weighted": sk_w_B, "unweighted": sk_u_B}},
            "A": {"skill": {"weighted": sk_w_A, "unweighted": sk_u_A}}}


PROD = {"A": (10.0, 8.0, 4.0, 2.0), "B": (11.0, 9.0, 5.0, 3.0)}


def test_c1_rule_per_group():
    ok = p5.decide(_sb(PROD, {"A": (9.9, 7.9, 4.1, 2.0), "B": (11.04, 9.0, 5.0, 3.0)}, PROD), _eps(0, 2, .1, 0, 0, 0))["C1_post_break"]
    assert ok["international"]["keep"] and not ok["domestic"]["keep"]                 # domestic worse on fold A
    bad = p5.decide(_sb(PROD, {"A": (9.9, 8.1, 4.0, 2.0), "B": (11.0, 9.0, 5.0, 3.0)}, PROD), _eps(0, 2, .1, 0, 0, 0))["C1_post_break"]
    assert not bad["international"]["keep"]                                            # monthly worse on fold A
    tol = p5.decide(_sb(PROD, {"A": (9.9, 7.9, 4.0, 2.0), "B": (11.06, 9.0, 5.0, 3.0)}, PROD), _eps(0, 2, .1, 0, 0, 0))["C1_post_break"]
    assert not tol["international"]["keep"]                                            # fold B worse by more than 0.05


def test_c3_needs_a_tenth_of_a_point_and_no_domestic_loss():
    d = lambda c3: p5.decide(_sb(PROD, PROD, c3), _eps(0, 2, .1, 0, 0, 0))["C3_poisson_weights"]["keep"]
    assert d({"A": (9.89, 8.0, 4.0, 2.0), "B": (11.0, 9.0, 5.0, 3.0)})
    assert not d({"A": (9.95, 8.0, 4.0, 2.0), "B": (11.0, 9.0, 5.0, 3.0)})            # gain below 0.10
    assert not d({"A": (9.8, 8.0, 4.0, 2.01), "B": (11.0, 9.0, 5.0, 3.0)})            # Domestic worse


def test_c2_lever_test_and_skill_gains():
    d = lambda e: p5.decide(_sb(PROD, PROD, PROD), e)["C2_recency_eps"]["keep"]
    assert d(_eps(0.5, 1.5, 0.12, 0.10, 0.05, 0.055))
    assert not d(_eps(0.1, 0.9, 0.12, 0.10, 0.05, 0.05))                              # slope interval excludes 1
    assert not d(_eps(0.5, 1.5, 0.105, 0.10, 0.05, 0.05))                             # gain below 0.01
    assert not d(_eps(0.5, 1.5, 0.12, 0.10, 0.03, 0.05))                              # fold A loses more than 0.01


def test_saved_result_was_produced_under_the_preregistration_as_it_stands():
    cfg = load_config()
    r = json.load(open(cfg["paths"]["processed"] / "phase5" / "candidates.json"))
    assert r["prereg_sha256"] == p5.sha256(cfg["root"] / p5.PREREG)
    assert r["production_matches_saved"]
    assert r["decisions"] == p5.decide(r["stageb"], r["eps"])


def test_pooled_fit_with_one_series_is_the_production_fit():
    """(b): with a single series, the shared-month fit is the production forecaster exactly (same rows, terms, offset)."""
    import numpy as np, pandas as pd
    from atp import forecaster as fc
    cfg = load_config(); D, _, _ = p5._forecaster_frame(cfg)
    j24 = json.load(open(cfg["paths"]["processed"] / "phase2" / "step24_forecaster.json"))
    spec, F, start = set(j24["chosen_spec"]), cfg["folds"]["A"], pd.Timestamp("2023-01-07")
    d = D[D.key == "KUWAIT"]
    a = fc.fit_predict(d, F["train_end"], start, F["holdout_end"], spec, j24["eps_used_in_folds"]["A"])[0]
    b = p5.fit_predict_pooled_months(d, F["train_end"], start, F["holdout_end"], spec, j24["eps_used_in_folds"]["A"])[0]
    assert np.allclose(a.pred.to_numpy(), b.pred.to_numpy(), rtol=1e-6)


def test_saved_b_c_result_matches_its_preregistration_and_rules():
    cfg = load_config()
    r = json.load(open(cfg["paths"]["processed"] / "phase5" / "candidates_bc.json"))
    assert r["prereg_sha256"] == p5.sha256(cfg["root"] / p5.PREREG_BC)
    b, c = r["b_gcc_seasonal"], r["c_resident_split"]
    assert b["decision"]["keep"] == (b["decision"]["gain_A"] >= 0.10 and b["decision"]["worse_B"] <= p5.RULE_TOL)
    same = (c["A"]["difference"]["estimate"] > 0) == (c["B"]["difference"]["estimate"] > 0)
    assert c["decision"]["keep"] == (c["A"]["excludes_zero"] and c["B"]["excludes_zero"] and same)
