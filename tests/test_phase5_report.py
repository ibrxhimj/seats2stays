"""Step 5.10: the Phase 5 report is what the code builds from the saved results, and the results it summarises still
match their pre-registrations and agree with one another."""
from atp import phase5_report as pr
from atp.config import load_config


def test_saved_report_is_what_the_code_builds():
    cfg = load_config()
    assert (cfg["root"] / pr.OUT).read_text() == pr.build(pr.load(cfg)), "rerun `python -m atp.phase5 report`"


def test_preregistrations_and_consistency():
    r = pr.load(load_config())
    assert all(p["matches"] for p in pr.preregs(r)), "a pre-registration changed after its result was saved"
    assert r["rebuild_compare"]["ok"] and r["review_compare"]["ok"]
    kept = [n for n, k in pr.decisions(r) if k]
    assert not kept                                   # 5.8 is reported as not triggered: no 5.4 candidate passed
    assert not r["benchmark"]["verdict"]["lightgbm_more_accurate"] and r["sobol"]["ranking"]["confirmed"]
    assert pr.accuracy(r)["all_reviewed"]             # every accuracy figure quoted was re-derived identically by the reviewer
