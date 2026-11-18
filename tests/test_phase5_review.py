"""Step 5.3: the independent reviewer's re-derivation (review/results.json) against our outputs. Every quantity must agree
to a relative 1e-6, or be a bootstrap end within its random-draw tolerance, or be one of the differences explained in
reports/phase5_review.md. If a later change moves a number the reviewer checked, rerun `make review` (the reviewer's own
code re-derives it from the raw files) rather than editing results.json."""
import json
import pandas as pd
import pytest
from atp import phase5 as p5
from atp.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_reviewer_agrees_with_our_outputs(cfg):
    theirs = json.load(open(cfg["root"] / p5.REVIEW / "results.json"))
    d = p5.compare_review(p5.ours_for_review(cfg), theirs)
    assert d["ok"], {k: r for k, r in d["rows"].items() if r["status"] in ("DIFFERENT", "NOT OURS")}
    by = lambda s: {k for k, r in d["rows"].items() if r["status"] == s}
    assert by("random draws") <= set(p5.APPROX) and by("by definition") <= set(p5.APART)
    assert len(by("same")) >= 100 and len(theirs) == 114


def test_status_rules():
    s = p5._review_status
    assert s("x", 0.31, 0.31 * (1 + 1e-9)) == "same" and s("x", 0.31, 0.32) == "DIFFERENT"
    assert s("n", 505, 505.0) == "same"                                          # int vs float
    assert s("eps_ci95_lo", 0.057, 0.045) == "random draws" and s("eps_ci95_lo", 0.057, 0.10) == "DIFFERENT"
    assert s("m", "2024-01-01", "2024-01") == "same" and s("m", "2024-01-01", "2024-02") == "DIFFERENT"
    assert s("d", "2025-06-06", "2025-06-07") == "DIFFERENT"
    assert s("l", ["2025-06-24", "2026-02-28"], ["2025-06-24", "2026-02-28"]) == "same"
    assert s("c", {"A": ["2025-06-24", "2025-06-26"]}, {"A": ["2025-06-23", "2025-06-26"]}) == "DIFFERENT"
    assert s("narrowbody_seats", 174.5, 173.4) == "by definition"                  # explained in the report
    d = p5.compare_review({"a": 1.0}, {"a": 1.0, "b": 2.0})
    assert not d["ok"] and d["rows"]["b"]["status"] == "NOT OURS"


def test_review_inputs_carry_no_actuals(cfg, tmp_path):
    d = p5.review_inputs(cfg, tmp_path / "inputs")
    for f in "AB":
        for kind, n_markets in (("endtoend", 46), ("stageb", 45)):
            t = pd.read_csv(d / f"backtest_{kind}_{f}.csv")
            assert list(t.columns) == ["market", "date", "pred_guests"] and t.market.nunique() == n_markets
    assert list(pd.read_csv(d / "forecast_test_window.csv", nrows=1).columns) == ["series", "date", "checkins_forecast"]
    for name, want in cfg["raw_sha256"].items():                                 # the raw files exactly as received
        assert p5.sha256(d / "raw" / name) == want
