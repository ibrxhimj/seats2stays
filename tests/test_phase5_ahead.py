"""Step 5.7: accuracy by months ahead. The saved result is the one the pre-registration (as it stands) produces, checked
against an independent recomputation from the tables Tab 3 already shows."""
import json
import re
import numpy as np
import pandas as pd
from atp import ahead as ah, phase5 as p5
from atp.config import load_config


def _saved():
    cfg = load_config()
    raw = json.load(open(cfg["paths"]["processed"] / "phase5" / "ahead.json"))
    return cfg, json.loads(json.dumps(raw), object_pairs_hook=lambda p: {(int(k) if k.isdigit() else k): v for k, v in p})


def test_saved_result_matches_its_preregistration_and_rule():
    cfg, r = _saved()
    assert r["prereg_sha256"] == p5.sha256(cfg["root"] / ah.PREREG_AHEAD)
    assert r["consistency_max_abs_diff"] < 1e-6
    assert r["grows_with_lead"] == ah.grows(r["guests"])
    for f in "AB":                                                      # contrast of Amendment A1 is what it says
        v = r["guests"][f]["by_nationality"]
        assert np.isclose(r["guests"][f]["first_vs_later"]["estimate"], np.mean([v[h] for h in ah.LEADS[1:]]) - v[1])


def test_by_nationality_error_is_recomputed_from_the_tab_three_table():
    """Independent of ahead.py's own reading of the predictions: backtest_monthly.csv, months numbered within each fold."""
    cfg, r = _saved()
    bm = pd.read_csv(cfg["paths"]["processed"] / "phase4" / "backtest_monthly.csv")
    bm = bm[bm.market != "DOMESTIC"]
    bm["h"] = bm.groupby("fold").month.transform(lambda m: m.rank(method="dense")).astype(int)
    for f in "AB":
        for h in ah.LEADS:
            d = bm[(bm.fold == f) & (bm.h == h)]
            want = (d.actual_guest_nights - d.rebuilt_guest_nights).abs().sum() / d.actual_guest_nights.sum() * 100
            assert np.isclose(r["guests"][f]["by_nationality"][h], want, rtol=1e-9)
            tot = d.rebuilt_guest_nights.sum() / d.actual_guest_nights.sum() - 1
            assert np.isclose(r["guests"][f]["total"][h]["signed_error_pct"], tot * 100, rtol=1e-9)
    both = bm.assign(a=bm.actual_guest_nights, p=bm.rebuilt_guest_nights)
    for h in ah.LEADS:
        d = both[both.h == h]
        assert np.isclose(r["guests"]["pooled"]["by_nationality"][h], (d.a - d.p).abs().sum() / d.a.sum() * 100, rtol=1e-9)


def test_coverage_uses_the_other_folds_band():
    """Pre-registered: the band learned on the other fold. Over all markets this is exactly Step 3.5's cross-fold coverage."""
    cfg, r = _saved()
    s35 = json.load(open(cfg["paths"]["processed"] / "phase3" / "step35_bands.json"))["level_diagnostics"]["guests"]
    assert np.isclose(r["guests"]["A"]["coverage_all_markets"], s35["coverage_B_to_A"])
    assert np.isclose(r["guests"]["B"]["coverage_all_markets"], s35["coverage_A_to_B"])


def test_slope_rule():
    mk = lambda lo_a, lo_b: {f: {"slope": 1.0, "slope_ci": [lo, 2.0]} for f, lo in (("A", lo_a), ("B", lo_b))}
    assert ah.grows(mk(0.1, 0.2))["grows"]
    assert not ah.grows(mk(0.1, -0.1))["grows"]                          # both folds are needed


def test_tab_three_block_is_the_saved_result():
    from atp.simulator import Simulator
    K = Simulator().know(); A = K["ahead"]; _, r = _saved()
    assert A["chart"]["leads"] == ah.LEADS and all(len(A["chart"][k][f]) == 7 for k in ("country", "total", "coverage") for f in "AB")
    assert A["chart"]["country"]["A"][0] == round(r["guests"]["A"]["by_nationality"][1], 1)
    assert [m[:7] for m in A["chart"]["months"]["A"]] == [m[:7] for m in r["months"]["A"]] and len(A["points"]) == 3
    assert not any(re.search(r"(?<![\w.])-\d", t) for t in A["points"] + [A["method"]])      # typographic minus only
    assert "Months 8 to 12 are untested" in " ".join(A["points"])
