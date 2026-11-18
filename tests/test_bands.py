"""Step 3.5: effect band (flight lever) and level band (forecast error) — calibrated, separate, honest."""
import json
import numpy as np
import pandas as pd
import pytest
from atp import bands as bd, engine as en
from atp.config import load_config

cfg = load_config(); O = cfg["paths"]["processed"] / "phase3"
if not (O / "step35_bands.json").exists():
    pytest.skip("run `make phase3` first", allow_module_level=True)
J = json.load(open(O / "step35_bands.json"))
E = en.Engine(cfg); B = bd.Bands(cfg)


def test_effect_band_comes_from_the_test_window():
    e = J["effect_band"]
    s = e["source"]["slope_se_cluster"]
    assert np.isclose(e["eps_lo"], e["eps"] * (1 - bd.Z80 * s)) and np.isclose(e["eps_hi"], e["eps"] * (1 + bd.Z80 * s))
    assert e["per_seat_lo"] < e["per_seat"] < e["per_seat_hi"]


def test_level_band_passes_cross_fold_coverage():
    for kind, d in J["level_diagnostics"].items():
        assert min(d["coverage_A_to_B"], d["coverage_B_to_A"]) >= 0.80
        assert max(d["own_market_bands_coverage_A_to_B"], d["own_market_bands_coverage_B_to_A"]) < 0.80   # why tiers
    lv = pd.read_csv(O / "level_bands.csv")
    assert (lv.ratio_lo < 1).all() and (lv.ratio_hi > 1).all()
    assert J["tier_counts"] == {"26–46": 21, "11–25": 15, "top 10": 10}


def test_bands_on_a_scenario():
    s = {"changes": [{"type": "route", "market": "INDIA", "start": "2026-10-01", "end": "2026-12-31", "seats_pct": 20},
                     {"type": "demand", "markets": "DOMESTIC", "start": "2026-10-01", "end": "2026-12-31", "pct": 5}]}
    r = B.run(E, s)
    for k in ("d_checkins", "d_guest_nights"):
        lo, hi = r.effect_band["summary"][k]
        assert lo < r.summary[k] < hi
    m = r.effect_band["monthly"]
    assert (m.d_checkins_lo <= m.d_checkins + 1e-9).all() and (m.d_checkins <= m.d_checkins_hi + 1e-9).all()
    dom = m.xs("DOMESTIC", level="market")                     # the assumed demand change: baseline-level part only
    lo_r, hi_r = B._level_ratios("checkins", ["DOMESTIC"])
    nz = dom.d_checkins != 0
    assert np.allclose(dom.d_checkins_lo[nz], dom.d_checkins[nz] * lo_r[0]) and np.allclose(dom.d_checkins_hi[nz], dom.d_checkins[nz] * hi_r[0])
    p = r.effect_band["parts"]["d_checkins"]                                  # both parts present for a flight change
    assert p["lever"][0] < 0 < p["lever"][1] and p["baseline_level"][0] < 0 < p["baseline_level"][1]


def test_assumption_only_scenarios_have_no_lever_part():
    r = B.run(E, {"changes": [{"type": "stay_length", "markets": "DOMESTIC", "start": "2026-07-01", "end": "2026-08-31", "pct": 10}]})
    p = r.effect_band["parts"]["d_guest_nights"]
    assert p["lever"] == [0, 0] or np.allclose(p["lever"], 0)
    lo, hi = r.effect_band["summary"]["d_guest_nights"]
    lo_r, hi_r = B._level_ratios("guests", ["DOMESTIC"])
    v = r.summary["d_guest_nights"]
    assert np.isclose(lo, v * lo_r[0]) and np.isclose(hi, v * hi_r[0]) and "baseline-level part only" in r.effect_band["note"]


def test_level_errors_are_close_to_independent_across_markets():
    for kind in ("checkins", "guests"):
        out = []
        for f in "AB":
            d = bd._fold_monthly(cfg, f, kind); d["e"] = d.y / d.pred - 1
            for _, g in d.groupby("month"):
                w = g.pred / g.pred.sum()
                out.append((g.y.sum() / g.pred.sum() - 1, np.sqrt((w ** 2 * g.e ** 2).sum()), (w * g.e.abs()).sum()))
        o = np.array(out); rms = np.sqrt((o ** 2).mean(axis=0))
        assert rms[0] < 0.5 * rms[2] and abs(rms[0] - rms[1]) < 0.02          # total error ≈ independent, far below full correlation


def test_level_band_flags_months_beyond_seven():
    t = pd.read_csv(O / "baseline_level_total.csv", index_col=0)
    assert (t.guests_base_lo < t.guests_base).all() and (t.guests_base < t.guests_base_hi).all()
    assert t.level_band_calibrated.loc["2026-09"] and not t.level_band_calibrated.loc["2026-10"]
    mm = pd.read_csv(O / "baseline_level_monthly.csv", index_col=[0, 1])
    assert (mm.checkins_base_lo < mm.checkins_base).all() and (mm.checkins_base < mm.checkins_base_hi).all()


def test_effect_band_flags_months_beyond_seven():
    early = B.run(E, {"changes": [{"type": "route", "market": "CHINA", "start": "2026-05-01", "end": "2026-05-31", "seats_pct": 20}]})
    late = B.run(E, {"changes": [{"type": "route", "market": "CHINA", "start": "2026-11-01", "end": "2026-11-30", "seats_pct": 20}]})
    assert early.effect_band["months_beyond_7"] == [] and not any("more than 7 months" in f for f in early.flags)
    assert late.effect_band["months_beyond_7"][0] == "2026-11" and late.effect_band["share_beyond_7_months"] > 0.99
    assert any("more than 7 months" in f for f in late.flags)
    m = late.effect_band["monthly"]
    assert not m.level_band_calibrated.xs("2026-11", level="month").any() and m.level_band_calibrated.xs("2026-05", level="month").all()
    for d in J["level_diagnostics"].values():                    # evidence behind the flag text: no growth with lead to month 7
        for f in "AB":
            e = d["median_abs_log_error_by_months_ahead"][f]
            assert e["7"] <= max(e.values()) and e["7"] < e["3"]
