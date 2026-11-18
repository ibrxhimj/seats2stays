"""Step 3.6: sensitivity results are consistent with the engine's rules and reproducible."""
import json
import numpy as np
import pandas as pd
import pytest
from atp import bands as bd, engine as en
from atp.config import load_config

cfg = load_config(); O = cfg["paths"]["processed"] / "phase3"
if not (O / "step36_sensitivity.json").exists():
    pytest.skip("run `make phase3` first", allow_module_level=True)
J = json.load(open(O / "step36_sensitivity.json"))
L = pd.DataFrame(J["levers"]).set_index(["lever", "change_pct"])


def test_lever_elasticities():
    eps = json.load(open(O / "effect_band.json"))["eps"]
    flight = J["guest_night_shares"]["flight markets"]
    up, dn = L.loc[("seats", 10), "elasticity"], L.loc[("seats", -10), "elasticity"]
    assert abs(up - eps * flight) < 0.05 and dn > up                     # ε × flight-market share; concave lever
    for lever in ("stay length", "non-flight demand"):                    # scale every guest night
        assert abs(L.loc[(lever, 10), "elasticity"] - 1) < 0.01
    assert L.loc[("load factor", 10), "local_passengers_pct"] < 10        # planes are nearly full: capped
    assert abs(L.loc[("transfer share", 10), "elasticity"]) > up          # strongest flight-side lever
    # ranking number = elasticity at ±1%: within the trusted range for every lever, nearly symmetric
    sm = L.elasticity_small_change
    assert abs(sm.loc[("seats", 10)] - eps * flight) < 0.02 and abs(sm.loc[("load factor", 10)] - sm.loc[("seats", 10)]) < 0.01
    assert abs(sm.loc[("transfer share", 10)] - sm.loc[("transfer share", -10)]) < 0.03
    assert L.loc[("transfer share", 10), "pct_of_checkin_change_beyond_trusted_range"] > 50    # why ±10% is not the ranking number
    assert L.loc[("seats", 10), "pct_of_checkin_change_beyond_trusted_range"] == 0
    assert list(J["lever_ranking_by_small_change_elasticity"])[2] == "transfer share"
    assert (L.lo <= L.d_guest_nights).all() and (L.d_guest_nights <= L.hi).all()


def test_market_season_table():
    ms = pd.read_csv(O / "sens_market_month.csv")
    assert len(ms) == 45 * 12 and (ms.d_guest_nights > 0).all() and (ms.lo < ms.d_guest_nights).all()
    nf = ms[ms.rule == "new-route lever (no flight data)"]
    assert nf.market.nunique() == 12
    r = pd.read_csv(O / "sens_market_ranking.csv", index_col=0)
    assert r.guest_nights_per_1000_seats.is_monotonic_decreasing


def test_seats_needed_hits_the_target():
    E = en.Engine(cfg); B = bd.Bands(cfg)
    for q in J["seats_needed"]:
        s = {"changes": [{"type": "new_route", "market": q["market"], "city": "added daily flight", "airline": "generic",
                          "start": q["window"][0], "end": q["window"][1], "flights_per_week": q["flights_per_week"],
                          "seats_per_flight": q["seats_per_flight"]}]}
        assert abs(E.run(s).summary["d_guest_nights"] - q["target_guest_nights"]) < 0.01 * q["target_guest_nights"]
        lo, hi = q["flights_per_week_range"]; llo, lhi = q["flights_per_week_range_lever_only"]
        assert lo <= llo < q["flights_per_week"] < lhi <= hi                # full band (lever + level) ⊇ lever only
        for fpw, i in ((lo, 1), (hi, 0)):                                     # at each end the band's end meets the target
            s["changes"][0]["flights_per_week"] = fpw
            assert abs(B.run(E, s).effect_band["summary"]["d_guest_nights"][i] - q["target_guest_nights"]) < 0.01 * q["target_guest_nights"]


def test_uncertainty_tornado_and_evidence():
    t = J["uncertainty_tornado"]
    lev = np.ptp(t["lever_test_window_eps"]); base = np.ptp(t["baseline_level"])
    assert lev > base and t["combined"][0] <= t["lever_test_window_eps"][0] and t["combined"][1] >= t["lever_test_window_eps"][1]
    ev = J["local_passenger_evidence"]
    assert ev["local_per_seat"][1][0] > 0                                  # load factor × local share moves check-ins
