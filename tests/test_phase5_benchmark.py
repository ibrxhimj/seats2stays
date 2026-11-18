"""Step 5.5: the LightGBM benchmark. It must see only what the forecaster sees, and the saved result must be the one the
pre-registration (as it stands) produces."""
import json
import numpy as np
import pandas as pd
import pytest
from atp import benchmark as bm, forecaster as fc, phase5 as p5
from atp.config import load_config

FAST = dict(bm.PRIMARY, n_estimators=25)


@pytest.fixture(scope="module")
def frame():
    cfg = load_config(); D, intl, flight = p5._forecaster_frame(cfg)
    return cfg, D, intl, flight


def test_inputs_are_calendar_and_seats_only():
    assert set(bm.CAL + bm.SEATS) == {"month", "dow", "ramadan", "main_break", "festival", "yearend", "f1", "adipec", "home_holiday",
                                      "school_break", "recent", "log_seats", "rel_seats"}
    assert not {"y", "arrivals", "checkins", "occupancy", "guests"} & set(bm.CAL + bm.SEATS)


def test_no_hotel_figure_from_the_horizon_reaches_the_prediction(frame):
    """Replace every check-in after the training end with zeros: the predictions must not move."""
    cfg, D, _, _ = frame; F = cfg["folds"]["A"]
    a = bm.fit_predict_lgbm(D, F["train_end"], F["holdout_end"], bm.SPEC, params=FAST, seeds=(1,))
    D2 = D.assign(y=np.where(D.date > F["train_end"], 0.0, D.y))
    b = bm.fit_predict_lgbm(D2, F["train_end"], F["holdout_end"], bm.SPEC, params=FAST, seeds=(1,))
    assert np.allclose(a.pred.to_numpy(), b.pred.to_numpy(), rtol=0, atol=0)


def test_seat_scenario_moves_only_series_with_flights(frame):
    cfg, D, _, flight = frame; F = cfg["folds"]["A"]
    a = bm.fit_predict_lgbm(D, F["train_end"], F["holdout_end"], bm.SPEC, params=FAST, seeds=(1,))
    same = bm.fit_predict_lgbm(D, F["train_end"], F["holdout_end"], bm.SPEC, params=FAST, seeds=(1,), seat_scale=1.0)
    up = bm.fit_predict_lgbm(D, F["train_end"], F["holdout_end"], bm.SPEC, params=FAST, seeds=(1,), seat_scale=1.5)
    assert np.allclose(a.pred, same.pred)
    ch = (up.pred - a.pred).abs().groupby(a.key).max()
    assert (ch[~ch.index.isin(flight)] == 0).all() and (ch[ch.index.isin(flight)] > 0).any()


def test_the_wrong_specification_is_refused(frame):
    cfg, D, _, _ = frame; F = cfg["folds"]["A"]
    with pytest.raises(AssertionError):
        bm.fit_predict_lgbm(D, F["train_end"], F["holdout_end"], bm.SPEC | {"C5"}, params=FAST, seeds=(1,))


def test_month_block_interval_is_zero_for_the_same_forecast():
    d = pd.date_range("2024-08-01", "2025-02-28")
    p = pd.DataFrame({"key": "X", "date": d, "y": 10.0, "pred": 8.0})
    assert bm.month_block_interval(p, p, {"X"}) == [0.0, 0.0]


def test_verdict_rule():
    mk = lambda a, b, ia, ib: {"A": {"ours": {"intl_daily_nat": 10}, "lightgbm": {"intl_daily_nat": 10 + a}, "diff_interval_95": ia},
                               "B": {"ours": {"intl_daily_nat": 10}, "lightgbm": {"intl_daily_nat": 10 + b}, "diff_interval_95": ib}}
    assert bm.verdict(mk(-1, -2, [-3, 1], [-4, 0]))["lightgbm_more_accurate"]
    assert not bm.verdict(mk(-1, 2, [-3, 1], [1, 4]))["lightgbm_more_accurate"]            # one fold each is not a win
    v = bm.verdict(mk(2, 1, [1, 3], [-1, 3]))
    assert v["ours_lower_on_both_folds"] and "clearly so on A" in v["label"]


def test_saved_result_matches_its_preregistration_and_our_saved_scores():
    cfg = load_config(); P = cfg["paths"]["processed"]
    r = json.load(open(P / "phase5" / "benchmark.json")); j24 = json.load(open(P / "phase2" / "step24_forecaster.json"))
    assert r["prereg_sha256"] == p5.sha256(cfg["root"] / bm.PREREG_BENCH)
    for f in "AB":
        for k, v in j24["chosen_metrics"][f].items():
            assert abs(r[f]["ours"][k] - v) < 1e-9
    assert r["verdict"] == bm.verdict(r)
    assert len(r["A"]["grid"]) + len(r["A"]["grid_extended"]) == 10
    for f in "AB":                                                      # the best of ten is the best of the ten
        allg = r[f]["grid"] + r[f]["grid_extended"]
        assert r[f]["best_of_ten"]["intl_daily_nat"] == min(g["intl_daily_nat"] for g in allg)
        lo, hi = r[f]["best_of_ten"]["diff_interval_95"]; assert lo <= r[f]["best_of_ten"]["intl_daily_nat"] - r[f]["ours"]["intl_daily_nat"] <= hi
