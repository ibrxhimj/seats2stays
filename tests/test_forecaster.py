"""Stage A daily forecaster (Step 2.4): features, estimator behaviour, frozen test forecast."""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from atp import forecaster as fc
from atp.config import load_config

cfg = load_config(); P = cfg["paths"]["processed"]; O = P / "phase2"


def _cal():
    if not (P / "calendar.parquet").exists():
        pytest.skip("run `make data` first")
    return pd.read_parquet(P / "calendar.parquet")


def test_main_breaks_keep_only_eid_and_national_day():
    b = fc.uae_breaks(_cal()).set_index("date")
    assert b.loc["2024-04-10", "main_break"] == 1 and b.loc["2024-06-16", "main_break"] == 1      # Eid al-Fitr, Eid al-Adha
    assert b.loc["2024-12-02", "main_break"] == 1                                                 # National Day
    assert b.loc["2024-07-07", "main_break"] == 0 and b.loc["2024-07-07", "all_break"] == 1      # Islamic New Year: minor
    assert b.loc["2024-09-15", "main_break"] == 0                                                 # Mawlid: minor
    assert (b.main_break <= b.all_break).all()
    # split: Eid (Gulf-wide) vs National Day (UAE residents only); together they are exactly the main breaks
    assert b.loc["2024-04-10", "eid_break"] == 1 and b.loc["2024-04-10", "nd_break"] == 0
    assert b.loc["2024-12-02", "nd_break"] == 1 and b.loc["2024-12-02", "eid_break"] == 0
    assert (np.maximum(b.eid_break, b.nd_break) == b.main_break).all()


def test_home_holiday_window_and_frame():
    hh = pd.read_csv(cfg["paths"]["reference"] / "home_holidays.csv", parse_dates=["start", "end"])
    assert hh.end.ge(hh.start).all() and hh.source.notna().all() and not hh.duplicated(["nationality", "start"]).any()
    names = set(pd.read_csv(cfg["paths"]["reference"] / "festival_groups.csv").nationality)
    assert set(hh.nationality) <= names                                                             # keys match the series
    h = fc.home_holiday_daily(hh)
    cn = set(h[h.nationality == "CHINA"].date)
    assert pd.Timestamp("2025-09-30") in cn and pd.Timestamp("2025-10-08") in cn                 # day before … last day
    assert pd.Timestamp("2025-09-29") not in cn and pd.Timestamp("2025-10-09") not in cn
    cal = _cal()
    s = pd.DataFrame({"key": ["CHINA", "DOMESTIC"] * 2, "date": pd.to_datetime(["2025-10-03"] * 2 + ["2025-08-15"] * 2),
                      "y": 0.0, "seats_7d": 1.0, "exclude": False})
    D = fc.build_frame(s, cal, fc.uae_breaks(cal), pd.DataFrame(columns=["nationality", "date", "festival"]),
                       pd.DataFrame({"date": cal.date[:1], "f1": [0.0], "adipec": [0.0]}), h).set_index(["key", "date"])
    assert len(D) == 4 and D.loc[("CHINA", "2025-10-03"), "home_holiday"] == 1                    # no row duplication
    assert D.loc[("DOMESTIC", "2025-10-03"), "home_holiday"] == 0                                  # only its own market
    assert D.loc[("DOMESTIC", "2025-08-15"), "school_break"] == 1                                  # UAE summer school break …
    assert D.loc[("CHINA", "2025-08-15"), "school_break"] == 0                                     # … moves residents only


def _toy(n_days=900, eps=0.3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-07", periods=n_days, freq="D")
    seats = 1000 * np.exp(0.4 * np.sin(np.arange(n_days) / 60))
    d = pd.DataFrame({"key": "X", "date": dates, "seats_7d": seats, "exclude": False})
    cal = pd.DataFrame({"date": dates, "month": dates.month, "dow": dates.dayofweek, "is_ramadan": False})
    br = pd.DataFrame({"date": dates, "all_break": 0.0, "main_break": 0.0})
    D = fc.build_frame(d.assign(y=0.0), cal, br, pd.DataFrame(columns=["nationality", "date", "festival"]),
                       pd.DataFrame({"date": dates[:1], "f1": [0.0], "adipec": [0.0]}))
    mu = np.exp(np.log(20) + 0.3 * (D.dow == 4) + eps * np.log(D.seats_7d))
    return D.assign(y=rng.poisson(mu).astype(float)), mu


def test_fit_recovers_calendar_and_honours_fixed_eps():
    D, mu = _toy()
    te = D.date.iloc[700]
    pr, beta, info = fc.fit_predict(D, te, D.date.min(), D.date.max(), set(), 0.3)
    assert info["converged"] and abs(beta["w4"] - 0.3) < 0.08                                  # Friday effect recovered
    assert abs(fc.wmape(mu[D.date > te], pr.pred) ) < 6                                          # forecasts the true mean
    # seat floor: a full closure cannot drive the forecast to zero
    Dz = D.copy(); Dz.loc[Dz.date > te, "seats_7d"] = 0.0
    prz, _, inf = fc.fit_predict(Dz, te, D.date.min(), D.date.max(), set(), 0.3)
    assert (prz.pred > 0).all() and np.isclose(prz.pred.min() / pr.pred.min(), (inf["seat_floor"] / D.seats_7d[D.date > te].min()) ** 0.3, rtol=0.2)


def test_recent_level_and_frozen_trend():
    D, mu = _toy(seed=1)
    te = D.date.iloc[700]
    D2 = D.copy(); D2.loc[(D2.date > te - pd.Timedelta(days=182)) & (D2.date <= te), "y"] *= 1.3   # level jump at the end
    pr, _, inf = fc.fit_predict(D2, te, D.date.min(), D.date.max(), {"C4b"}, 0.3)
    assert inf["level"] > 1.05 and abs(pr.pred.sum() / mu[D.date > te].sum() - 1.3) < 0.08       # forecast sits at the new level
    D3 = D.assign(y=(D.y * np.exp(0.5 * (D.date - D.date.min()).dt.days / 365.25)).round())          # strong growth
    pr, beta, _ = fc.fit_predict(D3, te, D.date.min(), D.date.max(), {"C4a"}, 0.0)
    assert beta["trend"] > 0.3                                                                       # trend is in the fit …
    fr = pr[(pr.date.dt.dayofweek == 4) & (pr.date.dt.to_period("M") == pr.date.iloc[40].to_period("M"))]
    assert len(fr) >= 4 and np.allclose(fr.pred, fr.pred.iloc[0])                                   # … but frozen in the forecast


def test_poisson_mad_matches_simulation():
    rng = np.random.default_rng(0)
    for mu in (0.7, 3.7, 50.0):
        assert np.isclose(fc.poisson_mad(np.array([mu]))[0], np.abs(rng.poisson(mu, 400_000) - mu).mean(), rtol=0.01)


def test_step24_contract():
    if not (O / "step24_forecaster.json").exists():
        pytest.skip("run `make phase2` first")
    j = json.load(open(O / "step24_forecaster.json"))
    f = O / "stagea_test_forecast.csv"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == j["final_model"]["sha256"]                 # frozen, unchanged
    t = pd.read_csv(f, parse_dates=["date"])
    assert len(t) == 46 * 212 and "y" not in t and t.checkins_forecast.gt(0).all()                 # never carries actuals
    assert t.date.min() == cfg["dates"]["test_start"] and t.date.max() == cfg["dates"]["test_end"]
    assert "intl_daily_nat" in j["chosen_metrics"]["A"] and j["metric_for_level_terms"] == "intl_monthly_nat"
    for f in ("A", "B"):
        p = pd.read_parquet(O / f"stagea_pred_{f}.parquet")
        assert p.date.min() == cfg["folds"][f]["holdout_start"] and p.date.max() == cfg["folds"][f]["holdout_end"]
    assert not j["non_converged_fits"]


def test_step26_contract():
    if not (O / "step26_endtoend.json").exists():
        pytest.skip("run `make phase2` first")
    j = json.load(open(O / "step26_endtoend.json"))
    # the rebuilt Stage B reproduces the Phase 1 numbers when fed the actual check-ins
    v = pd.read_csv(P / "phase1" / "validation_metrics.csv", index_col=[0, 1])
    sel = json.load(open(P / "phase1" / "final_summary.json"))["selection"]          # Phase 1's chosen models
    for f in ("A", "B"):
        assert abs(j["folds"][f]["phase1_actual_checkins"]["intl_daily_x_nationality"] - v.loc[(f, sel["intl_model"]), "intl_daily_x_nationality"]) < 0.01
        assert abs(j["folds"][f]["phase1_actual_checkins"]["domestic_daily"] - v.loc[(f, sel["dom_model"]), "domestic_daily"]) < 0.01
        p = pd.read_parquet(O / f"endtoend_pred_{f}.parquet")
        assert p.date.min() >= cfg["folds"][f]["holdout_start"] and p.date.max() <= cfg["folds"][f]["holdout_end"]
    # test window scored from the archived forecast that was scored once (v1), never a later or re-fitted one
    ref = json.load(open(cfg["paths"]["reference"] / "frozen_test_forecast.json"))
    arch = cfg["paths"]["reference"].parent.parent / ref["file"]
    assert hashlib.sha256(arch.read_bytes()).hexdigest() == ref["sha256"] == j["test_window"]["frozen_forecast_sha256_verified"]
    assert j["test_window"]["current_forecaster_sha256_unscored"] == json.load(open(O / "step24_forecaster.json"))["final_model"]["sha256"]
    assert set(j["criteria"]) == {"a_end_to_end_beats_same_day_last_year", "c_test_stagea_beats_benchmarks", "d_lever_check_passed"}


def test_forecaster_series_start_and_no_log_zero():
    from atp.phase2 import _series, _load
    fd, fm, hi = _load(cfg)
    s = _series(cfg, fd, hi)
    assert s.date.min() == pd.Timestamp("2023-01-07")                      # first full 7-day seat window, never earlier
    assert s.loc[s.seats_7d.notna(), "seats_7d"].ge(0).all()
    tr = s[(s.date <= cfg["folds"]["final"]["train_end"]) & s.seats_7d.notna()]
    floor = 0.05 * tr.groupby("key").seats_7d.transform("mean")
    assert np.isfinite(np.log(np.maximum(tr.seats_7d, floor))).all()      # the seat term is always finite
