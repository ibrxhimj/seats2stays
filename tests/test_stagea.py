"""Stage A building blocks: year-on-year alignment, fixed-effect estimator, airline defaults identities."""
import json
import numpy as np
import pandas as pd
import pytest
from atp import stagea as sa
from atp.config import load_config


def _toy_panel():
    months = pd.date_range("2023-01-01", "2025-06-01", freq="MS")
    rows = [{"nationality_key": c, "month": m, "seats": 100 * (1 + i) * (1.1 ** k), "pax": 80 * (1 + i) * (1.1 ** k),
             "p2p": 40 * (1 + i) * (1.1 ** k), "checkins": 10.0 * (1 + i), "source": "daily"}
            for i, c in enumerate("AB") for k, m in enumerate(months)]
    return pd.DataFrame(rows).drop(index=5).reset_index(drop=True)          # a missing month for country A


def test_yoy_uses_calendar_months_not_positions():
    y = sa.yoy(_toy_panel())
    a = y[y.nationality_key == "A"].set_index("month")
    # growth is 10%/month ⇒ exactly 12·log(1.1) year-on-year wherever both months exist
    assert np.allclose(a.loc["2024-03-01":, "d_seats"].dropna(), 12 * np.log(1.1))
    assert np.isnan(a.loc["2024-06-01", "d_seats"])                           # base month 2023-06 is missing
    assert "2023-06-01" not in a.index.astype(str)                            # the missing row stays missing
    assert np.isnan(a.loc["2023-07-01", "d_seats"])                           # no 2022 in the panel
    # lead / lag = the same country's YoY change one calendar month later / earlier
    assert np.isnan(a.loc["2024-05-01", "d_seats_lead1"]) and np.isclose(a.loc["2024-04-01", "d_seats_lead1"], 12 * np.log(1.1))
    assert np.isnan(a.loc["2024-07-01", "d_seats_lag1"]) and np.isclose(a.loc["2024-08-01", "d_seats_lag1"], 12 * np.log(1.1))


def test_truncate_blocks_information_past_cutoff():
    y = sa.yoy(sa.truncate(_toy_panel(), pd.Timestamp("2024-12-31")))
    assert y.month.max() == pd.Timestamp("2024-12-01")
    assert y.loc[y.month == "2024-12-01", "d_seats_lead1"].isna().all()           # January 2025 is not visible
    assert y.loc[y.month == "2024-11-01", "d_seats_lead2"].isna().all()


def test_trusted_range_selects_on_seats_only():
    d = _sim(3)
    rng_ = sa.trusted_range(d, 0.05); kept = sa.trim(d, rng_)
    assert np.isclose(len(kept) / len(d), 0.90, atol=0.01) and kept.d_seats.between(*rng_).all()
    shuffled = d.assign(d_p2p=np.random.default_rng(0).permutation(d.d_p2p.to_numpy()),
                        d_checkins=np.random.default_rng(1).normal(size=len(d)))
    assert sa.trusted_range(shuffled, 0.05) == rng_ and sa.trim(shuffled, rng_).month.equals(kept.month)   # outcome plays no role


def test_paired_ratio_and_disjoint_difference():
    d = _sim(4)
    num = sa.fe_elasticity(d.assign(y=0.5 * d.d_p2p), "y", ["d_seats"], n_boot=200)
    den = sa.fe_elasticity(d, "d_p2p", ["d_seats"], n_boot=200)
    r = sa.ratio(num, den)
    assert np.isclose(r["estimate"] * den["estimate"], num["estimate"])
    assert np.allclose(r["boot_draws"], 0.5)                                       # same countries in every paired draw
    half = sa.fe_elasticity(d[d.nationality_key.isin([f"C{i}" for i in range(16)])], "d_p2p", ["d_seats"], n_boot=200)
    with pytest.raises(AssertionError):
        sa.ratio(num, half)                                                        # different country sets → refused
    other = sa.fe_elasticity(d[~d.nationality_key.isin(half["countries"])], "d_p2p", ["d_seats"], n_boot=200, seed=9)
    assert np.isclose(sa.difference(half, other)["estimate"], half["estimate"] - other["estimate"])
    with pytest.raises(AssertionError):
        sa.difference(half, den)                                                   # overlapping groups → refused


def _sim(seed, n_c=33, n_m=26, true=0.7):
    """Panel shaped like ours (33 countries × 26 months) where seats co-move with a common monthly shock
    and each country has its own persistent error (so clustering matters)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2024-01-01", periods=n_m, freq="MS"); cs = [f"C{i}" for i in range(n_c)]
    shock = pd.Series(rng.normal(0, .3, n_m), index=months); ce = pd.Series(rng.normal(0, .1, n_c), index=cs)
    d = pd.DataFrame([(c, m) for c in cs for m in months], columns=["nationality_key", "month"])
    s = shock.reindex(d.month).to_numpy()
    d["d_seats"] = rng.normal(0, .3, len(d)) + 0.8 * s
    d["d_p2p"] = true * d.d_seats + 1.5 * s + ce.reindex(d.nationality_key).to_numpy() * d.d_seats + rng.normal(0, .1, len(d))
    return d


def test_month_fixed_effects_remove_common_shocks():
    d = _sim(1)
    naive = np.polyfit(d.d_seats, d.d_p2p, 1)[0]
    r = sa.fe_elasticity(d, "d_p2p", ["d_seats"], n_boot=300)
    se = (r["ci_cluster"][1] - r["ci_cluster"][0]) / 3.92
    assert naive > 1.2                                                          # without month effects: badly biased
    assert abs(r["estimate"] - 0.7) < 3 * se                                    # with them: unbiased up to noise
    assert abs(np.diff(r["ci_boot"])[0] / np.diff(r["ci_cluster"])[0] - 1) < 0.35   # the two interval methods agree


def test_cluster_interval_coverage():
    """Over 100 simulated panels the 95% country-clustered interval should contain the truth ~95% of the time."""
    hits = [lo < 0.7 < hi for lo, hi in (sa.fe_elasticity(_sim(s), "d_p2p", ["d_seats"], n_boot=0)["ci_cluster"] for s in range(100))]
    assert np.mean(hits) >= 0.88


def test_elasticity_decomposes_exactly():
    """log p2p = log seats + log LF + log local share ⇒ ε₁ = 1 + β_LF + β_share on the same sample."""
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"nationality_key": np.repeat(list("ABCDEFGH"), 20),
                      "month": np.tile(pd.date_range("2024-01-01", periods=20, freq="MS"), 8)})
    d["d_seats"] = rng.normal(0, .3, len(d)); d["d_pax"] = d.d_seats + rng.normal(0, .1, len(d))
    d["d_p2p"] = d.d_pax + rng.normal(0, .1, len(d))
    e = sa.fe_elasticity(d, "d_p2p", ["d_seats"], n_boot=0)["estimate"]
    lf = sa.fe_elasticity(d.assign(y=d.d_pax - d.d_seats), "y", ["d_seats"], n_boot=0)["estimate"]
    sh = sa.fe_elasticity(d.assign(y=d.d_p2p - d.d_pax), "y", ["d_seats"], n_boot=0)["estimate"]
    assert np.isclose(e, 1 + lf + sh)


def test_airline_defaults_contract():
    cfg = load_config(); p = cfg["paths"]["processed"] / "phase2" / "airline_defaults.csv"
    if not p.exists():
        pytest.skip("run `make phase2` first")
    d = pd.read_csv(p)
    assert not d.duplicated(["nationality_key", "moy"]).any() and set(d.moy) == set(range(1, 13))
    assert d[["seats_per_flight", "load_factor", "transfer_share", "local_share"]].notna().all().all()
    assert np.allclose(d.local_share, 1 - d.transfer_share - d.transit_share)
    assert d.local_share.between(0, 1).all() and (d.seats_per_flight > 0).all()
    j = json.load(open(p.parent / "step21_airline.json"))
    assert abs(j["diagnostics"]["decomposition"]["identity_gap"]) < 1e-9


def test_step22_contract():
    cfg = load_config(); o = cfg["paths"]["processed"] / "phase2"
    if not (o / "step22_checkins.json").exists():
        pytest.skip("run `make phase2` first")
    j = json.load(open(o / "step22_checkins.json")); t = pd.read_csv(o / "elasticities.csv")
    assert set(j["folds"]) == {"A", "B", "final"} and len(t) == 15
    for f, v in j["folds"].items():
        plain, lead = v["eps_total_plain"]["estimate"], v["eps_total_lead"]["estimate"]
        assert np.isclose(v["eps_total_default"]["estimate"], min(plain, lead))                      # conservative rule
        assert np.isclose(v["eps2"]["estimate"] * v["eps1"]["estimate"], v["eps_total_default"]["estimate"])  # ε₂·ε₁ = ε_total
        for k in ("eps_total_plain", "eps_total_lead", "eps1"):              # each fold's own training window only,
            assert pd.Timestamp(v[k]["last_month"]) <= cfg["folds"][f]["train_end"]   # so never the official test window
        assert cfg["folds"][f]["train_end"] < cfg["folds"]["final"]["holdout_start"]
    assert j["final_range"][0] == j["folds"]["final"]["eps_total_default"]["estimate"]


# ----------------------------------------------------------------------------- Step 2.3: festivals and lever test
def _fest_inputs():
    cfg = load_config(); RD, P = cfg["paths"]["reference"], cfg["paths"]["processed"]
    if not (P / "calendar.parquet").exists():
        pytest.skip("run `make data` first")
    off = pd.read_csv(RD / "calendar_events_official.csv", parse_dates=["start_date", "end_date"])
    return sa.festival_dates(RD, off), pd.read_csv(RD / "festival_groups.csv"), pd.read_parquet(P / "calendar.parquet"), cfg


def test_festival_dates_match_their_sources():
    from dateutil.easter import easter
    fest, grp, cal, cfg = _fest_inputs()
    f = fest.set_index(["festival", "year"]).date
    assert all(f["easter", y] == pd.Timestamp(easter(y)) for y in range(2022, 2027))
    assert f["eid_al_fitr", 2025] == pd.Timestamp("2025-03-30") and f["eid_al_adha", 2024] == pd.Timestamp("2024-06-16")
    assert f["lunar_new_year", 2025] == pd.Timestamp("2025-01-29") and f["diwali", 2024] == pd.Timestamp("2024-10-31")
    assert f["orthodox_easter", 2024] == pd.Timestamp(easter(2024, 2)) == pd.Timestamp("2024-05-05")
    # every hotel nationality (45) has exactly one group; groups only name festivals that exist
    cm = pd.read_csv(cfg["paths"]["reference"] / "country_map.csv")
    assert set(grp.nationality) == set(cm.nationality) and grp.nationality.is_unique and len(grp) == 45
    named = set(";".join(grp.festivals.dropna()).split(";")) - {"ramadan"}
    assert named <= set(fest.festival)


def test_festival_panel_counts():
    fest, grp, cal, _ = _fest_inputs()
    fp = sa.festival_panel(grp, fest, cal).set_index(["nationality_key", "month"])
    # Lunar New Year 29 Jan 2025, window 27 Jan…5 Feb → 5 days in January, 5 in February; 10 Feb 2024 → all 10 in Feb
    assert fp.loc[("CHINA", "2025-01-01"), "festival_days"] == 5 and fp.loc[("CHINA", "2025-02-01"), "festival_days"] == 5
    assert fp.loc[("CHINA", "2024-02-01"), "festival_days"] == 10 and fp.loc[("JAPAN", "2025-01-01"), "festival_days"] == 0
    # Ramadan 1446 = 1…29 March 2025, only for Muslim-majority markets
    assert fp.loc[("SAUDI ARABIA", "2025-03-01"), "ramadan_days"] == 29 and fp.loc[("GERMANY", "2025-03-01"), "ramadan_days"] == 0
    # a country never counts more days than the month has, even where windows overlap
    assert (fp.festival_days <= 31).all()


def test_lever_skill_reference_points():
    d = _sim(5, true=0.7).assign(d_checkins=lambda x: x.d_p2p)
    r_noise_free = d.assign(d_checkins=0.7 * d.d_seats)
    s = sa.lever_skill(r_noise_free, {"true": 0.7, "zero": 0.0}, n_boot=0)["skill"]
    assert np.isclose(s["true"], 1.0) and np.isclose(s["zero"], 0.0)         # perfect lever explains everything
    s2 = sa.lever_skill(d, {"true": 0.7, "double": 1.4}, n_boot=50)
    assert s2["skill"]["true"] > s2["skill"]["double"]                     # over-stating the lever is penalised


def test_estimator_drops_empty_controls():
    d = _sim(6).assign(empty=0.0)
    r = sa.fe_elasticity(d, "d_p2p", ["d_seats", "empty"], n_boot=0)
    assert r["dropped_controls"] == ["empty"] and np.isclose(r["estimate"], sa.fe_elasticity(d, "d_p2p", ["d_seats"], n_boot=0)["estimate"])


def test_step23_contract():
    cfg = load_config(); o = cfg["paths"]["processed"] / "phase2"
    if not (o / "step23_lever_test.json").exists():
        pytest.skip("run `make phase2` first")
    j = json.load(open(o / "step23_lever_test.json")); rows = pd.read_csv(o / "lever_test_rows.csv", parse_dates=["month"])
    F = cfg["folds"]
    assert rows.month.min() > F["B"]["train_end"] and rows.month.max() < F["final"]["holdout_start"]   # held out, never test window
    for k in ("primary", "secondary_foldA", "sensitivity_all_rows"):
        assert pd.Timestamp(j[k]["test_months"][1]) < F["final"]["holdout_start"]
    lo, hi = j["decision"]["simulator_band"]
    assert lo <= j["decision"]["default_eps_total"] <= hi


# ----------------------------------------------------------------------------- Step 2.5: new-route method
def test_leave_one_out_borrowing():
    from atp.phase2 import _loo
    r = pd.Series({"A": 0.1, "B": 0.2, "C": 0.4, "D": 0.8})
    d = _loo(r).set_index("nationality")
    assert np.isclose(d.loc["A", "predicted"], 0.4) and np.isclose(d.loc["D", "predicted"], 0.2)   # median of the OTHERS
    assert (d.error_factor >= 1).all() and np.isclose(d.loc["A", "error_factor"], 4.0)
    g = pd.Series({"A": "x", "B": "x", "C": "y", "D": "y"})
    assert np.allclose(_loo(r, g).predicted, _loo(r).predicted)            # groups too small → falls back to overall median


def test_step25_contract():
    cfg = load_config(); o = cfg["paths"]["processed"] / "phase2"
    if not (o / "step25_new_route.json").exists():
        pytest.skip("run `make phase2` first")
    j = json.load(open(o / "step25_new_route.json"))
    resp = j["eps"] + (j["new_city_test"]["trusted range (primary)"]["other_coefs"]["g_new"] if j["new_routes_get_own_response"] else 0)
    assert np.isclose(j["checkins_per_added_seat"], resp * j["ratio_median"])
    assert j["band"][0] < j["checkins_per_added_seat"] < j["band"][1] <= j["cap_local_pax_per_seat"]
    assert j["new_city_exposure"]["max_seat_identity_gap"] < 1e-9                  # continuing + new + closed = total seats
    assert pd.Timestamp(j["window"][1]) <= cfg["folds"]["final"]["train_end"]      # training months only
    assert len(j["no_flight_markets_current_checkins_per_month"]) == 12


# ----------------------------------------------------------------------------- Step 2.7: scenario levers (Phase 3 inputs)
def test_lever_no_change_gives_no_change():
    dA, ok = sa.seat_lever(1200.0, 10_000.0, 0.0, 0.31, 0.34)
    assert dA == 0 and ok


def test_lever_is_monotone_in_seats():
    dS = np.linspace(-10_000, 30_000, 401)
    for A in (500.0, 1500.0, 18_000.0):                     # incl. a market where the cap binds (check-ins > seats)
        dA, _ = sa.seat_lever(A, 10_000.0, dS, 0.31, 0.34)
        assert np.all(np.diff(dA) >= -1e-12)


def test_lever_shape_cap_and_full_cancellation():
    A, S, eps, lo, hi = 1500.0, 10_000.0, 0.31, -0.29, 1.09
    r = np.array([-0.2, 0.0, 0.5, 1.0])                                         # inside the trusted range: the estimated log form
    dA, ok = sa.seat_lever(A, S, r * S, eps, 1.0)
    assert ok.all() and np.allclose(dA, A * ((1 + r) ** eps - 1))
    dA, ok = sa.seat_lever(A, S, -S, eps, 1.0)                                 # full cancellation keeps (1−ε)(1+lo)^ε·A
    assert not ok and np.isclose(A + dA, (1 - eps) * (1 + lo) ** eps * A) and A + dA > 0
    for edge in (lo, hi):                                                      # continuous, with the same slope, at both edges
        e = 1e-6
        f = lambda x: float(sa.seat_lever(A, S, x * S, eps, 1.0)[0])
        assert abs(f(edge + e) - f(edge - e)) < 1e-6 * A
        assert np.isclose((f(edge + 2 * e) - f(edge + e)) / e, (f(edge - e) - f(edge - 2 * e)) / e, rtol=1e-3)
    dA, _ = sa.seat_lever(18_000.0, S, 1000.0, eps, 0.34)                      # A·g = 540 > 1000·0.34 → capped at 340
    assert np.isclose(dA, 340.0)
    for dS in (-9000.0, -5000.0, 2000.0, 30_000.0):                            # never more than the local passengers carried
        dA, _ = sa.seat_lever(A, S, dS, eps, 0.34)
        assert abs(dA) <= abs(dS) * 0.34 + 1e-9


def test_lever_rejects_impossible_inputs_and_flags_range():
    with pytest.raises(ValueError):
        sa.seat_lever(1000.0, 10_000.0, -10_001.0, 0.31, 0.34)    # cutting more seats than exist
    with pytest.raises(ValueError):
        sa.seat_lever(1000.0, 0.0, 500.0, 0.31, 0.34)              # no seats → must use the new-route lever
    _, ok = sa.seat_lever(1000.0, 10_000.0, np.array([-2900.0, -3100.0, 10_900.0, 11_100.0]), 0.31, 0.34)
    assert ok.tolist() == [True, False, True, False]                # trusted range −29% … +109%


def test_new_route_lever():
    assert np.isclose(sa.new_route_lever(9130.0, 0.048, 0.337), 9130 * 0.048)
    assert np.isclose(sa.new_route_lever(1000.0, 0.5, 0.337), 337.0)          # capped by local passengers per seat
    assert sa.new_route_lever(0.0, 0.048, 0.337) == 0
    with pytest.raises(ValueError):
        sa.new_route_lever(-10.0, 0.048, 0.337)


def test_step27_contract():
    cfg = load_config(); o = cfg["paths"]["processed"] / "phase2"
    if not (o / "guests_per_seat.csv").exists():
        pytest.skip("run `make phase2` first")
    t = pd.read_csv(o / "guests_per_seat.csv"); j = json.load(open(o / "step27_guests_per_seat.json"))
    fl = t[t.has_flight_data]
    assert len(t) == 45 and len(fl) == 33
    exp = np.minimum(j["eps"] * fl.checkins_per_seat, fl.local_pax_per_seat)
    assert np.allclose(fl.marginal_guest_nights_per_1000_added_seats, 1000 * exp * fl.stay_nights)          # marginal = min(ε·A/S, cap)·stay
    assert (fl.cap_binds == (j["eps"] * fl.checkins_per_seat > fl.local_pax_per_seat)).all()
    assert (t.band_low <= t.marginal_guest_nights_per_1000_added_seats + 1e-9).all() and (t.marginal_guest_nights_per_1000_added_seats <= t.band_high + 1e-9).all()
    assert abs(j["stay_data_vs_model_median_ratio"] - 1) < 0.05                                             # Stage B stays agree with the data
    rep = (cfg["paths"]["reports"] / "phase2_report.md").read_text()
    s = rep.split("## Draft summary analysis (≤ 1,000 characters)")[1].strip().split("\n\n")[0]
    assert len(s) <= 1000


def test_weighted_fixed_effects_equal_wls_with_month_dummies():
    import statsmodels.api as sm_
    d = _sim(7).assign(w=lambda x: np.random.default_rng(3).uniform(0.2, 5, len(x)))
    X = pd.get_dummies(d.month.astype(str), drop_first=False).astype(float).assign(d_seats=d.d_seats)
    ref = sm_.WLS(d.d_p2p, X, weights=d.w).fit().params["d_seats"]
    assert np.isclose(sa.fe_elasticity(d, "d_p2p", ["d_seats"], w="w", n_boot=0)["estimate"], ref)


def test_no_leakage_from_after_the_cutoff():
    """Corrupt every check-in and seat after a fold's cut-off: the fold's ε, trusted range and forecasts must not move."""
    from atp.phase2 import _load, _primary, _series, LEADS
    from atp import forecaster as fc
    cfg = load_config()
    if not (cfg["paths"]["processed"] / "flights_country_daily.parquet").exists():
        pytest.skip("run `make data` first")
    fd, fm, hi = _load(cfg); end = cfg["folds"]["A"]["train_end"]
    fd2 = fd.assign(seats=np.where(fd.date > end, fd.seats * 2 + 10, fd.seats))
    hi2 = hi.assign(arrivals=np.where(hi.date > end, hi.arrivals * 3 + 50, hi.arrivals))
    for spec in (["d_seats"], LEADS):
        (p1, r1), (p2, r2) = (_primary(sa.monthly_panel(f, h), end) for f, h in ((fd, hi), (fd2, hi2)))
        assert r1 == r2
        assert np.isclose(sa.fe_elasticity(p1, "d_checkins", spec, n_boot=0)["estimate"],
                          sa.fe_elasticity(p2, "d_checkins", spec, n_boot=0)["estimate"])
    S = _series(cfg, fd, hi); S2 = S.assign(y=np.where(S.date > end, S.y * 5 + 100, S.y))
    for key in ("INDIA", "CHINA", "DOMESTIC"):
        a, b = (fc.fit_predict(x[x.key == key].assign(month=x.date.dt.month, dow=x.date.dt.dayofweek, is_ramadan=False,
                                                        main_break=0.0, all_break=0.0, festival=0.0, yearend=0.0, f1=0.0, adipec=0.0,
                                                        **{f"m{m}": (x.date.dt.month == m).astype(float) for m in range(2, 13)},
                                                        **{f"w{w}": (x.date.dt.dayofweek == w).astype(float) for w in range(1, 7)},
                                                        ramadan=0.0),
                                 end, pd.Timestamp("2023-01-07"), cfg["folds"]["A"]["holdout_end"], {"C4b"}, 0.3)[0] for x in (S, S2))
        assert np.allclose(a.pred, b.pred)


def test_closure_days_flag_interruptions_only():
    d = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    twice = np.where(d.dayofweek.isin([0, 4]), 300.0, 0.0)                        # flies Mon + Fri only
    daily = np.full(len(d), 500.0)
    daily[(d >= "2024-06-10") & (d <= "2024-06-16")] = 0.0                         # 7-day airspace closure
    daily[d == "2024-03-05"] = 0.0                                                 # one cancelled day
    seasonal = np.where((d.month >= 5) & (d.month <= 8), 0.0, 200.0)               # summer gap
    launch = np.where(d >= "2024-07-01", 150.0, 0.0)                               # new route
    fd = pd.concat([pd.DataFrame({"nationality_key": k, "date": d, "seats": v})
                    for k, v in (("TWICE", twice), ("DAILY", daily), ("SEASONAL", seasonal), ("LAUNCH", launch))])
    cl = sa.closure_days(fd, d.min(), d.max()).to_frame(index=False)
    assert set(cl.nationality_key) == {"DAILY"} and len(cl) == 7
    assert cl.date.min() == pd.Timestamp("2024-06-10") and cl.date.max() == pd.Timestamp("2024-06-16")
    # real data: the June 2025 Israel closure is excluded from the airline defaults, other countries keep every day
    cfg = load_config()
    if not (cfg["paths"]["processed"] / "phase2" / "airline_defaults.csv").exists():
        pytest.skip("run `make phase2` first")
    ad = pd.read_csv(cfg["paths"]["processed"] / "phase2" / "airline_defaults.csv").set_index(["nationality_key", "moy"])
    assert ad.loc[("ISRAEL", 6), "days"] == 17 and ad.loc[("JAPAN", 1), "days"] == 31
