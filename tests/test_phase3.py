"""Phase 3, Step 3.1: forward calendar (verified dates, complete horizon) and the baseline flight schedule."""
import numpy as np
import pandas as pd
import pytest
from atp import phase3 as p3
from atp import stagea as sa
from atp.config import load_config

cfg = load_config(); P = cfg["paths"]["processed"]; O = P / "phase3"; RD = cfg["paths"]["reference"]


def _need(f):
    if not (O / f).exists():
        pytest.skip("run `make phase3` first")


def test_national_day_corrections_reach_the_calendar():
    c = pd.read_parquet(P / "calendar.parquet").set_index("date")
    # 2023: MoHRE 2–4 Dec (Sat–Mon) → a 3-day long break; 2025: MoHRE/FAHR Mon 1–Tue 2 Dec → 29 Nov–2 Dec long break
    assert c.loc["2023-12-04", "is_national_day"] and c.loc["2023-12-02":"2023-12-04", "is_long_break"].all()
    assert c.loc["2025-12-01", "is_national_day"] and not c.loc["2025-12-03", "is_public_holiday"]
    assert c.loc["2025-11-29":"2025-12-02", "is_long_break"].all() and c.loc["2025-11-29", "long_break_length"] == 4


def test_horizon_calendar_is_complete_and_sourced():
    _need("horizon_events.csv")
    t = pd.read_csv(O / "horizon_events.csv", parse_dates=["start", "end"])
    assert not p3.check_horizon(t, pd.read_csv(RD / "home_holidays.csv"))
    assert t.source.notna().all() and set(t.status) <= {"official", "scheduled", "estimated", "rule window", "fixed date"}
    est = t[t.status.isin(["estimated", "scheduled"])].set_index("name")
    assert set(est.index) == {"ramadan", "national_day"}                       # the only dates still to confirm
    assert est.loc["ramadan", "start"] == pd.Timestamp("2027-02-08")
    # spot checks against the announcements
    start = lambda k, n: t.loc[(t.kind == k) & (t.name == n), "start"].tolist()
    assert start("UAE public holiday", "eid_al_adha") == [pd.Timestamp("2026-05-27")]
    assert start("UAE public holiday", "mawlid") == [pd.Timestamp("2026-08-28")]
    assert start("main moving festival", "lunar_new_year") == [pd.Timestamp("2027-02-04")]   # 6 Feb − 2 days
    assert start("school break", "school_spring_break") == [pd.Timestamp("2026-03-09")]     # moved by the government on 4 Mar
    assert start("Ramadan", "ramadan")[0] == pd.Timestamp("2026-02-18")                     # overlaps the horizon start
    ko = t[(t.kind == "home holiday") & (t.applies_to == "SOUTH KOREA")].iloc[0]
    assert ko.start == pd.Timestamp("2026-09-23") and ko.end == pd.Timestamp("2026-09-27")     # day before … Sunday


def test_route_baseline_rules_on_known_cases():
    _need("route_baseline.csv")
    st = pd.read_csv(O / "route_status.csv").set_index(["nationality_key", "city", "airline"])
    assert (st.loc[st.index.get_level_values("airline") == "Wizz Air Abu Dhabi", "status"] == "ended (airline exited)").all()
    assert st.loc[("SPAIN", "Malaga", "Etihad Airways"), "status"] == "seasonal"           # summer route, back in season
    osa = st.loc[("JAPAN", "Osaka-Kansai", "Etihad Airways")]                              # published seasonal (Nov 2025)
    assert osa.status == "seasonal" and osa.status_from_data == "ended (route stopped)" and "AeroRoutes" in osa.override_source
    assert st.loc[("EGYPT", "Dabah", "Etihad Airways"), "status"] == "seasonal"
    assert st.loc[("CHINA", "Shanghai", "Etihad Airways"), "status"] == "ended (route stopped)"
    assert st.loc[("CHINA", "Haikou", "Hainan Airlines"), "status"] == "resumed"            # year-round before its pause
    assert st.loc[("TURKEY", "Izmir", "SunExpress"), "status"] == "established"            # winter route: gaps kept
    assert st.loc[("UNITED STATES OF AMERICA", "Atlanta", "Etihad Airways"), "status"] == "new"
    b = pd.read_csv(O / "route_baseline.csv")
    assert (b.loc[b.status.str.startswith("ended"), "seats"] == 0).all()
    atl = b[(b.city == "Atlanta") & (b.airline == "Etihad Airways")].set_index("moy")
    assert atl.loc[[3, 4, 5, 6], "filled"].all() and np.allclose(atl.loc[[3, 4, 5, 6], "seats"], atl.loc[3, "seats"])
    assert not atl.loc[[7, 12], "filled"].any()                                             # flown months keep their own value
    osb = b[(b.city == "Osaka-Kansai") & (b.airline == "Etihad Airways")].set_index("moy").seats
    assert (osb.loc[[12, 1, 2]] == 0).all() and (osb.loc[4:10] > 0).all()                  # off 20 Nov - 15 Mar
    assert np.isclose(osb.loc[3], osb.loc[4] * 16 / 31) and np.isclose(osb.loc[11], osb.loc[10] * 19 / 30)
    izm = b[(b.city == "Izmir") & (b.airline == "SunExpress")].set_index("moy")
    assert izm.loc[[6, 7, 8], "seats"].eq(0).all()                                          # no summer flights invented
    assert b.local_share.dropna().between(0, 1).all() and (b[["seats", "pax", "p2p"]] >= 0).all().all()


def test_route_sum_reproduces_airline_defaults():
    b, _, _ = p3.route_baseline(cfg)
    fd = pd.read_parquet(P / "flights_country_daily.parquet"); end = cfg["dates"]["test_end"]
    ad = sa.airline_defaults(fd, end - pd.DateOffset(years=1) + pd.Timedelta(days=1), end).set_index(["nationality_key", "moy"])
    naive = b.groupby(["nationality_key", "moy"]).naive.sum()
    assert np.allclose(naive.reindex(ad.index), ad.seats / ad.days, rtol=0, atol=1e-6)


def test_daily_baseline_grid():
    _need("baseline_seats_daily.parquet")
    d = pd.read_parquet(O / "baseline_seats_daily.parquet")
    h0, h1 = cfg["dates"]["horizon_start"], cfg["dates"]["horizon_end"]
    n = (h1 - h0).days + 1
    assert d.groupby("nationality_key").date.agg(lambda s: s.is_monotonic_increasing and len(s) == n and s.min() == h0).all()
    assert d.seats_7d.notna().all()                                     # lead-in days: no gap and no 28 Feb 2026 dip
    b = pd.read_csv(O / "route_baseline.csv").groupby(["nationality_key", "moy"]).seats.sum()
    mid = d[d.date.dt.day == 15].assign(moy=lambda x: x.date.dt.month).set_index(["nationality_key", "moy"])
    assert np.allclose(mid.seats_7d, b.reindex(mid.index))              # mid-month: the month's own per-day rate


def test_spring_breaks_reach_the_calendar():
    c = pd.read_parquet(P / "calendar.parquet").set_index("date")
    for a, z in [("2023-03-27", "2023-04-09"), ("2024-03-25", "2024-04-14"), ("2025-03-24", "2025-04-06"), ("2026-03-09", "2026-03-22")]:
        assert c.loc[a:z, "is_school_break"].all() and c.loc[a:z, "is_school_spring_break"].all()
        assert not c.loc[pd.Timestamp(z) + pd.Timedelta(days=1), "is_school_break"]           # restart Monday
    assert c.loc["2026-03-01":"2026-03-19", "is_ramadan"].all()                               # Ramadan 1447 inside the horizon


def test_zero_seat_days_stay_finite():
    from atp import forecaster as fc
    d = pd.DataFrame({"seats_7d": [0.0, 5.0, 100.0]})
    off = fc._offset(d, 0.31, 20.0)
    assert np.isfinite(off).all() and off[0] == off[1]                                       # floored, never log 0
    _need("step31_world.json")
    import json
    z = json.load(open(O / "step31_world.json"))["zero_seat_check"]
    assert set(z) == {"Uzbekistan Jul", "Uzbekistan Aug"}
    for v in z.values():                                                                     # floor ≈ the lever's cancellation rule
        assert abs(v["forecaster_keeps"] - v["lever_full_cancellation_keeps"]) < 0.05


def test_news_and_overrides_are_sourced():
    n = pd.read_csv(RD / "horizon_news.csv")
    assert n.source.notna().all() and n.status_2026_10_03.notna().all() and n.in_baseline.notna().all()
    ov = pd.read_csv(RD / "route_overrides.csv")
    assert ov.source.str.len().gt(40).all() and (pd.to_datetime(ov.published) <= cfg["dates"]["test_end"]).all()


def test_production_refit():
    _need("step32_refit.json")
    import json
    j = json.load(open(O / "step32_refit.json"))
    assert j["reproduces_phase2_final"]["max_rel_diff"] < 1e-9                  # same procedure as the Phase 2 final model
    assert j["reproduces_phase2_final"]["rows"] == j["reproduces_phase2_final"]["of"]
    assert j["stageb_reproduces_phase1_test"]["max_abs_diff_guests"] < 1e-6     # persisted Stage B = Phase 1 final fit
    assert not j["non_converged"] and j["train_end"] == "2026-02-28"
    assert j["war_day_checkins"]["normal_world"] < j["war_day_checkins"]["actual"]   # stranded passengers kept out of memory
    h0, h1 = cfg["dates"]["horizon_start"], cfg["dates"]["horizon_end"]
    b = pd.read_parquet(O / "baseline_checkins_daily.parquet")
    assert b.groupby("series").date.agg(lambda s: s.min() == h0 and s.max() == h1 and s.is_unique and len(s) == (h1 - h0).days + 1).all()
    assert b.series.nunique() == 46 and np.isfinite(b.checkins).all() and (b.checkins > 0).all()
    fl = b[b.seat_floor_active]
    assert set(fl.series) == {"UZBEKISTAN"} and fl.date.dt.month.isin([7, 8, 9]).all()  # only the zero-seat summer (+7-day tail)
    assert (b.beyond_7_months == (b.date >= pd.Timestamp(j["beyond_7_months_from"]))).all()
    g = pd.read_parquet(O / "baseline_guests_daily.parquet")
    assert g.market.nunique() == 46 and (g.guests >= 0).all() and len(g) == len(b)
    p = json.load(open(O / "stageb_params.json"))
    assert len(p["theta"]) == 46 and len(p["theta"]["DOMESTIC"]) == 12                # domestic keeps its (frozen) drift
