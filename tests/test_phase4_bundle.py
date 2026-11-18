"""Step 4.1: the simulator's data — saved peak, app-ready tables, facts, and the bundle (only what the simulator needs,
no raw competition data, identical to its sources, and the engine on the bundle reproduces the engine on the project)."""
import json
import os
import sys
import numpy as np
import pandas as pd
import pytest
from atp import engine as en, bands as bd, scenario as sc, phase3 as p3, phase4 as p4
from atp.config import load_config

cfg = load_config(); P = cfg["paths"]["processed"]; ROOT = cfg["root"]
if not (P / "phase4" / "facts.json").exists():
    pytest.skip("run `python -m atp.phase4 data` first", allow_module_level=True)

PRESETS = ["data/reference/scenarios/etihad_2026_launches.json", "data/reference/scenarios/european_carriers_return.json",
           "data/processed/phase3/presets/march_2026_replay_low.json"]
ALL_KINDS = {"name": "every change type", "changes": [
    {"type": "route", "markets": "UNITED KINGDOM", "start": "2026-11-01", "end": "2026-12-31", "seats_pct": -20},
    {"type": "new_route", "market": "DENMARK", "city": "Copenhagen", "airline": "Etihad Airways", "start": "2026-12-01",
     "end": "2027-02-28", "flights_per_week": 2, "aircraft_class": "narrowbody"},
    {"type": "load_factor", "markets": "INDIA", "start": "2026-06-01", "end": "2026-08-31", "pct": 5},
    {"type": "transfer_share", "markets": "INDIA", "start": "2026-10-01", "end": "2026-12-31", "points": -10},
    {"type": "demand", "markets": "Europe", "start": "2026-11-01", "end": "2027-01-31", "pct": -10},
    {"type": "stay_length", "markets": "CHINA", "start": "2026-07-01", "end": "2026-08-31", "pct": 10}]}


class Opened:
    """Records every file opened under a folder (Python audit hook; hooks cannot be removed, so it is switchable)."""
    on = False; files = set()

    @classmethod
    def start(cls):
        cls.files = set(); cls.on = True

    @classmethod
    def stop(cls):
        cls.on = False; return set(cls.files)


def _hook(ev, args):
    if Opened.on and ev == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        Opened.files.add(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_hook)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("b") / "bundle"
    p4.build_bundle(out, cfg)
    return out


# ----------------------------------------------------------------------------- peak
def test_peak_file_equals_recomputation():
    j = json.load(open(P / "phase3" / "peak_guests.json"))
    v, d = p3.observed_peak(cfg)
    assert j["peak_guests"] == v and j["date"] == str(d.date())


def test_engine_no_longer_reads_hotel_tables():
    Opened.start()
    E = en.Engine(cfg); E.run(ALL_KINDS)
    seen = {os.path.basename(f) for f in Opened.stop()}
    assert not seen & p4.FORBIDDEN_NAMES, seen & p4.FORBIDDEN_NAMES
    assert E.params.peak_guests > 0


# ----------------------------------------------------------------------------- app-ready tables and facts
def test_occupancy_basis():
    ob = pd.read_csv(P / "phase4" / "occupancy_basis.csv")
    assert list(ob.month) == [str(m) for m in pd.period_range("2026-03", "2027-02", freq="M")]
    assert (ob.rooms == 34526).all()
    s = pd.read_csv(P / "phase4" / "scad_monthly.csv").set_index("month")
    for r in ob.itertuples():                                              # same calendar month, a year before the data end
        src = s.loc[r.ratio_source_month]
        assert r.ratio_source_month[5:] == r.month[5:] and "2025-03" <= r.ratio_source_month <= "2026-02"
        assert np.isclose(r.guests_per_occupied_room, src.scad_guest_nights / (src.rooms * src.occupancy_pct / 100 * src.days))
        assert np.isclose(r.guest_nights_per_occupancy_point, r.guests_per_occupied_room * r.rooms * r.days / 100)
        assert np.isclose(r.dct_guest_nights_per_occupancy_point, r.guest_nights_per_occupancy_point * r.dct_over_scad)
    ref = s[s.in_reference]                                               # the latest 12 months with both sources
    assert len(ref) == 12 and ref.dct_over_scad.notna().all() and ref.index.max() == s.dct_over_scad.dropna().index.max()
    assert (s.dct_over_scad.dropna().index >= ref.index.min()).sum() == 12
    assert np.isclose(ob.dct_over_scad.iloc[0], ref.dct_over_scad.mean()) and ob.dct_over_scad.nunique() == 1
    assert 1.4 < ob.guests_per_occupied_room.min() and ob.guests_per_occupied_room.max() < 1.8
    r = s.dct_over_scad.dropna()                                          # DCT's guest nights sit close to SCAD's
    assert len(r) >= 24 and 0.85 < r.min() and r.max() < 1.1


def test_backtest_monthly_reproduces_reported_wmape():
    bt = pd.read_csv(P / "phase4" / "backtest_monthly.csv")
    m = pd.read_csv(P / "phase2" / "endtoend_metrics.csv").set_index(["fold", "model"])
    w = lambda d: float((d.actual_guest_nights - d.rebuilt_guest_nights).abs().sum() / d.actual_guest_nights.sum() * 100)
    for f in ("A", "B"):
        d = bt[bt.fold == f]; i = d[d.market != "DOMESTIC"]
        tot = i.groupby("month")[["actual_guest_nights", "rebuilt_guest_nights"]].sum()
        e = m.loc[(f, "end_to_end")]
        assert np.isclose(w(tot), e.intl_monthly_total, atol=1e-9)
        assert np.isclose(w(i), e.intl_monthly_x_nationality, atol=1e-9)
        assert np.isclose(w(d[d.market == "DOMESTIC"]), e.domestic_monthly, atol=1e-9)


def test_facts_match_their_sources():
    fx = json.load(open(P / "phase4" / "facts.json"))
    assert set(fx) == set(p4.FACTS)
    for k, f in fx.items():
        assert f["source"] == p4.FACTS[k]
        v = p4.read_fact(P, f["source"])
        assert v == f["value"] or np.isclose(v, f["value"], rtol=0, atol=1e-12), k
    v = lambda k: fx[k]["value"]
    assert round(v("eps"), 2) == 0.31 and round(v("eps_lo"), 2) == 0.20 and round(v("eps_hi"), 2) == 0.43
    assert round(v("backtest_A_monthly_total"), 1) == 3.4 and round(v("backtest_B_monthly_total"), 1) == 6.8
    assert round(v("offset_10") * 100) == 40 and v("peak_guests") > 0


# ----------------------------------------------------------------------------- the bundle
def test_bundle_has_only_manifest_files(bundle):
    files = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
    assert files == set(p4.MANIFEST) | {"bundle_manifest.json"}


def test_bundle_has_no_raw_or_full_tables(bundle):
    files = [p for p in bundle.rglob("*") if p.is_file()]
    assert not any(p.suffix in (".xlsx", ".xls", ".xlsm") for p in files)
    assert not any(p.name in p4.FORBIDDEN_NAMES for p in files)
    assert not (bundle / "data" / "raw").exists() and not (bundle / "data" / "interim").exists()
    raw = cfg["paths"]["raw"]
    raw_names = set(cfg["raw_files"].values())
    assert not any(p.name in raw_names for p in files)
    if all((raw / n).exists() for n in raw_names):                       # also by content, when the raw files are here
        raw_hash = {p4.sha256(raw / n) for n in raw_names}
        assert not {p4.sha256(p) for p in files} & raw_hash


def test_bundle_files_identical_to_sources(bundle):
    m = json.load(open(bundle / "bundle_manifest.json"))["files"]
    for rel, v in m.items():
        assert v["sha256"] == p4.sha256(bundle / rel) == p4.sha256(ROOT / rel), rel


def test_bundle_refuses_unsafe_targets(tmp_path):
    with pytest.raises(ValueError):
        p4.build_bundle(ROOT / "data" / "processed" / "x", cfg)
    (tmp_path / "keep.txt").write_text("not a bundle")
    with pytest.raises(ValueError):
        p4.build_bundle(tmp_path, cfg)


def test_engine_on_bundle_reproduces_project(bundle):
    cb = load_config(root=bundle)
    assert cb["root"] == bundle.resolve() and not (bundle / "data" / "raw").exists()
    E0, B0 = en.Engine(cfg), bd.Bands(cfg)                                # the project, run first (not recorded)
    ref = [B0.run(E0, sc.load(ROOT / rel)) for rel in PRESETS] + [B0.run(E0, ALL_KINDS)]
    Opened.start()                                                         # the bundle, recorded
    E1, B1 = en.Engine(cb), bd.Bands(cb)
    got = [B1.run(E1, sc.load(bundle / rel)) for rel in PRESETS] + [B1.run(E1, ALL_KINDS)]
    opened = Opened.stop()
    runs = list(zip(got, ref))
    data_opened = {f for f in opened if "/data/" in f or f.endswith("config.yaml")}
    outside = {f for f in data_opened if not f.startswith(str(bundle.resolve()))}
    assert not outside, f"engine on the bundle opened files outside it: {sorted(outside)[:5]}"
    for r1, r0 in runs:
        pd.testing.assert_frame_equal(r1.daily, r0.daily)
        assert r1.flags == r0.flags
        assert json.dumps(r1.summary, sort_keys=True, default=float) == json.dumps(r0.summary, sort_keys=True, default=float)
        assert json.dumps(r1.effect_band["summary"], default=float) == json.dumps(r0.effect_band["summary"], default=float)
    etihad = runs[0][0].summary["d_guest_nights"]
    assert round(etihad, -2) == 37100                                       # Phase 3 report, showcase 2
