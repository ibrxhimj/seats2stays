"""Phase 4 — interactive simulator: the data it runs on.

Step 4.1 (`data`): app-ready tables built from the frozen Phase 0–3 outputs (nothing is re-modelled), written to
data/processed/phase4/:
  occupancy_basis.csv  per horizon month: SCAD's hotel rooms and guests per occupied room, so a change in guest nights
                       can also be read as occupancy points (SCAD, Abu Dhabi hotel establishments, monthly releases).
                       Rooms = the latest published count; guests per occupied room = the same calendar month in the
                       12 months before the data end (Mar 2025 – Feb 2026), matching the "normal year" baseline.
                       DCT's guest nights (the competition data) run at 91–95% of SCAD's published totals in the latest
                       12 months with both (Aug 2024 – Jul 2025, mean 92.9%; `in_reference` in scad_monthly.csv — 2023
                       sits on a different basis, 97–103%), so a change is first put on SCAD's basis:
                       points = Δ DCT guest nights / DCT-over-SCAD / (guests per occupied room × rooms × days) × 100.
  scad_monthly.csv     every SCAD month: rooms, occupancy, guest nights, guests per occupied room, and DCT's guest
                       nights for the same month where DCT publishes guests (to Jul 2025), with the ratio.
  backtest_monthly.csv the Phase 2 end-to-end back-test (guest nights rebuilt from flights and the calendar alone, model
                       frozen before each held-out period) summed to market × month; it reproduces the reported WMAPE
                       exactly (tested).
  passport_vs_flight.csv  UAE residents and eight nationalities on National Day and Eid breaks (the country catch).
  facts.json           every headline number the simulator shows that is not computed live, each with the file and
                       key it comes from (tested against its source), so no number is typed by hand.

Step 4.1 (`bundle`): the app bundle — a copy of only the files the simulator needs, in the same folder structure, so the
same code runs on it unchanged (`load_config(root=bundle)` or ATP_ROOT). No raw competition files, no Excel files and
no full daily hotel or flight tables (tested). Each file is identical to its source (sha256 in bundle_manifest.json).
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_config

# ----------------------------------------------------------------------------- what the simulator needs
# Engine runtime: traced (every file opened by Engine, Bands, all presets, every change type and the reverse
# calculator), then the full hotel tables replaced by phase3/peak_guests.json.
ENGINE_FILES = {
    "config.yaml": "paths, dates, horizon",
    "data/processed/calendar.parquet": "Stage B calendar (checkout-day effects)",
    "data/processed/external_events.parquet": "Stage B Grand Prix days",
    "data/processed/phase2/step23_lever_test.json": "seat lever ε and its trusted range",
    "data/processed/phase2/step25_new_route.json": "new-route lever (nationalities without flight data)",
    "data/processed/phase2/step26_endtoend.json": "ε's out-of-sample interval",
    "data/processed/phase3/aircraft_seats.csv": "narrowbody / widebody seats per flight",
    "data/processed/phase3/baseline_checkins_daily.parquet": "normal-year check-ins by market and day",
    "data/processed/phase3/baseline_guests_daily.parquet": "normal-year guests in-house by market and day",
    "data/processed/phase3/effect_band.json": "effect range (80%)",
    "data/processed/phase3/level_bands.csv": "level range (80%) by market size",
    "data/processed/phase3/market_tiers.csv": "market size tiers",
    "data/processed/phase3/peak_guests.json": "highest guests in-house ever observed (capacity flag)",
    "data/processed/phase3/route_baseline.csv": "normal-year route schedule",
    "data/processed/phase3/stageb_memory.parquet": "Stage B 45-day memory before the horizon",
    "data/processed/phase3/stageb_params.json": "Stage B parameters",
    "data/processed/phase3/step32_refit.json": "months flagged as beyond validation",
    "data/reference/scad_region_map.csv": "nationality → SCAD world region",
    "data/reference/nationality_places.csv": "map: each nationality's largest city (where its marker sits)",
}
PRESET_FILES = {
    "data/reference/scenarios/etihad_2026_launches.json": "preset: Etihad's 2026 launches (sourced)",
    "data/reference/scenarios/european_carriers_return.json": "preset: fewer European flights (sourced)",
    "data/processed/phase3/presets/march_2026_replay_low.json": "preset: March 2026 replayed, lower seat assumption",
    "data/processed/phase3/presets/march_2026_replay_high.json": "preset: March 2026 replayed, higher seat assumption",
}
APP_FILES = {
    "data/processed/phase3/sens_levers.csv": "What moves guests: lever ranking (tornado)",
    "data/processed/phase3/sens_market_month.csv": "What moves guests: guest nights per 1,000 seats by market × month",
    "data/processed/phase3/sens_market_ranking.csv": "What moves guests: market ranking and rule flags",
    "data/processed/phase3/sens_season.csv": "What moves guests: guest nights per 1,000 seats by month",
    "data/processed/phase3/march_2026_replay_by_region.csv": "How we know: March 2026 replay vs SCAD",
    "data/processed/phase2/endtoend_metrics.csv": "How we know: back-test WMAPE at every level, with benchmark",
    "data/processed/phase2/endtoend_per_market.csv": "How we know: back-test error by market",
    "data/processed/phase4/backtest_monthly.csv": "How we know: actual vs rebuilt monthly guest nights",
    "data/processed/phase4/passport_vs_flight.csv": "How we know: National Day and Eid break effects, UAE residents and nationalities",
    "data/processed/phase4/occupancy_basis.csv": "occupancy points per horizon month",
    "data/processed/phase4/facts.json": "headline numbers with their sources",
    "data/processed/phase5/limitations.json": "How we know: the one limitations list (Step 5.9)",
    "data/processed/phase5/ahead.json": "How we know: accuracy by months ahead (Step 5.7)",
    "data/processed/phase5/sobol.json": "How we know: Sobol check of the lever ranking and of the range (Step 5.6)",
}
MANIFEST = {**ENGINE_FILES, **PRESET_FILES, **APP_FILES}
# never in a bundle: raw competition files, Excel, full daily hotel / flight tables, interim caches
FORBIDDEN_NAMES = {"hotel_intl_daily.parquet", "hotel_dom_daily.parquet", "hotel_monthly.parquet", "flights_rows.parquet",
                   "flights_country_daily.parquet", "flights_country_monthly.parquet"}

SCAD_XLSX = "scad/SCAD_Hotel_Master_Database_2022_2026.xlsx"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- occupancy basis (SCAD)
def scad_monthly(cfg) -> pd.DataFrame:
    t = pd.read_excel(cfg["paths"]["external"] / SCAD_XLSX, sheet_name="T1_Emirate_Key_Indicators")
    t["month"] = pd.to_datetime(t.date).dt.to_period("M")
    s = pd.DataFrame({"month": t.month, "rooms": t.num_rooms, "occupancy_pct": t.occupancy_rate_pct,
                      "scad_guest_nights": t.guest_nights_thousands * 1000, "days": t.month.dt.days_in_month})
    s["guests_per_occupied_room"] = s.scad_guest_nights / (s.rooms * s.occupancy_pct / 100 * s.days)
    P = cfg["paths"]["processed"]
    hi = pd.read_parquet(P / "hotel_intl_daily.parquet"); hd = pd.read_parquet(P / "hotel_dom_daily.parquet")
    end = cfg["dates"]["hotel_train_end"]
    gi = hi[hi.is_target & (hi.date <= end)].groupby(hi.date.dt.to_period("M")).guests.sum()
    gd = hd[hd.date <= end].groupby(hd.date.dt.to_period("M")).guests.sum()
    dct = (gi + gd).rename("dct_guest_nights")
    s = s.merge(dct, left_on="month", right_index=True, how="left")
    s["dct_over_scad"] = s.dct_guest_nights / s.scad_guest_nights
    s = s.sort_values("month").reset_index(drop=True)
    # the ratio used: the latest 12 months with both sources. The two series shift level at Jan 2024 (2023: 0.97–1.03,
    # DCT above SCAD in 4 months; from 2024: 0.90–0.95, and SCAD's guests per room steps up too), so an average over
    # every month would mix two bases (independent review, Step 5.3)
    s["in_reference"] = s.index.isin(s.dropna(subset=["dct_over_scad"]).index[-12:])
    return s


def occupancy_basis(cfg, s: pd.DataFrame) -> pd.DataFrame:
    """Per horizon month: rooms (latest count), guests per occupied room (same calendar month, 12 months before the
    data end — the normal-year reference) and DCT's guest nights as a share of SCAD's (average over the latest 12
    months with both, `in_reference`), so a change in DCT guest nights is put on SCAD's basis before it becomes
    occupancy points."""
    end = cfg["dates"]["test_end"].to_period("M")
    ref = s[(s.month > end - 12) & (s.month <= end)]
    ref = ref.set_index(ref.month.dt.month)
    assert len(ref) == 12, f"SCAD reference year incomplete: {len(ref)} months"
    latest = s.dropna(subset=["rooms"]).iloc[-1]
    months = pd.period_range(cfg["dates"]["horizon_start"], cfg["dates"]["horizon_end"], freq="M")
    out = pd.DataFrame({"month": months.astype(str), "days": months.days_in_month,
                        "rooms": int(latest.rooms), "rooms_source_month": str(latest.month),
                        "guests_per_occupied_room": [ref.loc[m.month, "guests_per_occupied_room"] for m in months],
                        "ratio_source_month": [str(ref.loc[m.month, "month"]) for m in months]})
    out["guest_nights_per_occupancy_point"] = out.guests_per_occupied_room * out.rooms * out.days / 100
    out["dct_over_scad"] = float(s.loc[s.in_reference, "dct_over_scad"].mean())
    out["dct_guest_nights_per_occupancy_point"] = out.guest_nights_per_occupancy_point * out.dct_over_scad
    return out


# ----------------------------------------------------------------------------- back-test, monthly
def backtest_monthly(cfg) -> pd.DataFrame:
    P = cfg["paths"]["processed"] / "phase2"
    d = pd.concat([pd.read_parquet(P / f"endtoend_pred_{f}.parquet") for f in ("A", "B")], ignore_index=True)
    d["month"] = d.date.dt.to_period("M").astype(str)
    m = d.groupby(["fold", "market", "month"], as_index=False)[["y", "p"]].sum()
    return m.rename(columns={"y": "actual_guest_nights", "p": "rebuilt_guest_nights"})


# ----------------------------------------------------------------------------- passport vs departure country
def passport_vs_flight(cfg) -> pd.DataFrame:
    """Check-ins on break days ÷ the same weekdays either side, for UAE residents and eight nationalities: National Day is a
    holiday for residents only, Eid across the Gulf (Phase 2 external-factor audit). Shows that flights count departure
    countries and hotels count passports, and that the two are not the same people."""
    R = json.load(open(cfg["paths"]["processed"] / "phase2" / "step27_limitations.json", encoding="utf-8"))["resident_evidence"]
    nd, ed = R["nd_break"]["ratio"], R["eid_break"]["ratio"]
    return pd.DataFrame({"market": list(nd), "national_day": [nd[k] for k in nd], "eid": [ed[k] for k in nd]})


# ----------------------------------------------------------------------------- facts (headline numbers with sources)
def _j(file, *path, label):
    return {"kind": "json", "file": file, "path": list(path), "label": label}


def _c(file, where, column, label):
    return {"kind": "csv", "file": file, "where": where, "column": column, "label": label}


def _a(file, column, agg, label, where=None):
    return {"kind": "agg", "file": file, "column": column, "agg": agg, "where": where or {}, "label": label}


E2E = "phase2/endtoend_metrics.csv"
LC = ("test_window", "lever_check")
FACTS = {
    "eps": _j("phase3/effect_band.json", "eps", label="seat lever ε: % change in check-ins per 1% change in seats (estimated on seats; applied to local passengers, ε₁ ≈ 1)"),
    "eps_lo": _j("phase3/effect_band.json", "eps_lo", label="seat lever, 80% range, low end (unseen test months)"),
    "eps_hi": _j("phase3/effect_band.json", "eps_hi", label="seat lever, 80% range, high end"),
    "countries_estimated": _c("phase2/elasticities.csv", {"fold": "final", "parameter": "eps_total_default"}, "n_countries",
                              label="countries the lever is estimated on"),
    "trusted_lo": _j("phase2/step23_lever_test.json", "trusted_range_final", 0, label="trusted range of year-on-year seat change, 5th percentile (fraction)"),
    "trusted_hi": _j("phase2/step23_lever_test.json", "trusted_range_final", 1, label="trusted range of year-on-year seat change, 95th percentile (fraction)"),
    "lever_check_slope": _j("phase2/step26_endtoend.json", *LC, "calibration_slope", "estimate",
                            label="lever check on unseen months: predicted vs actual slope (1 = right size)"),
    "lever_check_slope_lo": _j("phase2/step26_endtoend.json", *LC, "calibration_slope", "ci_boot", 0, label="lever check slope, 95% low (country-cluster bootstrap)"),
    "lever_check_slope_hi": _j("phase2/step26_endtoend.json", *LC, "calibration_slope", "ci_boot", 1, label="lever check slope, 95% high (country-cluster bootstrap)"),
    "lever_check_countries": _j("phase2/step26_endtoend.json", *LC, "n_countries", label="countries in the lever check"),
    "lever_check_skill": _j("phase2/step26_endtoend.json", *LC, "skill", "default 0.31", label="lever skill on unseen months"),
    "one_for_one_skill": _j("phase2/step26_endtoend.json", *LC, "skill", "naive one-for-one (1.0)",
                            label="skill of 'one passenger = one guest' on unseen months"),
    "new_route_per_seat": _j("phase2/step25_new_route.json", "checkins_per_added_seat", label="check-ins per added seat, markets without flight data"),
    "new_route_per_seat_lo": _j("phase2/step25_new_route.json", "band", 0, label="check-ins per added seat, 80% low"),
    "new_route_per_seat_hi": _j("phase2/step25_new_route.json", "band", 1, label="check-ins per added seat, 80% high"),
    "full_cancellation_keeps": _j("phase3/step34_engine.json", "checks", "full_cancellation_keeps", "rule",
                                  label="share of a market's check-ins kept if all its flights are cancelled"),
    "narrowbody_seats": _c("phase3/aircraft_seats.csv", {"aircraft_class": "narrowbody"}, "seats_per_flight", label="narrowbody seats per flight"),
    "widebody_seats": _c("phase3/aircraft_seats.csv", {"aircraft_class": "widebody"}, "seats_per_flight", label="widebody seats per flight"),
    "peak_guests": _j("phase3/peak_guests.json", "peak_guests", label="highest guests in-house ever observed"),
    "peak_date": _j("phase3/peak_guests.json", "date", label="date of the highest level"),
    "beyond_validation_from": _j("phase3/step32_refit.json", "beyond_7_months_from", label="first month more than 7 months after the data"),
    "share_flight_markets": _j("phase3/step36_sensitivity.json", "guest_night_shares", "flight markets",
                               label="share of guest nights from nationalities with flight data"),
    "share_domestic": _j("phase3/step36_sensitivity.json", "guest_night_shares", "domestic", label="share of guest nights from UAE residents"),
    **{f"offset_{s}": _j("phase3/step38_showcase.json", "recovery_planner", "offset_rule", f"{s}%", "central",
                         label=f"increase in local passengers needed to offset a {s}% demand shortfall (fraction: 0.18 = +18%)") for s in (5, 10, 20)},
    **{f"offset_{s}_lo": _j("phase3/step38_showcase.json", "recovery_planner", "offset_rule", f"{s}%", "eps_high",
                            label=f"increase needed to offset a {s}% shortfall, 80% low (high ε; fraction)") for s in (5, 10, 20)},
    **{f"offset_{s}_hi": _j("phase3/step38_showcase.json", "recovery_planner", "offset_rule", f"{s}%", "eps_low",
                            label=f"increase needed to offset a {s}% shortfall, 80% high (low ε; fraction; above +1.09 is beyond the trusted range)") for s in (5, 10, 20)},
    **{f"backtest_{f}_{lvl}": _c(E2E, {"fold": f, "model": "end_to_end"}, col, label=f"back-test WMAPE %, {lab}, period {f}")
       for f in ("A", "B") for lvl, col, lab in [("monthly_total", "intl_monthly_total", "monthly international total"),
                                                  ("monthly_market", "intl_monthly_x_nationality", "monthly, market by market"),
                                                  ("daily_market", "intl_daily_x_nationality", "daily, market by market"),
                                                  ("domestic_monthly", "domestic_monthly", "UAE residents, monthly")]},
    **{f"benchmark_{f}_{lvl}": _c(E2E, {"fold": f, "model": "same_day_last_year_guests"}, col,
                                  label=f"benchmark (same day last year) WMAPE %, {lab}, period {f}")
       for f in ("A", "B") for lvl, col, lab in [("monthly_total", "intl_monthly_total", "monthly international total"),
                                                  ("monthly_market", "intl_monthly_x_nationality", "monthly, market by market")]},
    "march_checkins_vs_normal": _j("phase3/step37_validation.json", "march_2026", "total", "observed_vs_normal",
                                   label="March 2026 hotel check-ins as a share of the normal year (SCAD)"),
    "march_flights_share_lo": _j("phase3/step37_validation.json", "march_2026", "total", "share_of_drop_explained_by_flights", 0,
                                 label="share of March 2026's fall in check-ins that flights explain, low (seat assumptions)"),
    "march_flights_share_hi": _j("phase3/step37_validation.json", "march_2026", "total", "share_of_drop_explained_by_flights", 1,
                                 label="share of March 2026's fall in check-ins that flights explain, high (seat assumptions)"),
    "coverage_A_to_B": _j("phase3/step35_bands.json", "level_diagnostics", "guests", "coverage_A_to_B",
                          label="share of market-months inside the 80% range, range taken from period A, tested on B"),
    "coverage_B_to_A": _j("phase3/step35_bands.json", "level_diagnostics", "guests", "coverage_B_to_A",
                          label="share of market-months inside the 80% range, range taken from period B, tested on A"),
    "noflight_share_pct": _j("phase2/step27_limitations.json", "noflight_share_pct", label="share of international check-ins from nationalities with no flight data, last 12 training months (%)"),
    "domestic_share_pct": _j("phase2/step27_limitations.json", "resident_evidence", "domestic_share_of_checkins_12m_pct",
                             label="UAE residents' share of check-ins, last 12 training months (%)"),
    "rooms": _a("phase4/occupancy_basis.csv", "rooms", "first", label="hotel rooms in Abu Dhabi (SCAD, latest month)"),
    "guests_per_room_lo": _a("phase4/occupancy_basis.csv", "guests_per_occupied_room", "min",
                             label="guests per occupied room used, lowest month (SCAD, Mar 2025 – Feb 2026)"),
    "guests_per_room_hi": _a("phase4/occupancy_basis.csv", "guests_per_occupied_room", "max",
                             label="guests per occupied room used, highest month (SCAD, Mar 2025 – Feb 2026)"),
    **{f"dct_over_scad_{k}": _a("phase4/scad_monthly.csv", "dct_over_scad", agg, where={"in_reference": True},
                                label=f"DCT guest nights ÷ SCAD's, {what} (the 12 months used, Aug 2024 – Jul 2025)")
       for k, agg, what in (("lo", "min", "lowest month"), ("hi", "max", "highest month"), ("mean", "mean", "average"))},
}


def read_fact(processed, spec: dict):
    """Evaluate one fact from its source (used to build facts.json and, independently, by the tests)."""
    f = processed / spec["file"]
    if spec["kind"] == "json":
        v = json.load(open(f, encoding="utf-8"))
        for k in spec["path"]:
            v = v[k]
        return v
    d = pd.read_csv(f)
    for k, x in spec["where"].items():
        d = d[d[k] == x]
    if spec["kind"] == "csv":
        assert len(d) == 1, f"{spec['file']} {spec['where']}: {len(d)} rows"
        v = d[spec["column"]].iloc[0]
    else:
        col = d[spec["column"]].dropna()
        v = col.iloc[0] if spec["agg"] == "first" else getattr(col, spec["agg"])()
    return v.item() if isinstance(v, np.generic) else v


def facts(cfg) -> dict:
    P = cfg["paths"]["processed"]
    return {k: {"value": read_fact(P, s), "label": s["label"], "source": s} for k, s in FACTS.items()}


# ----------------------------------------------------------------------------- steps
def step_data():
    cfg = load_config(); out = cfg["paths"]["processed"] / "phase4"; out.mkdir(exist_ok=True)
    s = scad_monthly(cfg)
    s.assign(month=s.month.astype(str)).to_csv(out / "scad_monthly.csv", index=False)
    ob = occupancy_basis(cfg, s); ob.to_csv(out / "occupancy_basis.csv", index=False)
    bt = backtest_monthly(cfg); bt.to_csv(out / "backtest_monthly.csv", index=False)
    passport_vs_flight(cfg).to_csv(out / "passport_vs_flight.csv", index=False)
    fx = facts(cfg); json.dump(fx, open(out / "facts.json", "w", encoding="utf-8"), indent=1)
    r = s.loc[s.in_reference, "dct_over_scad"]
    print(f"OCCUPANCY: {ob.rooms.iloc[0]:,} rooms (SCAD {ob.rooms_source_month.iloc[0]}); guests per occupied room "
          f"{ob.guests_per_occupied_room.min():.2f}–{ob.guests_per_occupied_room.max():.2f} over the horizon "
          f"(from {ob.ratio_source_month.iloc[0]} … {ob.ratio_source_month.iloc[-1]}); one occupancy point ≈ "
          f"{ob.guest_nights_per_occupancy_point.min():,.0f}–{ob.guest_nights_per_occupancy_point.max():,.0f} guest nights a month")
    print(f"  DCT guest nights ÷ SCAD's, months used: {r.min():.3f}–{r.max():.3f} (mean {r.mean():.3f}, {len(r)} months, "
          f"{s.month[s.in_reference].min()}…{s.month[s.in_reference].max()}; all {s.dct_over_scad.notna().sum()} months "
          f"{s.dct_over_scad.min():.3f}–{s.dct_over_scad.max():.3f})")
    print(f"BACK-TEST: {len(bt):,} market-months over folds {', '.join(sorted(bt.fold.unique()))}")
    print(f"FACTS: {len(fx)} headline numbers with sources → phase4/facts.json")


def build_bundle(out, cfg=None) -> dict:
    """Copy the MANIFEST files into `out` (same structure) and write bundle_manifest.json. Refuses to write inside the
    project's data folder or over a folder that is not an earlier bundle."""
    cfg = cfg or load_config(); root = cfg["root"]
    out = Path(out); out = (out if out.is_absolute() else root / out).resolve()
    if out == root or root / "data" in [out, *out.parents]:
        raise ValueError(f"refusing to build a bundle at {out}")
    if out.exists():
        if any(out.iterdir()) and not (out / "bundle_manifest.json").exists():
            raise ValueError(f"{out} exists and is not a bundle: not overwriting")
        shutil.rmtree(out)
    files = {}
    for rel, why in MANIFEST.items():
        src = root / rel
        if not src.exists():
            raise FileNotFoundError(f"{rel} is missing — run `make phase3` / `python -m atp.phase4 data` first")
        dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[rel] = {"sha256": sha256(dst), "bytes": dst.stat().st_size, "purpose": why}
        assert files[rel]["sha256"] == sha256(src), f"copy of {rel} differs from its source"
    m = {"note": "Seats2Stays (ATP 2026) simulator bundle: only the files the simulator needs, copied unchanged from the project "
                 "(same folder structure). Aggregated, derived figures; competition data are licensed for use within "
                 "the competition only. No raw competition files.", "files": files}
    json.dump(m, open(out / "bundle_manifest.json", "w", encoding="utf-8"), indent=1)
    return m


def step_bundle(argv):
    a = argparse.ArgumentParser(prog="atp.phase4 bundle"); a.add_argument("--out", default="build/bundle")
    out = a.parse_args(argv).out
    m = build_bundle(out)
    n = sum(v["bytes"] for v in m["files"].values())
    print(f"BUNDLE: {len(m['files'])} files, {n / 1e6:.1f} MB → {out}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1]
    if cmd == "data":
        step_data()
    elif cmd == "bundle":
        step_bundle(sys.argv[2:])
    else:
        raise SystemExit(f"unknown step {cmd!r}: data | bundle")
