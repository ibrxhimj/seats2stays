"""Phase 0 build: raw Excel → validated model-ready parquet tables + data card.

Run:  make data      (or: PYTHONPATH=src python -m atp.build)
Exit code 1 if any blocking check fails.
"""
import sys
import time
from . import flights, hotels, reference, checks, datacard
from .config import load_config
from .io import read_raw, sha256
from .ledger import Ledger


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    led = Ledger()
    raw = {k: read_raw(cfg, k, led) for k in cfg["raw_files"]}
    digests = {f: sha256(cfg["paths"]["raw"] / f) for f in cfg["raw_files"].values()}

    rows = flights.clean_rows(raw["flights"], cfg, led)
    dq = flights.day_quality(rows, cfg, led)
    fd = flights.country_daily(rows, dq, cfg, led)
    fm = flights.country_monthly(rows, dq, cfg, led)
    hi, _ = hotels.build_international(raw["intl_train"], raw["intl_test"], cfg, led)
    hd = hotels.build_domestic(raw["dom_train"], raw["dom_test"], cfg, led)
    hm = hotels.monthly(hi, hd, led)
    cmap = reference.country_map(hi.nationality.unique(), rows, led)
    ev = reference.events_table(cfg, led)
    ext = reference.external_events(cfg, led)
    cal = reference.calendar(ev, cfg, led)

    T = {"flights_rows": rows, "flights_day_quality": dq, "flights_country_daily": fd, "flights_country_monthly": fm,
         "hotel_intl_daily": hi, "hotel_dom_daily": hd, "hotel_monthly": hm, "country_map": cmap,
         "calendar_events": ev, "calendar": cal, "external_events": ext}

    from .io import to_num
    raw_totals = {c: to_num(raw["flights"][src]).fillna(0).sum() for c, src in
                  [("seats", "Total Seats"), ("pax", "Total PAX"), ("p2p", "Total P2P"), ("transfer", "Total Transfer"),
                   ("transit", "Total Transit")]}
    raw_totals["pax_excl_infant"] = rows.pax_excl_infant.sum()  # one missing value filled by rule F7
    raw_totals["intl_guests"] = to_num(raw["intl_train"]["Guests"]).sum()
    raw_totals["intl_arrivals"] = to_num(raw["intl_train"]["New Arrivals"]).sum() + to_num(raw["intl_test"]["New Arrivals"]).sum()
    raw_totals["dom_guests"] = to_num(raw["dom_train"]["Guests"]).sum()
    checks.run_all(T, raw_totals, cfg, led)

    out = cfg["paths"]["processed"]
    for name, df in T.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    cmap.to_csv(cfg["paths"]["reference"] / "country_map.csv", index=False)
    datacard.write(T, led, cfg, digests, cfg["paths"]["reports"] / "data_card.md")

    fails = led.failures()
    warns = [c for c in led.checks if c[1] == "warn" and not c[2]]
    print(f"Phase 0 build: {len(T)} tables, {len(led.rules)} rules, {len(led.checks)} checks "
          f"({len(fails)} failed, {len(warns)} warnings) in {time.time() - t0:.0f}s")
    for c in fails:
        print("  FAIL", c[0], "—", c[3])
    for c in warns:
        print("  WARN", c[0], "—", c[3])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
