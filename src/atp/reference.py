"""Reference tables: departure-country ↔ nationality map, and the event calendar.

Calendar design: Islamic dates in the UAE follow moon sighting, so algorithmic (Umm al-Qura) dates
can be off by ±1 day. Every event is a row in an EVENTS table with a `source` column. The build
writes the algorithmic table as a template; if `calendar_events_official.csv` exists, each
(event, key_year) it contains REPLACES the algorithmic row. Day-level flags are derived only from
the events table — there is a single source of truth for every holiday.
"""
import pandas as pd
from hijridate import Hijri, Gregorian
from .ledger import Ledger

# ISO3 and UN M49 sub-region (objective grouping, used later only for pooling / cold-start)
COUNTRIES = {
    "ARMENIA": ("ARM", "Western Asia"), "AUSTRALIA": ("AUS", "Australia and New Zealand"),
    "AUSTRIA": ("AUT", "Western Europe"), "AZERBAIJAN": ("AZE", "Western Asia"),
    "BAHRAIN": ("BHR", "Western Asia"), "BELGIUM": ("BEL", "Western Europe"),
    "BRAZIL": ("BRA", "Latin America and the Caribbean"), "CANADA": ("CAN", "Northern America"),
    "CHINA": ("CHN", "Eastern Asia"), "CZECHIA": ("CZE", "Eastern Europe"),
    "DENMARK": ("DNK", "Northern Europe"), "EGYPT": ("EGY", "Northern Africa"),
    "FINLAND": ("FIN", "Northern Europe"), "FRANCE": ("FRA", "Western Europe"),
    "GERMANY": ("DEU", "Western Europe"), "INDIA": ("IND", "Southern Asia"),
    "IRELAND": ("IRL", "Northern Europe"), "ISRAEL": ("ISR", "Western Asia"),
    "ITALY": ("ITA", "Southern Europe"), "JAPAN": ("JPN", "Eastern Asia"),
    "JORDAN": ("JOR", "Western Asia"), "KAZAKHSTAN": ("KAZ", "Central Asia"),
    "KUWAIT": ("KWT", "Western Asia"), "LEBANON": ("LBN", "Western Asia"),
    "MEXICO": ("MEX", "Latin America and the Caribbean"), "MOROCCO": ("MAR", "Northern Africa"),
    "NETHERLANDS": ("NLD", "Western Europe"), "NORWAY": ("NOR", "Northern Europe"),
    "OMAN": ("OMN", "Western Asia"), "PAKISTAN": ("PAK", "Southern Asia"),
    "PHILIPPINES": ("PHL", "South-eastern Asia"), "POLAND": ("POL", "Eastern Europe"),
    "QATAR": ("QAT", "Western Asia"), "ROMANIA": ("ROU", "Eastern Europe"),
    "RUSSIAN FEDERATION": ("RUS", "Eastern Europe"), "SAUDI ARABIA": ("SAU", "Western Asia"),
    "SOUTH AFRICA": ("ZAF", "Sub-Saharan Africa"), "SOUTH KOREA": ("KOR", "Eastern Asia"),
    "SPAIN": ("ESP", "Southern Europe"), "SWEDEN": ("SWE", "Northern Europe"),
    "SWITZERLAND": ("CHE", "Western Europe"), "TURKEY": ("TUR", "Western Asia"),
    "UNITED KINGDOM": ("GBR", "Northern Europe"), "UNITED STATES OF AMERICA": ("USA", "Northern America"),
    "UZBEKISTAN": ("UZB", "Central Asia"),
}
GCC = {"BAHRAIN", "KUWAIT", "OMAN", "QATAR", "SAUDI ARABIA"}


def country_map(hotel_nats, flight_rows: pd.DataFrame, led: Ledger) -> pd.DataFrame:
    fl = flight_rows.groupby("nationality_key").agg(flight_country_name=("country", "first"),
                                                    first_flight_date=("date", "min"),
                                                    n_departure_cities=("city", "nunique"),
                                                    n_airlines=("airline", "nunique"))
    rows = []
    for n in sorted(hotel_nats):
        iso3, m49 = COUNTRIES[n]
        has = n in fl.index
        rows.append(dict(nationality=n, iso3=iso3, m49_subregion=m49, is_gcc=n in GCC, has_flight_data=has,
                         **(fl.loc[n].to_dict() if has else dict(flight_country_name=None, first_flight_date=pd.NaT,
                                                                     n_departure_cities=0, n_airlines=0))))
    m = pd.DataFrame(rows)
    led.check("map:all_nationalities_known", set(hotel_nats) <= set(COUNTRIES), "every hotel nationality has ISO3 + M49 sub-region")
    led.check("map:all_flight_countries_mapped", set(fl.index) <= set(hotel_nats),
              f"all {len(fl)} flight departure countries map 1:1 to a hotel nationality")
    no_data = m.loc[~m.has_flight_data, "nationality"].tolist()
    led.fact("no_flight_data_nationalities", no_data)
    led.rule("R1", "country_map", f"{int(m.has_flight_data.sum())} nationalities have flights from their country in the "
             f"pack; {len(no_data)} have NO FLIGHT DATA IN THE PACK ({', '.join(x.title() for x in no_data)}). This does "
             "not mean they have no direct route to Abu Dhabi (several do in reality); their flights are simply not in "
             "the file, so their check-ins cannot be linked to seats", len(m))
    return m


def _h2g(y, m, d):
    return pd.Timestamp(Hijri(y, m, d).to_gregorian().isoformat())


def algorithmic_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    h0 = Gregorian(start.year, start.month, start.day).to_hijri().year - 1
    h1 = Gregorian(end.year, end.month, end.day).to_hijri().year + 1
    ev = []
    for hy in range(h0, h1 + 1):
        r0 = _h2g(hy, 9, 1)
        shawwal1 = _h2g(hy, 10, 1)
        ev += [("ramadan", hy, r0, shawwal1 - pd.Timedelta(days=1)),
               ("eid_al_fitr", hy, shawwal1, shawwal1 + pd.Timedelta(days=2)),
               ("arafat_day", hy, _h2g(hy, 12, 9), _h2g(hy, 12, 9)),
               ("eid_al_adha", hy, _h2g(hy, 12, 10), _h2g(hy, 12, 12)),
               ("islamic_new_year", hy, _h2g(hy, 1, 1), _h2g(hy, 1, 1)),
               ("mawlid", hy, _h2g(hy, 3, 12), _h2g(hy, 3, 12))]
    for y in range(start.year, end.year + 1):
        ev += [("new_year", y, pd.Timestamp(y, 1, 1), pd.Timestamp(y, 1, 1)),
               ("commemoration_day", y, pd.Timestamp(y, 12, 1), pd.Timestamp(y, 12, 1)),
               ("national_day", y, pd.Timestamp(y, 12, 2), pd.Timestamp(y, 12, 3))]
    df = pd.DataFrame(ev, columns=["event", "key_year", "start_date", "end_date"])
    df["source"] = df.event.map(lambda e: "fixed_gregorian" if e in ("new_year", "commemoration_day", "national_day")
                                else "algorithmic_umm_al_qura")
    df = df[(df.end_date >= start) & (df.start_date <= end)].reset_index(drop=True)
    return df


def events_table(cfg: dict, led: Ledger) -> pd.DataFrame:
    start, end = cfg["dates"]["calendar_start"], cfg["dates"]["calendar_end"]
    alg = algorithmic_events(start, end)
    alg["active"] = True
    ref = cfg["paths"]["reference"]
    alg.to_csv(ref / "calendar_events_algorithmic.csv", index=False, date_format="%Y-%m-%d")
    key = ["event", "key_year"]
    need = {"event", "key_year", "start_date", "end_date", "source"}
    off_path = ref / "calendar_events_official.csv"
    if off_path.exists():
        off = pd.read_csv(off_path, parse_dates=["start_date", "end_date"])
        led.check("calendar:official_schema", need <= set(off.columns), f"official file columns {sorted(off.columns)}")
        off["active"] = off.get("active", True)
        off["active"] = off["active"].astype(str).str.lower().isin(["true", "1", "yes"])
        merged = alg.set_index(key)
        o = off.set_index(key)[["start_date", "end_date", "source", "active"]]
        changed = [(k, merged.loc[k, "start_date"].date(), o.loc[k, "start_date"].date())
                   for k in o.index if k in merged.index and merged.loc[k, "start_date"] != o.loc[k, "start_date"]]
        merged.update(o)
        merged["active"] = merged["active"].astype(bool)
        new = o.loc[~o.index.isin(merged.index)]
        ev = pd.concat([merged, new]).reset_index()
        led.rule("C2", "calendar_events", f"Official UAE public-holiday dates (user-supplied, moon-sighting based) override "
                 f"the algorithm: {len(o)} rows supplied, {len(changed)} start dates differ from the algorithm, "
                 f"{int((~o.active).sum())} rows deactivated (not on the official list)", len(o))
        led.fact("calendar_official_changes", [f"{k[0]} {k[1]}: {a} → {b}" for k, a, b in changed])
        led.check("calendar:official_dates_supplied", True, f"{len(o)} official rows loaded")
    else:
        ev = alg
        led.rule("C2", "calendar_events", "NO official date file — Islamic events use algorithmic Umm al-Qura dates", 0)
        led.check("calendar:official_dates_supplied", False, "official moon-sighting dates not yet supplied", "warn")
    sb_path = ref / "school_breaks.csv"
    if sb_path.exists():
        sb = pd.read_csv(sb_path, parse_dates=["start_date", "end_date"])
        sb["active"] = sb["active"].astype(str).str.lower().isin(["true", "1", "yes"])
        sb["end_confirmed"] = sb["end_confirmed"].astype(str).str.lower().isin(["true", "1", "yes"])
        # Restart rule: schools reopen on a Monday (Tuesday if that Monday is a public holiday);
        # any other sourced restart date snaps to the closest Monday.
        hol = set()
        for r in ev[ev.active & ~ev.event.isin(["ramadan"])].itertuples():
            hol |= set(pd.date_range(r.start_date, r.end_date))
        snapped = []
        for i, r in sb.iterrows():
            restart = r.end_date + pd.Timedelta(days=1)
            ok = restart.dayofweek == 0 or (restart.dayofweek == 1 and restart - pd.Timedelta(days=1) in hol)
            if not ok:
                back = restart - pd.Timedelta(days=restart.dayofweek)
                fwd = back + pd.Timedelta(days=7)
                new_restart = back if (restart - back) <= (fwd - restart) else fwd
                sb.loc[i, "end_date"] = new_restart - pd.Timedelta(days=1)
                snapped.append(f"{r.event} {r.key_year}: restart {restart.date()} → {new_restart.date()}")
        led.rule("C5", "calendar_events", "School restart must be a Monday (Tuesday if that Monday is a public holiday, "
                 "e.g. 2 Jan 2024 after New Year); otherwise snapped to the closest Monday. Snapped: "
                 + ("none" if not snapped else "; ".join(snapped)), len(snapped))
        restarts = sb.end_date + pd.Timedelta(days=1)
        led.check("calendar:school_restart_monday_rule",
                  all(d.dayofweek == 0 or (d.dayofweek == 1 and d - pd.Timedelta(days=1) in hol) for d in restarts),
                  "every school restart is a Monday, or a Tuesday after a Monday public holiday")
        ev = pd.concat([ev, sb], ignore_index=True)
        led.rule("C4", "calendar_events", "School breaks (national MoE/unified calendar; Indian/Pakistani-curriculum "
                 "schools run Apr–Mar and differ): " + "; ".join(f"{r.event.replace('school_', '').replace('_break', '')} "
                 f"{r.start_date.date()} (±{r.start_tolerance_days}d)…{r.end_date.date()}{'' if r.end_confirmed else ' (end unconfirmed)'}" for r in sb.itertuples())
                 + ". Break START varies by school (±tolerance); END is common to all schools", len(sb))
    n_inactive = int((~ev.active).sum())
    ev = ev[ev.active].drop(columns="active")
    ev["n_days"] = (ev.end_date - ev.start_date).dt.days + 1
    ram = ev[ev.event == "ramadan"]
    led.check("calendar:ramadan_length", ram.n_days.between(29, 30).all(), f"Ramadan lengths {sorted(int(x) for x in ram.n_days.unique())} days")
    win = ev[(ev.end_date >= cfg["dates"]["hotel_start"]) & (ev.start_date <= cfg["dates"]["test_end"])]
    is_school = win.event.str.startswith("school_")
    shaky = win[(~is_school & win.source.str.contains("algorithmic|needs_confirmation|approximate"))
                | (is_school & ~win.get("end_confirmed", pd.Series(True, index=win.index)).fillna(True).astype(bool))]
    led.check("calendar:modelling_window_sources", shaky.empty,
              "events inside 2022-01…2026-02 with unconfirmed dates (school breaks judged on END date): "
              + ("none" if shaky.empty else "; ".join(f"{r.event} {r.start_date.date()} [{r.source}]" for r in shaky.itertuples())), "warn")
    led.rule("C1", "calendar_events", "Event windows: Ramadan (fasting month), Eid al-Fitr and Eid al-Adha (official public-"
             "holiday windows, may include 29 Ramadan), Arafat, Islamic New Year and Mawlid (Cabinet-shifted day where "
             f"applicable), New Year, Commemoration Day, National Day; {n_inactive} inactive rows removed", len(ev))
    return ev.sort_values(["start_date", "event"]).reset_index(drop=True)


def calendar(ev: pd.DataFrame, cfg: dict, led: Ledger) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(cfg["dates"]["calendar_start"], cfg["dates"]["calendar_end"])})
    d["year"], d["month"], d["day"] = d.date.dt.year, d.date.dt.month, d.date.dt.day
    d["dow"] = d.date.dt.dayofweek            # 0 = Monday
    d["dow_name"] = d.date.dt.day_name()
    d["days_in_month"] = d.date.dt.days_in_month
    d["is_weekend_uae"] = d.dow.isin([5, 6])  # Sat–Sun since 1 Jan 2022
    public = {"eid_al_fitr", "arafat_day", "eid_al_adha", "islamic_new_year", "mawlid", "new_year",
              "commemoration_day", "national_day"}
    d["holiday_name"] = ""
    d["event_source"] = ""
    for e in sorted(ev.event.unique()):
        d[f"is_{e}"] = False
    d["ramadan_day"] = 0
    d["school_break_days_left"] = 0
    d["is_school_break_possible"] = False   # nominal start − tolerance … end (some schools already off)
    d["is_school_break_certain"] = False    # nominal start + tolerance … end (all schools off)
    for _, r in ev.iterrows():
        m = d.date.between(r.start_date, r.end_date)
        d.loc[m, f"is_{r.event}"] = True
        d.loc[m, "event_source"] = d.loc[m, "event_source"].where(d.loc[m, "event_source"] == "", d.loc[m, "event_source"] + ";") + r.source
        if r.event == "ramadan":
            d.loc[m, "ramadan_day"] = (d.loc[m, "date"] - r.start_date).dt.days + 1
        if r.event.startswith("school_"):
            tol = pd.Timedelta(days=int(r.start_tolerance_days))
            wide = d.date.between(r.start_date - tol, r.end_date)
            d.loc[wide, "school_break_days_left"] = (r.end_date - d.loc[wide, "date"]).dt.days + 1
            d.loc[wide, "is_school_break_possible"] = True
            d.loc[d.date.between(r.start_date + tol, r.end_date), "is_school_break_certain"] = True
        if r.event in public:
            d.loc[m, "holiday_name"] = d.loc[m, "holiday_name"].where(d.loc[m, "holiday_name"] == "", d.loc[m, "holiday_name"] + ";") + r.event
    d["is_public_holiday"] = d.holiday_name != ""
    d["is_school_break"] = d[[c for c in d if c in ("is_school_winter_break", "is_school_summer_break", "is_school_spring_break")]].any(axis=1)
    # long break = public holiday(s) joined with directly adjacent weekend days (the actual time off)
    off = (d.is_public_holiday | d.is_weekend_uae).to_numpy()
    grp = (off != pd.Series(off).shift(fill_value=False).to_numpy()).cumsum()
    blocks = pd.DataFrame({"g": grp, "off": off, "hol": d.is_public_holiday})
    has_hol = blocks.groupby("g").hol.transform("any") & blocks.off
    length = blocks.groupby("g").off.transform("size").where(has_hol, 0)
    d["is_long_break"] = has_hol & (length >= 3)
    d["long_break_length"] = length.where(d.is_long_break, 0).astype(int)
    led.rule("C3", "calendar", f"Daily calendar {d.date.min().date()}…{d.date.max().date()}: weekday, UAE weekend (Sat–Sun), "
             "event flags, `ramadan_day`, `is_public_holiday`, `is_long_break` (holiday + adjacent weekend, ≥3 days) with "
             "`long_break_length`; school breaks as nominal (`is_school_break`), `is_school_break_possible` (start −7d) and "
             "`is_school_break_certain` (start +7d), plus `school_break_days_left` counted to the common END date "
             "(family return wave); `event_source`", len(d))
    return d


def external_events(cfg: dict, led: Ledger) -> pd.DataFrame:
    """Audit log of real-world events checked against the data (evidence + decision per event)."""
    ext = pd.read_csv(cfg["paths"]["reference"] / "external_events.csv", parse_dates=["start_date", "end_date"])
    led.rule("R2", "external_events", "External events audited against the data: " + "; ".join(
        f"{r.event} ({r.start_date.date()}) → {r.decision.split('.')[0]}" for r in ext.itertuples()), len(ext))
    return ext
