import shutil
import pandas as pd
from atp.config import load_config
from atp.ledger import Ledger
from atp.reference import events_table, calendar


def test_official_file_overrides_algorithm(tmp_path):
    cfg = load_config()
    cfg["paths"]["reference"] = tmp_path
    alg = events_table(cfg, Ledger())
    r = alg[(alg.event == "ramadan") & (alg.key_year == 1447)].iloc[0]
    shifted = pd.DataFrame([{"event": "ramadan", "key_year": 1447,
                             "start_date": (r.start_date + pd.Timedelta(days=1)).date(),
                             "end_date": r.end_date.date(), "source": "official_test"}])
    shifted.to_csv(tmp_path / "calendar_events_official.csv", index=False)
    led = Ledger()
    ev = events_table(cfg, led)
    new = ev[(ev.event == "ramadan") & (ev.key_year == 1447)].iloc[0]
    assert new.start_date == r.start_date + pd.Timedelta(days=1) and new.source == "official_test"
    cal = calendar(ev, cfg, Ledger())
    assert not cal.loc[cal.date == r.start_date, "is_ramadan"].item()          # old first day no longer Ramadan
    assert cal.loc[cal.date == new.start_date, "ramadan_day"].item() == 1
    assert any("1447" in s for s in led.facts["calendar_official_changes"])


def test_school_break_tolerance_flags():
    import pandas as pd
    from atp.config import load_config
    p = load_config()["paths"]["processed"] / "calendar.parquet"
    if not p.exists():
        return
    c = pd.read_parquet(p).set_index("date")
    # 2024-25 winter break: nominal 16 Dec 2024, ±7 days, common end 5 Jan 2025
    assert c.loc["2024-12-09", "is_school_break_possible"] and not c.loc["2024-12-08", "is_school_break_possible"]
    assert not c.loc["2024-12-22", "is_school_break_certain"] and c.loc["2024-12-23", "is_school_break_certain"]
    assert c.loc["2025-01-05", "school_break_days_left"] == 1 and c.loc["2025-01-06", "school_break_days_left"] == 0
    assert c.loc["2024-12-09", "school_break_days_left"] == 28
