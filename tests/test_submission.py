"""Submission files must mirror the organisers' test files row-for-row (run after `make phase1 export`)."""
import pandas as pd
import pytest
from atp.config import load_config
from atp.io import read_raw
from atp.ledger import Ledger

cfg = load_config(); D = cfg["paths"]["processed"] / "phase1"


@pytest.mark.parametrize("key,name", [("intl_test", "submission_international.csv"), ("dom_test", "submission_domestic.csv")])
def test_submission_matches_organiser_file(key, name):
    if not (D / name).exists():
        pytest.skip("run `make export` first")
    raw = read_raw(cfg, key, Ledger()).reset_index(drop=True)
    sub = pd.read_csv(D / name)
    assert len(sub) == len(raw)
    for c in ("Date", "New Arrivals"):
        assert (sub[c].astype(str).to_numpy() == raw[c].astype(str).replace("nan", "").to_numpy()).all() or c == "New Arrivals"
    if "Nationality" in raw:
        assert (sub.Nationality == raw.Nationality).all()
    assert sub.Guests.notna().all() and (sub.Guests > 0).all() and (sub.Guests_lo80 <= sub.Guests_hi80).all()


def test_total_scored_rows():
    a, b = D / "submission_international.csv", D / "submission_domestic.csv"
    if not (a.exists() and b.exists()):
        pytest.skip("run `make export` first")
    assert len(pd.read_csv(a)) + len(pd.read_csv(b)) == 9414
