"""Raw-file access: checksum verification + cached typed copies. Raw files are never modified."""
import hashlib
import pandas as pd
from .ledger import Ledger


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(cfg: dict, key: str, led: Ledger) -> pd.DataFrame:
    """Read one raw Excel file as strings (no silent type coercion); cache by checksum."""
    fname = cfg["raw_files"][key]
    path = cfg["paths"]["raw"] / fname
    digest = sha256(path)
    expected = cfg["raw_sha256"][fname]
    led.check(f"raw_checksum:{fname}", digest == expected, f"sha256 {digest[:12]}… (expected {expected[:12]}…)")
    cache = cfg["paths"]["interim"] / f"{path.stem}.{digest[:12]}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_excel(path, sheet_name="Export", dtype=str)
    df.to_parquet(cache, index=False)
    return df


def to_num(s: pd.Series) -> pd.Series:
    """'*' and blanks → NaN; anything else must parse as a number (raises otherwise)."""
    s2 = s.replace("*", pd.NA)
    out = pd.to_numeric(s2, errors="coerce")
    bad = out.isna() & s2.notna()
    if bad.any():
        raise ValueError(f"non-numeric values in {s.name}: {s2[bad].unique()[:5]}")
    return out
