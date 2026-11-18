"""Load config.yaml and resolve paths relative to the project root.

The root is the project folder by default. A different root (the Phase 4 app bundle, which has the same folder
structure but only the files the simulator needs) is chosen with `load_config(root=...)` or the ATP_ROOT environment
variable. Folders are created only under the project root: a bundle is read-only and must not grow empty data/raw or
data/interim folders.
"""
import os
from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None, root: Path | str | None = None) -> dict:
    root = Path(root or os.environ.get("ATP_ROOT") or ROOT).resolve()
    cfg = yaml.safe_load(open(path or root / "config.yaml", encoding="utf-8"))
    cfg["root"] = root
    cfg["paths"] = {k: root / v for k, v in cfg["paths"].items()}
    cfg["dates"] = {k: pd.Timestamp(v) for k, v in cfg["dates"].items()}
    cfg["folds"] = {f: {k: pd.Timestamp(v) for k, v in d.items()} for f, d in cfg["folds"].items()}
    if root == ROOT:
        for p in cfg["paths"].values():
            p.mkdir(parents=True, exist_ok=True)
    return cfg
