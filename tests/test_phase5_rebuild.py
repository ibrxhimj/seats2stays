"""Step 5.2: comparing a rebuilt copy with the snapshot. Files that differ only in bytes but hold the same values count
as reproduced; a real change in a number, a table value, a table's layout or a missing file does not. The rebuild itself
(scripts/clean_rebuild.sh) takes as long as every phase; its result is reports/phase5_rebuild.md."""
import json
import math
import shutil
import numpy as np
import pandas as pd
import pytest
from atp import phase5 as p5


def _project(root, value=1.0, table=None, extra_json=None):
    (root / "data/processed/phase5").mkdir(parents=True); (root / "reports").mkdir()
    t = table if table is not None else pd.DataFrame({"market": ["A", "B"], "x": [1.0, 2.0]})
    t.to_csv(root / "data/processed/t.csv", index=False)
    json.dump(extra_json or {"a": [1.0, {"b": 2.0}]}, open(root / "data/processed/j.json", "w"))
    (root / "reports/phase1_report.md").write_text("same")
    files = {f"data/processed/{n}": {"group": "data", "sha256": p5.sha256(root / "data/processed" / n), "bytes": 1} for n in ("t.csv", "j.json")}
    snap = {"numbers": {"n": {"value": value, "label": "x", "source": {"kind": "json"}}}, "files": files}
    json.dump(snap, open(root / "data/processed/phase5/snapshot.json", "w"))


@pytest.fixture
def two(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"; return a, b


def test_identical_copies_are_reproduced(two):
    a, b = two; _project(a); _project(b)
    d = p5.compare_rebuild(a, b)
    assert d["ok"] and d["numbers_identical"] == 1 and len(d["files"]["identical"]) == 2 and d["reports"] == {"phase1_report.md": "identical"}


def test_floating_point_noise_is_reproduced_but_a_real_change_is_not(two):
    a, b = two
    _project(a); _project(b, value=1.0 + 1e-12, table=pd.DataFrame({"market": ["A", "B"], "x": [1.0, 2.0 + 1e-10]}))
    d = p5.compare_rebuild(a, b)
    assert d["ok"] and d["files"]["same_content"] == ["data/processed/t.csv"]
    shutil.rmtree(b); _project(b, value=1.01)
    assert not p5.compare_rebuild(a, b)["ok"]
    shutil.rmtree(b); _project(b, table=pd.DataFrame({"market": ["A", "B"], "x": [1.0, 2.5]}))
    d = p5.compare_rebuild(a, b); assert not d["ok"] and d["files"]["off"]["data/processed/t.csv"] == pytest.approx(0.2)


def test_layout_text_and_missing_files_are_differences(two):
    a, b = two
    _project(a); _project(b, table=pd.DataFrame({"market": ["A", "C"], "x": [1.0, 2.0]}))
    assert p5.compare_rebuild(a, b)["files"]["off"]["data/processed/t.csv"] == math.inf          # a changed label
    shutil.rmtree(b); _project(b, table=pd.DataFrame({"market": ["A"], "x": [1.0]}))
    assert p5.compare_rebuild(a, b)["files"]["off"]["data/processed/t.csv"] == math.inf          # a missing row
    shutil.rmtree(b); _project(b, extra_json={"a": [1.0]})
    assert not p5.compare_rebuild(a, b)["ok"]                                                    # a json that lost a key
    shutil.rmtree(b); _project(b); (b / "data/processed/j.json").unlink()
    snap = json.load(open(b / "data/processed/phase5/snapshot.json")); del snap["files"]["data/processed/j.json"]
    json.dump(snap, open(b / "data/processed/phase5/snapshot.json", "w"))
    d = p5.compare_rebuild(a, b); assert not d["ok"] and d["files"]["missing"] == ["data/processed/j.json"]


def test_arrays_compare_by_value(tmp_path):
    x = tmp_path / "x.npz"; y = tmp_path / "y.npz"
    np.savez(x, e=np.array([0.31, 0.44])); np.savez(y, e=np.array([0.31, 0.44 * (1 + 1e-9)]))
    assert p5._max_rel_table(x, y) < p5.REBUILD_TOL
    np.savez(y, e=np.array([0.31, 0.45]))
    assert p5._max_rel_table(x, y) > p5.REBUILD_TOL
