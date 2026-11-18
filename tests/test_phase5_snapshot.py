"""Step 5.1: the snapshot. Every headline number has a label and a source; recomputing them today gives exactly the
snapshot (nothing has drifted); the comparison flags a changed number or data file, and lists changed code without
failing on it."""
import copy
import json
import pytest
from atp import phase5 as p5
from atp.phase4 import FACTS
from atp.config import load_config

cfg = load_config()
PATH = cfg["paths"]["processed"] / p5.OUT / "snapshot.json"
if not PATH.exists():
    pytest.skip("run `python -m atp.phase5 snapshot` first", allow_module_level=True)
SNAP = json.load(open(PATH))


def test_every_number_has_a_label_and_a_source():
    n = SNAP["numbers"]
    assert set(FACTS) | set(p5.EXTRA) <= set(n)                                   # the app's facts and the reports' numbers
    assert any(v["source"]["kind"] == "simulator" for v in n.values())
    for k, v in n.items():
        assert v["label"] and v["source"]["kind"] in ("json", "csv", "agg", "simulator"), k
        assert v["value"] is not None and v["value"] == v["value"], k            # no missing or NaN


def test_the_raw_files_are_the_pinned_ones():
    for name, want in cfg["raw_sha256"].items():
        assert SNAP["files"][f"data/raw/{name}"]["sha256"] == want


def test_nothing_has_drifted_since_the_snapshot():
    d = p5.compare(SNAP, {"numbers": p5.numbers(cfg), "files": p5.files(cfg)})
    assert d["ok"], {k: v for k, v in d.items() if v and k.startswith(("numbers", "data"))}


def test_the_comparison_flags_what_matters():
    new = copy.deepcopy(SNAP)
    assert p5.compare(SNAP, new)["ok"]
    k = next(iter(new["numbers"])); new["numbers"][k]["value"] = 1e9
    assert not p5.compare(SNAP, new)["ok"]
    new = copy.deepcopy(SNAP); f = next(x for x, v in new["files"].items() if v["group"] == "data"); new["files"][f]["sha256"] = "0" * 64
    d = p5.compare(SNAP, new); assert not d["ok"] and d["data_changed"] == [f]
    new = copy.deepcopy(SNAP); f = next(x for x, v in new["files"].items() if v["group"] == "code"); new["files"][f]["sha256"] = "0" * 64
    d = p5.compare(SNAP, new); assert d["ok"] and d["code_changed"] == [f]     # code may change; the numbers decide
    new = copy.deepcopy(SNAP); new["numbers"][k]["value"] = SNAP["numbers"][k]["value"] * (1 + 1e-12) if isinstance(SNAP["numbers"][k]["value"], float) else SNAP["numbers"][k]["value"]
    assert p5.compare(SNAP, new)["ok"]                                            # float noise is not a change
