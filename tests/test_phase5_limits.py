"""Step 5.9: one limitations list. It is the one Tab 3, docs/TECHNICAL.md and the report show; every item the plan asked for is in
it; its numbers come from the saved results."""
import json
import re
import pytest
from atp import limits as lm
from atp.config import load_config

CFG = load_config()
_SAVED = CFG["paths"]["processed"] / "phase5" / "limitations.json"
if not _SAVED.exists():
    pytest.skip("run `python -m atp.phase5 limits` first", allow_module_level=True)
SAVED = json.load(open(_SAVED, encoding="utf-8"))
IDS = {i["id"] for i in SAVED}


def test_the_saved_list_is_what_the_code_builds_now():
    assert lm.build(CFG) == SAVED


def test_every_planned_item_is_there():
    # 5.9 plan: disruptions and checkouts, stranded travellers, the uncalibrated total range; and what 5.3–5.7 turned up
    assert {"checkouts", "stranded", "width", "horizon", "trim", "drift", "residents", "newroute", "uncertainty", "benchmark"} <= IDS
    assert {"gateway", "demand", "prices", "follow", "noisy", "deep", "stays", "sustained"} <= IDS   # the earlier Tab 3 list
    assert len(IDS) == len(SAVED) and {i["group"] for i in SAVED} == set(lm.GROUPS)


def test_tab_three_readme_and_report_show_the_same_list():
    from atp.simulator import Simulator
    K = Simulator().know()["limits"]
    assert [x for g in K for x in g["items"]] == [{"title": i["title"], "text": i["text"]} for g in lm.GROUPS for i in SAVED if i["group"] == g]
    readme = (CFG["root"] / "docs" / "TECHNICAL.md").read_text(encoding="utf-8")
    block = readme[readme.index(lm.START):readme.index(lm.END)]
    rep = (CFG["root"] / "reports" / "limitations.md").read_text()
    for i in SAVED:
        assert f"**{i['title']}** {i['text']}" in block and f"**{i['title']}** {i['text']}" in rep


def test_text_is_clean():
    for i in SAVED:
        assert not re.search(r"(?<![\w.])-\d", i["text"]) and "nan" not in i["text"].lower() and "{" not in i["text"]
    assert sum(i["deck"] for i in SAVED) <= 8


def test_stranded_numbers_are_the_data():
    s = lm.stranded(CFG)["2025-06-24"]
    assert 1.0 < s["ratio"] < 1.5 and s["share_of_month_checkins_pct"] < 1.0
    t = next(i for i in SAVED if i["id"] == "stranded")["text"]
    assert f"×{s['ratio']:.2f}" in t and f"{s['share_of_month_checkins_pct']:.1f}%" in t
