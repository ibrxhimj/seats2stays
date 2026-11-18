"""Step 4.5 in a real browser: the four add-a-change forms build exactly the change the engine takes (and what the page
shows equals a direct engine run), errors come back in plain words, a change can be edited, scenario A compares with
what is on screen, the by-day view shows the record, downloads work, and a country without flights opens the new-route
form. Skipped without Playwright + Chromium."""
import json
import math
from pathlib import Path
from atp import scenario as sc

WIN = {"start": "2026-11-01", "end": "2027-02-28"}


def open_form(page, kind=None):
    page.click("#add-change")
    if kind:
        page.click(f"#fd-kinds label:has-text('{kind}')")


def submit(page):
    page.click("#fd-form .fd-add")


def added(page):
    page.wait_for_selector("#form-dialog", state="hidden", timeout=30000)


def last_run(page):
    return [b for u, b in page.posts if u == "run"][-1]


def strips(page):
    return page.locator("#strips .strip").count()


def sig3(x):
    return 0 if x == 0 else round(x, 2 - math.floor(math.log10(abs(x))))


def fmt(x):
    return ("−" if x < 0 else "+") + f"{abs(x):,.0f}"


def test_route_form(page):
    meta = page.api("/api/meta")
    r = next(m for m in meta["markets"] if m["key"] == "UNITED KINGDOM")["routes"][0]
    open_form(page); submit(page)                                  # unchanged value: refused, nothing added
    assert "what this route has today" in page.inner_text("#fd-error") and strips(page) == 1
    page.fill("[name=value]", "14"); submit(page); added(page)
    want = {"type": "route", "markets": "UNITED KINGDOM", "city": r["city"], "airline": r["airline"], "flights_per_week": 14, **WIN}
    assert last_run(page) == {"changes": [want], "response": "central"} and strips(page) == 2
    assert page.inner_text("#big") == page.api("/api/run", {"changes": [want]})["headline"]["text"]
    open_form(page); page.click("label:has-text('Seats per plane')"); page.fill("[name=value]", "300"); submit(page); added(page)
    assert last_run(page)["changes"][0]["seats_per_flight"] == 300 and not page.errors


def test_new_route_form_default_is_the_briefs_example(page):
    open_form(page, "Open a new route"); submit(page); added(page)
    assert last_run(page)["changes"] == page.api("/api/presets/new_route")["changes"]      # Mexico City, Etihad, 2 a week, widebody
    assert page.inner_text("#big") == "+1,660" and "New route: Mexico City (Etihad)" in page.inner_text("#strips")
    open_form(page, "Open a new route")                                                     # a mix of the user's own
    page.select_option("[name=airline]", "IndiGo"); page.click("label:has-text('Narrowbody')")
    page.click(".fd-mix summary"); page.fill("[name=load_factor]", "80"); page.fill("[name=local_share]", "50"); submit(page); added(page)
    ch = last_run(page)["changes"][0]
    assert (ch["airline"], ch["aircraft_class"], ch["load_factor"], ch["local_share"]) == ("IndiGo", "narrowbody", 0.8, 0.5)
    assert page.inner_text("#big") == page.api("/api/run", {"changes": [ch]})["headline"]["text"]


def test_a_refused_change_says_why_in_plain_words(page):
    open_form(page, "Open a new route")
    page.select_option("[name=market]", "UNITED KINGDOM"); page.fill("[name=city]", "London"); page.select_option("[name=airline]", "Etihad Airways")
    submit(page)
    page.wait_for_function("document.getElementById('fd-error').textContent !== ''")          # the server's answer
    err = page.inner_text("#fd-error")
    assert "already flies London" in err and "new_route" not in err and not err.startswith("change ")
    assert page.is_visible("#form-dialog") and strips(page) == 1                            # still open, nothing added
    page.fill("[name=fpw]", "0"); submit(page)
    assert "between 1 and 100" in page.inner_text("#fd-error")


def test_planes_and_connections(page):
    open_form(page, "Planes and connections"); submit(page); added(page)
    assert last_run(page)["changes"] == [{"type": "load_factor", "markets": "flight markets", "pct": 5, **WIN}]
    assert page.inner_text("#big") == page.api("/api/run", {"changes": last_run(page)["changes"]})["headline"]["text"]
    open_form(page, "Planes and connections"); page.click("label:has-text('Passengers connecting onwards')")
    assert page.input_value("[name=value]") == "-5"                                          # fewer connecting is the useful default
    page.select_option("[name=scope]", "Europe"); submit(page); added(page)
    assert last_run(page)["changes"] == [{"type": "transfer_share", "markets": "Europe", "points": -5, **WIN}]
    assert not page.errors


def test_demand_and_stays(page):
    open_form(page, "Demand and stays"); page.select_option("[name=scope]", "DOMESTIC"); page.fill("[name=value]", "-10"); submit(page); added(page)
    assert last_run(page)["changes"] == [{"type": "demand", "markets": "DOMESTIC", "pct": -10, **WIN}]
    assert page.locator("#strips .strip.demand").count() == 1                                # an assumption, drawn as one
    assert "lead" in page.get_attribute("#lead", "class") and "loss" in page.get_attribute("#lead", "class")
    assert page.inner_text("#big").startswith("−") and "No flights change" in page.inner_text("#stream")
    open_form(page, "Demand and stays"); page.click("label:has-text('Length of stay')"); page.select_option("[name=scope]", "Europe"); submit(page); added(page)
    assert last_run(page)["changes"] == [{"type": "stay_length", "markets": "Europe", "pct": 10, **WIN}]


def test_a_change_can_be_edited(page):
    open_form(page); page.fill("[name=value]", "14"); submit(page); added(page)
    assert page.locator("#strips .strip.form [data-edit]").count() == 1
    page.click("#strips .strip.form [data-edit]")
    assert page.text_content("#fd-title") == "Edit this change" and page.input_value("[name=value]") == "14" and not page.is_visible("#fd-kinds")
    page.fill("[name=value]", "7"); submit(page); added(page)
    assert strips(page) == 2 and last_run(page)["changes"][0]["flights_per_week"] == 7
    assert "7 flights a week" in page.inner_text("#strips")
    page.click("#presets [data-preset='new_route']"); page.wait_for_function("document.getElementById('big').textContent === '+1,660'")
    page.click("#strips [data-edit]"); assert page.input_value("[name=city]") == "Mexico City"      # a preset's route opens in its form


def test_pin_a_and_compare(page):
    page.click("#pin"); assert page.is_visible("#pinned .pinned") and not page.is_visible("#compare") is False
    a = page.api("/api/run", {"changes": page.api("/api/presets/etihad")["changes"]})
    open_form(page, "Demand and stays"); page.fill("[name=value]", "-10"); submit(page); added(page)
    now = last_run(page)["changes"]
    b = page.api("/api/run", {"changes": now})
    text = page.inner_text("#compare")
    assert "Etihad's 2026 launches" in text and a["headline"]["text"] in text and b["headline"]["text"] in text
    assert fmt(sig3(b["headline"]["d_guest_nights"] - a["headline"]["d_guest_nights"])) in text
    assert page.locator("#months line[stroke='#15222A']").count() == 12                      # A: one dark tick per month
    page.focus("#lever-slider"); page.keyboard.press("ArrowRight")                           # A stays while the slider moves
    page.wait_for_function("document.getElementById('lead').textContent.includes('extra seats')")
    assert page.is_visible("#compare") and "Etihad's 2026 launches" in page.inner_text("#compare")
    page.click("#compare [data-unpin]"); assert not page.is_visible("#compare") and page.locator("#months line[stroke='#15222A']").count() == 0


def test_by_day_view_shows_the_record(page):
    page.click("#when-view [data-view=day]")
    assert "Highest ever recorded: " in page.text_content("#months") and "Normal year" in page.inner_text("#legend")
    box = page.locator("#months").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + 100)
    assert " guests (" in page.text_content("#months")
    page.focus("#lever-slider"); page.keyboard.press("ArrowRight")
    page.wait_for_selector("#when-note", state="visible", timeout=2000)                       # the nights wait for the confirmed run
    page.wait_for_selector("#when-note", state="hidden", timeout=30000)
    assert "Highest ever recorded" in page.text_content("#months") and not page.errors
    page.click("#when-view [data-view=month]"); assert "Change in guest nights" in page.inner_text("#legend")


def test_downloads(page, tmp_path):
    with page.expect_download() as d:
        page.click("#dl-csv")
    assert d.value.suggested_filename == "etihad_s_2026_launches_by_market_and_month.csv"
    csv = Path(d.value.path()).read_text()
    assert csv.startswith("market,month,guest_nights_normal_year,guest_nights_scenario,change_guest_nights") and "United States,2026-" in csv
    with page.expect_download() as d:
        page.click("#dl-json")
    p = tmp_path / "s.json"; d.value.save_as(p)
    assert sc.load(p)["changes"] == page.api("/api/presets/etihad")["changes"]                # the file runs as it is


def test_a_country_without_flights_opens_the_new_route_form(page):
    page.click("#presets [data-preset='new_route']"); page.wait_for_function("document.getElementById('big').textContent === '+1,660'")
    page.click("#list li.pickable:has-text('Mexico')")
    assert page.is_visible("#form-dialog") and page.input_value("[name=market]") == "MEXICO" and page.input_value("[name=city]") == "Mexico City"
    assert page.is_checked("#fd-kinds [value=new]")


def test_clear_all(page):
    open_form(page, "Demand and stays"); submit(page); added(page)
    page.click("#clear"); page.wait_for_function("document.getElementById('lead').textContent.startsWith('No changes yet')")
    assert "No changes yet" in page.inner_text("#strips") and not page.is_visible("#clear")
