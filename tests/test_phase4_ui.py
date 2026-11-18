"""Tab 1 in a real browser: the page loads without errors, each lever step shows the curve's number without a request,
release confirms the same numbers with a full run, and a preset keeps its note. Skipped without Playwright + Chromium."""
import pytest


def test_page_loads_with_a_real_result(page):
    assert page.inner_text("#big") == "+37,100" and "Etihad" in page.inner_text("#strips")
    assert page.inner_text("#trust-range").endswith("%") and not page.errors


def test_lever_steps_are_the_curve_and_release_confirms_them(page):
    curve = page.evaluate("fetch('/api/lever?market=UNITED%20KINGDOM&season=winter').then(r => r.json())")
    page.wait_for_timeout(300)
    seen = []
    page.on("request", lambda r: "/api/" in r.url and seen.append(r.url))
    page.focus("#lever-slider")
    for k in range(1, 8):
        page.keyboard.press("ArrowRight"); page.wait_for_timeout(30)
        assert page.inner_text("#big") == curve["steps"][curve["today_index"] + k]["headline"]["text"]
    live = {s: page.inner_text(s) for s in ("#big", "#lead", "#range-text", "#occ", "#list", "#why")}
    assert seen == []                                             # dragging never waits for the network
    page.wait_for_timeout(1500)
    assert any("/api/run" in u for u in seen)                     # release: a full engine run …
    assert {s: page.inner_text(s) for s in live} == live          # … that shows exactly the same
    assert max(page.evaluate("window.__frames")) < 16 and not page.errors


def test_a_preset_keeps_its_note(page):
    page.click("#presets [data-preset='march']")
    page.wait_for_function("document.getElementById('big').textContent.startsWith('−')", timeout=30000)
    assert "Flights explain" in page.inner_text("#lead") and not page.errors
