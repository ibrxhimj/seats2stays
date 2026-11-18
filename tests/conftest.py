import json
import socket
import sys
import threading
import time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---- public copy without the competition data: tests that need it are reported as skipped, not failed.
# The data pack is licensed for the competition only and is not in the repository; with the raw files in data/raw/
# and the pipeline run (`make data phase1 phase2 phase3 phase4data`), every test runs as normal.
HAVE_DATA = (ROOT / "data" / "raw" / "flight_data.xlsx").exists()
NO_DATA = "needs the ATP 2026 competition data (not included; see data/raw/README.md)"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    if not HAVE_DATA and rep.failed:
        rep.outcome, rep.longrepr = "skipped", (str(item.path), 0, f"Skipped: {NO_DATA}")
    return rep


# ---- browser tests (Phase 4): one app server and one Chromium for the whole run; every test gets its own fresh page.
@pytest.fixture(scope="session")
def app_url():
    uvicorn = pytest.importorskip("uvicorn")
    from atp import api
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(300):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture(scope="session")
def browser():
    sync = pytest.importorskip("playwright.sync_api")
    p = sync.sync_playwright().start()
    try:
        b = p.chromium.launch()
    except Exception as e:                                        # no browser on this machine
        p.stop(); pytest.skip(f"no Chromium: {e}")
    yield b
    b.close(); p.stop()


@pytest.fixture
def page(browser, app_url):
    """A fresh page on the loaded app. page.errors: console errors; page.posts: the JSON bodies of every POST so far."""
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
    pg = ctx.new_page()
    pg.errors, pg.posts = [], []
    pg.on("console", lambda m: m.type == "error" and pg.errors.append(m.text))
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("request", lambda r: r.method == "POST" and pg.posts.append((r.url.rsplit("/", 1)[-1], json.loads(r.post_data or "null"))))
    pg.goto(app_url + "/")
    pg.wait_for_selector("body[data-ready='1']", timeout=60000)
    pg.api = lambda path, body=None: pg.evaluate(
        "([p, b]) => fetch(p, b === null ? {} : {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(b)}).then(r => r.json())", [path, body])
    yield pg
    ctx.close()
