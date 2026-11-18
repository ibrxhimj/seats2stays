"""Step 4.8: packaging. The Dockerfile, the pinned requirements, and the claim behind them: the app runs from the bundle
with only the pinned runtime packages, and offline (everything it shows is vendored). Docker itself is not needed here:
these tests check the Dockerfile's stages, and run what each stage runs."""
import re
import subprocess
import sys
import textwrap
import pytest
from atp import phase4 as p4
from atp.config import load_config

cfg = load_config(); ROOT = cfg["root"]
if not (cfg["paths"]["processed"] / "phase4" / "facts.json").exists():
    pytest.skip("run `python -m atp.phase4 data` first", allow_module_level=True)
DOCKERFILE = (ROOT / "Dockerfile").read_text()


def instructions(text):
    """Dockerfile instructions with line continuations joined, comments dropped."""
    out, cur = [], ""
    for line in text.splitlines():
        if line.lstrip().startswith("#") and not cur:
            continue
        cur += line.rstrip("\\").rstrip() + " "
        if not line.rstrip().endswith("\\"):
            out.append(cur.strip()); cur = ""
    return [i for i in out if i]


INS = instructions(DOCKERFILE)


# ----------------------------------------------------------------------------- the Dockerfile
def test_dockerfile_has_three_stages_on_the_pinned_python():
    froms = [i.split()[1:] for i in INS if i.startswith("FROM ")]
    assert [f[-1] for f in froms] == ["deps", "bundle", "runtime"]
    assert froms[0][0] == "python:3.12-slim" and froms[2][0] == "python:3.12-slim" and froms[1][0] == "deps"


def test_every_copy_source_exists_and_none_is_raw_data():
    for i in INS:
        if not i.startswith("COPY "):
            continue
        parts = i.split()[1:]
        if parts[0].startswith("--from="):
            continue
        for src in [p for p in parts[:-1] if not p.startswith("--")]:
            assert (ROOT / src).exists(), f"COPY source {src} is missing"
            assert not re.search(r"data/(raw|interim)", src) and not src.endswith((".xlsx", ".xls"))
    assert (ROOT / "web" / "index.html").exists()


def test_the_runtime_is_non_root_with_a_health_check_and_the_bundle():
    run = DOCKERFILE.split("AS runtime", 1)[1]
    assert "USER app" in run and "HEALTHCHECK" in run and 'CMD ["python", "-m", "atp.api"]' in run
    assert "ATP_ROOT=/app/bundle" in run and "COPY --from=bundle /out/bundle /app/bundle" in run
    assert "EXPOSE 8000" in run and "PORT=8000" in run
    assert "COPY src /app/src" in run and "COPY web /app/web" in run
    assert "data/processed" not in run                       # the runtime never sees the project's tables, only the bundle


def test_dockerignore_keeps_raw_data_out():
    ig = (ROOT / ".dockerignore").read_text().split()
    assert {"data/raw", "data/interim", ".git", "build"} <= set(ig)


def test_compose_and_makefile_use_the_image():
    comp = (ROOT / "docker-compose.yml").read_text()
    assert "build: ." in comp and '"8000:8000"' in comp
    mk = (ROOT / "Makefile").read_text()
    assert "docker build -t seats2stays ." in mk and "docker run --rm -p 8000:8000 seats2stays" in mk


# ----------------------------------------------------------------------------- the pinned requirements
def test_requirements_are_exactly_pinned():
    for name in ("requirements.txt", "requirements-app.txt"):
        for line in (ROOT / name).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-r "):
                continue
            assert re.fullmatch(r"[A-Za-z0-9_.\-]+==\d+(\.\d+)*", line), f"{name}: {line!r} is not an exact pin"
    assert "-r requirements-app.txt" in (ROOT / "requirements.txt").read_text()
    app = {l.split("==")[0].lower() for l in (ROOT / "requirements-app.txt").read_text().splitlines() if "==" in l}
    assert {"numpy", "pandas", "scipy", "pyarrow", "pyyaml", "fastapi", "uvicorn"} <= app
    assert not app & {"statsmodels", "openpyxl", "hijridate", "pytest", "playwright"}      # build and test tools stay out of the image


# ----------------------------------------------------------------------------- what the bundle stage runs, and the runtime
@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("img") / "bundle"
    p4.build_bundle(out, cfg)
    return out


RUNTIME_ONLY = textwrap.dedent("""
    import sys
    BLOCKED = {"statsmodels", "openpyxl", "hijridate", "tabulate", "sklearn", "pytest", "playwright"}
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"{name} is not in the app image")
    sys.meta_path.insert(0, Block())
    from fastapi.testclient import TestClient
    from atp import api
    from atp.simulator import Simulator
    s = Simulator()
    r = s.run(s.preset("etihad")); assert r["headline"]["text"] == "+37,100", r["headline"]["text"]
    assert s.moves()["levers"]["rows"] and s.know()["series"] and s.reverse("INDIA", "2026-11-01", "2027-02-28", 20000, "narrowbody")
    c = TestClient(api.app)
    for path in ("/api/meta", "/api/moves", "/api/know", "/api/lever?market=INDIA&season=winter", "/api/presets/europe"):
        assert c.get(path).status_code == 200, path
    assert c.post("/api/run", json=s.preset("march")).status_code == 200
    assert c.get("/").status_code == 200 and "Seats2Stays" in c.get("/").text
    print("OK")
""")


def test_the_app_runs_from_the_bundle_with_only_the_runtime_packages(bundle):
    env = {"ATP_ROOT": str(bundle), "PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", RUNTIME_ONLY], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-1500:]
    assert not (bundle / "data" / "raw").exists() and not (bundle / "data" / "interim").exists()      # a bundle is never written to


# ----------------------------------------------------------------------------- offline
def test_every_asset_the_page_references_is_vendored():
    web = ROOT / "web"
    html = (web / "index.html").read_text(); css = (web / "css" / "app.css").read_text()
    for ref in re.findall(r'(?:src|href)="([^"#]+)"', html):
        if not ref.startswith(("http", "data:", "#")):
            assert (web / ref).exists(), ref
    for ref in re.findall(r"url\(([^)]+)\)", css):
        ref = ref.strip("'\" ")
        if not ref.startswith("data:"):
            assert (web / "css" / ref).resolve().exists(), ref
    for js in (web / "js").glob("*.js"):
        assert not re.search(r"https?://(?!www\.w3\.org)", js.read_text()), f"{js.name} calls out to the network"


def test_the_app_makes_no_request_beyond_itself_on_any_tab(page, app_url):
    pg = page.context.new_page(); outside, bad = [], []
    pg.on("request", lambda r: outside.append(r.url) if not r.url.startswith((app_url, "data:", "blob:")) else None)
    pg.on("response", lambda r: bad.append((r.status, r.url)) if r.status >= 400 else None)
    pg.goto(app_url + "/"); pg.wait_for_selector("body[data-ready='1']")
    for t in ("moves", "know", "simulator"):
        pg.click(f'.tabs a[data-tab="{t}"]'); pg.wait_for_timeout(600)
    assert outside == [] and bad == []
    loaded = pg.evaluate("[...document.fonts].filter(f => f.status === 'loaded').map(f => f.family)")
    assert "Barlow" in loaded and "Barlow Condensed" in loaded
