"""Phase 4, Step 4.2 — the simulator's web API (FastAPI). Run: `python -m atp.api` (or `make app`), then open
http://localhost:8000. The data root is the project, or the app bundle when ATP_ROOT points at it (Docker).

  GET  /api/health           ready, and whether the presets are computed
  GET  /api/meta             markets, routes, regions, aircraft, presets, headline facts — everything the forms need
  GET  /api/presets/{id}     a preset's list of changes (etihad, europe, march, new_route)
  POST /api/run              a scenario {"changes": [...], "response": central | low | high} → the full result (chain, sentence, headline with range,
                             occupancy, by market, by month, by day, notes)
  GET  /api/lever?market=&season=&response=   the live seats lever: every slider position for one market and season, each the
                             exact result of the market-wide seats change (closed form, equal to a full run)
  GET  /api/moves            the "What moves guests" tab: lever ranking, market × month grid, planner rules
  GET  /api/know             the "How we know" tab: back-test, the country catch, March 2026 replay, assumptions, limits
  POST /api/reverse          {market, start, end, target, aircraft: today | narrowbody | widebody} → flights a week needed
                             (null when the target is out of reach of what the simulator accepts; target must be > 0)
  POST /api/export           a scenario → the result by market and month as CSV
  GET  /api/data/{name}      precomputed tables for "What moves guests" and "How we know"

A scenario the engine cannot honour returns 422 with a plain message: {"error": "…"}. The web front end (web/) is
served at / when present.
"""
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from .scenario import ScenarioError
from .simulator import Simulator

WEB = Path(os.environ.get("ATP_WEB") or Path(__file__).resolve().parents[2] / "web")
state = {"sim": None, "warm": False}


def _warm():
    state["sim"].warm(); state["warm"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["sim"] = Simulator()                                   # engine and ranges loaded once
    threading.Thread(target=_warm, daemon=True).start()          # presets computed in the background
    yield


app = FastAPI(title="Seats2Stays — flight → hotel demand simulator", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=2000)


@app.exception_handler(ScenarioError)
async def _scenario_error(request: Request, exc: ScenarioError):
    return JSONResponse(status_code=422, content={"error": str(exc)})


def sim() -> Simulator:
    if state["sim"] is None:                                      # e.g. the app used without its lifespan
        state["sim"] = Simulator()
    return state["sim"]


@app.get("/api/health")
def health():
    return {"ok": True, "presets_ready": state["warm"]}


@app.get("/api/meta")
def meta():
    return sim().meta()


@app.get("/api/presets/{pid}")
def preset(pid: str):
    return sim().preset(pid)


@app.post("/api/run")
def run(scenario: dict = Body(...)):
    return sim().run(scenario)


@app.get("/api/lever")
def lever(market: str, season: str = "winter", response: str = "central"):
    return sim().lever(market, season, response)


@app.get("/api/moves")
def moves():
    return sim().moves()


@app.get("/api/know")
def know():
    return sim().know()


@app.post("/api/reverse")
def reverse(q: dict = Body(...)):
    try:
        args = dict(market=str(q["market"]), start=str(q["start"]), end=str(q["end"]), target=float(q["target"]),
                    aircraft=str(q.get("aircraft") or q.get("aircraft_class") or "today"),
                    seats_per_flight=float(q["seats_per_flight"]) if q.get("seats_per_flight") else None)
    except KeyError as e:
        raise ScenarioError(f"reverse needs market, start, end and target (missing {e.args[0]})")
    except (TypeError, ValueError):
        raise ScenarioError("the target (and seats_per_flight, if given) must be numbers")
    return sim().reverse(**args)


@app.post("/api/export")
def export(scenario: dict = Body(...)):
    csv = sim().export_csv(scenario)
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(scenario.get("name", "scenario")))[:60] or "scenario"
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}_by_market_and_month.csv"'})


@app.get("/api/data/{name}")
def data(name: str):
    try:
        return sim().data(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown table {name!r}")


if WEB.is_dir():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
