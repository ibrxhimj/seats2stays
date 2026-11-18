"""Phase 4, Step 4.2 — simulator core: everything the web app shows, computed by the Phase 3 engine.

One `Simulator` holds the engine and the calibrated ranges (loaded once) and answers four questions:

  run(scenario)      the result of a list of changes: the conversion chain (seats → arriving passengers → inbound passengers →
                     check-ins → guest nights) with what leaves or joins it at each step, a plain-words sentence, the
                     headline change with its 80% range, occupancy points, and the change by source market, by month
                     and by day (against the highest level ever observed), plus the engine's flags as plain notes.
  reverse(...)       how many weekly flights from one market a target of extra guest nights needs (fast version of
                     sensitivity.seats_needed: the same lever and Stage B, evaluated in closed form — Stage B is linear
                     in check-ins, so guest nights = Σ_day nights-per-check-in(day) × extra check-ins(day); tested
                     against the exact solver).
  export_csv(...)    the result by market and month, for download.
  meta() / data()    reference lists for the forms (markets, routes, aircraft, presets) and the precomputed tables of the
                     "What moves guests" and "How we know" tabs.

Numbers: model estimates (check-ins, guest nights, ranges) are shown to 3 significant figures, schedule inputs (seats,
passengers, local passengers) exactly — the Phase 3 report's rule. Raw values travel with every display string.
State: the scenario list lives in the browser; the server is stateless apart from caches, and one lock serialises
engine use (the engine is not built for concurrent runs).
"""
import json
import math
import re
import threading
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .config import load_config
from . import bands as bd, engine as en, scenario as sc, sensitivity as se

MINUS = "−"
LABELS = {"UNITED STATES OF AMERICA": "United States", "RUSSIAN FEDERATION": "Russia", "DOMESTIC": "UAE residents"}

PRESETS = [
    {"id": "etihad", "file": "data/reference/scenarios/etihad_2026_launches.json", "label": "Etihad's 2026 launches",
     "blurb": "Etihad's new routes into our markets in 2026, each with a published source."},
    {"id": "europe", "file": "data/reference/scenarios/european_carriers_return.json", "label": "Fewer European flights",
     "blurb": "Three European airlines' announced winter routes do not start on 24–25 October.",
     "note": "Those routes are worth the opposite: about {gain} guest nights by February."},
    {"id": "march", "file": "data/processed/phase3/presets/march_2026_replay_low.json", "label": "March 2026, replayed",
     "blurb": "The March 2026 disruption: Etihad's interim schedules, a fall in demand, and longer stays.",
     "note": "Flights explain {lo}–{hi} of the fall in check-ins, depending on how many seats other airlines flew; "
             "the rest is a direct fall in demand."},
    {"id": "new_route", "label": "A new twice-weekly route",
     "blurb": "The brief's question: a new route from a country that has no flights to Abu Dhabi in the data.",
     "template": {"type": "new_route", "market": "MEXICO", "city": "Mexico City", "airline": "Etihad Airways",
                  "start": "2026-11-01", "end": "2027-02-28", "flights_per_week": 2, "aircraft_class": "widebody"}},
]
# The check-in response (how strongly a country's hotel check-ins follow its inbound passengers: ε, and the per-seat rate
# for countries without flight data) is the chain's one estimated step. The app shows every result at the central
# estimate by default; a planner can look at it at either end of the validated 80% interval. The 80% range stays the
# model's range, taken around the result shown.
RESPONSES = {"central": ("Central", "the central estimate", None),
             "low": ("Low", "the low end of its 80% range", "lo"),
             "high": ("High", "the high end of its 80% range", "hi")}
SEASONS = {"winter": ("2026-11-01", "2027-02-28", "Winter", "Nov 2026 to Feb 2027"),
           "summer": ("2026-06-01", "2026-09-30", "Summer", "Jun to Sep 2026"),
           "year": ("2026-03-01", "2027-02-28", "Whole year", "Mar 2026 to Feb 2027")}
MAX_STEPS = 200                       # slider positions per market and season (step = whole weekly flights)
ROUTE_MAX_FLIGHTS = 100               # the most flights a week the new-route form takes for one route
REVERSE_FROM = 1e-3                   # the reverse sum's search starts at a sliver of a flight a week

DATA = {"levers": "phase3/sens_levers.csv", "market_month": "phase3/sens_market_month.csv",
        "market_ranking": "phase3/sens_market_ranking.csv", "season": "phase3/sens_season.csv",
        "march_regions": "phase3/march_2026_replay_by_region.csv", "backtest_metrics": "phase2/endtoend_metrics.csv",
        "backtest_markets": "phase2/endtoend_per_market.csv", "backtest_monthly": "phase4/backtest_monthly.csv",
        "occupancy": "phase4/occupancy_basis.csv"}


# "What moves guests": the levers in plain words (table name → label, what it is, who it acts on), and the rule behind each
# heat-map cell (the code is the position in CELL_RULES; 0 = the ordinary seat lever, nothing to flag).
LEVERS = [("non-flight demand", "Demand not caused by flights", "Advisories, events, regional conditions; one-for-one by definition", "all guests"),
          ("stay length", "Length of stay", "Nights per check-in; one-for-one by definition", "all guests"),
          ("transfer share", "Passengers connecting onwards", "More connecting means fewer hotel guests", "flights"),
          ("seats", "Seats", "More or fewer flights, or bigger planes", "flights"),
          ("load factor", "How full planes are", "Passengers per seat", "flights")]
CELL_RULES = ["seat lever", "beyond trusted range (straight line)", "capped by local passengers",
              "from full cancellation (no baseline seats)", "new-route lever (no flight data)"]
CELL_RULE_TEXT = [None,
                  "Beyond the range the seat lever was tested on: extended in a straight line, not estimated.",
                  "Capped: check-ins cannot rise by more than the extra inbound passengers.",
                  "No flights this month in the normal year: the answer starts from a full cancellation.",
                  "No flights from this country in the data: the hotel response is borrowed from other markets."]


# ----------------------------------------------------------------------------- formatting
def label(market: str) -> str:
    return LABELS.get(market, market.title())


def sig3(x: float) -> float:
    """Round to 3 significant figures (model estimates)."""
    if not x or not math.isfinite(x):
        return 0.0
    return round(x, 2 - int(math.floor(math.log10(abs(x)))))


def _sign(s: str, v: float, signed: bool) -> str:
    s = s.replace("-", MINUS)
    return ("+" + s) if signed and v > 0 else s


def est(x: float, signed: bool = False) -> str:
    """A model estimate: 3 significant figures (37,117 → 37,100; 2.8543 → 2.85)."""
    v = sig3(x)
    if v == 0:
        return "0"
    dec = 0 if abs(v) >= 100 else 2 - int(math.floor(math.log10(abs(v))))
    return _sign(f"{v:,.{dec}f}", v, signed)


def exact(x: float, signed: bool = False) -> str:
    """A schedule input (seats, passengers): exact, whole numbers."""
    v = int(round(x))
    return _sign(f"{v:,}", v, signed)


def pct(share: float, dec: int = 0) -> str:
    """A share as a percentage; never shows 0% or 100% unless it is exactly that (99.6%, not 100%)."""
    v = share * 100
    s = f"{v:.{dec}f}"
    while float(s) in (0.0, 100.0) and v not in (0.0, 100.0) and dec < 3:
        dec += 1; s = f"{v:.{dec}f}"
    return s.replace("-", MINUS) + "%"


def flights(x: float) -> str:
    """Weekly flights are whole: 37.2 → "37"; below one a week → "under 1"."""
    return "under 1" if x < 0.5 else f"{round(x):,}"


def month_label(m: str) -> str:
    return pd.Period(m, "M").strftime("%b %Y")


def compact(o, sig: int = 6):
    """For transport: floats to `sig` significant figures (display texts are made before this)."""
    if isinstance(o, dict):
        return {k: compact(v, sig) for k, v in o.items()}
    if isinstance(o, list):
        return [compact(v, sig) for v in o]
    if isinstance(o, float) and o != 0 and math.isfinite(o):
        return round(o, sig - 1 - int(math.floor(math.log10(abs(o)))))
    return o


def clean(o):
    """JSON-safe: numpy scalars to Python, NaN/inf to None, DataFrames not allowed."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


# ----------------------------------------------------------------------------- flags → plain notes
def _markets_in(flag: str) -> str:
    """The market list an engine flag ends with, in display names ("United States Of America…" → "United States and others")."""
    m = re.search(r"\(([^()]*)\)\s*$", flag)
    if not m:
        return ""
    names = [x.strip() for x in m.group(1).split(",") if x.strip()]
    more = any(x.endswith("…") for x in names)
    out = ", ".join(label(x.rstrip("…").upper()) for x in names)
    return out + (" and others" if more else "")


NOTE_RULES = [  # (id, level, pattern, text or function of the matching flags)
    ("empty", None, r"^empty scenario", None),
    ("capacity", "warning", r"in-house guests above the highest level ever observed", None),       # built in _notes
    ("double_count", "warning", r"observed-level preset combined",
     "This combines an observed month with flight changes over the same days: the observed level already includes "
     "that month's flight cuts, so they would count twice."),
    ("no_effect", "warning", r"no seats in the selected markets in this window",
     "A change has no effect: there are no seats in that market and window."),
    ("beyond_validation", "info", r"more than 7 months after the data|reaches beyond 7 months after the data end", None),
    ("beyond_range", "info", r"change outside the range the lever was estimated on",
     "Some changes go beyond the tested range of inbound-passenger changes (−29% to +109%): the effect there is "
     "extended, not estimated, so the range is wider."),
    ("cap", "info", r"limited to the change in local passengers \(cap\)",
     lambda fl: "Capped: check-ins cannot rise by more than the extra inbound passengers"
                + (f" ({_markets_in(fl[0])})." if _markets_in(fl[0]) else ".")),
    ("zero_baseline", "info", r"market without baseline seats",
     lambda fl: "Some markets have no flights in the normal year on these days: the effect is measured from a full "
                "cancellation of their usual service" + (f" ({_markets_in(fl[0])})." if _markets_in(fl[0]) else ".")),
    ("no_flight_data", "info", r"has no flight data — new-route lever only|nationality without flight data: new-route lever", None),
    ("new_route_mix", "info", r"^new_route .* taken from",
     "New routes take their load factor and share of passengers ending their trip here from the airline's own routes in that market, "
     "else its Abu Dhabi network, else the market."),
    ("aircraft", "info", r"aircraft not stated by the source",
     "Where a source names no aircraft, typical seats are assumed (narrowbody 174, widebody 302)."),
    ("assumption", "info", r"assumption lever",
     "Demand and stay-length changes are assumptions you set, not estimated from flight data: their range covers only "
     "the uncertainty of the normal year."),
    ("short", "info", r"shorter than 28 days",
     "A change lasts less than four weeks: the seat lever was measured on sustained, month-long changes, so read the "
     "effect as indicative."),
    ("unchanged", "info", r"have no flight data and are left unchanged",
     "Some markets in your selection have no flights in the data, so flight changes leave them unchanged."),
    ("lf_cap", "info", r"capped at full planes", "Load factor is capped at full planes on some routes."),
    ("tr_cap", "info", r"capped at 99% connecting|capped just below 100% connecting",
     "Some routes reach just below 100% connecting; the market's other routes take up the rest, so the market share is "
     "the one you asked for."),
    ("tr_miss", "warning", r"transfer_share: target not reached",
     lambda fl: "The connecting share asked for cannot be reached on some days: routes with no connecting passengers "
                "cannot connect. " + (fl[0].split("; ", 1)[1].rstrip(")").capitalize() + "." if "; " in fl[0] else "")),
]


# ----------------------------------------------------------------------------- the simulator
class Simulator:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        P = self.cfg["paths"]["processed"]; self.root = self.cfg["root"]
        self.E = en.Engine(self.cfg); self.B = bd.Bands(self.cfg); self.w = self.E.world
        self.facts = json.load(open(P / "phase4" / "facts.json", encoding="utf-8"))
        self.occ = pd.read_csv(P / "phase4" / "occupancy_basis.csv").set_index("month")
        self._occ_point = self.occ.dct_guest_nights_per_occupancy_point.to_dict()
        reg = pd.read_csv(self.cfg["paths"]["reference"] / "scad_region_map.csv")
        self.region = reg.dropna(subset=["scad_region"]).set_index("nationality").scad_region.to_dict()
        pl = pd.read_csv(self.cfg["paths"]["reference"] / "nationality_places.csv").set_index("nationality")
        self.place = {m: {"place": r.place, "lat": float(r.lat), "lon": float(r.lon)} for m, r in pl.iterrows()}
        self._lock = threading.Lock(); self._cache = OrderedDict(); self._levers = {}; self._moves = None; self._know = None
        self._base = None                                                    # normal-year inputs by market and day
        self.presets = {p["id"]: p for p in PRESETS}
        self._preset_of = {self._key(self.preset(p["id"])["changes"]): p["id"] for p in PRESETS if "file" in p}

    @staticmethod
    def _key(changes) -> str:
        """Canonical key: 1.0 and 1 are the same number (a browser sends 1), so a scenario sent back from JSON
        finds its cached result and its preset."""
        def canon(o):
            if isinstance(o, float) and o.is_integer():
                return int(o)
            if isinstance(o, dict):
                return {k: canon(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [canon(v) for v in o]
            return o
        return json.dumps(canon(changes), sort_keys=True, default=str)

    def fact(self, k):
        return self.facts[k]["value"]

    # ------------------------------------------------------------------ reference lists
    def meta(self) -> dict:
        w = self.w; r = w.routes[w.routes.seats > 0]
        routes = {}
        for (m, city, airline), g in r.groupby(["nationality_key", "city", "airline"]):
            routes.setdefault(m, []).append({
                "city": city, "airline": airline, "months": sorted(int(x) for x in g.moy),
                "flights_per_week": float(np.average(g.flights_per_week, weights=g.seats)),
                "seats_per_flight": float(np.average(g.seats_per_flight, weights=g.seats))})
        markets = [{"key": m, "label": label(m), "region": self.region.get(m), "has_flight_data": m in w.flight, **self.place[m],
                    "routes": sorted(routes.get(m, []), key=lambda x: -x["flights_per_week"])} for m in w.markets]
        presets = []
        for p in PRESETS:
            q = {k: p[k] for k in ("id", "label", "blurb")}
            if "file" in p:
                q["changes"] = len(self.preset(p["id"])["changes"])
            presets.append(q)
        a = r.groupby("airline")[["seats", "pax", "p2p"]].sum()                # what a new route of each airline would inherit
        airlines = [{"name": n, "load_factor": float(v.pax / v.seats), "local_share": float(v.p2p / v.pax)}
                    for n, v in a.iterrows() if v.seats > 0 and v.pax > 0]
        network = self._network()
        return clean({"horizon": [str(w.h0.date()), str(w.h1.date())], "data_end": str(w.data_end.date()),
                      "airlines": airlines, "network": network,
                      "beyond_validation_from": self.fact("beyond_validation_from")[:7],
                      "markets": markets, "regions": sorted(w.regions), "aircraft": w.aircraft,
                      "groups": ["all", "international", "flight markets"], "presets": presets,
                      "peak_guests": self.E.params.peak_guests, "peak_date": self.fact("peak_date"),
                      "abu_dhabi": {"lat": 24.45, "lon": 54.38},
                      "seasons": [{"id": k, "label": v[2], "span": v[3], "window": [v[0], v[1]]} for k, v in SEASONS.items()],
                      "responses": [{"id": k, "label": v[0], "text": v[1], **self.response_params(k)} for k, v in RESPONSES.items()],
                      "facts": {k: {"value": v["value"], "label": v["label"]} for k, v in self.facts.items()}})

    def _network(self) -> dict:
        """The whole network today: passengers per seat, and the share of passengers who connect onwards."""
        r = self.w.routes[self.w.routes.seats > 0]
        return {"load_factor": float(r.pax.sum() / r.seats.sum()),
                "transfer_share": float((r.transfer.sum() + r.transit.sum()) / r.pax.sum())}

    def preset(self, pid: str) -> dict:
        p = self.presets.get(pid)
        if p is None:
            raise sc.ScenarioError(f"unknown preset {pid!r}")
        if "template" in p:
            return {"name": p["id"], "title": p["label"], "changes": [dict(p["template"])]}
        s = sc.load(self.root / p["file"])
        return {"name": p["id"], "title": p["label"], "changes": s["changes"]}

    def data(self, name: str) -> list:
        if name not in DATA:
            raise KeyError(name)
        d = pd.read_csv(self.cfg["paths"]["processed"] / DATA[name])
        if "market" in d:
            d.insert(d.columns.get_loc("market") + 1, "label", d.market.map(label))
        return clean(d.to_dict("records"))

    # ------------------------------------------------------------------ "What moves guests"
    def moves(self) -> dict:
        """Everything the "What moves guests" tab shows, from the Phase 3 sensitivity tables and the headline facts (the
        same files the bundle carries; nothing is re-run): the lever ranking (each lever moved ±10% over the year, with
        its 80% range), the market × month grid of guest nights per 1,000 added seats (one extra daily narrowbody), and
        four planner rules whose numbers are taken from those tables."""
        with self._lock:
            if self._moves is None:
                self._moves = self._build_moves()
            return self._moves

    def _build_moves(self) -> dict:
        P = self.cfg["paths"]["processed"] / "phase3"
        lv = pd.read_csv(P / "sens_levers.csv"); mm = pd.read_csv(P / "sens_market_month.csv")
        rk = pd.read_csv(P / "sens_market_ranking.csv"); se = pd.read_csv(P / "sens_season.csv")
        # ---- the levers, ranked by what a 1% change is worth (ties by name, as the Phase 3 report ranks them)
        elas = lv.groupby("lever").elasticity_small_change.apply(lambda x: float(x.abs().mean()))
        order = sorted(LEVERS, key=lambda t: (-round(elas[t[0]], 6), t[0]))
        up_seats = lv[(lv.lever == "seats") & (lv.change_pct > 0)].iloc[0]
        base = float(up_seats.d_guest_nights / (up_seats.guest_nights_pct / 100))
        rows = []
        for key, name, what, acts in order:
            sides = {}
            for sd, sign in (("down", -1), ("up", 1)):
                r = lv[(lv.lever == key) & (lv.change_pct * sign > 0)].iloc[0]
                note = None
                if key == "load factor" and isinstance(r.capped_or_flagged, str) and "capped at full planes" in r.capped_or_flagged:
                    note = (f"Planes cannot be more than full, so a {r.change_pct:+.0f}% change in how full they are reaches only "
                            f"{pct(r.local_passengers_pct / 100, 1)} more passengers.")
                elif r.pct_of_checkin_change_beyond_trusted_range >= 1:
                    note = (f"{pct(r.pct_of_checkin_change_beyond_trusted_range / 100)} of this change lies beyond the range the lever "
                            f"was tested on: read it as direction and size, not a forecast.")
                sides[sd] = {"change_pct": float(r.change_pct), "d_guest_nights": float(r.d_guest_nights), "lo": float(r.lo),
                             "hi": float(r.hi), "pct": float(r.guest_nights_pct), "note": note}
            rows.append({"id": key, "name": name, "what": what, "acts": acts, "elasticity": round(elas[key], 2),
                         "opposite": bool((lv[lv.lever == key].elasticity_small_change < 0).all()), **sides})
        fl, dom = self.fact("share_flight_markets"), self.fact("share_domestic")
        el = {r["id"]: r["elasticity"] for r in rows}
        # ---- the grid: one extra daily narrowbody, one month at a time
        months = sorted(mm.month.unique())
        seats = mm.groupby("month").seats_added.first().loc[months]
        cells_of = {m: g.set_index("month").loc[months] for m, g in mm.groupby("market")}
        grid = []
        for r in rk.itertuples():
            g = cells_of[r.market]
            cells = [[round(float(c.guest_nights_per_1000_seats), 1), round(float(c.lo), 1), round(float(c.hi), 1),
                      round(float(c.d_guest_nights), 1), CELL_RULES.index(c.rule)] for c in g.itertuples()]    # per 1,000 seats; range and total in guest nights
            grid.append({"market": r.market, "label": label(r.market), "flights": r.market in self.w.flight,
                         "year": [round(float(r.guest_nights_per_1000_seats), 1), round(float(r.lo), 1), round(float(r.hi), 1), CELL_RULES.index(r.rule)],
                         "cells": cells})
        grid.sort(key=lambda g: (not g["flights"], -g["year"][0]))
        season = [round(float(v), 1) for v in se.set_index("month").guest_nights_per_1000_seats.loc[months]]
        # ---- the planner rules (numbers from the tables above)
        ok = [g for g in grid if g["flights"] and g["year"][3] == 0]
        top, low = ok[0], ok[-1]
        hi_m, lo_m = int(np.argmax(season)), int(np.argmin(season))
        net = self._network()
        off = {k: self.fact(k) for k in ("offset_10", "offset_10_lo", "offset_10_hi")}
        ratio = lambda a, b: f"{a / b:.2g}"
        per1000 = lambda v: f"{sig3(v):,.0f}"
        for r_ in rows:
            if r_["id"] == "transfer share":                  # a relative change: say what it is on today's average
                r_["what"] += f". +10% means, on average, {pct(net['transfer_share'])} → {pct(net['transfer_share'] * 1.1)} of passengers"
        beyond = [g for g in grid if g["flights"] and g["year"][3] == 1 and g["year"][0] > top["year"][0]]    # ranked above it, marked † in the table
        named = [f"{g['label']} ({per1000(g['year'][0])})" for g in beyond]
        above = (f" The table also ranks {', '.join(named[:-1]) + ' and ' if len(named) > 1 else ''}{named[-1]} higher, marked †: a straight-line extension "
                 f"beyond the tested range, to be read as direction and size, not a forecast.") if beyond else ""
        lo_t, hi_t = self.E.params.lo, self.E.params.hi
        rules = [
            {"id": "where", "title": "Rank routes by hotel nights per seat, not by seats.",
             "evidence": f"One extra daily narrowbody from {top['label']} brings about {per1000(top['year'][0])} guest nights per 1,000 seats, the most of any "
                         f"country within the model's tested range; from {low['label']}, {per1000(low['year'][0])}. A seat's worth differs about "
                         f"{round(top['year'][0] / low['year'][0])}-fold between markets.{above}"},
            {"id": "when", "title": "Time new capacity for the strong months.",
             "evidence": f"A seat added in {pd.Period(months[hi_m]).strftime('%B')} brings about {per1000(season[hi_m])} guest nights per 1,000 seats; "
                         f"one added in {pd.Period(months[lo_m]).strftime('%B')}, {per1000(season[lo_m])} ({pct(1 - season[lo_m] / season[hi_m])} less)."},
            {"id": "who", "title": "Ask airlines how many passengers stay, not only how many seats.",
             "evidence": f"{pct(net['transfer_share'])} of passengers connect onwards at Abu Dhabi on average. One percent fewer connecting passengers "
                         f"is worth {el['transfer share']:.2f}% more guest nights, {ratio(el['transfer share'], el['seats'])} times what 1% more seats is worth ({el['seats']:.2f}%)."},
            {"id": "beyond", "title": "Work on demand and length of stay as hard as on flights.",
             "evidence": f"One percent more demand, or longer stays, is worth {el['stay length']:.2f}% more guest nights, {ratio(el['stay length'], el['seats'])} times "
                         f"what 1% more seats is worth. Making up a 10% fall in demand with seats alone would take about {pct(off['offset_10'])} more seats "
                         f"(80% range {pct(off['offset_10_lo'])} to {pct(off['offset_10_hi'])})."}]
        lead = (f"Of the flight levers, connecting passengers matter most: 1% fewer is {el['transfer share']:.2f}% more guest nights, "
                f"{ratio(el['transfer share'], el['seats'])} times what 1% more seats gives ({el['seats']:.2f}%). Demand and length of stay "
                f"move guest nights one for one by definition: making up a 10% fall in demand with seats alone would take about {pct(off['offset_10'])} more seats.")
        m0, m1 = months[0], months[-1]
        return compact(clean({
            "span": f"{month_label(m0)} to {month_label(m1)}", "lead": lead,
            "levers": {"base_guest_nights": base, "rows": rows,
                       "scope": f"Seats, how full planes are and connecting passengers act only through flights. Nationalities with flights make up "
                                f"{pct(fl)} of guest nights; UAE residents {pct(dom)}."},
            "grid": {"cell": ["per_1000_seats", "range_low", "range_high", "guest_nights", "rule"], "year": ["per_1000_seats", "range_low", "range_high", "rule"],
                     "months": months, "month_labels": [month_label(m) for m in months], "seats_added": [round(float(v), 1) for v in seats],
                     "seats_per_flight": float(self.w.aircraft["narrowbody"]), "season": season, "rows": grid, "rules": CELL_RULE_TEXT[1:],
                     "dagger": f"† Beyond the tested range. In most months one extra daily flight is a bigger increase for this country than the model was "
                               f"tested on ({pct(lo_t)} to +{pct(hi_t)}), so its figure is extended in a straight line, not estimated: "
                               f"read it as direction and size, not a forecast."},
            "rules": rules}), 7)

    # ------------------------------------------------------------------ "How we know"
    def know(self) -> dict:
        """Everything the "How we know" tab shows, from the frozen Phase 2/3 tables and the headline facts (nothing is re-run):
        the back-test of guest nights rebuilt from flights, the calendar and normal stays against what hotels reported
        (two held-out periods, with the benchmark and the 80% range), the passport-versus-departure-country evidence,
        the March 2026 replay by world region, every assumption of the chain, and the limits."""
        with self._lock:
            if self._know is None:
                self._know = self._build_know()
            return self._know

    def _build_know(self) -> dict:
        P, F = self.cfg["paths"]["processed"], self.fact
        bm = pd.read_csv(P / "phase4" / "backtest_monthly.csv")
        tier = pd.read_csv(P / "phase3" / "market_tiers.csv").set_index("market").tier
        lb = pd.read_csv(P / "phase3" / "level_bands.csv"); lb = lb[lb.kind == "guests"].set_index(["level", "tier"])
        wm = pd.read_csv(P / "phase2" / "endtoend_per_market.csv").set_index(["fold", "market"]).e2e_wmape_monthly
        mr = pd.read_csv(P / "phase3" / "march_2026_replay_by_region.csv")
        pv = pd.read_csv(P / "phase4" / "passport_vs_flight.csv").set_index("market")
        w = lambda v: f"{v:.1f}%"                                                    # an error, in percent
        two = lambda a, b: f"{a} · {b}"
        # ---- the two held-out periods
        folds = []
        for f in ("A", "B"):
            ms = sorted(bm[bm.fold == f].month.unique())
            folds.append({"id": f, "months": ms, "label": f"{month_label(ms[0])} – {month_label(ms[-1])}",
                          "trained_to": month_label(str(self.cfg["folds"][f]["train_end"])[:7])})
        # ---- actual against rebuilt, with the 80% range (actual ÷ rebuilt, by market size; the total's is the full range seen)
        def rows(g, lo, hi):
            g = g.sort_values("month")
            return [[round(float(a), 1), round(float(r), 1), round(float(r) * lo, 1), round(float(r) * hi, 1)]
                    for a, r in zip(g.actual_guest_nights, g.rebuilt_guest_nights)]
        tot = bm[bm.market != "DOMESTIC"].groupby(["fold", "month"], as_index=False)[["actual_guest_nights", "rebuilt_guest_nights"]].sum()
        tl, th = lb.loc[("total-month", "all markets")][["ratio_lo", "ratio_hi"]]
        series = [{"key": "ALL", "label": "All international guests", "flights": None,
                   "error": [F("backtest_A_monthly_total"), F("backtest_B_monthly_total")],
                   "range_note": "The range for the total is the widest gap seen in these 14 months: seven months a period is too few to give an 80% range.",
                   "rows": {f["id"]: rows(tot[tot.fold == f["id"]], tl, th) for f in folds}}]
        for m in sorted(bm.market.unique(), key=label):
            lo, hi = lb.loc[("market-month", tier[m])][["ratio_lo", "ratio_hi"]]
            series.append({"key": m, "label": label(m), "flights": None if m == "DOMESTIC" else m in self.w.flight, "error": [float(wm[(f["id"], m)]) for f in folds],
                           "range_note": None, "rows": {f["id"]: rows(bm[(bm.fold == f["id"]) & (bm.market == m)], lo, hi) for f in folds}})
        # ---- the stat row and the headline
        a_t, b_t, a_m, b_m = (F(f"backtest_{x}") for x in ("A_monthly_total", "B_monthly_total", "A_monthly_market", "B_monthly_market"))
        ba_t, bb_t, ba_m, bb_m = (F(f"benchmark_{x}") for x in ("A_monthly_total", "B_monthly_total", "A_monthly_market", "B_monthly_market"))
        skill, naive, n_c = F("lever_check_skill"), F("one_for_one_skill"), F("lever_check_countries")
        sg = lambda v: f"{v * 100:+.0f}%".replace("-", MINUS)
        stats = [
            {"id": "total", "title": "Monthly international guest nights", "big": two(w(a_t), w(b_t)),
             "versus": f"Same day last year: {two(w(ba_t), w(bb_t))}",
             "note": "How far the rebuilt month is from the reported one (WMAPE), on months the model never saw."},
            {"id": "country", "title": "One country at a time, month by month", "big": two(w(a_m), w(b_m)),
             "versus": f"Same day last year: {two(w(ba_m), w(bb_m))}",
             "note": "Better than repeating last year, but not by much: this is why every country's figure carries a wider range."},
            {"id": "lever", "title": "Seat changes → check-ins, on unseen months", "big": f"{F('lever_check_slope'):.2f}",
             "versus": f"95% range {F('lever_check_slope_lo'):.2f} to {F('lever_check_slope_hi'):.2f}, {n_c} countries",
             "note": f"Predicted against actual change (1.00 is the right size). It removes {pct(skill)} of the squared error of "
                     f"“every country grows alike”; “one passenger = one guest” adds {pct(-naive)}."},
            {"id": "range", "title": "Actual inside the 80% range", "big": two(pct(F("coverage_B_to_A")), pct(F("coverage_A_to_B"))),
             "versus": "Market-months, range taken from the other period",
             "note": "The share of country-months where what hotels reported fell inside the range we show. Its width was set "
                     "so that both periods reach 80%; a third period would be needed to confirm it."}]
        lead = (f"Trained on earlier years, then given only the flights that actually operated and the calendar, never the hotel figures it was "
                f"predicting, the simulator "
                f"rebuilt monthly international guest nights within {w(a_t)} and {w(b_t)} of what hotels reported, in two {len(folds[0]['months'])}-month "
                f"periods it had not seen. One country at a time it is about {a_m:.0f}% off, a little better than repeating last year "
                f"({ba_m:.0f}%, {bb_m:.0f}%), so country figures carry wider ranges.")
        # ---- the country catch
        nd, ed = pv.national_day, pv.eid
        expat = ["PHILIPPINES", "INDIA", "PAKISTAN", "EGYPT"]
        catch = {"lead": "Flights count the country a plane left from; hotels count the guest's passport. They are not the same people, so a passenger is never treated as a guest.",
                 "rows": [{"label": label(m), "national_day": float(nd[m]), "eid": float(ed[m])} for m in pv.index],
                 "points": [
                     f"UAE residents of every passport form one group, “UAE residents” ({F('domestic_share_pct'):.0f}% of check-ins). On National Day, a holiday for residents only, "
                     f"that group rises ×{nd['DOMESTIC']:.2f} while the Philippines, India, Pakistan and Egypt fall (×{nd[expat].min():.2f} to ×{nd[expat].max():.2f}): residents are not counted under their passport. "
                     f"No flight change moves them.",
                     f"On Eid, a holiday across the Gulf, the same passports surge (Philippines ×{ed['PHILIPPINES']:.2f}, Saudi Arabia ×{ed['SAUDI ARABIA']:.2f}) while the United Kingdom "
                     f"(×{ed['UNITED KINGDOM']:.2f}) and Germany (×{ed['GERMANY']:.2f}) do not: many guests of one passport arrive from elsewhere in the Gulf, not from the home country.",
                     f"So flights are joined to guests by country name, and the link is learned as a market response across {F('countries_estimated')} countries, "
                     f"not as a head count. Cancel all of a country's flights and about {pct(F('full_cancellation_keeps'))} of its check-ins still come."]}
        # ---- March 2026: what flights alone would have done, by world region
        short = {"UAE residents (Domestic)": "UAE residents"}
        march = {"observed": F("march_checkins_vs_normal"), "share": [F("march_flights_share_lo"), F("march_flights_share_hi")],
                 "lead": (f"In March 2026 hotel check-ins fell to {pct(F('march_checkins_vs_normal'))} of a normal March. Flights alone explain "
                          f"{pct(F('march_flights_share_lo'))} to {pct(F('march_flights_share_hi'))} of that fall; the rest is a direct fall in demand, which is a separate lever."),
                 "rows": [{"region": short.get(r.region, r.region.replace("excluding", "excl.")), "markets": int(r.markets), "flight_markets": int(r.flight_markets),
                           "normal_checkins": float(r.normal_checkins), "observed": float(r.observed_vs_normal),
                           "flights": [float(r.flights_only_low_seats), float(r.flights_only_high_seats)] if r.flight_markets else None}
                          for r in mr.sort_values("normal_checkins", ascending=False).itertuples()]}
        # ---- every assumption of the chain
        net = self._network(); eps, e_lo, e_hi = F("eps"), F("eps_lo"), F("eps_hi")
        up = lambda e: f"+{(1.1 ** e - 1) * 100:.1f}%"
        assumptions = [
            {"step": "Seats", "what": "Each route keeps last year's schedule for the same calendar month, unless the data show it ended or is new",
             "value": "Last year's schedule", "source": "Flight schedules at Abu Dhabi airport", "change": "Add, cancel or resize any route, or open a new one"},
            {"step": "Arriving passengers", "what": "Passengers per seat on each route; a plane cannot fill beyond its seats",
             "value": f"{pct(net['load_factor'])} on average", "source": "Passenger counts by route", "change": "“How full planes are”, or per new route"},
            {"step": "Inbound passengers", "what": "Passengers who connect onwards (transfer and transit) never reach a hotel",
             "value": f"{pct(net['transfer_share'])} connect onwards on average", "source": "Passenger counts by route", "change": "“Passengers connecting onwards”"},
            {"step": "Hotel check-ins", "what": "How a country's check-ins respond when its inbound passengers change (ε)",
             "value": f"10% more passengers: {up(eps)} check-ins (80% range {up(e_lo)} to {up(e_hi)})",
             "source": f"Estimated across {F('countries_estimated')} countries on monthly changes; tested from {pct(F('trusted_lo'))} to +{pct(F('trusted_hi'))}",
             "change": "Shown as the 80% range on every result"},
            {"step": "Countries without flights", "what": "Check-ins per added seat where the data hold no flights from that country",
             "value": f"{F('new_route_per_seat'):.3f} (80% range {F('new_route_per_seat_lo'):.3f} to {F('new_route_per_seat_hi'):.3f})",
             "source": "New routes in other countries", "change": "Any country and airline; load factor and local share if known"},
            {"step": "Guest nights", "what": "Nights per check-in: each nationality's own stay, by season",
             "value": "Own stay by nationality and season", "source": "Hotel history, 2022 to Feb 2026", "change": "“Demand and stays”: longer or shorter stays"},
            {"step": "Aircraft", "what": "Seats per flight for a new route or a bigger plane",
             "value": f"{F('narrowbody_seats'):.0f} narrowbody · {F('widebody_seats'):.0f} widebody", "source": "Aircraft in the schedule data", "change": "Narrowbody or widebody on any route"},
            {"step": "Normal year", "what": "The comparison: March 2026 to February 2027 on last year's schedules, without the March 2026 disruption",
             "value": "Flagged from Oct 2026 (over 7 months past the data)", "source": "Frozen on the data to February 2026", "change": "“Demand and stays” for any shock"},
            {"step": "Occupancy", "what": "Guest nights are put on the hotel statistics' basis, then divided by room-nights",
             "value": f"{F('rooms'):,.0f} rooms · {F('guests_per_room_lo'):.2f} to {F('guests_per_room_hi'):.2f} guests a room · DCT nights {pct(F('dct_over_scad_mean'))} of SCAD's",
             "source": "SCAD hotel statistics", "change": "Fixed"}]
        # ---- the limits: the one list of Step 5.9 (reports/limitations.md), grouped
        from . import limits as lm
        limits = lm.tab3(json.load(open(P / "phase5" / "limitations.json", encoding="utf-8")))
        # ---- the Sobol check (Step 5.6): the levers moved together, and which uncertainty dominates a result
        from . import sobol as so
        sobol = so.tab3(json.load(open(P / "phase5" / "sobol.json", encoding="utf-8")), {k: n for k, n, _, _ in LEVERS})
        # ---- accuracy by months ahead (Step 5.7)
        from . import ahead as ah
        ahead = ah.tab3(self._json_int_keys(json.load(open(P / "phase5" / "ahead.json", encoding="utf-8"))), folds)
        return compact(clean({"lead": lead, "folds": folds, "stats": stats, "series": series, "catch": catch, "march": march, "ahead": ahead, "sobol": sobol,
                              "assumptions": assumptions, "limits": limits}), 7)

    @staticmethod
    def _json_int_keys(o):
        """JSON object keys are strings; the months-ahead tables are keyed by integer."""
        return json.loads(json.dumps(o), object_pairs_hook=lambda p: {(int(k) if k.isdigit() else k): v for k, v in p})

    # ------------------------------------------------------------------ a scenario
    def _check(self, scenario: dict) -> dict:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("changes", []), list):
            raise sc.ScenarioError("a scenario is {\"changes\": [ … ]}")
        if not all(isinstance(c, dict) for c in scenario.get("changes", [])):
            raise sc.ScenarioError("each change is an object, e.g. {\"type\": \"route\", …}")
        resp = scenario.get("response", "central")
        if resp not in RESPONSES:
            raise sc.ScenarioError(f"response must be one of {', '.join(RESPONSES)}")
        return {"name": str(scenario.get("name", "scenario")), "changes": scenario.get("changes", []), "response": resp}

    def response_params(self, resp: str = "central") -> dict:
        """ε and the new-route rate the result is shown at."""
        e, end = self.B.effect, RESPONSES[resp][2]
        return {"eps": e[f"eps_{end}"] if end else e["eps"], "per_seat": e[f"per_seat_{end}"] if end else e["per_seat"]}

    def run(self, scenario: dict) -> dict:
        s = self._check(scenario)
        key = self._key({"changes": s["changes"], "response": s["response"]})
        with self._lock:
            if key not in self._cache:
                r = self.B.run(self.E, s, **self.response_params(s["response"]))
                self._cache[key] = self._payload(r, s)
                while len(self._cache) > 16:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            out = dict(self._cache[key])
        out["name"] = s["name"]
        out["preset"] = self._preset_of.get(self._key(s["changes"]))       # the note applies only to the preset as published
        out["note"] = self._preset_note(out["preset"], out)
        return out

    def warm(self, levers: bool = True):
        """Compute the presets once, then every flight market's winter lever (called at start-up, in the background),
        so a preset click or a first drag answers at once."""
        for p in PRESETS:
            if "file" in p:
                self.run(self.preset(p["id"]))
        if levers:
            for m in self.w.flight:
                try:
                    self.lever(m, "winter")
                except sc.ScenarioError:
                    pass

    # ------------------------------------------------------------------ the live lever
    def baseline_inputs(self) -> pd.DataFrame:
        if self._base is None:
            self._base = self.E.resolve({"changes": []})[0]
        return self._base

    def lever(self, market: str, season: str, response: str = "central") -> dict:
        """The live seats lever for one market and season: every slider position (whole weekly flights, spread over the
        market's current routes) with the result as the full engine would give it — the market-wide `route` change
        `seats_pct`, in closed form (same lever, cap, ranges and Stage B; equal to a full run, tested)."""
        if season not in SEASONS:
            raise sc.ScenarioError(f"unknown season {season!r} ({', '.join(SEASONS)})")
        if response not in RESPONSES:
            raise sc.ScenarioError(f"response must be one of {', '.join(RESPONSES)}")
        key = (market, season, response)
        with self._lock:
            if key not in self._levers:
                lv = _Lever(self, market, *SEASONS[season][:2], eps=self.response_params(response)["eps"])
                self._levers[key] = compact(clean({**lv.curve(season), "response": response}))
            return self._levers[key]

    def _preset_note(self, pid, out) -> str | None:
        p = self.presets.get(pid) if pid else None
        if not p or "note" not in p:
            return None
        if pid == "europe":
            return p["note"].format(gain=est(-out["headline"]["d_guest_nights"], signed=True))
        if pid == "march":
            return p["note"].format(lo=pct(self.fact("march_flights_share_lo")), hi=pct(self.fact("march_flights_share_hi")))
        return p["note"]

    def _payload(self, r: en.Result, s: dict) -> dict:
        S, eb, d = r.summary, r.effect_band, r.daily
        seats, pax, local = S["d_seats"], S["d_passengers"], S["d_local_passengers"]
        ckf, ckd, ck, gn = S["d_checkins_flights"], S["d_checkins_demand"], S["d_checkins"], S["d_guest_nights"]
        lo, hi = eb["summary"]["d_guest_nights"]
        gross = float((d.seats_scen - d.seats_base).abs().sum())
        mixed = (gross - abs(seats)) / 2 > 0.05 * gross                   # over 5% of the seats moved the other way
        stays = [c for c in s["changes"] if c.get("type") == "stay_length"]
        flights = any(abs(x) > 0.5 for x in (seats, pax, local))
        months = self._by_month(r)
        head = self._headline(gn, lo, hi, ck, *eb["summary"]["d_checkins"], S["pct_guest_nights"],
                              S["d_guest_nights_after_horizon"], months)
        rp = self.response_params(s["response"]); lab = RESPONSES[s["response"]]
        return clean({
            "empty": not s["changes"],
            "response": {"id": s["response"], "label": lab[0], "text": lab[1], "eps": rp["eps"], "per_seat": rp["per_seat"]},
            "headline": {**head, "range_parts": eb["parts"]["d_guest_nights"], "range_note": eb["note"]},
            "chain": self._chain(S, mixed, flights),
            "sentence": self._sentence(S, lo, hi, mixed, flights, stays),
            "by_market": self._by_market(eb["by_market"]),
            "by_month": months,
            "daily": self._daily(d),
            "notes": self._notes(r.flags),
        })

    # ------------------------------------------------------------------ pieces of the result
    def _chain(self, S, mixed: bool, flights: bool) -> dict:
        seats, pax, local = S["d_seats"], S["d_passengers"], S["d_local_passengers"]
        ckf, ckd, ck, gn = S["d_checkins_flights"], S["d_checkins_demand"], S["d_checkins"], S["d_guest_nights"]
        sgn = 1.0 if (local or seats or pax or ck or gn) >= 0 else -1.0    # direction: the flight change, else the effect
        loss = sgn < 0
        ratio = lambda a, b: a / b if abs(b) > 1e-9 else None
        stages = [
            {"key": "seats", "label": "Seats", "value": seats, "text": exact(seats, True)},
            {"key": "passengers", "label": "Arriving passengers", "value": pax, "text": exact(pax, True),
             "ratio": ratio(pax, seats), "ratio_text": None if ratio(pax, seats) is None else f"{pct(pax / seats)} of seats filled"},
            {"key": "local", "label": "Inbound passengers", "value": local, "text": exact(local, True),
             "ratio": ratio(local, pax), "ratio_text": None if ratio(local, pax) is None else f"{pct(local / pax)} end their trip here"},
            {"key": "checkins", "label": "Hotel check-ins", "value": ck, "text": est(ck, True), "from_flights": ckf, "from_demand": ckd,
             "ratio": ratio(ckf, local), "ratio_text": None if ratio(ckf, local) is None else f"{ckf / local:.2f} per inbound passenger"},
            {"key": "guest_nights", "label": "Guest nights", "value": gn, "text": est(gn, True),
             "ratio": ratio(gn, ck), "ratio_text": None if ratio(gn, ck) is None else f"{est(gn / ck)} nights per check-in"},
        ]
        steps = []                                                         # what leaves (exit) or joins (inlet) between stages
        for i, (a, b, out_txt, in_txt, sub) in enumerate([
                (seats, pax, ("of the lost seats were empty" if loss else "seats fly empty"),
                 ("emptier planes" if loss else "fuller planes"), None),
                (pax, local, ("were connecting onwards" if loss else "passengers connect onwards"),
                 ("more passengers connect onwards" if loss else "fewer passengers connect onwards"), "transfer and transit"),
                (local, ckf, ("were not hotel guests" if loss else "no extra hotel stay"), None,
                 "residents, family visits, other lodging")]):
            x = sgn * (a - b)
            if not flights or abs(x) < 0.5:
                steps.append(None); continue
            base = abs(a) if abs(a) > 1e-9 else None
            steps.append({"kind": "exit" if x > 0 else "inlet", "value": abs(x),
                          "share": abs(x) / base if base else None,
                          "text": est(abs(x)) if i == 2 else exact(abs(round(a) - round(b))),         # shown stages add up
                          "label": out_txt if x > 0 else in_txt, "sub": sub})
        demand = None if abs(ckd) < 0.5 else {"kind": "inlet" if ckd > 0 else "exit", "value": abs(ckd), "text": est(abs(ckd)),
                                                "label": "from demand, not flights"}
        per100 = None
        if flights and not mixed and abs(seats) > 0.5 and abs(ckd) < 0.5:
            f = 100 / abs(seats)
            per100 = {k: abs(v) * f for k, v in (("passengers", pax), ("local", local), ("checkins", ckf), ("guest_nights", gn))}
            per100["text"] = (f"For every 100 lost seats: {per100['passengers']:.0f} passengers, {per100['local']:.0f} of them "
                              f"ending their trip in Abu Dhabi, {est(per100['checkins'])} fewer hotel check-ins, "
                              f"{est(per100['guest_nights'])} fewer guest nights." if loss else
                              f"For every 100 extra seats: {per100['passengers']:.0f} passengers, {per100['local']:.0f} end their "
                              f"trip in Abu Dhabi, {est(per100['checkins'])} check in to a hotel, {est(per100['guest_nights'])} guest nights.")
        return {"stages": stages, "steps": steps, "demand": demand, "per_100_seats": per100, "mixed": mixed,
                "direction": "loss" if loss else "gain", "has_flight_change": flights}

    def _sentence(self, S, lo, hi, mixed, flights, stays) -> list:
        seats, pax, local = S["d_seats"], S["d_passengers"], S["d_local_passengers"]
        ckf, ckd, ck, gn = S["d_checkins_flights"], S["d_checkins_demand"], S["d_checkins"], S["d_guest_nights"]
        if abs(gn) < 0.5 and abs(ck) < 0.5 and not flights:
            return [{"text": "No changes yet: this is the normal year, March 2026 to February 2027. Pick a preset or add a change.",
                     "strong": False}]
        out = []; T = lambda t, strong=False: out.append({"text": t, "strong": strong})
        if flights:
            # the verdict only: the stream below shows every step (seats, passengers, who ends the trip here)
            if mixed:
                T(f"These changes add and remove seats (net {exact(seats, True)}): about {est(ckf, True)} hotel check-ins")
            elif seats > 0.5 and pax >= local >= 0:
                T(f"{exact(seats)} extra seats bring about {est(ckf)} more hotel check-ins")
            elif seats < -0.5 and pax <= local <= 0:
                T(f"Losing {exact(-seats)} seats means about {est(-ckf)} fewer hotel check-ins")
            else:
                what = ("Fuller planes" if pax > 0.5 else "Emptier planes" if pax < -0.5 else
                        "Fewer passengers connecting onwards" if local > 0 else "More passengers connecting onwards")
                T(f"{what} {'bring' if ckf > 0 else 'mean'} about {est(abs(ckf))} {'more' if ckf > 0 else 'fewer'} hotel check-ins")
        if abs(ckd) > 0.5:
            T(f"; demand from other causes {'adds' if ckd > 0 else 'removes'} about {est(abs(ckd))} more" if flights else
              f"Demand from causes other than flights {'adds' if ckd > 0 else 'removes'} about {est(abs(ckd))} hotel check-ins")
        if abs(ck) > 0.5:
            if stays:
                T(f", at {est(gn / ck)} guest nights per check-in with the stay-length change")
        elif stays:
            if len(stays) == 1:
                c = stays[0]
                T(f"Guests checking in from {pd.Timestamp(c['start']):%-d %b} to {pd.Timestamp(c['end']):%-d %b %Y} stay "
                  f"{abs(c['pct']):g}% {'longer' if c['pct'] > 0 else 'shorter'}")
            else:
                T("With the stay-length changes")
        T(": "); T(f"{est(gn, True)} guest nights", True); T(f" (80% range {est(lo, True)} to {est(hi, True)}).")
        return out

    def _by_market(self, bm: pd.DataFrame) -> list:
        rows = [{"market": m, "label": label(m), "region": self.region.get(m), "has_flight_data": m in self.w.flight,
                 "d_guest_nights": v.d_guest_nights, "lo": v.lo, "hi": v.hi, "d_checkins": v.d_checkins,
                 "text": est(v.d_guest_nights, True)} for m, v in bm.iterrows() if abs(v.d_guest_nights) >= 0.5]
        return sorted(rows, key=lambda x: -abs(x["d_guest_nights"]))

    def _headline(self, gn, lo, hi, ck, ck_lo, ck_hi, pct_gn, after, months) -> dict:
        occ = max(months, key=lambda m: abs(m["occupancy_points"])) if months else None
        return {"d_guest_nights": gn, "lo": lo, "hi": hi, "d_checkins": ck, "checkins_lo": ck_lo, "checkins_hi": ck_hi,
                "pct_guest_nights": pct_gn, "after_horizon": after,
                "text": est(gn, True), "range_text": f"{est(lo, True)} to {est(hi, True)}",
                "occupancy": None if occ is None or abs(gn) < 0.5 else {
                    "month": occ["month"], "points": occ["occupancy_points"],
                    "text": self._occ_text(occ["occupancy_points"], occ["month"])}}

    def _month_rows(self, months, d, lo, hi, base) -> list:
        """One row per month: the change with its 80% range, the all-market level (normal year and scenario, with the
        indicative total-month level band), occupancy points and the beyond-validation flag."""
        t_lo, t_hi = self.B._total["guests"]
        beyond = self.fact("beyond_validation_from")[:7]
        pt = self._occ_point
        out = []
        for mo, x, a, b, g in zip(months, d, lo, hi, base):
            scen = g + x
            out.append({"month": mo, "label": month_label(mo), "d_guest_nights": x, "lo": a, "hi": b,
                        "baseline": g, "scenario": scen, "level_lo": scen * t_lo, "level_hi": scen * t_hi,
                        "occupancy_points": x / pt[mo], "beyond_validation": mo >= beyond})
        return out

    def _by_month(self, r: en.Result) -> list:
        bm = r.effect_band["by_month"]
        base = r.monthly(by_market=False)["guests_base"].reindex(bm.index)
        return self._month_rows(list(bm.index), bm.d_guest_nights, bm.lo, bm.hi, base)

    @staticmethod
    def _occ_text(points: float, month: str) -> str:
        v = round(points, 1)
        if v == 0:
            return f"under 0.1 occupancy points in {month_label(month)}"
        return f"≈ {_sign(f'{v:.1f}', v, True)} occupancy points in {month_label(month)}"

    def _daily(self, d: pd.DataFrame) -> dict:
        t = d.groupby("date")[["guests_base", "guests_scen"]].sum()
        peak = self.E.params.peak_guests
        over = t[t.guests_scen > peak]
        return {"dates": [x.strftime("%Y-%m-%d") for x in t.index], "baseline": t.guests_base.round(1).tolist(),
                "scenario": t.guests_scen.round(1).tolist(), "peak": peak, "peak_date": self.fact("peak_date"),
                "days_above_peak": int(len(over)),
                "max": float(t.guests_scen.max()), "max_date": t.guests_scen.idxmax().strftime("%Y-%m-%d")}

    def _notes(self, flags: list) -> list:
        groups = OrderedDict()
        for f in flags:
            for nid, level, pat, txt in NOTE_RULES:
                if re.search(pat, f):
                    groups.setdefault(nid, []).append(f); break
            else:
                groups.setdefault(f"other:{f[:40]}", []).append(f)
        notes = []
        for nid, fl in groups.items():
            rule = next((x for x in NOTE_RULES if x[0] == nid), None)
            level = rule[1] if rule else "info"
            if level is None:
                continue
            if nid == "capacity":
                m = re.search(r"on (\d+) days, max ([\d,]+) on (\S+)", fl[0])
                n = int(m.group(1)) if m else 0
                text = (f"On {'one day' if n == 1 else f'{n} days'} hotels would hold more guests than ever recorded "
                        f"({self.E.params.peak_guests:,.0f} on {pd.Timestamp(self.fact('peak_date')):%-d %b %Y}), up to "
                        f"{m.group(2)} on {pd.Timestamp(m.group(3)):%-d %b %Y}: check room capacity.") if m else fl[0]
            elif nid == "beyond_validation":
                share = next((re.search(r"^(\d+%) of the guest-night change", x) for x in fl if re.search(r"^\d+%", x)), None)
                start = month_label(self.fact("beyond_validation_from")[:7])
                text = ((f"{share.group(1)} of this change" if share else "Part of this change") +
                        f" falls from {start}, more than seven months beyond the data: the normal year is less certain there.")
            elif nid == "no_flight_data":
                mk = sorted({label(x.upper()) for f in fl for x in re.findall(r"([A-Z][a-z]+(?: [A-Z][a-z]+)*) has no flight data", f)} |
                            {x.strip() for f in fl if "market-days" in f for x in _markets_in(f).split(",") if x.strip()})
                who = ", ".join(mk) if mk else "Some markets"
                text = (f"{who} {'has' if len(mk) == 1 else 'have'} no flights in the data: a new route there uses the rate "
                        f"borrowed from {self.fact('countries_estimated')} markets, {est(self.fact('new_route_per_seat'))} check-ins "
                        f"per added seat (80% range {est(self.fact('new_route_per_seat_lo'))}–{est(self.fact('new_route_per_seat_hi'))}), "
                        f"and its planes fill like the network's ({pct(self.w.network_load_factor)}).")
            elif rule is None:
                text = fl[0]
            else:
                text = rule[3](fl) if callable(rule[3]) else rule[3]
            notes.append({"id": nid, "level": level, "text": text, "count": len(fl), "details": fl})
        return sorted(notes, key=lambda n: n["level"] != "warning")

    # ------------------------------------------------------------------ download
    def export_csv(self, scenario: dict) -> str:
        s = self._check(scenario)
        with self._lock:
            r = self.B.run(self.E, s, **self.response_params(s["response"]))
        m = r.effect_band["monthly"].join(r.monthly()[["guests_base", "guests_scen"]])
        m = m.reset_index()
        out = pd.DataFrame({"market": m.market.map(label), "month": m.month, "guest_nights_normal_year": m.guests_base.round(1),
                            "guest_nights_scenario": m.guests_scen.round(1), "change_guest_nights": m.d_guest_nights.round(1),
                            "change_80pct_low": m.d_guest_nights_lo.round(1), "change_80pct_high": m.d_guest_nights_hi.round(1),
                            "change_checkins": m.d_checkins.round(1), "beyond_7_months_after_data": ~m.level_band_calibrated,
                            "check_in_response": s["response"]})
        out = out[(out.change_guest_nights != 0) | (out.change_checkins != 0)]
        return out.to_csv(index=False)

    # ------------------------------------------------------------------ reverse question
    def reverse(self, market: str, start: str, end: str, target: float, aircraft: str = "today",
                seats_per_flight: float | None = None) -> dict:
        """Extra flights a week for `target` extra guest nights from `market`, with check-ins between start and end.
        aircraft="today": more flights on the market's current routes, at today's average aircraft size — the live
        slider's lever (the market-wide `route` change), inverted in closed form. aircraft="narrowbody"/"widebody" (or
        seats_per_flight): one added route of that size with the market's own passenger mix — the Phase 3 method
        (sensitivity.seats_needed, closed form). Markets with no flights in the window always take an added route
        (narrowbody unless stated). Every path is tested against the full engine.

        Answers stay inside what the simulator itself accepts: on today's routes up to double the country's flights (the
        slider's range), on an added route up to 100 flights a week or double the country's flights, whichever is more
        (the new-route form's range). A target beyond that is "out of reach", with the most the ceiling would win."""
        if not (math.isfinite(target) and target > 0):
            raise sc.ScenarioError("the target must be a positive number of guest nights")
        try:
            pd.Timestamp(start), pd.Timestamp(end)
        except (ValueError, TypeError):
            raise sc.ScenarioError(f"start and end must be dates such as 2026-11-01 (got {start!r}, {end!r})")
        if seats_per_flight:
            spf, kind = float(seats_per_flight), "route"
        elif aircraft in self.w.aircraft:
            spf, kind = float(self.w.aircraft[aircraft]), "route"
        elif aircraft == "today":
            spf, kind = float(self.w.aircraft["narrowbody"]), "today"
        else:
            raise sc.ScenarioError(f"aircraft must be today, {' or '.join(sorted(self.w.aircraft))}")
        with self._lock:
            ev = None
            if kind == "today" and market in self.w.flight:
                try:
                    ev = _Lever(self, market, start, end)
                except sc.ScenarioError:
                    ev = None                                                # no seats in this window: an added route
            if ev is None:
                ev = _ReverseModel(self, market, start, end, spf)
            lever = isinstance(ev, _Lever)
            top = float(max(1, math.floor(2 * ev.current_fpw if lever else max(ROUTE_MAX_FLIGHTS, 2 * ev.current_fpw))))
            out = ev.solve(float(target), top)
            at_top = ev.gain(top) if out["flights_per_week"] is None else None
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        f = out["flights_per_week"]
        size = ev.seats_per_flight if lever else spf
        cls = next((k for k, v in self.w.aircraft.items() if np.isclose(v, spf)), None)
        new_route = not lever and not (market in self.w.flight and ev.current_fpw > 0)
        tested = self.E.params.hi * ev.current_fpw if lever else None              # the most flights the lever was tested on
        out.update({"market": market, "label": label(market), "window": [start, end], "target_guest_nights": target,
                    "aircraft": "today" if lever else (cls or "custom"), "seats_per_flight": size,
                    "current_flights_per_week": ev.current_fpw, "new_route": new_route, "flights_per_week_max": top,
                    "guest_nights_at_max": at_top, "tested_max_flights": tested,
                    "outside_tested_range": bool(lever and f is not None and f > tested),
                    "seats_added": None if f is None else f / 7 * days * size})
        if f is None:
            out["text"] = (f"That target is out of reach for {label(market)} in this window: even {flights(top)} more flights a week "
                           f"would win only about {est(at_top)} guest nights. Try a smaller target" + (", or a new route." if lever else "."))
            return clean(out)
        a, b = out["flights_per_week_range"]
        rng = f"80% range {flights(a)} to {flights(b) if b is not None else 'more than ' + flights(top)}"
        size_txt = f"{size:,.0f} seats"
        if lever:
            what = f"flights a week on today's routes (average {size_txt}; {rng})"
        elif new_route:
            what = f"flights a week on a new {cls or ''} route ({size_txt}; {rng})".replace("new  route", "new route")
        else:
            what = f"{cls + ' ' if cls else ''}flights a week ({size_txt}; {rng})"
        out["text"] = f"To win {est(target)} more guest nights from {label(market)}, Abu Dhabi needs about {flights(f)} more {what}."
        return clean(out)


def _root(fn, target: float, top: float):
    """Flights a week at which the rising `fn` reaches `target` (the exact solver's bracket and tolerance): None when even
    `top` flights fall short, and the search's lower end when a sliver of a flight is already enough."""
    if fn(top) < target:
        return None
    if fn(REVERSE_FROM) >= target:
        return REVERSE_FROM
    return brentq(lambda x: fn(x) - target, REVERSE_FROM, top, xtol=0.01)


class _ReverseModel:
    """The engine's arithmetic for one added route in one market, as a function of flights per week.

    Resolving the route at 1 flight a week gives, per day, the added seats and local passengers (both proportional to
    flights a week) and the baseline check-ins and local passengers. The engine's steps 1–3 (lever with cap, zero-seat
    rule, new-route lever) then give the extra check-ins per day, and Stage B — linear in check-ins, no stay change —
    turns a check-in on day t into w[t] guest nights (unit-impulse response, stays after the horizon included)."""

    def __init__(self, sim: Simulator, market: str, start: str, end: str, spf: float):
        E = sim.E; self.E, self.B, self.market = E, sim.B, market
        probe = {"changes": [{"type": "new_route", "market": market, **se.GENERIC, "start": start, "end": end,
                              "flights_per_week": 1.0, "seats_per_flight": spf}]}
        d, _ = E.resolve(probe)
        d = d[d.market == market].sort_values("date")
        on = ((d.date >= pd.Timestamp(start)) & (d.date <= pd.Timestamp(end))).to_numpy()
        d = d[on]
        self.A0 = d.checkins_base.to_numpy(); self.L0 = d.p2p_base.to_numpy()
        self.k = (d.p2p_scen - d.p2p_base).to_numpy()                       # local passengers per flight a week
        self.ks = (d.seats_scen - d.seats_base).to_numpy()                  # seats per flight a week
        self.S0 = d.seats_base.to_numpy()
        self.nrs = d.new_route_seats.to_numpy(); self.nrl = d.new_route_local.to_numpy()
        self.fl = bool(d.has_flight_data.iloc[0])
        self.Lref = np.array([E.ref_local.get((market, t.month), np.nan) for t in d.date])
        nh = len(E.world.days); pos = ((d.date - E.world.h0).dt.days).to_numpy()
        self.w = np.empty(len(pos))
        for i, p in enumerate(pos):                                          # nights from one check-in on each day
            a = np.zeros(nh); a[p] = 1.0
            self.w[i] = E.stageb.guests(market, a, memory=False).sum()
        e = self.B.effect
        self.ends = [(e[f"eps_{x}"], e[f"per_seat_{x}"], "line") for x in ("lo", "hi")]
        if self.fl and (self.L0 == 0).any():                                 # the zero-seat rule: both cut rules (Bands.end_runs)
            self.ends += [(e[f"eps_{x}"], e[f"per_seat_{x}"], "log") for x in ("lo", "hi")]
        self.floor = e["cut_floor"]
        r = E.world.routes[E.world.routes.nationality_key == market].groupby("moy").n_flights.sum()
        self.current_fpw = float(np.mean([r.get(t.month, 0.0) for t in d.date]) * 7)

    def checkins(self, f: float, eps=None, per_seat=None, cuts="line") -> np.ndarray:
        p = self.E.params
        eps = p.eps if eps is None else eps; per_seat = p.per_seat if per_seat is None else per_seat
        G = lambda v: en.lever_G(v, eps, p.lo, p.hi, cuts, self.floor)
        L1 = self.L0 + f * self.k
        dA = np.zeros(len(L1))
        if self.fl:
            std = self.L0 > 0
            x = np.divide(L1, self.L0, out=np.ones(len(L1)), where=std)
            raw = self.A0 * (G(x) - 1)
            dA[std] = (np.sign(raw) * np.minimum(np.abs(raw), np.abs(L1 - self.L0)))[std]
            zero = (self.L0 == 0) & (L1 > 0)
            if zero.any():
                G0 = G(0.0)
                raw0 = self.A0[zero] / G0 * (G(L1[zero] / self.Lref[zero]) - G0)
                raw0 = np.where(np.isfinite(raw0), raw0, per_seat * (self.S0 + f * self.ks)[zero])
                dA[zero] = np.minimum(raw0, L1[zero])
        else:
            nrs = f * self.nrs
            lps = np.divide(f * self.nrl, nrs, out=np.zeros(len(nrs)), where=nrs > 0)
            dA = np.where(nrs > 0, np.minimum(per_seat, lps) * nrs, 0.0)
        return dA

    def gain(self, f, **kw) -> float:
        return float(self.w @ self.checkins(f, **kw))

    def _band(self, f) -> tuple[float, list]:
        c = self.gain(f)
        return c, [c] + [self.gain(f, eps=e, per_seat=ps, cuts=cu) for e, ps, cu in self.ends]

    def band_end(self, f, i) -> float:                                      # Bands.run, guest nights, end i (0 low, 1 high)
        c, vals = self._band(f)
        dn, up = self.B._level_dev("guests", {self.market: c})
        return c - float(np.hypot(c - min(vals), dn)) if i == 0 else c + float(np.hypot(max(vals) - c, up))

    def lever_end(self, f, fn) -> float:
        return fn(self._band(f)[1])

    def solve(self, target: float, top: float) -> dict:
        root = lambda fn: _root(fn, target, top)
        return {"flights_per_week": root(self.gain),
                "flights_per_week_range": [root(lambda x: self.band_end(x, 1)), root(lambda x: self.band_end(x, 0))],
                "flights_per_week_range_lever_only": [root(lambda x: self.lever_end(x, max)), root(lambda x: self.lever_end(x, min))]}


class _Lever:
    """The market-wide seats lever in closed form: every route of `market` scaled by x = 1 + seats_pct/100 on the days
    between start and end — exactly the engine's `route` change with `seats_pct` (scenario.resolve multiplies each
    route-day's seats, passengers and local passengers by x, so the market's local passengers become x·L0 every day).

    Engine steps for that market, with x the same every day: extra check-ins = A0·(G(x) − 1), capped by |L0·(x − 1)|
    (days without baseline seats are unchanged); no demand change. Stage B is linear in check-ins and unchanged in
    stays, so guest nights = Σ_day check-ins(day) × nights-per-check-in(day), split by month (unit-impulse responses).
    The 80% range follows Bands.run: the lever at both ends of ε (and, for cuts beyond the tested range, the log form
    at both ends), the market's baseline-level error, combined in quadrature."""

    def __init__(self, sim: Simulator, market: str, start: str, end: str, eps: float | None = None):
        self.sim, self.E, self.B, self.market = sim, sim.E, sim.B, market
        self.eps = sim.E.params.eps if eps is None else eps                 # the response the curve is shown at
        w = sim.w; self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        if market not in w.flight:
            raise sc.ScenarioError(f"{label(market)} has no flight data: use “Open a new route”")
        if self.start < w.h0 or self.end > w.h1 or self.start > self.end:
            raise sc.ScenarioError(f"the window {start}…{end} is outside the horizon")
        d = sim.baseline_inputs()
        d = d[(d.market == market) & (d.date >= self.start) & (d.date <= self.end)].sort_values("date")
        self.L0, self.A0 = d.p2p_base.to_numpy(), d.checkins_base.to_numpy()
        self.S0, self.P0 = d.seats_base.to_numpy(), d.pax_base.to_numpy()
        if not (self.S0 > 0).any():
            raise sc.ScenarioError(f"{label(market)} has no flights in this window: use “Open a new route”")
        # nights from one check-in on each day, by horizon month, plus stays running past the horizon
        nh = len(w.days); pos = ((d.date - w.h0).dt.days).to_numpy()
        self.months = [str(m) for m in pd.period_range(w.h0, w.h1, freq="M")]
        mi = pd.PeriodIndex(w.days, freq="M").astype(str).map({m: i for i, m in enumerate(self.months)}).to_numpy()
        self.R = np.zeros((len(pos), len(self.months) + 1))
        for i, p_ in enumerate(pos):
            a = np.zeros(nh); a[p_] = 1.0
            g = self.E.stageb.guests(market, a, memory=False)
            self.R[i, :-1] = np.bincount(mi, weights=g[:nh], minlength=len(self.months)); self.R[i, -1] = g[nh:].sum()
        r = w.routes[w.routes.nationality_key == market].groupby("moy").n_flights.sum()
        self.current_fpw = float(np.mean([r.get(t.month, 0.0) for t in d.date]) * 7)
        self.seats_per_flight = float(self.S0.sum() / (self.current_fpw / 7 * len(d)))
        e = self.B.effect
        self.ends = [(e["eps_lo"], "line"), (e["eps_hi"], "line")]
        self.log_ends = [(e["eps_lo"], "log"), (e["eps_hi"], "log")]
        self.floor = e["cut_floor"]
        self.days = len(d)

    # ---- the engine's arithmetic
    def _checkins(self, x: np.ndarray, eps: float, cuts: str) -> tuple[np.ndarray, np.ndarray]:
        """Extra check-ins by step (rows) and day (columns), and whether the cap binds."""
        p = self.E.params
        g = en.lever_G(x, eps, p.lo, p.hi, cuts, self.floor if cuts == "log" else 0.0)
        raw = self.A0[None, :] * (g - 1)[:, None]
        cap = self.L0[None, :] * np.abs(x - 1)[:, None]
        std = self.L0 > 0
        dA = np.where(std, np.sign(raw) * np.minimum(np.abs(raw), cap), 0.0)
        return dA, std & (np.abs(raw) > cap + 1e-9)

    def evaluate(self, x) -> dict:
        x = np.atleast_1d(np.asarray(x, dtype=float)); p = self.E.params
        dA, capped = self._checkins(x, self.eps, "line")
        out = {"x": x, "ck": dA.sum(1), "mon": dA @ self.R, "capped": capped.any(1)}
        beyond_cut = (x - 1) < p.lo
        ends = []
        for eps, cuts in self.ends + self.log_ends:
            dAe, _ = self._checkins(x, eps, cuts)
            v = {"ck": dAe.sum(1), "mon": dAe @ self.R}
            if cuts == "log":                                    # only where Bands.end_runs adds the log form
                v = {k: np.where(beyond_cut[:, None] if a.ndim == 2 else beyond_cut, a, np.nan)
                     for k, a in v.items()}
            ends.append(v)
        out["ends"] = ends
        out["outside"] = ((x - 1) < p.lo) | ((x - 1) > p.hi)
        return out

    def _range(self, c: float, vals: list, kind: str) -> tuple[float, float]:
        vals = [v for v in vals if np.isfinite(v)]
        dn, up = self.B._level_dev(kind, {self.market: c} if c != 0 else {})
        return c - float(np.hypot(c - min(vals), dn)), c + float(np.hypot(max(vals) - c, up))

    def seats_added(self, f: float) -> float:
        return float(self.S0.sum() * f / self.current_fpw)

    # ---- one slider position as a result (same builders as a full run)
    def step(self, ev: dict, i: int, f: float) -> dict:
        sim, x = self.sim, ev["x"][i]
        mon, ck = ev["mon"][i], float(ev["ck"][i])
        gn, tail = float(mon.sum()), float(mon[-1])
        lo, hi = self._range(gn, [gn] + [float(np.nansum(e["mon"][i])) if np.isfinite(e["mon"][i]).all() else np.nan
                                         for e in ev["ends"]], "guests")
        ck_lo, ck_hi = self._range(ck, [ck] + [float(e["ck"][i]) for e in ev["ends"]], "checkins")
        m_lo, m_hi = [], []
        for j in range(len(self.months)):
            c = float(mon[j])
            vals = [c] + [float(e["mon"][i][j]) for e in ev["ends"]]
            a, b = self._range(c, vals, "guests") if c != 0 else (0.0, 0.0)
            m_lo.append(a); m_hi.append(b)
        S = {"d_seats": (x - 1) * self.S0.sum(), "d_passengers": (x - 1) * self.P0.sum(), "d_local_passengers": (x - 1) * self.L0.sum(),
             "d_checkins_flights": ck, "d_checkins_demand": 0.0, "d_checkins": ck, "d_guest_nights": gn}
        base_total = sim.baseline_inputs().guests_base.sum()
        flights = abs(x - 1) > 0
        rows = sim._month_rows(self.months, mon[:-1], m_lo, m_hi, self.base_month)
        empty = f == 0
        return {"flights_change": f, "seats_pct": (x - 1) * 100, "flights_per_week": self.current_fpw + f,
                "outside_tested_range": bool(ev["outside"][i]),
                "headline": sim._headline(gn, lo, hi, ck, ck_lo, ck_hi, gn / base_total * 100, tail, rows),
                "chain": sim._chain(S, False, flights),
                "sentence": sim._sentence(S, lo, hi, False, flights, []) if not empty else
                            [{"text": f"{label(self.market)} today: about {flights_text(self.current_fpw)} flights a week. "
                                      "Drag to add or remove flights.", "strong": False}],
                "by_market": [] if empty else [{"market": self.market, "label": label(self.market),
                                                "region": sim.region.get(self.market), "has_flight_data": True,
                                                "d_guest_nights": gn, "lo": lo, "hi": hi, "d_checkins": ck, "text": est(gn, True)}],
                "by_month": [{k: r_[k] for k in ("d_guest_nights", "lo", "hi", "occupancy_points")} for r_ in rows],
                "because": None if empty else self._because(S, bool(ev["capped"][i]), bool(ev["outside"][i]))}

    def _because(self, S, capped: bool, outside: bool) -> str:
        today = self.A0.sum() / self.L0.sum()
        extra = S["d_checkins_flights"] / S["d_local_passengers"]
        who = label(self.market)
        if S["d_local_passengers"] > 0:
            t = (f"{who} sends {today:.2f} hotel check-ins per inbound passenger on its own flights today (counting guests "
                 f"who arrive by any route); each extra inbound passenger adds {extra:.2f}, a little less with every added flight.")
            if capped:
                t += " Capped: check-ins cannot rise by more than the extra inbound passengers."
        else:
            t = (f"{who} sends {today:.2f} hotel check-ins per inbound passenger on its own flights today; each one lost removes "
                 f"only {extra:.2f}, because many travellers switch to other routes, through Dubai or a connection.")
        if outside:
            t += " Beyond the tested range the effect is extended in a straight line, not estimated."
        return t

    def curve(self, season: str) -> dict:
        c = self.current_fpw
        step = max(1, int(np.ceil(3 * c / MAX_STEPS)))
        fs = [-c] + [k for k in range(-int(np.floor(c / step)) * step, int(2 * c) + 1, step) if -c < k] if c >= 1 else [-c, 0.0, 1.0, 2.0]
        fs = sorted(set(float(v) for v in fs) | {0.0})
        x = 1 + np.array(fs) / c
        ev = self.evaluate(x)
        base = self.sim.w.guests.assign(m=lambda g: g.date.dt.to_period("M").astype(str)).groupby("m").guests.sum()
        self.base_month = [float(base[m]) for m in self.months]
        steps = [self.step(ev, i, f) for i, f in enumerate(fs)]
        tt = self.B.level[self.B.level.level == "total-month"].set_index("kind").loc["guests"]
        beyond = self.sim.fact("beyond_validation_from")[:7]
        lo_t, hi_t = self.E.params.lo, self.E.params.hi
        return {"market": self.market, "label": label(self.market), "season": season, "season_label": SEASONS[season][2],
                "season_span": SEASONS[season][3], "window": [str(self.start.date()), str(self.end.date())],
                "current_flights_per_week": c, "today_index": fs.index(0.0),
                "tested_range_flights": [lo_t * c, hi_t * c],
                "change": {"type": "route", "markets": self.market, "start": str(self.start.date()), "end": str(self.end.date())},
                "months": [{"month": m, "label": month_label(m), "baseline": b, "beyond_validation": m >= beyond,
                            "dct_guest_nights_per_occupancy_point": float(self.sim.occ.loc[m, "dct_guest_nights_per_occupancy_point"])}
                           for m, b in zip(self.months, self.base_month)],
                "level_ratio": [float(tt.ratio_lo), float(tt.ratio_hi)],
                "steps": steps}

    # ---- the reverse question on the same lever
    def gain(self, f, eps=None, cuts="line") -> float:
        dA, _ = self._checkins(np.array([1 + f / self.current_fpw]), self.E.params.eps if eps is None else eps, cuts)
        return float((dA @ self.R).sum())

    def _vals(self, f) -> tuple[float, list]:
        c = self.gain(f)
        vals = [c] + [self.gain(f, e, cu) for e, cu in self.ends]
        if (f / self.current_fpw) < self.E.params.lo:
            vals += [self.gain(f, e, cu) for e, cu in self.log_ends]
        return c, vals

    def solve(self, target: float, top: float) -> dict:
        def band_end(f, i):
            c, vals = self._vals(f)
            return self._range(c, vals, "guests")[i]

        root = lambda fn: _root(fn, target, top)
        return {"flights_per_week": root(self.gain),
                "flights_per_week_range": [root(lambda v: band_end(v, 1)), root(lambda v: band_end(v, 0))],
                "flights_per_week_range_lever_only": [root(lambda v: max(self._vals(v)[1])), root(lambda v: min(self._vals(v)[1]))]}


def flights_text(x: float) -> str:
    return flights(x)
