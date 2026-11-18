"""Phase 3, Step 3.3 — scenario inputs: what a user can change, how it is checked, and what the engine receives.

A scenario is a list of changes over the horizon (dates inclusive). Six kinds, each with a window (start, end):

  route           an existing route (market [+ city] [+ airline]): seats_pct (−100 = cancel), or flights_per_week,
                  or seats_per_flight (aircraft size), on the days it flies in the baseline. Passengers move with seats
                  (load factor and local share kept). Service in months a route does not fly = new_route.
  new_route       market, city, airline, flights_per_week, seats_per_flight or aircraft_class (narrowbody / widebody,
                  aircraft_seats.csv); load factor and local share default to the airline's own routes in the market,
                  else its AUH network, else the market (nationalities without flight data: the network's average load
                  factor and the Step 2.5 cap on local passengers per seat).
  load_factor     pct change of passengers per seat on all routes of the selected markets (local share kept); a route
                  cannot fill beyond its seats, so the change is capped at full planes (flagged).
  transfer_share  share of passengers connecting at AUH (transfer + transit), market level: `value` (0–1), `points`
                  (+/−) or `pct` (relative to the market's own share, capped at 99%); each route's share scales in proportion.
  demand          pct change of check-ins from causes other than flights (advisories, visas, events, 2026 conditions).
  stay_length     pct change of the average stay of guests checking in during the window (Stage B checkout hazard
                  solved so that their nights per check-in change by exactly pct).

Markets: a nationality key, a list, or a group — "all", "international", "flight markets", "DOMESTIC", or a SCAD
region ("Europe", "GCC", …; data/reference/scad_region_map.csv). Flight levers (route, load_factor, transfer_share)
default to "flight markets"; a group keeps its members with flight data (the others are left unchanged, flagged),
while naming a nationality without flight data is an error. new_route also works for nationalities without flight
data (new-route lever). flights_per_week and seats_per_flight set one route's schedule, so they need a single route.

`resolve` turns a scenario into one row per market and day — seats, passengers and local passengers before and after,
seats on new routes, demand and stay multipliers, and the baseline check-ins and guests (Step 3.2) — which is all the
engine (Step 3.4) needs. Errors stop a scenario; flags travel with the result and are shown to the user.
"""
import json
from functools import cached_property
import numpy as np
import pandas as pd
from .config import load_config

SHORT_DAYS = 28                      # ε comes from sustained (monthly) changes; shorter ones are flagged
FLOWS = ["seats", "pax", "p2p"]
KINDS = {"route", "new_route", "load_factor", "transfer_share", "demand", "stay_length"}


TRANSFER_ROUTE_CAP = 0.999           # a single route's connecting share stays just below 100%


def scale_shares(tau, pax, group, target, cap: float = TRANSFER_ROUTE_CAP):
    """Each route's connecting share τ scaled by one factor λ per group (market-day or market-month), so that the
    passenger-weighted group share reaches `target` while every route stays at or below `cap`:
        Σ pax·min(λ·τ, cap) / Σ pax = target.
    Routes keep their order (a mostly point-to-point route stays mostly point-to-point). The left side rises with λ, so
    λ is found by bisection; without any clipping λ = target / current share exactly. Groups whose routes all have τ = 0
    take the target on every route. Where even every route at `cap` cannot reach the target (routes with τ = 0 never
    connect), the group stops at its maximum. Returns (new τ per route, achieved share per route's group)."""
    tau = np.asarray(tau, float); pax = np.asarray(pax, float); target = np.asarray(target, float)
    gi = pd.factorize(np.asarray(group))[0]; n = gi.max() + 1 if len(gi) else 0
    P = np.bincount(gi, pax, minlength=n)
    share = lambda t: np.divide(np.bincount(gi, pax * t, minlength=n), P, out=np.zeros(n), where=P > 0)
    cur = share(tau)
    tgt = np.zeros(n); np.maximum.at(tgt, gi, target)                    # one target per group (constant within it)
    zero = cur <= 0
    lam = np.divide(tgt, cur, out=np.zeros(n), where=~zero)              # exact when nothing clips
    clipped = lambda l: share(np.minimum(l[gi] * tau, cap))
    short = (~zero) & (clipped(lam) < tgt - 1e-12)
    if short.any():
        pos = np.where(tau > 0, tau, np.inf); tmin = np.full(n, np.inf); np.minimum.at(tmin, gi, pos)
        lo = lam.copy(); hi = np.where(np.isfinite(tmin), cap / np.where(np.isfinite(tmin), tmin, 1.0), lam)
        for _ in range(80):                                              # bisection on every short group at once
            mid = (lo + hi) / 2
            ok = clipped(mid) >= tgt
            hi = np.where(short & ok, mid, hi); lo = np.where(short & ~ok, mid, lo)
        lam = np.where(short, hi, lam)
    new = np.where(zero[gi], target, np.minimum(lam[gi] * tau, cap))
    return new, share(new)[gi]


class ScenarioError(ValueError):
    pass


# ----------------------------------------------------------------------------- the world a scenario edits
class World:
    """Normal-year baseline (Steps 3.1–3.2) in the shape the scenario layer needs."""

    def __init__(self, cfg=None):
        cfg = cfg or load_config(); self.cfg = cfg
        P, RD = cfg["paths"]["processed"] / "phase3", cfg["paths"]["reference"]
        self.h0, self.h1, self.data_end = cfg["dates"]["horizon_start"], cfg["dates"]["horizon_end"], cfg["dates"]["test_end"]
        self.days = pd.date_range(self.h0, self.h1)
        self.routes = pd.read_csv(P / "route_baseline.csv")
        self.checkins = pd.read_parquet(P / "baseline_checkins_daily.parquet")
        self.guests = pd.read_parquet(P / "baseline_guests_daily.parquet")
        self.markets = sorted(self.checkins.series.unique())
        self.flight = sorted(self.routes.nationality_key.unique())
        reg = pd.read_csv(RD / "scad_region_map.csv").dropna(subset=["scad_region"])
        self.regions = reg.groupby("scad_region").nationality.apply(list).to_dict()
        self.aircraft = pd.read_csv(P / "aircraft_seats.csv").set_index("aircraft_class").seats_per_flight.to_dict()
        j25 = json.load(open(cfg["paths"]["processed"] / "phase2" / "step25_new_route.json", encoding="utf-8"))
        self.no_flight_local_per_seat = float(j25["cap_local_pax_per_seat"])
        # a new route from a market without flight data fills like the network does (passengers ÷ seats over the
        # normal year); it only sets the passenger count shown in the chain — check-ins use the borrowed per-seat rate
        self.network_load_factor = float(self.routes.pax.sum() / self.routes.seats.sum())

    @cached_property
    def route_keys(self) -> "_Keys":
        """Integer codes of the baseline route table (validation selects routes once per change)."""
        return _Keys(self.routes.rename(columns={"nationality_key": "market"}))

    def select(self, m) -> list:
        groups = {"all": self.markets, "international": [x for x in self.markets if x != "DOMESTIC"],
                  "flight markets": self.flight, **self.regions}
        out = []
        for x in ([m] if isinstance(m, str) else list(m)):
            if x in groups:
                out += groups[x]
            elif x in self.markets:
                out.append(x)
            else:
                raise ScenarioError(f"unknown market or group: {x!r}")
        return sorted(set(out))

    def route_days(self) -> pd.DataFrame:
        """Route × day flows over the horizon (each day carries its calendar month's per-day rate)."""
        r = self.routes[["nationality_key", "city", "airline", "moy", "n_flights"] + FLOWS]
        d = pd.DataFrame({"date": self.days, "moy": self.days.month})
        out = r.merge(d, on="moy").drop(columns="moy").rename(columns={"nationality_key": "market"})
        for x in FLOWS:
            out[f"{x}_base"] = out[x]
        return out.assign(new=False).sort_values(["market", "city", "airline", "date"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- loading and checking
def load(path) -> dict:
    s = json.load(open(path, encoding="utf-8"))
    s.setdefault("name", str(path)); s.setdefault("changes", [])
    return s


def _window(c, w: World):
    try:
        a, b = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
    except (KeyError, ValueError, TypeError) as e:
        raise ScenarioError(f"{c.get('type')}: start and end dates are required ({e})")
    if pd.isna(a) or pd.isna(b):
        raise ScenarioError(f"{c.get('type')}: start and end must be dates (YYYY-MM-DD)")
    if a > b:
        raise ScenarioError(f"{c['type']}: start {a.date()} is after end {b.date()}")
    if a < w.h0 or b > w.h1:
        raise ScenarioError(f"{c['type']}: window {a.date()}…{b.date()} is outside the horizon {w.h0.date()}…{w.h1.date()}")
    return a, b


# every number a change can carry, with the widest value the engine accepts (the app's forms accept less; beyond these a
# result is no longer a scenario but an extrapolation without meaning, e.g. +1,000,000% seats)
NUMBERS = {"seats_pct": (-100, 1000), "flights_per_week": (0, 1000), "seats_per_flight": (0, 900), "pct": (-100, 1000),
           "points": (-100, 100), "value": (0, 1), "load_factor": (0, 1), "local_share": (0, 1)}


def _numbers(c: dict, tag: str):
    for k, (lo, hi) in NUMBERS.items():
        if k not in c:
            continue
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ScenarioError(f"{tag}: {k} must be a number")
        if not lo <= v <= hi:
            raise ScenarioError(f"{tag}: {k} must be between {lo:g} and {hi:g}")


def _flight_markets(c: dict, w: World, tag: str = "", flags: list | None = None) -> list:
    """Markets a flight lever acts on: a named nationality must have flight data; a group (region, "international",
    the default "flight markets") keeps its members with flight data and leaves the others unchanged (flagged)."""
    sel = c.get("markets", c.get("market", "flight markets"))
    names = [sel] if isinstance(sel, str) else list(sel)
    mk = w.select(sel)
    bad = sorted(set(x for x in names if x in w.markets) - set(w.flight))
    if bad:
        raise ScenarioError(f"{tag}: no flight data for {', '.join(x.title() for x in bad)} — use demand or new_route")
    out = sorted(set(mk) & set(w.flight))
    dropped = sorted(set(mk) - set(out))
    if dropped and flags is not None:
        flags.append(f"{tag}: {len(dropped)} markets in the selection have no flight data and are left unchanged "
                     f"({', '.join(x.title() for x in dropped[:5])}{'…' if len(dropped) > 5 else ''})")
    if not out:
        raise ScenarioError(f"{tag}: the selection contains no market with flight data")
    return out


def validate(s: dict, w: World) -> list:
    """Raise ScenarioError on anything the engine cannot honour; return the flags to show with the result."""
    flags = []
    if not s["changes"]:
        flags.append("empty scenario: results equal the baseline")
    for i, c in enumerate(s["changes"]):
        k = c.get("type")
        if k not in KINDS:
            raise ScenarioError(f"change {i}: unknown type {k!r} (one of {sorted(KINDS)})")
        a, b = _window(c, w)
        tag = f"change {i} ({k}, {a.date()}…{b.date()})"
        _numbers(c, tag)
        if (b - a).days + 1 < SHORT_DAYS and k in {"route", "new_route", "load_factor", "transfer_share"}:
            flags.append(f"{tag}: shorter than {SHORT_DAYS} days — the seat lever was estimated on sustained changes")
        if b >= w.data_end + pd.offsets.MonthBegin(1) + pd.DateOffset(months=7):
            flags.append(f"{tag}: reaches beyond 7 months after the data end — the baseline level is less certain there")
        if k in {"demand", "stay_length"}:
            mk = w.select(c.get("markets", "all"))
            p = c.get("pct")
            if p is None or not np.isfinite(p) or p <= -100:
                raise ScenarioError(f"{tag}: pct must be a number above −100")
            flags.append(f"{tag}: assumption lever ({len(mk)} markets) — not estimated from flight data")
            continue
        if k == "new_route":
            for f in ("market", "city", "airline", "flights_per_week"):
                if f not in c:
                    raise ScenarioError(f"{tag}: {f} is required")
            if c["market"] not in w.markets or c["market"] == "DOMESTIC":
                raise ScenarioError(f"{tag}: {c['market']!r} is not a nationality in the hotel data")
            if not c["flights_per_week"] > 0:
                raise ScenarioError(f"{tag}: flights_per_week must be positive")
            for f in ("load_factor", "local_share"):
                if f in c and not c[f] > 0:
                    raise ScenarioError(f"{tag}: {f} must be above 0")
            if "seats_per_flight" not in c and c.get("aircraft_class") not in w.aircraft:
                raise ScenarioError(f"{tag}: give seats_per_flight or aircraft_class ({sorted(w.aircraft)})")
            if c["market"] not in w.flight:
                flags.append(f"{tag}: {c['market'].title()} has no flight data — new-route lever only (borrowed check-ins per seat)")
            if c.get("aircraft_assumed"):
                flags.append(f"{tag}: aircraft not stated by the source — {c.get('aircraft_class')} seats assumed")
            exist = w.routes[(w.routes.nationality_key == c["market"]) & (w.routes.city == c["city"]) & (w.routes.airline == c["airline"])
                             & w.routes.moy.isin(pd.date_range(a, b).month.unique())]
            if exist.seats.sum() > 0:
                raise ScenarioError(f"{tag}: {c['airline']} already flies {c['city']} in these months of the baseline — use a route change")
            continue
        mk = _flight_markets(c, w, tag, flags)
        months = pd.date_range(a, b).month.unique()
        if k == "route":
            RK = w.route_keys
            sel = RK.route(c, mk) & np.isin(w.routes.moy.to_numpy(), months)
            if not sel.any() or w.routes.loc[sel, "seats"].sum() == 0:
                raise ScenarioError(f"{tag}: no baseline seats on the selected route(s) in this window — use new_route to add service")
            opts = [x for x in ("seats_pct", "flights_per_week", "seats_per_flight") if x in c]
            if len(opts) != 1:
                raise ScenarioError(f"{tag}: give exactly one of seats_pct, flights_per_week, seats_per_flight")
            live = sel & (w.routes.seats.to_numpy() > 0)
            n_routes = np.unique(np.stack([RK.code[x][live] for x in ("market", "city", "airline")]), axis=1).shape[1]
            if opts[0] != "seats_pct" and n_routes > 1:
                raise ScenarioError(f"{tag}: {opts[0]} sets one route's schedule, but the selection matches {n_routes} routes — "
                                    f"give market, city and airline (or use seats_pct)")
            v = c[opts[0]]
            if (opts[0] == "seats_pct" and v < -100) or (opts[0] != "seats_pct" and v < 0):
                raise ScenarioError(f"{tag}: {opts[0]} out of range")
        if k == "load_factor" and (c.get("pct") is None or c["pct"] <= -100):
            raise ScenarioError(f"{tag}: pct must be above −100")
        if k == "transfer_share":
            if sum(x in c for x in ("value", "points", "pct")) != 1:
                raise ScenarioError(f"{tag}: give exactly one of value (0–1), points or pct")
            if "pct" in c and not c["pct"] > -100:
                raise ScenarioError(f"{tag}: pct must be above −100")
            if "value" in c and not 0 <= c["value"] < 1:
                raise ScenarioError(f"{tag}: value must be in [0, 1)")
    # the same new route listed twice over overlapping days (e.g. two presets combined) would add its seats twice
    nr = [(i, c) for i, c in enumerate(s["changes"]) if c.get("type") == "new_route"]
    for x, (i, a_) in enumerate(nr):
        for j, b_ in nr[x + 1:]:
            if (a_["market"], a_["city"], a_["airline"]) == (b_["market"], b_["city"], b_["airline"]) and \
                    pd.Timestamp(a_["start"]) <= pd.Timestamp(b_["end"]) and pd.Timestamp(b_["start"]) <= pd.Timestamp(a_["end"]):
                raise ScenarioError(f"changes {i} and {j}: new route {a_['city']} ({a_['airline']}) is added twice over overlapping days — "
                                    "keep one (combined presets may both contain it)")
    # an observed level (e.g. the March 2026 preset) already contains that period's flight cuts: adding flight
    # changes over the same days would count them twice
    obs = [c for c in s["changes"] if "scad_region" in c]
    fl_ = [c for c in s["changes"] if c["type"] in {"route", "new_route", "load_factor", "transfer_share"}]
    if obs and any(pd.Timestamp(f["start"]) <= pd.Timestamp(o["end"]) and pd.Timestamp(o["start"]) <= pd.Timestamp(f["end"])
                   for o in obs for f in fl_):
        flags.append("observed-level preset combined with flight changes over the same days: the observed level already "
                     "includes that period's flight cuts, so they would be counted twice")
    return flags


def _new_route_shares(w: World, market: str, airline: str, months) -> tuple[pd.DataFrame, str]:
    """Default load factor and local share for a new route, by calendar month: the airline's own routes in the market;
    else the airline's whole network at AUH; else the market's other airlines. A hub carrier such as Etihad carries
    mostly connecting passengers (local share ~26%) while low-cost carriers carry almost none, so the airline matters
    more than the market. Months without service take the nearest months with service."""
    r = w.routes
    for mask, src in [((r.nationality_key == market) & (r.airline == airline), f"{airline}'s own routes in {market.title()}"),
                      (r.airline == airline, f"{airline}'s network at AUH"),
                      (r.nationality_key == market, f"the {market.title()} market average")]:
        mo = r[mask].groupby("moy")[FLOWS].sum()
        mo = mo[(mo.seats > 0) & (mo.pax > 0)]
        if len(mo):
            near = lambda m_: mo.index[np.argmin([min((x - m_) % 12, (m_ - x) % 12) for x in mo.index])]
            return mo.loc[[m_ if m_ in mo.index else near(m_) for m_ in months]], src
    raise ScenarioError(f"new_route: no load factor or local share available for {market}")


# ----------------------------------------------------------------------------- resolution
class _Keys:
    """Route-day selection on integer codes (string comparisons over the ~55,000 route-days were the cost of a scenario
    with hundreds of changes); selects exactly the same rows as comparing the columns themselves."""

    def __init__(self, r: pd.DataFrame):
        if "date" in r:                                 # route × day table (resolve); the route table has no dates
            self.ns = r.date.to_numpy().astype("datetime64[ns]").view("i8"); self.new = r.new.to_numpy(bool)
        self.code, self.ix = {}, {}
        for col in ("market", "city", "airline"):
            codes, uniq = pd.factorize(r[col])
            self.code[col] = codes; self.ix[col] = {v: i for i, v in enumerate(uniq)}

    def win(self, a, b) -> np.ndarray:                  # == r.date.between(a, b)
        return (self.ns >= pd.Timestamp(a).value) & (self.ns <= pd.Timestamp(b).value)

    def isin(self, col: str, values) -> np.ndarray:
        return np.isin(self.code[col], [self.ix[col][v] for v in values if v in self.ix[col]])

    def route(self, c: dict, mk: list) -> np.ndarray:  # market in mk, and the change's city / airline if given
        m = self.isin("market", mk)
        for col in ("city", "airline"):
            if col in c:
                m &= self.isin(col, [c[col]])
        return m


def resolve(s: dict, w: World) -> tuple[pd.DataFrame, list]:
    """Scenario → market × day inputs for the engine (plus the validation flags)."""
    flags = validate(s, w)
    r = w.route_days(); K = _Keys(r)
    demand = pd.DataFrame(1.0, index=w.days, columns=w.markets)
    stay = demand.copy()
    for c in s["changes"]:
        k = c["type"]; a, b = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
        win = K.win(a, b)
        if k in {"demand", "stay_length"}:
            tgt = demand if k == "demand" else stay
            tgt.loc[a:b, w.select(c.get("markets", "all"))] *= 1 + c["pct"] / 100
            continue
        if k == "new_route":
            spf = c.get("seats_per_flight") or w.aircraft[c["aircraft_class"]]
            d = pd.date_range(a, b)
            seats = c["flights_per_week"] / 7 * spf
            if c["market"] in w.flight:
                rows, src = _new_route_shares(w, c["market"], c["airline"], d.month)
                lf = np.full(len(d), c["load_factor"]) if "load_factor" in c else (rows.pax / rows.seats).to_numpy()
                ls = np.full(len(d), c["local_share"]) if "local_share" in c else (rows.p2p / rows.pax).to_numpy()
                p2p = seats * lf * ls
                if "load_factor" not in c or "local_share" not in c:
                    flags.append(f"new_route {c['city']} ({c['airline']}): load factor {np.mean(lf):.0%} and local share {np.mean(ls):.0%} "
                                 f"taken from {src}")
            else:                                       # no flight data: the Step 2.5 local passengers per seat
                lf = np.full(len(d), c.get("load_factor", w.network_load_factor))
                p2p = seats * (lf * c["local_share"] if "local_share" in c else np.full(len(d), w.no_flight_local_per_seat))
            add = pd.DataFrame({"market": c["market"], "city": c["city"], "airline": c["airline"], "date": d,
                                "n_flights": c["flights_per_week"] / 7, "seats": seats, "pax": seats * lf, "p2p": p2p, "new": True,
                                **{f"{x}_base": 0.0 for x in FLOWS}})
            r = pd.concat([r, add], ignore_index=True); K = _Keys(r)
            continue
        mk = _flight_markets(c, w)
        touched = (win & K.isin("market", mk) & (r.seats.to_numpy() > 0)).sum()
        if k in {"load_factor", "transfer_share"} and touched == 0:
            flags.append(f"{k} ({a.date()}…{b.date()}): no seats in the selected markets in this window — no effect")
        if k == "route":
            m = win & K.route(c, mk) & ~K.new
            if "seats_pct" in c:
                f = np.full(m.sum(), 1 + c["seats_pct"] / 100)
            elif "flights_per_week" in c:
                f = np.where(r.loc[m, "n_flights"] > 0, c["flights_per_week"] / 7 / r.loc[m, "n_flights"].replace(0, np.nan), 0.0)
            else:
                spf = (r.loc[m, "seats"] / r.loc[m, "n_flights"]).replace([np.inf], np.nan)
                f = np.where(spf > 0, c["seats_per_flight"] / spf, 1.0)
            f = np.asarray(f, dtype=float)
            r.loc[m, FLOWS] = r.loc[m, FLOWS].to_numpy() * f[:, None]
            r.loc[m, "n_flights"] = r.loc[m, "n_flights"].to_numpy() * (f if "seats_per_flight" not in c else 1.0)
        elif k == "load_factor":
            m = win & K.isin("market", mk)
            # a plane cannot fill beyond its seats: cap at 100% (or the route's own baseline load factor, which can
            # exceed 100% slightly because lap infants are counted as passengers)
            cap = np.maximum(r.loc[m, "seats"], r.loc[m, "pax_base"] * r.loc[m, "seats"] / r.loc[m, "seats_base"].replace(0, np.nan)).fillna(r.loc[m, "seats"])
            want = r.loc[m, "pax"] * (1 + c["pct"] / 100)
            new_pax = np.minimum(want, cap)
            capped = int(((want > cap * 1.0001) & (r.loc[m, "seats"] > 0)).sum())
            r.loc[m, "p2p"] *= (new_pax / r.loc[m, "pax"].replace(0, np.nan)).fillna(1.0)
            r.loc[m, "pax"] = new_pax
            if capped:
                flags.append(f"load_factor {c['pct']:+}% ({a.date()}…{b.date()}): capped at full planes on {capped:,} route-days")
        elif k == "transfer_share":
            # market-level share (passenger-weighted, per day) moved to the target; each route's share scaled in
            # proportion, so routes keep their differences (a point-to-point route stays mostly point-to-point)
            m = win & K.isin("market", mk) & (r.pax.to_numpy() > 0)
            sub = r.loc[m, ["market", "date", "pax", "p2p"]]
            tau = 1 - sub.p2p / sub.pax
            g = sub.groupby(["market", "date"])
            t_mkt = 1 - g.p2p.transform("sum") / g.pax.transform("sum")
            if "pct" in c:                             # relative change of each market's own share, kept below 99%
                want = t_mkt * (1 + c["pct"] / 100)
                target = want.clip(upper=0.99)
                if (want > 0.99).any():
                    flags.append(f"transfer_share {c['pct']:+}% ({a.date()}…{b.date()}): capped at 99% connecting on "
                                 f"{int((want > 0.99).groupby([sub.market, sub.date]).any().sum())} market-days")
            else:
                target = pd.Series(c["value"], index=sub.index) if "value" in c else t_mkt + c["points"] / 100
            if (target < 0).any() or (target >= 1).any():
                bad = sub.loc[(target < 0) | (target >= 1), "market"].unique()
                raise ScenarioError(f"transfer_share: market share would leave [0, 1) for {', '.join(x.title() for x in bad)} "
                                    f"(current share {t_mkt[sub.market.isin(bad)].mean():.0%})")
            new, got = scale_shares(tau.to_numpy(), sub.pax.to_numpy(), sub.market.astype(str) + "|" + sub.date.astype(str),
                                    target.to_numpy())
            miss = got < target.to_numpy() - 5e-4
            if miss.any():
                kd = sub[miss].groupby(["market", "date"]).size().shape[0]
                flags.append(f"transfer_share: target not reached on {kd} market-days (routes with no connecting passengers "
                             f"cannot connect; reached {got[miss].min():.0%} at the least)")
            if (new >= TRANSFER_ROUTE_CAP - 1e-12).any():
                flags.append(f"transfer_share: {int((new >= TRANSFER_ROUTE_CAP - 1e-12).sum())} route-days capped just below "
                             "100% connecting; the other routes of the market rise more, so the market share is the one asked for")
            r.loc[m, "p2p"] = sub.pax * (1 - new)
    g = r.groupby(["market", "date"])
    agg = g[[f"{x}_base" for x in FLOWS] + FLOWS].sum().rename(columns={x: f"{x}_scen" for x in FLOWS})
    newr = r[r.new].groupby(["market", "date"])[["seats", "p2p"]].sum().rename(columns={"seats": "new_route_seats", "p2p": "new_route_local"})
    grid = pd.MultiIndex.from_product([w.markets, w.days], names=["market", "date"])
    out = agg.reindex(grid).join(newr).fillna(0.0)
    out["has_flight_data"] = out.index.get_level_values("market").isin(w.flight)
    out["demand_mult"] = demand.stack().reorder_levels([1, 0]).reindex(grid).to_numpy()
    out["stay_mult"] = stay.stack().reorder_levels([1, 0]).reindex(grid).to_numpy()
    ck = w.checkins.set_index(["series", "date"]).rename_axis(["market", "date"])
    out["checkins_base"] = ck.checkins.reindex(grid).to_numpy()
    out["beyond_7_months"] = ck.beyond_7_months.reindex(grid).to_numpy()
    out["guests_base"] = w.guests.set_index(["market", "date"]).guests.reindex(grid).to_numpy()
    return out.reset_index(), flags
