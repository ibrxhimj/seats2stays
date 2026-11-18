"""Phase 3, Step 3.4 — scenario engine: a scenario's inputs (Step 3.3) → extra or lost check-ins and guest nights.

Chain, per market and day (all levers estimated and validated in Phases 1–2):

  seats → passengers (load factor) → local passengers L (passengers minus transfer and transit)
        → check-ins A:  A_scen = A_base · dm + ΔA,  ΔA = sign(F)·min(|F|, |L_scen − L_base|),  F = A_base·(G(L_scen/L_base) − 1)·dm
                        (flight lever ε = 0.31, dm = demand multiplier for causes other than flights; the flight effect
                        is taken in the scenario's demand conditions and capped there — audit E4)
        → guests in-house: Stage B applied to the scenario's check-ins (stays scaled if the stay length changes)

  G(x) = (x)^ε inside the trusted range of changes (−29% … +109%), continued in a straight line beyond it (no
  extrapolation of the curve), and the flight-attributed change in check-ins never exceeds the change in local
  passengers, in absolute value, also when demand changes too (cap: |ΔA| ≤ |ΔL|).
  With seats alone changing, L moves one-for-one with seats (ε₁ ≈ 1), so this is exactly the validated seat lever.
  (Step 3.7: for cuts beyond the range the log form continued was closer on the unseen Wizz Air exit; the straight line
  stays the central rule — training months did not favour either — and the effect band spans both.)

Days whose market has no baseline seats (Uzbekistan in Jul–Aug: its only summer route pauses) start from the
full-cancellation point of the market's reference seats (the nearest months with service), so cancelling and
restoring the same service cancel out. Nationalities without flight data take the new-route lever (borrowed
check-ins per added seat, Step 2.5). Every rule that is not plain lever arithmetic raises a flag.

Effects are computed day by day, so a ten-day change moves ten days (no dilution into a monthly average), and guest
nights include the stays that run past the end of the horizon.
"""
import json
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .config import load_config
from . import scenario as sc, stageb as sbm


def lever_G(x, eps, lo, hi, cuts: str = "line", floor: float = 0.0):
    """Check-ins multiplier for local passengers multiplied by x (x ≥ 0): x^ε inside [1+lo, 1+hi], straight beyond.
    cuts="log" is the alternative rule for cuts beyond the range (used only for the effect band, Step 3.7): the log
    form continued, flat below `floor` (the deepest cut in the training data)."""
    x = np.asarray(x, dtype=float); r = x - 1
    rc = np.clip(r, lo, hi)
    g = (1 + rc) ** eps + eps * (1 + rc) ** (eps - 1) * (r - rc)
    return np.where(r < lo, np.maximum(x, floor) ** eps, g) if cuts == "log" else g


@dataclass
class Params:
    eps: float
    lo: float
    hi: float
    per_seat: float            # new-route lever: check-ins per added seat (no-flight markets)
    eps_band: tuple
    per_seat_band: tuple
    peak_guests: float         # highest daily guests in-house ever observed (capacity flag)

    @classmethod
    def load(cls, cfg):
        P = cfg["paths"]["processed"]
        j23 = json.load(open(P / "phase2" / "step23_lever_test.json", encoding="utf-8")); j25 = json.load(open(P / "phase2" / "step25_new_route.json", encoding="utf-8"))
        pk = P / "phase3" / "peak_guests.json"                     # saved by `python -m atp.phase3 peak`
        if not pk.exists():
            raise FileNotFoundError(f"{pk} is missing: run `python -m atp.phase3 peak`")
        peak = json.load(open(pk, encoding="utf-8"))["peak_guests"]
        lo, up = j23["trusted_range_final"]
        from .stagea import lever_interval
        band = lever_interval(json.load(open(P / "phase2" / "step26_endtoend.json", encoding="utf-8")))     # ε's out-of-sample 80% interval
        return cls(float(j23["decision"]["default_eps_total"]), float(lo), float(up), float(j25["checkins_per_added_seat"]),
                   tuple(float(x) for x in band), tuple(j25["band"]), float(peak))


def _bracket(f, c0: float, lo: float = 1e-4, hi: float = 1e4):
    """Interval around the root of a decreasing f, found by doubling (or halving) from c0 inside [lo, hi]; None if the
    root is not bracketed there. The root is then found to the same tolerance as on the full interval."""
    fa = f(c0)
    if fa == 0:
        return c0, c0
    step, a = (2.0 if fa > 0 else 0.5), c0
    while True:
        b = a * step
        if not lo <= b <= hi:
            return None
        fb = f(b)
        if (fb > 0) != (fa > 0) or fb == 0:
            return (a, b) if a < b else (b, a)
        a, fa = b, fb


class StageB:
    """Stage B with the persisted Phase 1 parameters (Step 3.2), over the horizon plus a 44-day tail (a stay covers at most 45 nights: the arrival night and 44 more)."""

    def __init__(self, cfg, w: sc.World):
        P = cfg["paths"]["processed"]
        p = json.load(open(P / "phase3" / "stageb_params.json", encoding="utf-8"))
        self.theta = {n: np.array(t) for n, t in p["theta"].items()}
        spec = p["spec"]; te = pd.Timestamp(spec["train_end"])
        self.h0 = w.h0
        self.idx = pd.date_range(w.h0 - pd.Timedelta(days=sbm.K - 1), w.h1 + pd.Timedelta(days=sbm.K - 1))
        cal = pd.read_parquet(P / "calendar.parquet")
        ext = pd.read_parquet(P / "external_events.parquet")
        gp = ext[ext.event.str.startswith("abu_dhabi_grand_prix")]
        gp_days = pd.Series(True, index=pd.DatetimeIndex(np.concatenate(
            [pd.date_range(r.start_date - pd.Timedelta(days=1), r.end_date) for r in gp.itertuples()])))
        X, names = sbm.design(cal, self.idx, te, {"long_break": spec["intl"]["long_break"], "grand_prix": spec["intl"]["grand_prix"], "gp_days": gp_days})
        Xd, dnames = sbm.design(cal, self.idx, te, {"long_break": spec["dom"]["long_break"]})
        assert names == spec["intl"]["names"] and dnames == spec["dom"]["names"]
        self.e = {"intl": np.exp(X @ np.array(p["intl_beta"])), "dom": np.exp(Xd @ np.array(p["dom_beta"]))}
        self.gamma = {"intl": None if p["intl_gamma"] is None else np.array(p["intl_gamma"]),
                      "dom": None if p["dom_gamma"] is None else np.array(p["dom_gamma"])}
        mem = pd.read_parquet(P / "phase3" / "stageb_memory.parquet")
        self.memory = mem.pivot(index="date", columns="market", values="arrivals").reindex(self.idx[self.idx < w.h0])
        assert self.memory.notna().all().all(), "Stage B memory incomplete"
        self.tau = {g: sbm.make_tau(self.idx, te, spec[g].get("drift_shape", "trend")) for g in ("intl", "dom")}
        dows = self.idx.dayofweek.to_numpy()
        self.lagdow = np.column_stack([np.r_[np.full(k, -1), dows[:len(dows) - k]] for k in range(sbm.K)])
        self.n_h = (w.h1 - w.h0).days + 1

    def guests(self, market: str, checkins: np.ndarray, hazard_scale: np.ndarray | None = None, memory: bool = True) -> np.ndarray:
        """Guests in-house from h0 to h1 + 44 days for horizon check-ins (memory before, none after the horizon).
        hazard_scale (horizon length): checkout hazard multiplier for guests checking in on each day."""
        mem = self.memory[market].to_numpy() if memory else np.zeros(sbm.K - 1)
        a = np.r_[mem, checkins, np.zeros(sbm.K - 1)]
        # lag matrix L[i, k] = a[i - k] (0 before the start), built as one sliding window (same values as stacking
        # 45 shifted copies, which was the cost of solving the stay lever)
        L = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(np.r_[np.zeros(sbm.K - 1), a], sbm.K)[:, ::-1])
        m = sbm.Market(market, L, np.zeros(len(a)), self.lagdow)
        g = "dom" if market == "DOMESTIC" else "intl"
        cs = None if hazard_scale is None else np.r_[np.ones(sbm.K - 1), hazard_scale, np.ones(sbm.K - 1)]
        th = self.theta[market]
        if len(th) == 1 + sbm.NH + 1:
            m.tau = self.tau[g]
        return sbm.predict(m, th, self.e[g], self.gamma[g], cohort_scale=cs)[sbm.K - 1:]

    def hazard_scale(self, market: str, checkins: np.ndarray, stay_mult: np.ndarray) -> np.ndarray:
        """Per check-in day hazard multiplier so that guests checking in on days with stay multiplier v stay exactly
        v times as many nights on average (solved per distinct v, on those days' own check-ins and calendar)."""
        from scipy.optimize import brentq
        out = np.ones(len(stay_mult))
        for v in np.unique(stay_mult[stay_mult != 1]):
            on = stay_mult == v
            a = np.where(on, checkins, 0.0)
            base = self.guests(market, a, memory=False).sum()
            if base <= 0:
                continue
            f = lambda c: self.guests(market, a, np.where(on, c, 1.0), memory=False).sum() / base - v
            br = _bracket(f, 1.0 / v)                         # start next to the answer (stays ×v ≈ checkouts ÷v)
            if br is None:                                    # not bracketed inside [1e-4, 1e4]: the full check
                lo_, hi_ = f(1e4) + v, f(1e-4) + v             # shortest and longest average stays the model allows
                if not lo_ < v < hi_:
                    raise sc.ScenarioError(f"stay_length ×{v:.2f} for {market.title()} is outside what the stay model can represent "
                                           f"(×{lo_:.2f} … ×{hi_:.2f}: a stay counts its arrival night and at most 44 more, 45 nights in all; the limits are approximate, at hazard scales 10⁻⁴ and 10⁴)")
                br = (1e-4, 1e4)
            out[on] = br[0] if br[0] == br[1] else brentq(f, *br, xtol=1e-10)
        return out


@dataclass
class Result:
    name: str
    daily: pd.DataFrame
    summary: dict
    flags: list = field(default_factory=list)
    effect_band: dict | None = None          # filled by bands.Bands.run (Step 3.5)

    def monthly(self, by_market=True) -> pd.DataFrame:
        d = self.daily.assign(month=self.daily.date.dt.to_period("M").astype(str))
        cols = ["seats_base", "seats_scen", "p2p_base", "p2p_scen", "checkins_base", "checkins_scen", "d_checkins_flights",
                "d_checkins_demand", "d_checkins", "guests_base", "guests_scen", "d_guest_nights"]
        return d.groupby((["market"] if by_market else []) + ["month"])[cols].sum()


class Engine:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.world = sc.World(self.cfg)
        self.params = Params.load(self.cfg)
        self.stageb = StageB(self.cfg, self.world)
        # reference local passengers for days without baseline seats: nearest calendar months with service
        loc = self.world.routes.groupby(["nationality_key", "moy"]).p2p.sum().unstack(fill_value=0.0)
        ref = {}
        for mk, row in loc.iterrows():
            pos = row[row > 0]
            for mo, v in row.items():
                if v == 0 and len(pos):
                    dist = pos.index.map(lambda m: min((m - mo) % 12, (mo - m) % 12))
                    ref[(mk, mo)] = float(pos[dist == dist.min()].mean())
        self.ref_local = ref
        self._resolved = {}                  # scenario → resolved inputs (identical for every lever end of one result)

    def resolve(self, scenario: dict) -> tuple[pd.DataFrame, list]:
        """sc.resolve with a small cache: a result and its range re-run the engine 3–5 times on the same scenario, and
        resolving is the slow part; each caller gets its own copy, so cached inputs are never modified."""
        key = json.dumps(scenario.get("changes", []), sort_keys=True, default=str)
        if key not in self._resolved:
            if len(self._resolved) >= 8:
                self._resolved.pop(next(iter(self._resolved)))
            self._resolved[key] = sc.resolve(scenario, self.world)
        d, flags = self._resolved[key]
        return d.copy(), list(flags)

    def run(self, scenario: dict, eps: float | None = None, per_seat: float | None = None, cuts: str = "line",
            floor: float = 0.0) -> Result:
        """cuts/floor: the extrapolation rule for cuts beyond the trusted range (default: straight line; "log" is the
        alternative the effect band spans, Step 3.7)."""
        p = self.params; eps = p.eps if eps is None else eps; per_seat = p.per_seat if per_seat is None else per_seat
        if cuts not in ("line", "log") or (cuts == "log" and not floor > 0):
            raise ValueError("cuts must be 'line' or 'log' (with a positive floor)")
        G = lambda v: lever_G(v, eps, p.lo, p.hi, cuts, floor)
        d, flags = self.resolve(scenario)
        A0, L0, L1 = d.checkins_base.to_numpy(), d.p2p_base.to_numpy(), d.p2p_scen.to_numpy()
        fl = d.has_flight_data.to_numpy()
        dm = d.demand_mult.to_numpy()
        raw_all = np.zeros(len(d)); cap_all = np.full(len(d), np.inf)
        # 1. markets with baseline local passengers: the lever; the change in check-ins is capped by the change in local
        #    passengers (applied below, after the demand multiplier, so the cap holds in the scenario's demand conditions)
        std = fl & (L0 > 0)
        x = np.divide(L1, L0, out=np.ones(len(d)), where=std)
        raw = A0 * (G(x) - 1)
        raw_all[std] = raw[std]; cap_all[std] = np.abs(L1 - L0)[std]
        d["outside_trusted_range"] = std & (L1 != L0) & ((x - 1 < p.lo) | (x - 1 > p.hi))
        # 2. flight markets on days without baseline seats: from the full-cancellation point of the reference seats
        zero = fl & (L0 == 0) & (L1 > 0)
        if zero.any():
            Lref = np.array([self.ref_local.get((m, t.month), np.nan) for m, t in zip(d.market[zero], d.date[zero])])
            G0 = G(0.0)
            raw0 = A0[zero] / G0 * (G(L1[zero] / Lref) - G0)
            # a market with no service in any month has no reference: fall back to the new-route lever
            raw0 = np.where(np.isfinite(raw0), raw0, per_seat * d.seats_scen.to_numpy()[zero])
            raw_all[zero] = raw0; cap_all[zero] = L1[zero]
        d["zero_baseline_rule"] = zero
        # 3. nationalities without flight data: the new-route lever on the added seats, capped by their local passengers
        nf = ~fl & (d.new_route_seats.to_numpy() > 0)
        raw_all[nf] = (per_seat * d.new_route_seats.to_numpy())[nf]; cap_all[nf] = d.new_route_local.to_numpy()[nf]
        d["new_route_lever"] = nf
        # 4. demand (causes other than flights) scales the whole market; the flight effect is taken in the scenario's demand
        #    conditions and only then capped, so it never exceeds the change in local passengers whatever the demand
        flight = raw_all * dm
        dA = np.sign(flight) * np.minimum(np.abs(flight), cap_all)
        d["cap_binds"] = std & (np.abs(flight) > cap_all + 1e-9)
        d["flight_raw"] = raw_all; d["flight_cap"] = cap_all     # uncapped flight effect at demand 1, and its cap (for exact inversions)
        d["checkins_scen"] = A0 * dm + dA
        d["d_checkins_flights"] = dA                       # flight effect in the scenario's demand conditions (capped)
        d["d_checkins_demand"] = A0 * (dm - 1)             # demand effect on the baseline flights
        d["d_checkins"] = d.checkins_scen - A0
        # 5. guests in-house: Stage B on the scenario check-ins (linear in check-ins; stays scaled if asked)
        nh, sbm_ = self.stageb.n_h, self.stageb
        dG_h = np.zeros(len(d)); tail = 0.0; tail_m = {}
        d = d.sort_values(["market", "date"]).reset_index(drop=True)
        for mk, g in d.groupby("market", sort=False):
            st = g.stay_mult.to_numpy()
            if np.allclose(g.d_checkins, 0) and np.all(st == 1):
                continue
            b = sbm_.guests(mk, g.checkins_base.to_numpy())
            a_s = g.checkins_scen.to_numpy()
            s = sbm_.guests(mk, a_s, None if np.all(st == 1) else sbm_.hazard_scale(mk, a_s, st))
            dG_h[g.index] = (s - b)[:nh]; tail_m[mk] = float((s - b)[nh:].sum()); tail += tail_m[mk]
        d["d_guest_nights"] = dG_h
        d["guests_scen"] = d.guests_base + dG_h
        tot = d.groupby("date")[["guests_base", "guests_scen"]].sum()
        over = tot[tot.guests_scen > p.peak_guests]
        if len(over):
            flags.append(f"in-house guests above the highest level ever observed ({p.peak_guests:,.0f}) on {len(over)} days, "
                         f"max {over.guests_scen.max():,.0f} on {over.guests_scen.idxmax().date()} — check hotel capacity")
        for col, msg in [("outside_trusted_range", "change outside the range the lever was estimated on (−29% … +109%): straight-line continuation"),
                         ("cap_binds", "check-in change limited to the change in local passengers (cap)"),
                         ("zero_baseline_rule", "market without baseline seats: effect measured from the full-cancellation point of its reference seats"),
                         ("new_route_lever", "nationality without flight data: new-route lever (borrowed check-ins per added seat)")]:
            if d[col].any():
                mk = sorted(d.loc[d[col], "market"].unique())
                flags.append(f"{msg} — {int(d[col].sum())} market-days ({', '.join(m.title() for m in mk[:6])}{'…' if len(mk) > 6 else ''})")
        dL = float((d.p2p_scen - d.p2p_base).sum())
        dAf = float(d.d_checkins_flights.sum()); dAt = float(d.d_checkins.sum()); dGt = float(dG_h.sum()) + tail
        summary = {"eps": eps, "d_seats": float((d.seats_scen - d.seats_base).sum()), "d_passengers": float((d.pax_scen - d.pax_base).sum()),
                   "d_local_passengers": dL, "d_checkins_flights": dAf, "d_checkins_demand": float(d.d_checkins_demand.sum()),
                   "d_checkins": dAt, "d_guest_nights": dGt, "d_guest_nights_after_horizon": tail,
                   "checkins_per_extra_local_passenger": dAf / dL if abs(dL) > 1e-9 else None,
                   "guest_nights_per_extra_checkin": dGt / dAt if abs(dAt) > 1e-9 else None,
                   "baseline_checkins": float(d.checkins_base.sum()), "baseline_guest_nights": float(d.guests_base.sum()),
                   "pct_checkins": dAt / float(d.checkins_base.sum()) * 100, "pct_guest_nights": dGt / float(d.guests_base.sum()) * 100}
        bm = d.groupby("market")[["d_checkins_flights", "d_checkins", "d_guest_nights"]].sum()
        bm["d_guest_nights"] += pd.Series(tail_m).reindex(bm.index).fillna(0.0)          # incl. stays ending after the horizon
        summary["by_market"] = bm[(bm != 0).any(axis=1)].to_dict("index")
        return Result(scenario.get("name", "scenario"), d, summary, flags)
