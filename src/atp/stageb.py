"""Stage B — occupancy engine: daily check-ins → guests in-house, per market (45 nationalities + domestic).

Model (per market n, day t):

    G[n,t] = S0[n] · Σ_{k=0}^{K-1} A[n,t-k] · exp( −c[dow(t−k)] · M[n,t,k] )
    M[n,t,k] = Σ_{j=1}^{k} ΔH[n,j] · exp( x[t−k+j]ᵀ β )          (cumulative checkout hazard)

  A        check-ins (observed; Phase 0 table)
  S0       arrival-night reporting ratio (≈1; absorbs definitional differences)
  ΔH[n,j]  baseline checkout hazard on night j of the stay, > 0 (10 bands: nights 1..7, 8–14, 15–21, 22–44)
  x[d]     calendar of the CHECKOUT day d: month, weekday, Ramadan, long break, [Grand Prix]
  β        shared by all international markets (domestic has its own)
  c        optional arrival-weekday scale of the whole stay (domestic weekend-trip pattern)
  δ        optional per-market drift of the checkout rate, hazard × exp(δ·τ_t), frozen after the train end:
           a linear trend ("trend") or a recent step over the last 182 training days ("step"); see make_tau

Guarantees by construction: survival is non-increasing in k and ≤ S0 for any parameter values
(every factor multiplies a positive hazard), predictions are linear in arrivals (twice the check-ins →
twice the guests, which is what the scenario engine needs), and zero check-ins give zero guests.
Fitting: one joint bounded least-squares problem (scipy trust-region reflective) over every market's
parameters (log S0 + 10 hazard bands [+ drift δ]) and the shared β, with an exact, block-sparse Jacobian
(fit_group). A fit is converged when its projected relative gradient (the first-order stationarity measure under the
bounds) is at or below GRAD_TARGET; one that stops above it, for any reason, continues in further rounds and is reported
as not converged if it never gets there (Phase 5.2: one validation fit stopped far from its optimum at the step limit,
so a clean rebuild got a different model; audit: a tolerance stop alone no longer counts as converged). This is a local
check: it does not by itself show the optimum is global or that every parameter is separately identified.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls
from .lags import stock_flow_matrix

K = 45
GRAD_TARGET = 1e-3       # convergence target: projected relative gradient ‖P(∇SSE/2)‖ / SSE of a fit (Phase 5.2; audit)
MAX_ROUNDS = 10          # at most this many rounds of max_nfev steps
STEP_DAYS = 182          # recent-level window of the "step" drift shape (same as Stage A's C4b)
BANDS = np.array([min(k, 8) if k <= 7 else (8 if k <= 14 else 9 if k <= 21 else 10) for k in range(1, K)]) - 1
NH = 10


# ----------------------------------------------------------------------------- calendar design
def design(cal: pd.DataFrame, idx: pd.DatetimeIndex, freeze_after: pd.Timestamp, spec: dict) -> tuple[np.ndarray, list]:
    c = cal.set_index("date").reindex(idx)
    cols, names = [], []
    for m in range(2, 13):
        cols.append((c.month == m).to_numpy(float)); names.append(f"month_{m:02d}")
    for w, nm in zip(range(1, 7), ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        cols.append((c.dow == w).to_numpy(float)); names.append(f"checkout_{nm}")
    cols.append(c.is_ramadan.to_numpy(float)); names.append("ramadan")
    if spec.get("long_break", True):
        cols.append(c.is_long_break.to_numpy(float)); names.append("long_break")
    if spec.get("post_break", False):                       # Phase 5.4 candidate C1: the two days after a long break
        lb_ = c.is_long_break.astype(bool)
        last = lb_ & ~lb_.shift(-1, fill_value=False)
        cols.append((last.shift(1, fill_value=False) | last.shift(2, fill_value=False)).to_numpy(float)); names.append("post_break")
    if spec.get("grand_prix", False):
        cols.append(spec["gp_days"].reindex(idx, fill_value=False).to_numpy(float)); names.append("grand_prix")
    return np.column_stack(cols), names


# ----------------------------------------------------------------------------- one market
@dataclass
class Market:
    name: str
    L: np.ndarray          # T × K check-ins by lag
    y: np.ndarray          # T guests (0 where unknown)
    dow_arr: np.ndarray    # T × K weekday of the arrival day (t−k)
    theta: np.ndarray = field(default=None)   # [log S0, log ΔH_band × NH, (drift δ)]
    tau: np.ndarray = field(default=None)     # years relative to end of training, frozen at 0 after it


def build_markets(hi: pd.DataFrame, hd: pd.DataFrame, idx: pd.DatetimeIndex) -> dict:
    out = {}
    series = {n: g.set_index("date") for n, g in hi.groupby("nationality")}
    series["DOMESTIC"] = hd.set_index("date")
    dows = idx.dayofweek.to_numpy()
    lagdow = np.column_stack([np.r_[np.full(k, -1), dows[:len(dows) - k]] for k in range(K)])
    for n, s in series.items():
        s = s.reindex(idx)
        a = s.arrivals.fillna(0).to_numpy()
        L = np.column_stack([np.r_[np.zeros(k), a[:len(a) - k]] for k in range(K)])
        y = s.guests.fillna(0).to_numpy()
        out[n] = Market(n, L, y, lagdow)
    return out


def make_tau(idx: pd.DatetimeIndex, train_end: pd.Timestamp, shape: str = "trend") -> np.ndarray:
    """Time variable of the per-market drift δ (checkout hazard × exp(δ·τ)), 0 at and after the train end (frozen):
    "trend" τ = years before the train end (linear in time); "step" τ = −1 before the last STEP_DAYS training days,
    0 within them (a recent change in level, the Stage B counterpart of Stage A's C4b)."""
    if shape == "step":
        return np.where(idx > pd.Timestamp(train_end) - pd.Timedelta(days=STEP_DAYS), 0.0, -1.0)
    if shape != "trend":
        raise ValueError(f"unknown drift shape {shape!r}")
    return np.minimum((idx - pd.Timestamp(train_end)).days.to_numpy() / 365.25, 0.0)


def predict(m: Market, theta, e, cgam=None, jac=False, X=None, cohort_scale=None):
    """Guests for every day; optionally the Jacobian wrt theta, β (via X) and the arrival-weekday γ.
    If len(theta) == 1 + NH + 1, the last entry δ is a per-market drift of the checkout rate:
    hazard × exp(δ · τ_t), τ = years before the end of training (0 afterwards → frozen, no extrapolation).
    cohort_scale (scenario use only, length T): multiplies the checkout hazard of everyone who checked in on that
    day (< 1 = longer stays); 1 everywhere reproduces the fitted model exactly."""
    T = len(e)
    drift = len(theta) == 1 + NH + 1
    S0 = np.exp(theta[0]); dH = np.exp(theta[1:1 + NH])[BANDS]
    if drift:
        e = e * np.exp(theta[-1] * m.tau)
        if jac:
            X = np.column_stack([X, m.tau])
    M = np.zeros((T, K))
    if jac:
        Mb = np.zeros((NH, T, K)); p = X.shape[1]; Q = np.zeros((T, K, p)); eX = e[:, None] * X
    for k in range(1, K):
        M[1:, k] = M[:-1, k - 1]; M[:, k] += dH[k - 1] * e
        if jac:
            Mb[:, 1:, k] = Mb[:, :-1, k - 1]; Mb[BANDS[k - 1], :, k] += dH[k - 1] * e
            Q[1:, k] = Q[:-1, k - 1]; Q[:, k] += dH[k - 1] * eX
    C = np.ones((T, K)) if cgam is None else np.exp(np.r_[0.0, cgam])[np.clip(m.dow_arr, 0, 6)]
    if cohort_scale is not None:                                   # cohort of day t−k gets cohort_scale[t−k]
        cs = np.asarray(cohort_scale, dtype=float)
        # cs_lag[t, k] = cs[t − k] (1 before the start): one sliding window, same values as stacking K shifted copies
        C = C * np.lib.stride_tricks.sliding_window_view(np.r_[np.ones(K - 1), cs], K)[:T, ::-1]
    W = m.L * np.exp(-C * M)
    pred = S0 * W.sum(1)
    if not jac:
        return pred
    WC = W * C
    Jth = np.column_stack([pred, -S0 * np.einsum("tk,btk->tb", WC, Mb)])
    Jb = -S0 * np.einsum("tk,tkp->tp", WC, Q)
    if drift:
        Jth = np.column_stack([Jth, Jb[:, -1]]); Jb = Jb[:, :-1]
    Jg = None
    if cgam is not None:
        Jg = np.column_stack([-S0 * (WC * M * (m.dow_arr == w)).sum(1) for w in range(1, 7)])
    return pred, Jth, Jb, Jg


def init_theta(m: Market, ok: np.ndarray) -> np.ndarray:
    """Start from a monotone fixed curve (cumulative-sum NNLS), converted to banded hazards."""
    C = np.cumsum(m.L, 1)
    u, _ = nnls(C[ok], m.y[ok])
    S = np.cumsum(u[::-1])[::-1]
    S0 = max(S[0], 1e-3)
    Sn = np.clip(S / S0, 1e-6, 1)
    dH = np.maximum(-np.diff(np.log(Sn)), 1e-4)
    return np.r_[np.log(S0), [np.log(dH[BANDS == b].mean()) for b in range(NH)]]


LB_TH = np.r_[-3, np.full(NH, -12)]
UB_TH = np.r_[3, np.full(NH, 4)]


@dataclass
class FitResult:
    markets: dict
    beta: np.ndarray
    gamma: np.ndarray | None
    names: list
    sse_path: list
    grad_rel: float


def fit_group(markets: dict, X: np.ndarray, ok: np.ndarray, arrival_dow: bool = False,
              max_nfev: int = 300, drift: bool = False, idx: pd.DatetimeIndex = None, drift_shape: str = "trend",
              weights: dict | None = None) -> FitResult:
    """Joint least squares over ALL parameters (each market's θ + shared β [+ γ]) with an exact,
    block-sparse Jacobian: market n's θ only touches market n's rows; β and γ touch every row.
    weights (optional, Phase 5.4 candidate C3): per market, an array over all days; observation weights of the squared
    residuals (None = unweighted, the production fit)."""
    from scipy.sparse import csr_matrix, hstack as sp_hstack, block_diag
    p = X.shape[1]; names = list(markets); N = len(names); nth = 1 + NH + (1 if drift else 0); ng = 6 if arrival_dow else 0
    masks = {n: ok & (markets[n].y > 0) for n in names}      # suppressed/unknown guests are never targets
    t_end = idx[ok].max() if drift else None
    for n in names:
        th = init_theta(markets[n], masks[n])
        if drift:
            markets[n].tau = make_tau(idx, t_end, drift_shape)
            th = np.r_[th, 0.0]
        markets[n].theta = th
    lbt = np.r_[LB_TH, -1.5] if drift else LB_TH; ubt = np.r_[UB_TH, 1.5] if drift else UB_TH
    z0 = np.r_[np.concatenate([markets[n].theta for n in names]), np.zeros(p + ng)]
    lb = np.r_[np.tile(lbt, N), np.full(p + ng, -5.0)]
    ub = np.r_[np.tile(ubt, N), np.full(p + ng, 5.0)]
    Y = np.concatenate([markets[n].y[masks[n]] for n in names])
    sw = None if weights is None else np.concatenate([np.sqrt(np.asarray(weights[n], dtype=float)[masks[n]]) for n in names])
    sse_path = []

    def split(z):
        th = z[:N * nth].reshape(N, nth); b = z[N * nth:N * nth + p]; g = z[N * nth + p:] if ng else None
        return th, b, g

    def fun(z):
        th, b, g = split(z); e = np.exp(X @ b)
        r = np.concatenate([predict(markets[n], th[i], e, g)[masks[n]] for i, n in enumerate(names)]) - Y
        if sw is not None:
            r = r * sw
        sse_path.append(float(r @ r)); return r

    def jac(z):
        th, b, g = split(z); e = np.exp(X @ b); blocks, shared = [], []
        for i, n in enumerate(names):
            _, Jth, Jb, Jg = predict(markets[n], th[i], e, g, jac=True, X=X)
            mk = masks[n]; blocks.append(csr_matrix(Jth[mk]))
            shared.append(np.hstack([Jb, Jg])[mk] if ng else Jb[mk])
        J = sp_hstack([block_diag(blocks, format="csr"), csr_matrix(np.vstack(shared))], format="csr")
        return J if sw is None else csr_matrix(J.multiply(sw[:, None]))

    def solve(z):
        return least_squares(fun, z, jac=jac, bounds=(lb, ub), method="trf", tr_solver="lsmr", x_scale="jac",
                             max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-8)

    def rel_grad(r):
        """Projected relative gradient ‖P(∇)‖ / SSE: the first-order (KKT) stationarity measure for a bound-constrained
        least-squares fit. A coordinate at a bound counts only where the gradient points into the feasible side, i.e.
        where moving off the bound would still lower the SSE; at an interior point it is the plain gradient."""
        grad = jac(r.x).T @ r.fun
        at_lo, at_hi = r.x <= lb + 1e-6, r.x >= ub - 1e-6
        pg = np.where(at_lo, np.minimum(grad, 0.0), np.where(at_hi, np.maximum(grad, 0.0), grad))
        return float(np.linalg.norm(pg) / (r.fun @ r.fun))

    # Convergence is decided by one criterion: the projected relative gradient at or below GRAD_TARGET. A fit that stops
    # above it — for any reason, out of steps or a step/function tolerance met first — continues from where it stopped,
    # in rounds of max_nfev steps, up to MAX_ROUNDS; if the criterion is still not met the fit is reported as not
    # converged (the callers list non-converged fits). Fits that meet it in the first round are unchanged.
    r = solve(np.clip(z0, lb + 1e-9, ub - 1e-9)); grad_rel = rel_grad(r); rounds = 1; nfev = r.nfev
    while grad_rel > GRAD_TARGET and rounds < MAX_ROUNDS:
        r = solve(r.x); grad_rel = rel_grad(r); rounds += 1; nfev += r.nfev
    th, b, g = split(r.x)
    for i, n in enumerate(names):
        markets[n].theta = th[i]
    res = FitResult(markets, b, g, [], [sse_path[0], float(r.fun @ r.fun)], grad_rel)
    res.status, res.nfev, res.rounds = r.status, nfev, rounds
    res.converged = grad_rel <= GRAD_TARGET
    res.at_bound = int(((r.x <= lb + 1e-6) | (r.x >= ub - 1e-6)).sum())
    return res


def predict_all(res: FitResult, X: np.ndarray) -> dict:
    e = np.exp(X @ res.beta)
    return {n: predict(m, m.theta, e, res.gamma) for n, m in res.markets.items()}


def survival_curve(theta, beta, x_rows: np.ndarray, gamma_arr: float = 0.0) -> np.ndarray:
    """Share of a cohort still in-house after k nights, for a given sequence of checkout-day calendars."""
    dH = np.exp(theta[1:])[BANDS]
    e = np.exp(x_rows[1:K] @ beta)
    return np.exp(theta[0]) * np.exp(-np.exp(gamma_arr) * np.r_[0.0, np.cumsum(dH * e)])


def assert_lag_windows(hi, hd, dates: pd.DatetimeIndex):
    """Phase 0 guard reused: every date we fit or predict must have its full 45-day arrival history."""
    series = {n: g.set_index("date").arrivals for n, g in hi.groupby("nationality")}
    series["DOMESTIC"] = hd.set_index("date").arrivals
    for s in series.values():
        stock_flow_matrix(s, dates, K)
