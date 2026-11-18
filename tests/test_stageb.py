"""Stage B model invariants — the physical guarantees the model claims, tested directly."""
import pytest
import numpy as np
import pandas as pd
from atp import stageb as sb

rng = np.random.default_rng(0)
T = 200
IDX = pd.date_range("2024-01-01", periods=T)


def _market(arrivals):
    L = np.column_stack([np.r_[np.zeros(k), arrivals[:T - k]] for k in range(sb.K)])
    dows = IDX.dayofweek.to_numpy()
    lagdow = np.column_stack([np.r_[np.full(k, -1), dows[:T - k]] for k in range(sb.K)])
    return sb.Market("X", L, np.zeros(T), lagdow)


def _random_params(p=8):
    theta = np.r_[rng.normal(0, 0.2), rng.normal(-1.2, 0.8, sb.NH)]
    beta = rng.normal(0, 1.0, p)               # deliberately large calendar effects
    X = (rng.random((T, p)) < 0.3).astype(float)
    return theta, beta, X


def test_survival_never_increases_and_never_exceeds_arrival_night():
    for _ in range(200):
        theta, beta, X = _random_params()
        s = sb.survival_curve(theta, beta, X[:sb.K], gamma_arr=rng.normal(0, 1))
        assert np.all(np.diff(s) <= 1e-12) and s[0] == np.exp(theta[0]) and np.all(s <= s[0] + 1e-12)


def test_linear_in_arrivals_and_zero_in_zero_out():
    theta, beta, X = _random_params()
    a = rng.poisson(100, T).astype(float)
    e = np.exp(X @ beta); g = rng.normal(0, .5, 6)
    p1 = sb.predict(_market(a), theta, e, g)
    p2 = sb.predict(_market(2.5 * a), theta, e, g)
    assert np.allclose(p2, 2.5 * p1)
    assert np.allclose(sb.predict(_market(np.zeros(T)), theta, e, g), 0)


def test_single_cohort_matches_survival_curve():
    theta, beta, X = _random_params()
    a = np.zeros(T); a[50] = 1000.0
    pred = sb.predict(_market(a), theta, np.exp(X @ beta))
    s = sb.survival_curve(theta, beta, X[50:50 + sb.K])
    assert np.allclose(pred[50:50 + sb.K], 1000 * s)


def test_analytic_jacobian_matches_finite_differences():
    theta, beta, X = _random_params(p=4)
    a = rng.poisson(50, T).astype(float); m = _market(a); g = rng.normal(0, .3, 6)
    _, Jth, Jb, Jg = sb.predict(m, theta, np.exp(X @ beta), g, jac=True, X=X)
    h = 1e-6
    for i in range(len(theta)):
        d = np.zeros_like(theta); d[i] = h
        fd = (sb.predict(m, theta + d, np.exp(X @ beta), g) - sb.predict(m, theta - d, np.exp(X @ beta), g)) / (2 * h)
        assert np.allclose(fd, Jth[:, i], rtol=1e-4, atol=1e-5)
    for i in range(len(beta)):
        d = np.zeros_like(beta); d[i] = h
        fd = (sb.predict(m, theta, np.exp(X @ (beta + d)), g) - sb.predict(m, theta, np.exp(X @ (beta - d)), g)) / (2 * h)
        assert np.allclose(fd, Jb[:, i], rtol=1e-4, atol=1e-5)
    for i in range(6):
        d = np.zeros(6); d[i] = h
        fd = (sb.predict(m, theta, np.exp(X @ beta), g + d) - sb.predict(m, theta, np.exp(X @ beta), g - d)) / (2 * h)
        assert np.allclose(fd, Jg[:, i], rtol=1e-4, atol=1e-5)


def test_drift_jacobian_and_freeze():
    theta, beta, X = _random_params(p=4)
    theta = np.r_[theta, 0.4]                                   # δ = +0.4 / year
    a = rng.poisson(50, T).astype(float); m = _market(a)
    m.tau = np.minimum((np.arange(T) - 150) / 365.25, 0.0)     # frozen at 0 after day 150
    e = np.exp(X @ beta)
    _, Jth, Jb, _ = sb.predict(m, theta, e, jac=True, X=X)
    h = 1e-6; d = np.zeros_like(theta); d[-1] = h
    fd = (sb.predict(m, theta + d, e) - sb.predict(m, theta - d, e)) / (2 * h)
    assert np.allclose(fd, Jth[:, -1], rtol=1e-4, atol=1e-5)
    # after the freeze point, predictions do not depend on δ for cohorts that arrived after it
    late = np.arange(150 + sb.K, T)
    assert np.allclose(np.abs(Jth[late, -1]), 0, atol=1e-9)


def test_drift_shapes_freeze_after_train_end():
    idx = pd.date_range("2024-01-01", "2025-12-31"); te = pd.Timestamp("2025-07-31")
    tr, st = sb.make_tau(idx, te, "trend"), sb.make_tau(idx, te, "step")
    assert (tr[idx >= te] == 0).all() and (np.diff(tr[idx <= te]) > 0).all()
    assert (st[idx > te - pd.Timedelta(days=sb.STEP_DAYS)] == 0).all() and (st[idx <= te - pd.Timedelta(days=sb.STEP_DAYS)] == -1).all()
    with pytest.raises(ValueError):
        sb.make_tau(idx, te, "wiggle")



def test_a_fit_that_runs_out_of_steps_continues_until_it_converges():
    """Phase 5.2: a fit stopped by its step limit far from the optimum continues in rounds until it converges; one
    that converges in the first round runs exactly one round (unchanged)."""
    r = np.random.default_rng(5)
    theta = np.r_[r.normal(0, 0.2), r.normal(-1.2, 0.8, sb.NH)]; beta = r.normal(0, 1.0, 3)
    X = (r.random((T, 3)) < 0.3).astype(float) * 0.3
    data = []
    for i in range(3):
        m = _market(r.poisson(200, T).astype(float))
        data.append((f"M{i}", m.L, sb.predict(m, theta + r.normal(0, 0.05, len(theta)), np.exp(X @ beta)) * r.lognormal(0, 0.02, T), m.dow_arr))
    ok = np.arange(T) >= sb.K
    fit = lambda nfev: sb.fit_group({n: sb.Market(n, L, y, d) for n, L, y, d in data}, X, ok, max_nfev=nfev)
    short = fit(40)
    assert short.rounds > 1 and short.converged and short.grad_rel <= sb.GRAD_TARGET
    full = fit(300)
    assert full.rounds == 1 and full.converged
    for f in (short, full):                                                     # audit E8: one criterion decides convergence
        assert f.converged == (f.grad_rel <= sb.GRAD_TARGET) and hasattr(f, "at_bound")


def _fit_data(seed=5, n=3):
    r = np.random.default_rng(seed)
    theta = np.r_[r.normal(0, 0.2), r.normal(-1.2, 0.8, sb.NH)]; beta = r.normal(0, 1.0, 3)
    X = (r.random((T, 3)) < 0.3).astype(float) * 0.3
    data = []
    for i in range(n):
        m = _market(r.poisson(200 * (i + 1), T).astype(float))
        data.append((f"M{i}", m.L, sb.predict(m, theta + r.normal(0, 0.05, len(theta)), np.exp(X @ beta)) * r.lognormal(0, 0.05, T), m.dow_arr))
    return X, data


def test_unit_weights_reproduce_the_unweighted_fit_and_weights_change_it():
    """Phase 5.4 (candidate C3): observation weights enter as √w on residuals and Jacobian rows; all-ones weights are the
    production fit exactly, other weights give a different optimum."""
    X, data = _fit_data()
    ok = np.arange(T) >= sb.K
    fit = lambda w: sb.fit_group({n: sb.Market(n, L, y, d) for n, L, y, d in data}, X, ok, weights=w)
    a, b = fit(None), fit({n: np.ones(T) for n, *_ in data})
    assert np.allclose(a.beta, b.beta) and all(np.allclose(a.markets[n].theta, b.markets[n].theta) for n in a.markets)
    c = fit({n: 1.0 / np.maximum(y, 10) for n, L, y, d in data})
    assert c.converged and not np.allclose(a.beta, c.beta, atol=1e-6)


def test_post_break_term_marks_the_two_days_after_each_long_break():
    """Phase 5.4 (candidate C1): post_break = 1 on the two days immediately after the last day of a long break."""
    dates = pd.date_range("2025-03-25", "2025-04-10")
    brk = dates.isin(pd.date_range("2025-03-29", "2025-04-01"))
    cal = pd.DataFrame({"date": dates, "month": dates.month, "dow": dates.dayofweek, "is_ramadan": False, "is_long_break": brk})
    X, names = sb.design(cal, dates, dates[-1], {"long_break": True, "post_break": True})
    assert names[-1] == "post_break"
    assert list(dates[X[:, -1] == 1].strftime("%m-%d")) == ["04-02", "04-03"]
    X0, n0 = sb.design(cal, dates, dates[-1], {"long_break": True})
    assert "post_break" not in n0 and np.array_equal(X0, X[:, :-1])
