# Seats2Stays — maths audit pack

This pack is for an independent audit of the **mathematics and statistics** of a flight → hotel-demand simulator
(ATP 2026, DCT Abu Dhabi track). The raw competition data are licensed to the challenge and are **not included**;
everything else needed to check the maths is: the technical brief, the code that implements every equation, the
reports and the saved estimates and scores. This file is the guide to all of it.

## Read in this order

1. `Technical_Brief.pdf` — all equations, choices and results in about eight pages (sections 2–3 are the maths).
2. The code that implements each equation (table below).
3. `reports/` — how each phase was built and validated, with every number generated from saved outputs.
4. `results/` — the saved estimates and scores the brief quotes, so any number can be traced.
5. **Only after your own audit is complete:** `previous_review/`, an earlier independent re-derivation of 114 numbers
   from the raw files and its comparison with ours. Please do not open it before writing your findings, so that it
   cannot bias them; afterwards, compare your findings with it.

## Where each piece of maths lives

| Brief section | What | Code |
|---|---|---|
| 2.1 | ε: year-on-year log changes, month fixed effects (within transform), seat leads, trimming to the 5th–95th percentile of the seat change, country-cluster bootstrap, out-of-sample lever check (calibration slope, skill) | `src/atp/stagea.py` (`yoy`, `trusted_range`, `trim`, `_demean`, `fe_elasticity`, `lever_skill`, `lever_interval`); `src/atp/phase2.py` (`_window`, `_primary`, `_fold_estimates`, `_lever_check_test`) |
| 2.1 | Small-cluster check: wild cluster bootstrap-t; market-error independence | `scripts/audit_checks.py` → `results/audit_checks.json` |
| 2.2 | Lever function G(x): power law inside the tested range, tangent line outside, cap at local passengers; new-route rate | `src/atp/stagea.py` (`seat_lever`, `new_route_lever`); `src/atp/engine.py` (`lever_G`); `src/atp/phase2.py` (`step_newroute`, `_loo`) |
| 2.3 | Baseline check-in forecast: Poisson GLM with log link, ε × log seats as an offset | `src/atp/forecaster.py` |
| 2.4 | Survival model of guests in-house: discrete-time proportional hazards (cloglog), joint bounded nonlinear least squares, analytic block-sparse Jacobian | `src/atp/stageb.py` (`predict`, `fit_group`, `design`, `make_tau`) |
| 2.5 | Scenario chain, stay-length root-finding, occupancy points, flights-needed root | `src/atp/engine.py` (`run`, `hazard_scale`, `guests`); `src/atp/scenario.py`; `src/atp/sensitivity.py`; `src/atp/phase4.py` (`scad_monthly`, `occupancy_basis`) |
| 3 | 80% range: lever part (engine re-run at ε's interval ends), level part (tiered actual ÷ forecast bands calibrated across folds), quadrature | `src/atp/bands.py` (`calibrate_level`, `Bands.run`, `_level_dev`, `end_runs`) |
| 4 | Back-tests, WMAPE at every level, benchmarks | `src/atp/phase1.py` (`baselines`, `metrics`), `src/atp/forecaster.py` (`wmape`, `metrics`), `src/atp/phase2.py` (`step_endtoend`) |
| — | Sensitivity (one at a time) and the uncertainty tornado | `src/atp/sensitivity.py`, `src/atp/phase3.py` (`step_sensitivity`) |
| 3 | Error by months ahead: slope of WMAPE on months ahead, nationality bootstrap, coverage by month | `src/atp/ahead.py`; rules in `docs/phase5_7_preregistration.md` |
| 4 | Clean rebuild and comparison with the snapshot | `scripts/clean_rebuild.sh`, `src/atp/phase5.py` (`compare_rebuild`) |
| 5 | LightGBM benchmark (same inputs, rows, folds, WMAPE; month-block bootstrap) | `src/atp/benchmark.py`; rules in `docs/phase5_5_preregistration.md` |
| 5 | Sobol indices on the engine (exact vectorised form, gate against `Engine.run`) | `src/atp/sobol.py`; rules in `docs/phase5_6_preregistration.md` |
| 5 | The rejected candidates and their pre-set rules | `src/atp/phase5.py` (`candidates`, `candidates_bc`); rules in `docs/phase5_4_preregistration*.md` |
| — | Summary of all validation checks | `reports/phase5_report.md` |
| — | Data rules (suppression, low-coverage days, country join) | `src/atp/hotels.py`, `src/atp/flights.py`, `reports/data_card.md` |
| — | Dates, folds, constants | `config.yaml` |

`tests/` holds the unit tests that check the maths directly (e.g. the within transform equals month dummies, the
month effect removes a shared shock in a simulated panel, the survival model is linear in check-ins and monotone, the
engine's null scenario reproduces the baseline, the band rules).

## What we would most like checked

1. **Section 2.4.** Is the survival model correctly specified as a discrete-time proportional-hazards model, are S₀,
   the banded hazards and the shared calendar effects jointly identified, and is unweighted least squares on daily
   guest levels (rather than a Poisson or multinomial likelihood) defensible?
2. **Section 3.** Is the quadrature combination of lever and level uncertainty valid (delta method, independence,
   the worst-case bound given), and is the level band's calibration on two 7-month folds honest?
3. **Section 2.1.** Identification of ε (year-on-year, month fixed effects, seat leads), the "smaller of two
   estimates" rule, trimming on the regressor, the cluster bootstrap with 33 clusters, and the out-of-sample check.
4. **Section 2.2.** The tangent-line extrapolation and the new-route rate dA/dS = ε × A/S.
5. Anything stated with more confidence than the maths supports, and any error in an equation or in its code.

Please report: each problem, where it is (section and file/function), why it is wrong, how serious it is, and the fix.

## Suggested prompt

> You are a senior statistician and econometrician auditing a forecasting and scenario model. Read
> Technical_Brief.pdf, then check every equation against the code that implements it (README_AUDIT.md maps one to
> the other). Do not open previous_review/ until your own findings are written; then compare them with it. For each issue give: location, what is wrong, why, severity (critical / major / minor), and the fix.
> Distinguish errors in the maths or code from questionable modelling choices and from wording that overstates
> confidence. Check the numbers in the brief against results/ and reports/. Do not assume something is right because
> it is stated; verify it.
