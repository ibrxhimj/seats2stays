# Phase 5, Step 5.6 — global sensitivity (Sobol), pre-registered

Written 11 Oct 2026, before any Sobol index was computed. `python -m atp.phase5 sobol` records the sha256 of this file
with the results.

## The questions

1. **Levers.** When the levers move together, which ones account for most of the change in guest nights? Do they
   interact, or does each act on its own? Does this confirm or correct the Tab 2 ranking, which moves one lever at a
   time and ranks by the effect of a 1% change?
2. **Uncertainty.** For a typical result, which uncertainty dominates its range: the check-in response to seats (ε) or
   the baseline level the change scales with?

## The model being analysed

The Phase 3 scenario engine itself (`engine.Engine.run`), not a surrogate. For speed it is evaluated in an exact
vectorised form (`atp.sobol.FastEngine`) that uses two properties of the engine:

- every lever here is uniform over whole calendar months, and the route flows are the same on every day of a month, so
  seats → passengers → local passengers can be computed per route and month with the engine's own rules (full-plane cap,
  transfer share capped at 99%, route shares in proportion);
- Stage B is linear in check-ins, and the stay lever is solved so that the guests checking in on the changed days stay
  exactly ×v nights, so guest nights are a fixed weight per market and check-in day times check-ins (weights computed
  once from Stage B).

**Gate.** Before any index is computed, the fast form must reproduce `Engine.run` to a relative difference ≤ 1e-6 on
the total change in guest nights and on each market's change: for the ten Tab 2 runs (each lever ±10%), for eight
random combinations of all five levers, and for the reference results below at both ends of ε's 80% interval. If it
does not, no index is reported.

## Part 1 — levers

The five Tab 2 levers, each drawn independently and uniformly from −10% to +10% (the size Tab 2 shows), over the
whole horizon (Mar 2026–Feb 2027): seats, load factor and transfer share on all flight markets; non-flight demand and
stay length on all markets. When several move, they are applied in the order seats → load factor → transfer share →
demand → stay length (the order of a scenario's change list). ε at its central value. Output: the change in guest nights
over the horizon, including stays that end after it.

- **1a**, all five levers; **1b**, the three flight levers only (seats, load factor, transfer share), so their shares
  are not swamped by demand and stay length, which move guest nights one for one;
- **1c**, a check on size: 1a with ±25% instead of ±10%, where the full-plane cap, the 99% transfer cap and the
  straight line beyond the trusted range act more (descriptive).

Sobol indices with SALib 1.6.0: Saltelli sampling from a scrambled Sobol sequence (seed 1), N = 8,192 base samples,
second-order indices on (N × (2D + 2) model runs), 95% intervals from 1,000 bootstrap resamples (seed 1). Reported:
first-order S1 (the lever alone), total-order ST (the lever with all its interactions), second-order S2 for every pair,
and 1 − ΣS1 (the share due to interactions). Convergence: the indices with N = 4,096 are reported next to them.

**Reading rules (fixed now).**

- *Ranking.* Tab 2's order (demand = stay length > transfer share > seats > load factor, by |elasticity| at ±1%) is
  **confirmed** if ordering the levers by ST in 1a gives the same order, levers whose 95% intervals for ST overlap
  counting as tied. Otherwise it is **corrected**, and the Sobol order is shown on Tab 3 with the reason.
- *Interactions.* They **matter** if 1 − ΣS1 > 0.05 or any lever's ST − S1 > 0.05 (in 1a or 1b); otherwise the levers
  act on their own and the one-at-a-time ranking is enough.

## Part 2 — uncertainty

Two reference results: **R1**, all flight markets +10% seats Nov 2026–Feb 2027 (the reference of the Phase 3
uncertainty tornado); **R2**, the same for the flight market with the most baseline guest nights (India), where a
single market's level error does not average out.

Uncertain inputs, independent:

- **ε**, normal with mean 0.3117 and standard deviation 0.3117 × 0.2868 (the out-of-sample cluster standard error of the
  calibration slope), i.e. exactly the 80% interval 0.197–0.426 the app uses;
- **baseline level**, each market's own ratio actual ÷ forecast, log-normal with its 10th and 90th percentiles at the
  80% guest-night band of its size tier (Step 3.5), independent between markets (the bands' assumption, checked in
  `scripts/audit_checks.py`); the markets form one group.

Output: the change in guest nights, Σ over markets of (level ratio × the engine's change at that ε). Sobol indices for
the two groups (ε, baseline level), same settings as Part 1. **Reading rule:** an uncertainty **dominates** if its ST is
above 0.5. Set against the split the app's 80% range implies (lever and level half-widths, squared, as shares).

Not part of R1 or R2, so not analysed: the new-route rate (nationalities without flights) and the rule for cuts beyond
the trusted range (both are only in other scenarios' ranges).

## What follows

The result is a small block on Tab 3 (the levers' shares, whether they interact, which uncertainty dominates) and one
line in the deck. If the ranking is corrected, Tab 2's order is changed to the Sobol order and the reason stated.

## Amendment A1 — added in the audit of 11 Oct 2026, after the first run

Two gaps found when auditing the first run; both make the step stricter, neither changes a rule above.

1. **The gate did not cover the ±25% size check (1c).** Eight more gate cases are added: the four corners where all
   five levers sit at ±25% in alternating signs, and four random combinations within ±25% (seed 2). The same limit
   (1e-6) applies to all 30 cases.
2. **The app's level part is not always the independent-markets quadrature.** For a net effect over many markets the app
   never lets the level part fall below the all-market monthly band (`bands.Bands._level_dev`); for R1 that floor is what
   the app uses. A third uncertainty run, **R1 with the level shared**, draws one level ratio common to every market from
   the all-market 80% band (log-normal, same construction). Independent markets and one shared level are the two ends of
   how market errors can be related, so the level's share in R1 lies between the two runs. The reading rule (ST > 0.5)
   is applied to both.
