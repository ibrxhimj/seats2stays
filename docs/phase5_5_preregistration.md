# Phase 5, Step 5.5 — gradient-boosting benchmark, pre-registered

Written 11 Oct 2026, before any LightGBM model was fitted or scored. `python -m atp.phase5 benchmark` records the sha256 of
this file with the results.

## The question

Does a flexible machine-learning model, given the same information as our daily check-in forecaster, predict hotel
check-ins better than our model does? It is reported whatever the outcome.

## What is compared

**Our model:** the Step 2.4 daily check-in forecaster (one Poisson regression per nationality, with the check-in response
ε to seats), exactly as saved in `data/processed/phase2/stagea_pred_{A,B}.parquet`. Before anything is compared, its
scores are recomputed from those files and must equal the ones saved in `step24_forecaster.json`.

**The benchmark:** one LightGBM model (Poisson objective) over all 46 series (45 nationalities + Domestic), trained on the
same rows and scored on the same days.

## Same inputs, nothing more

The benchmark sees the same information as the forecaster and nothing the forecaster does not see:

- the series (as a category), month, weekday, Ramadan, the UAE main holiday breaks, the nationality's own festival window,
  the year-end window, the Grand Prix and ADIPEC days, the source market's home holidays, UAE school breaks (Domestic only)
  — the calendar terms of the chosen forecaster specification (`C1 C2 C3 C4b C6 C7`);
- the "last 182 training days" indicator (C4b), which depends on dates only;
- seats: the 7-day seats on the route (floored at 5% of the training mean, as in the forecaster), as log seats and as log
  seats relative to the series' training mean; empty for series without flights.

It never sees a hotel figure from the period it predicts: no lagged check-ins, no rolling means, no occupancy, no guests,
no check-ins from any day after the training end. A unit test checks this by replacing every holdout check-in with
zeros and requiring identical predictions.

Estimation rows, start date (7 Jan 2023), fold training ends, and the exclusion of flight-disruption days and days below
the seat floor are the forecaster's. The official test window (Aug 2025–Feb 2026) is not used.

## The folds and the measure

Folds A (train to Jul 2024, holdout Aug 2024–Feb 2025) and B (train to Dec 2024, holdout Jan–Jul 2025), the same as for
every other model in the project. The measure is WMAPE (`forecaster.metrics`), the same function that scored ours:

- **primary:** international daily × nationality (Step 2.4's headline);
- also: international monthly × nationality, international total daily and monthly, Domestic daily and monthly.

**Reading the result.** The benchmark wins a measure on a fold if its WMAPE is lower. We say "LightGBM is more accurate"
only if it wins the primary measure on both folds. A win on one fold, or by less than the month-block interval below, is
reported as a tie. The interval comes from resampling the months of each holdout (2,000 draws, seed 7) and recomputing the
difference in WMAPE (benchmark − ours); with 7 months per fold it is wide, and is shown so that nobody over-reads a gap.

## The benchmark's settings (fixed now, not tuned on the holdout)

`objective=poisson`, 400 trees, learning rate 0.03, 15 leaves, minimum 20 rows per leaf, row and column sampling 0.8,
L2 penalty 1.0, series as a categorical feature; the prediction is the mean of five runs (seeds 1–5, one thread,
deterministic). These are common defaults for data of this size, picked before looking at any result.

A **sensitivity grid** is reported in full and is not used to choose: leaves {7, 15, 31} × (200 trees at 0.05, 600 trees
at 0.03). If any setting beats our model, that is reported as it stands; the primary setting above stays the one the
conclusion is drawn from.

## Descriptive checks (no decision rule)

1. **Without seats.** The same benchmark without the two seat features, and ours with ε = 0 (the Step 2.4 sensitivity),
   to show how much of each model's accuracy comes from the seat link.
2. **What the benchmark implies about seats.** On the holdout, seats of every series with flights are multiplied by 0.9,
   1.1 and 1.5 and the benchmark is asked again. The implied elasticity, log(total predicted check-ins ratio) ÷ log(seat
   ratio), is set against the fold's ε (0.56 on A, 0.51 on B; the project's default range is 0.20–0.43).

## What follows from the result

- If the benchmark is no better than ours: the interpretable model gives up nothing in accuracy.
- If it is better: the case for keeping our model is what it can do and the benchmark cannot — carry a seat change through
  the chain to guest-nights with a stated elasticity, give a range, and be explained to a planner — and the size of the gap
  is stated plainly next to the lever's own check (check 2). The benchmark is not adopted, because it replaces only the
  check-in step and has no hotel-stay side (Stage B), no lever, and no uncertainty range.

## Amendment A1 — added after the first run (11 Oct 2026)

The first run (primary setting, then the grid above) was done before this amendment. The primary result was: international
daily × nationality WMAPE 38.42 (ours 33.11) on fold A and 34.83 (ours 31.76) on fold B. In the grid, accuracy improved
with larger models (up to 36.85 on A and 32.16 on B), so the primary setting may be too small for the data. To avoid
that being a reason to doubt the comparison, four larger settings are added as a **best-case check for the benchmark**:
leaves {31, 63} × (1,200 trees at 0.03 with 20 rows per leaf; 2,000 trees at 0.03 with 10 rows per leaf).

These four, like the first grid, are reported in full. They were chosen after seeing results, and the best setting of the
ten is picked with hindsight on the holdout, so it flatters the benchmark; it is shown to say how far it could get, not
as the benchmark's score. The primary setting and the reading rule above are unchanged, and the first-run primary numbers
are unchanged by this amendment (the run is deterministic).
