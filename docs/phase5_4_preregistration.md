# Phase 5, Step 5.4 — three candidate improvements, pre-registered

Written 11 Oct 2026, before any of the three was fitted or scored. The definitions, the folds, the metrics and the
decision rules below are fixed; nothing is tuned after results are seen. `python -m atp.phase5 candidates` runs all
three and records the sha256 of this file with the results, so the rules cannot change silently afterwards.

All three are judged on the two validation folds only (A: trained to Jul 2024, held out Aug 2024 – Feb 2025;
B: trained to Dec 2024, held out Jan – Jul 2025). The official test window is not used to choose anything.
Each is compared with the current production model, refitted the same way (same code, same folds).

## C1 — post-break checkout term (Stage B, survival model)

**Why.** The largest jump in daily guest error in fold B came after a holiday break (2 Apr 2025, the day after the Eid
al-Fitr break, +10 points): guests who came for the break leave together when it ends, and the model has no term for it
(Phase 2 report, "Carried forward").

**Definition.** One calendar dummy on the checkout day, `post_break` = 1 on the two days immediately after the last day
of each long break (`is_long_break` in the calendar), 0 otherwise. Added to the checkout calendar x of both groups:
international (shared β, with the kept long-break term and per-market recent step) and Domestic (its own β, with its
kept trend). Fitted exactly as the production model otherwise.

**Rule (the project's rule for a hotel-side calendar term, as tightened in the 4 Oct audit).** For each group
separately (international on daily × nationality WMAPE, Domestic on daily WMAPE): keep if fold A improves, fold B does
not worsen by more than 0.05 points, and the monthly WMAPE (international: monthly × nationality; Domestic: monthly)
does not worsen on either fold.

## C2 — recency weighting of ε (Stage A, the seat lever)

**Why.** ε fell from about 0.5 (2024) to 0.26 (2025 months), and the fold-B estimate (0.51) failed the out-of-sample
lever test on Jan – Jul 2025 (Phase 2, Step 2.3). Weighting recent months more is the standard response to a drifting
parameter.

**Definition.** The same estimation (trusted range, month fixed effects, plain and lead-controlled specifications, the
smaller of the two as default), by weighted least squares with weight 0.5^(a/12) for a country-month a months before
the last month of the estimation window (half-life 12 months, fixed in advance, not tuned).

**Rule (the Step 2.3 lever test, on the same fold-B rows).** Estimate on data to Dec 2024; test on Jan – Jul 2025
(trimmed to the fold-B trusted range, festival-adjusted with fold-B festival effects). Adopt only if the weighted ε
(a) passes the Step 2.3 test (calibration slope's 95% interval contains 1 and skill > 0) and (b) has skill at least
0.01 higher than the unweighted fold-B ε on these rows, and (c) on the fold-A version of the test (estimate to Jul 2024,
test Aug 2024 – Jul 2025) its skill is not lower than the unweighted fold-A ε's by more than 0.01.

## C3 — Poisson-type weighting of the survival-model fit (Stage B)

**Why.** Unweighted least squares on daily guest levels gives the largest markets the most weight; a Poisson
likelihood (variance proportional to the mean) is the usual alternative for counts (technical brief, question 4).

**Definition.** One step of iteratively reweighted least squares towards the Poisson quasi-likelihood: refit the
production model with each observation weighted by 1 / Ĝ, where Ĝ is the production (unweighted) fit's prediction for
that cell, floored at 10 (the suppression threshold). Everything else unchanged.

**Rule (the project's rule for a change that touches every market: the 4 Oct audit rule).** Keep only if international
daily × nationality WMAPE improves by at least 0.10 points on fold A, does not worsen by more than 0.05 on fold B, and
monthly × nationality WMAPE does not worsen on either fold; Domestic must not worsen on either fold (daily and monthly),
since the same estimator would be used for both groups.

## What happens next

A candidate that passes is carried through every later phase (Step 5.8), and every number it moves is listed against
the snapshot. A candidate that fails is reported with its numbers and dropped.
