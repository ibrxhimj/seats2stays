# Phase 5, Step 5.4 — candidates (b) and (c), pre-registered

Written 11 Oct 2026, before either was fitted or scored. `python -m atp.phase5 candidates-bc` records the sha256 of this
file with the results.

## How this relates to the first pre-registration

The Phase 5 plan named three candidates, all open items from earlier phases:

- (a) a post-holiday checkout term in the survival model, under the Phase 1 rule;
- (b) a GCC seasonal term (11 parameters) in the daily check-in forecaster, under the forecaster's rule;
- (c) a residents-vs-visitors split of ε, measured by how much each nationality rises on UAE-only holidays.

When Step 5.4 was first run, only (a) was recovered from the project's records (the Phase 2 report's "carried forward"
list), so `docs/phase5_4_preregistration.md` registered (a) and two substitutes: recency-weighted ε and a Poisson-type
weighted survival fit. All three were dropped (`reports/phase5_candidates.md`).

- (a) was run as C1. Its registered rule added one condition to the Phase 1 rule (monthly error not worse); under the
  Phase 1 rule alone the decision is the same, because fold-A daily error rose for both groups (international +0.03,
  Domestic +0.27 points).
- The two substitutes stay in the record as additional checks, with their results, whatever (b) and (c) give.
- (b) and (c) are registered here and run now, before any of their results are seen.

The folds, metrics and the rule that the official test window chooses nothing are unchanged.

## (b) GCC seasonal term (daily check-in forecaster)

**Definition.** The five GCC nationalities in the data (Bahrain, Kuwait, Oman, Qatar, Saudi Arabia) share one
month-of-year profile: 11 month parameters, estimated jointly on the five series, in place of each series' own 11
month dummies (55). Every other term of the production forecaster stays specific to each series (intercept, weekdays,
Ramadan, UAE breaks, festival, year-end, events, recent level, home holidays, the ε seat offset, the carried growth).
The other 41 series are unchanged.

The plan's wording is "a GCC seasonal term (11 parameters)" and the original definition is not recorded elsewhere.
Each series already has its own month dummies, so an added 11-parameter GCC month term would not be identified; a
shared GCC profile is the reading that gives exactly 11 parameters: pooling the seasonality of five small, similar
series to cut noise.

**Rule (the forecaster's rule for a shape term, Step 2.4).** International daily × nationality WMAPE at least 0.10
points better on fold A, and fold B not worse by more than 0.05 points; each fold with its own ε, as in Step 2.4.

## (c) Residents vs visitors split of ε

**Definition.** For each of the 33 nationalities with flights, a resident index: check-ins on National Day break days
(a UAE-only holiday) ÷ check-ins on the same weekdays 8–28 days either side (no breaks, no Ramadan), summed over the
National Day breaks in the fold's training data (from 7 Jan 2023 to the fold's training end). Nationalities with an
index above the median of the 33 are "resident-heavy", the rest "visitor-heavy". ε is estimated separately on each
group, with the fold's own default specification and estimation sample (Step 2.2), and the difference
(resident-heavy − visitor-heavy) gets a 95% interval from a stratified country bootstrap (2,000 draws per group,
seeds 1 and 2, as in Step 2.2).

**Rule (as planned).** The split replaces the single ε only if the 95% interval of the difference excludes 0 on both
fold A and fold B, with the same sign on both.

## What happens next

A candidate that passes is carried through every later phase (Step 5.8). One that fails is reported with its numbers.
