# Phase 5, Step 5.7 — accuracy by months ahead, pre-registered

Written 11 Oct 2026, before any number by months ahead was computed for this step. `python -m atp.phase5 ahead` records
the sha256 of this file with the results.

## The question

The simulator's baseline is forecast from the training data forward. Does the error grow the further ahead the month
is? This is the evidence behind "the level band did not grow with lead in the folds", and behind flagging results more
than 7 months past the data.

## Definitions

- **Months ahead (h):** months since the training end. Fold A (train to Jul 2024): Aug 2024 = 1 … Feb 2025 = 7. Fold B
  (train to Dec 2024): Jan 2025 = 1 … Jul 2025 = 7. The two folds are the project's only held-out periods with a
  full 7 months; the official test window (Aug 2025–Feb 2026) is not used, as everywhere in this phase.
- **What is forecast:** the saved end-to-end predictions (`endtoend_pred_{A,B}.parquet`: flights actually flown → check-ins
  → guests in-house) and the saved check-in forecasts (`stagea_pred_{A,B}.parquet`), summed to months. The monthly guest
  nights must equal `phase4/backtest_monthly.csv` (the numbers Tab 3 already shows) to 1e-6 before anything else is
  computed. As in every back-test, the forecast is given the seats that actually flew, so the lead measures how far the
  *baseline* (level, seasonality, stays) can be carried, not uncertainty about the flight schedule.
- **International:** the 45 nationalities (flight and no-flight), without Domestic, as in the Tab 3 headline.

## Measures, by h, for each fold and for the two folds pooled

1. **International total, monthly guest nights:** absolute error ÷ actual for the month (one month per h per fold), and
   the signed error (forecast ÷ actual − 1), to show whether the forecast drifts one way.
2. **By nationality, monthly guest nights:** WMAPE over the 45 nationality-months of that h (Σ|actual − forecast| ÷
   Σ actual), with a 95% interval from a nationality bootstrap (2,000 draws, seed 11). Also by market-size tier (top 10,
   11–25, 26–46 by actual check-ins, the tiers of Step 3.5).
3. **The same two for check-ins** (Stage A alone), to separate the two stages.
4. **Coverage by h:** the share of nationality-months whose actual guest nights fall inside the 80% band of their size
   tier (the Step 3.5 band, learned on the other fold). Descriptive: the band's width was set so both folds reach 80%.

## Rule for "the error grows with lead"

The slope of the nationality-month WMAPE on h (ordinary least squares over h = 1…7), per fold, with a 95% interval from
the same nationality bootstrap (the slope is recomputed in each draw). **The error grows with lead** if the slope's
interval lies above 0 on both fold A and fold B. Otherwise the finding is "no detectable growth in 1–7 months"; that is
a statement about the 7 months tested only. For the international total there is one month per h per fold, too few for
a test: it is shown, not tested.

## What follows

A Tab 3 block (error by months ahead, with the table behind it and one sentence on what it does and does not show) and
one line in the deck. If the error grows with lead, the app's flag after 7 months moves earlier and the band is widened
with lead (an amendment to Step 3.5); if not, the flag stays where it is and the untested 8–12 months stay a
limitation (Step 5.9).

## Amendment A1 — added after the first run (11 Oct 2026)

The first run (the measures and the rule above) showed the pooled error lowest in month 1 and 3–8 points higher in months
2–7, without a steady rise. The slope test alone would hide that step. A descriptive contrast is added: the average of
the nationality-month WMAPE in months 2–7 minus that of month 1, per fold, with the same nationality bootstrap (2,000
draws, seed 11, same resampled nationalities). It is reported next to the pre-registered rule and is not a decision
rule: the conclusion "no detectable growth with lead" rests on the pre-registered slope rule, and the contrast says
whether the step from month 1 is distinguishable from noise.
