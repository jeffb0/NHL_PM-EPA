# Rest, travel and schedule effects — results

*Run 2026-10-12 under docs/schedule/methodology.md (pre-registered the same day, before the run). Numbers are in reports/metrics/schedule_effects.json. No deviations.*

## Verdict

**None of the three blocks is adopted.** No block improves the game model out of sample, no block's likelihood-ratio test is significant, and no single coefficient with a predicted sign reaches |t| > 2 on wins.
The back-to-back flags the model already has remain the only schedule terms.

## Primary test (6,502 test games, 2021–2025; logistic step fitted on earlier seasons)

| logistic step | log loss | paired z against baseline | LR test p (2019–2025) | SC-1 (z > 2) | SC-2 (z > −2) | adopted |
|---|---|---|---|---|---|---|
| baseline (model margin + back-to-back flags) | 0.65854 | | | | | |
| + R, rest (3 features) | 0.65918 | −1.59 | 0.50 | fail | pass | no |
| + T, travel (4 features) | 0.65846 | +0.18 | 0.14 | fail | pass | no |
| + D, trip and density (4 features) | 0.65900 | −1.63 | 0.61 | fail | pass | no |
| + ALL (11 features) | 0.65955 | −1.48 | 0.29 | fail | pass | reported only |

Rest and trip/density make the forecasts slightly worse (fitted noise); travel is level. By season, travel helps in 2021, 2022 and 2025 and hurts in 2023 and 2024.

## Coefficients (one fit on all 8,359 games of 2019–2025, all features together)

Win is in log-odds; goal and expected-goal differential are in goals, home minus away. t-values in brackets.

| feature | predicted | home win | goal differential | expected-goal differential |
|---|---|---|---|---|
| home on a back-to-back (existing) | − | −0.25 (−2.5) | −0.31 (−2.6) | −0.12 (−2.2) |
| away on a back-to-back (existing) | + | +0.25 (+2.9) | +0.30 (+2.8) | +0.08 (+1.7) |
| `rest_adv`, per day | + | +0.03 (+0.7) | +0.01 (+0.3) | +0.05 (+2.2) |
| `layoff_h` | none | −0.18 (−1.3) | −0.08 (−0.5) | −0.10 (−1.3) |
| `layoff_a` | none | +0.06 (+0.4) | +0.02 (+0.1) | 0.00 (0.0) |
| `km_h`, per 1,000 km | − | −0.05 (−1.8) | −0.06 (−1.6) | −0.01 (−0.7) |
| `km_a`, per 1,000 km | + | 0.00 (+0.2) | −0.02 (−0.5) | +0.03 (+2.0) |
| `east_a`, per hour | + | +0.03 (+1.2) | +0.04 (+1.1) | +0.03 (+1.7) |
| `west_a`, per hour | + | −0.03 (−1.2) | −0.03 (−0.8) | −0.01 (−0.6) |
| `trip_a`, per game | + | +0.01 (+0.3) | −0.01 (−0.5) | +0.01 (+0.8) |
| `stand_h`, per game | none | −0.03 (−1.5) | −0.04 (−1.7) | −0.01 (−0.5) |
| `dense_h` | − | −0.09 (−1.2) | −0.14 (−1.7) | 0.00 (+0.1) |
| `dense_a` | + | +0.01 (+0.1) | +0.04 (+0.5) | +0.01 (+0.3) |

## Reading

* **The back-to-back term is real and already in the model:** about 0.25 log-odds each way, roughly six percentage points of win probability and 0.3 goals.
* **Rest beyond the back-to-back (H-R): no evidence on wins or goals.** There is a hint on expected goals (+0.05 per day of rest advantage, t = 2.2), a twentieth of a goal, which does not carry through to goals or wins.
* **Travel (H-T): the nearest thing to a signal, and it is weak.** Most signs are as predicted (a home team that travelled does worse; an away team east of its home zone does worse), but nothing reaches |t| = 2 on wins and the block test has p = 0.14.
  A West-coast team three zones east is estimated at about two points of win probability worse (0.03 × 3 log-odds), with a standard error as large as the estimate. Teams playing west of their home zone show the opposite sign to the prediction.
* **Road trips, homestands and three-in-four (H-D): no evidence.** The largest estimate (home team's third game in four nights, −0.14 goals, t = −1.7) is not significant.
* Thirty-three coefficients are shown; two or so with |t| > 2 are expected by chance. The two that appear are both on expected goals and neither is on wins.

## What the null means

These effects are bounded, not excluded. The standard errors put a rest day, or 1,000 km of travel, at no more than about three points of win probability, and the point estimates are well inside that.
Effects of that size cannot be told from zero with seven seasons of win/loss results, and adding them makes forecasts worse because their coefficients are mostly noise.
