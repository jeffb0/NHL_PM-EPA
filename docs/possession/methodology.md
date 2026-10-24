# Possession under-scaling — methodology and pre-registration

*Written 2026-10-03 before any rescaled model was fitted or scored. The design, tuning protocol, tests and rule in §4–§5 are fixed here; changes after results are seen are logged in §8.*

## 1. Question

Several earlier results say the possession component (PM-EPA, credited puck actions) is on too small a scale: the unshrunk team-level slope is 2.84 ± 1.21 for defensemen and 1.37 ± 0.61 for forwards (docs/calibration §5d),
the position-specific ridge puts about 1.6 on it for both positions while being held toward 1, and on-ice offense moves 3.3 (defensemen) to 4.4 (forwards) times as much as PM-EPA for the same signal (docs/transition).
With chemistry ruled out (docs/chemistry), under-scaled possession value is the leading candidate for what the preseason player prior is missing. Does multiplying it by a factor above 1 improve team forecasts?

## 2. Why the effect could be small (stated in advance)

The published forecast already adds each player's share of last season's unexplained team result (the portable residual, coefficient about 1). A rescaling changes a team's total only through roster turnover and
through *which* players on a team hold the value; for a stable roster most of the change is absorbed by a smaller residual. So the test is mainly about **allocation within and across rosters**, not about team totals.

## 3. Design

* **Rescaling.** In the projection panel the possession rate is multiplied by κ_F for forwards and κ_D for defensemen and the total rate is recomputed; everything downstream (the projection ridge, its targets, the team residual, the carried residual) is rebuilt from that panel.
* **Grid, fixed here:** κ_F ∈ {1, 1.5, 2, 3}, κ_D ∈ {1, 1.5, 2, 3, 4} (20 combinations; the earlier estimates lie inside it).
* **Tuning, once:** the preseason-only game model (docs/gamemodel, hyperparameters frozen, original rate model) on the 2019 and 2020 seasons; the pair with the lowest log loss is frozen. Those seasons are not test seasons.
  The earlier estimates in §1 used all seasons and are **not** used to choose κ.

## 4. Tests (test seasons 2021–2025)

1. **PS-1, preseason-only game model:** paired z on per-game log loss, tuned κ against κ = (1, 1), 6,502 games.
2. **PS-2, full game model** (in-season updating): the same comparison.
3. **Reported:** the standings backtest for the published configuration (M1 + C1 + C3 + pre-NHL priors) with and without the rescaling; the tuning surface; the result at the earlier all-season estimate (1.5, 1.5) and at a defensemen-only rescaling (1, 3) for context; log loss by season.

## 5. Gates and the implementation rule

| gate | rule |
|---|---|
| PS-1 | the rescaled preseason-only game model is better (paired z > 2) |
| PS-2 | the rescaled full game model is better (paired z > 2) |
| PS-3 | no harm: neither comparison has z < −2 |

The method is rated **sound** in advance (one or two scale parameters, motivated by three independent earlier results, tuned on separate seasons). By the author's standing rule: implement if PS-3 holds and neither PS-1 nor PS-2 has z ≤ −1;
a reliable improvement is claimed only for z > 2. If tuning selects κ = (1, 1) there is nothing to implement and the hypothesis is reported as not supported by the tuning seasons.
"Implement" means the standings forecast and the game model's forward priors are built from the rescaled panel; the published player values (`player_value_all_in.csv`, the author-preferred total) are **not** changed by this step.

## 6. What the result can say

A pass means possession value is worth more, relative to the other components, than the model gives it, in a way that matters for teams. A null with tuned κ > 1 means the rescaling is harmless but its information is already carried by the portable residual.
A tuned κ of 1 means the under-scaling seen in whole-sample team regressions does not show up as forecast value.

## 7. Limitations stated in advance

Two unusual tuning seasons and a coarse grid, so the selected κ is noisy; one scale per position for all players; the rescaling multiplies the estimate's noise as well as its signal; the game tests use the original rate model while the standings check uses the published configuration.

## 8. Deviations log

*(any change to §3–§5 after results are seen is recorded here with its reason)*

1. **Outcome, applied mechanically (2026-10-03):** tuning on 2019–2020 selected κ = (1, 1) (log loss rises monotonically with either factor on those seasons), so by §5 there is **nothing to implement** and the gates are recorded as not passed.
2. **Reporting detail.** Because the tuned model is the baseline, the reported standings check is shown for the context variant (1.5, 1.5) instead; §4 had said "with and without the rescaling".
3. **Not a change, a caution:** the context variants declared in §4 do better on the test seasons (below). They were fixed before the run but (1.5, 1.5) comes from estimates that used the test seasons, so this is **not** a clean test and does not trigger implementation.
