# Goaltending component — methodology and pre-registration

*Written 2026-09-29 before any goaltending model was fitted. The decision rules in §6 are fixed here; changes after
results are seen are logged in §9.*

## 1. What is being built

A goalie component (`goalie_gar`) in goals above replacement, all strengths, beside the skater components. It is a
separate table (`goalie_player_season`): goalies are not in the skater stints, so there is no overlap with any skater component.
The whole-model validation's crude goalie proxy is replaced by this component (§7).

## 2. Value definition

For each unblocked shot a goalie faces (goals, saves and misses that reach him: shots on goal and goals), the model gives a
goal probability; the goalie's value is goals saved relative to an average goalie, then relative to replacement:

`goalie_gar = Σ_shots [p_avg − p_goalie] − r_rep × shots_faced`

* `p_avg` uses the shot's expected-goal offset, the shooter's persistent finishing shift, the season × strength intercept and
  (in the team-effect variant) the defending team-season's effect. `p_goalie` adds the goalie's persistent effect and his
  season deviation.
* Shots with no goalie on the ice (empty net) are excluded; they belong to no goalie.
* **Replacement.** Fringe goalie-seasons are those with fewer than 500 shots faced (about 17 games); `r_rep` is the fringe
  pool's mean goals saved per shot, pooled over seasons (the same pooling rule as the skater defensive component).
* The uncertainty is the delta-method posterior sd of the goalie's effect times the sum of `p(1 − p)` over his shots (the
  replacement level's own uncertainty is not included).

## 3. Model

Penalized Laplace logistic mixed model (the machinery of `src/finishing/glmm.py`, SQUAREM-accelerated EM variance components):

`logit P(goal) = logit(xG_f) + T_shooter + cell[season, strength] + u_g + d_{g,season} (+ t_{team,season})`

* `xG_f` is the finishing-only xG of the finishing component and `T_shooter` the shooter's fitted shift from the finishing
  model (`fin_shot_shifts`), so shooter quality is held fixed; `cell` is a fixed effect; `u_g` (persistent) and `d` (season
  deviation) are random goalie effects; `t` is a random effect for the defending team-season (shot quality that xG does not
  measure).
* Variance components are estimated once by EM on the 2017–2021 seasons and then held fixed for every other fit (as in the
  finishing model). Two variants: **G1** (no team-season effect) and **G2** (with it). The variant is chosen on the 2022
  validation fold (D-type rule: G2 iff it improves held-out log-loss by paired game-block z > 2, else G1).
* Forward folds are refit on seasons up to the fold's training end, so their goalie effects contain no later information. The
  production fit uses all seasons; its values carry the same within-model leakage as the finishing component and are not used
  as forecast inputs anywhere in the validation.

## 4. Statistics

Held-out log-loss against the baseline (xG offset + shooter shift + cell, no goalie term) with paired game-block z; year-over-year
correlation of single-season shrunk effects (goalies with ≥ 500 shots in both seasons, season-centred); forward calibration
slope of next-season realized goals saved per shot on the prior-season persistent estimate (goalies with ≥ 400 shots in the
prediction season, weighted by shots); the whole-model comparison of §7.

## 5. Reported, not gated

Starters' (≥ 1,500 shots) mean and SD of goals above replacement; the fringe replacement level in goals per 100 shots; the
correlation between G1 and G2 goalie-season effects; the variance components; the largest and smallest goalie values.

## 6. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| GK-1 coverage | ≥ 99.5% of goals on shots not at an empty net (`shots_xg` empty_net = False) are attributed to a goalie in the shot table |
| GK-2 stability | season-centred year-over-year r of single-season shrunk effects (≥ 500 shots both seasons) ≥ 0.15 |
| GK-3 out of sample | on the test folds (2023, 2024, 2025), the chosen model beats the goalie-free baseline in log-loss, paired z > 2 |
| GK-4 calibration | forward slope of realized goals saved per shot on the prior-season persistent estimate in [0.7, 1.4] |
| GK-5 whole model | the rolling goalie term beats the crude goalie proxy `G` at predicting next-season team goal differential (B0 with each; paired z > 2) |

## 7. Whole-model use

The whole-model validation is rerun with the rolling goalie term (`G_rolling`: Σ over a team's goalies of the shots faced in
season T × the goalie's season-(T−1) persistent goals-saved rate; goalies with no prior fold estimate get the replacement
rate) in place of the proxy. The change in every model's out-of-sample error and in WM-2/WM-3 is reported.

## 8. Limitations stated in advance

* Goalie effects rest on xG that omits screens, traffic and pre-shot movement; the team-season effect absorbs some of it.
* Goalies change teams rarely and starters dominate their team's shots, so goalie and defense effects are weakly separated.
* Save quality is rare-event data: about 30,000 shots a season across roughly 100 goalies; expect wide intervals and strong shrinkage.
* Rest, back-to-back and workload effects are not modeled; the value is per shot faced.
* The replacement definition (< 500 shots) is a convention.

## 9. Deviations log

*(any change to §2–§7 after results are seen is recorded here with its reason)*

1. **Replacement level changed from the pooled mean of fringe goals saved to an unpenalized fringe effect (after the first run).**
   §2 defined `r_rep` as the fringe pool's mean goals saved per shot. In the first run this gave −0.06 goals per 100 shots, which
   is almost the league average by construction: the random effects of thinly observed goalies are shrunk to zero, so the mean of shrunk
   effects over the fringe pool cannot separate from the average goalie. The skater components solved the same problem with an
   unpenalized fringe block (finishing) or pooled unshrunk fringe rates (possession). The goalie model now carries an unpenalized effect for
   goalie-seasons with fewer than 500 shots, and replacement is a goalie with that effect and no other; goals above replacement are
   `Σ [p_rep − p_goalie]`. The first-run gate results (GK-3 z = 3.24, GK-4 slope 0.81, GK-2 r = 0.11, variant G2) are superseded by the reruns.
2. **A tried forecast change, reverted.** After the replacement change, a goalie's prior-season fringe status was added to the forecast (the
   unpenalized effect is a real part of the fitted model). It made the forecast worse (test-fold z = −0.69, forward slope 0.30), so the
   persistent-effect forecast of §3 (the pre-registered one) is what the gates use; the tried version is reported beside it.
3. **Variant flip (mechanical).** With the fringe effect in the model the D-type rule selects G1 (G2 vs G1 z = 1.43 on the 2022 fold), where the
   first run had selected G2. The rule was applied as written; the goalie-season values of the two variants correlate 0.99.
