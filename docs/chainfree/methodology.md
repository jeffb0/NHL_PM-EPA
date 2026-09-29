# Chain-free defense and special teams — methodology and pre-registration

*Written 2026-09-29 before any chain-free estimate was computed. The gates in §4 are fixed here; changes after results are seen
are logged in §7.*

## 1. Question

The defensive component (DPM) and the special-teams RAPM parts are fitted season by season with a **chained prior**: each
season's ridge prior mean is ρ times the player's coefficient the previous season (DPM ρ = 1; special teams ρ selected per state,
0.5 for power play and penalty kill). That helped held-out prediction inside those components, but it makes the estimated values
persistent partly by construction. The projection layer found the consequence: a model trained to predict next season's value
estimates learns and extrapolates that persistence, and its projections lose to last season's values when summed to team level
(PR-6). This work re-estimates the components **with ρ = 0** (each season fitted independently, ridge toward zero) and asks
whether the total and the projections behave better against real results.

## 2. Variants (defaults are not changed)

* `def_gar_rho0`: the valuation-ready DPM refit (fringe-count covariates, pooled replacement, DPM targets unchanged) with ρ = 0, same
  ridge α as the published fit.
* `st_<state>_rapm_gar_rho0` for power play, penalty kill, 4v4 and 3v3: the special-teams RAPM with ρ = 0, same α and same fringe
  replacement construction. The finishing part of the special-teams components is unchanged.
* **Chain-free totals:** the headline total, and the rolling-refit total of the whole-model validation, with those parts replaced.
  Nothing else changes; no other component uses a chained prior.

## 3. Evaluation

1. **Persistence by construction:** year-over-year correlation of the defense and special-teams values (season-centred, ≥ 500 5v5 minutes
   for defense, ≥ 100 state minutes for power play and penalty kill) under ρ = 1 and ρ = 0.
2. **Whole-model M protocol** (docs/wholemodel): the chain-free rolling-refit total in place of the chained one; paired z on team-season
   squared error, test seasons 2020–2025; the baselines B0, B1, B3.
3. **Projection layer** (docs/projection): the same ridge projections built on the chain-free component rates; the player-level gates PR-1 to
   PR-5 and the team-level gate PR-6 re-evaluated, and the projected-versus-last-season comparison.
4. **Forward slopes** of next-season deployment-adjusted on-ice goal differential on each chain-free component (as in the calibration extension).

## 4. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| CF-1 | the chain-free rolling total beats the chained rolling total at the team level (paired z > 2), or is not significantly worse (z > −2) |
| CF-2 | with chain-free components, PR-6 passes (projected rates beat last season's rolling-refit rates, paired z > 2) |
| CF-3 | with chain-free components, projected rates still beat Marcel at the player level (PR-2, z > 2) |

Reported, not gated: YOY under ρ = 1 and ρ = 0; the forward slopes; the change in player rankings between the two versions.

## 5. What the result can and cannot say

A better team-level result under ρ = 0 supports the chained-prior explanation of the projection failure and would recommend
chain-free components for forecasting uses; it would not by itself change the published defaults, which the model owner decides. A
worse or unchanged result would mean the failure comes from something else (the projection target, the component mix, or the noise
in single-season estimates).

## 6. Limitations stated in advance

* ρ = 0 estimates are noisier for players with few minutes (ridge toward zero with no history); the comparison mixes less persistence
  with more noise.
* The published intervals of DPM and the special-teams components come from the chained fits; chain-free values carry no new intervals.
* Chain-free special-teams values are fitted with the α chosen under the chained selection.

## 7. Deviations log

*(any change to §2–§4 after results are seen is recorded here with its reason)*

1. **Exploratory diagnostics added after CF-2 failed** (`src/chainfree/diagnostics.py`, not pre-registered, no gate): last-season versus
   multi-season sums at the team level, the extra shrinkage of the projection features, and within-season dispersion and correlation.
   They were added to test why removing the chain did not help; they are hypothesis-generating, not confirmatory.
2. **Two intermediate probes were run once and are not reported as results:** projecting with target-season minutes as a feature (team RMSE
   58, much worse) and setting projections for players with no history to zero (a 0.6-goal improvement). Neither was pursued.
