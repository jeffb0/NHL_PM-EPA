# Turnover Cost, Faceoffs and Penalties (5v5) — Methodology and Pre-Registration

*Written 2026-09-29. The turnover reconciliation test (§2) was run before the rules below were written and
shaped the turnover definition; everything else (faceoff and penalty models, all gates) is fixed here before
any of those models is fitted. Later deviations are logged in §9.*

## 1. Scope and purpose

PM-EPA, DPM and FIN cover 5v5 possession, defense and finishing. Three named slices of 5v5 value are still
claimed by nobody: the opponent's possession-start value after **our turnovers**, the value of **faceoff**
outcomes, and the value of **penalties** taken and drawn. DPM's valuation column already removes the first
two slices from defense so they can be claimed here. This document defines the components that claim them
(`src/components/`), on the same goals scale, replacement definition and shrinkage as the earlier components.
Special teams (PP/PK on-ice value) remains a separate, later component.

## 2. The turnover reconciliation test (run first)

Question: is PM-EPA's under-scaling on team 5v5 goal differential (joint slope 1.68) due to the unclaimed
turnover value? Two candidate slices, both from the λ = 1 minus λ = 0 credit difference in `credits*.parquet`:

* **T2 (charge only)** = difference of {giveaway, unobserved_loss, takeaway_against}: the opponent's start value
  after our turnovers, charged to the committers.
* **T1 (symmetric)** = T2 + the takeaway difference (also credits the taker with the value denied).

Team-season regressions of 5v5 goal differential (284 team-seasons; slopes ± SE; `reports/metrics/val_turnover_test.json`):

| Model | R² | PM-EPA | DPM | FIN | Added term |
|---|---|---|---|---|---|
| R0 current | 0.461 | 1.68 ± 0.43 | 1.17 ± 0.13 | 1.46 ± 0.25 | — |
| R1 + T2 | 0.517 | **1.05 ± 0.42** | 1.17 ± 0.13 | 2.05 ± 0.26 | T2 = 0.80 ± 0.14 |
| R2 + T1 | 0.519 | 0.99 ± 0.42 | 1.18 ± 0.13 | 2.07 ± 0.26 | T1 = 0.85 ± 0.15 |
| R3 + T2 + taker credit | 0.520 | 0.96 ± 0.43 | 1.19 ± 0.13 | 2.07 ± 0.26 | T2 = 0.86, taker credit 1.34 ± 1.04 |

**Finding:** most of PM-EPA's apparent under-scaling was the missing turnover charge. With T2 in the model the
possession slope falls from 1.68 to 1.05, R² rises from 0.46 to 0.52, and T2 itself lands near unit scale.
The symmetric taker credit is uninformative (slope 1.34 ± 1.04), so **the definition is T2 (charge only)**.
It has no counterfactual credit and is identity-based. In the same run, finishing's slope rose from 1.46 to
2.05 (finishing is now the most under-scaled component; this is left for the joint calibration).

**Overlap found by the identity check.** Summed over all players, T2 is −17.3k xG (5v5, 2017-2025) against
9.8k of turnover-born starts that DPM excludes (ratio 1.76). By component: takeaway-against matches almost
exactly (−2,861 vs 2,872), but T2 charges *every* giveaway and unobserved loss with the opponent's expected
start value, while DPM only excludes starts that were *observed* after a turnover (88k of 198k giveaways;
the rest land in DPM's `start_unobserved` slice). Charging T2 while leaving that slice in DPM would double
count. **Fix (decided before further results): DPM's valuation column also excludes `start_unobserved`.**
Against turnover-born plus unobserved starts (19.4k) the ratio is 0.89, so the accounting is consistent to
about 11%.

## 3. Turnover cost (TO)

* **Definition.** `TO_i` = 5v5 sum over player i's credit rows of (value at λ = 1 − value at λ = 0) for
  components giveaway, unobserved_loss and takeaway_against. The giveaway part is direct (the giver); the other
  two are pooled equally over the losing team's on-ice skaters at the loss, as in PM-EPA.
* **Estimation.** Per-60 rate with game-block bootstrap SE (as PM-EPA), empirical-Bayes shrinkage by season ×
  position × status (regular/fringe) with τ² from qualified regulars; goals = rate × TOI / 60.
* **Replacement.** The pooled raw fringe rate (unbiased, large pool), by season and position.
* **Scorer and rink bias.** Giveaways and takeaways are scorer judgments (methodology of PM-EPA). No rink
  adjustment is applied in v1. The home-versus-away rate gap and the team-changer YOY are reported as diagnostics.

## 4. Faceoffs (FO)

* **Value of a draw.** The ledger books the start value of a faceoff-started possession for the winner
  (unattributed in PM-EPA, excluded in DPM). With V(dot) the mean cross-fitted start value of 5v5
  faceoff-started possessions at a faceoff dot in the winner's frame, winning versus losing a draw at a
  location is worth `S = V(dot in own frame) + V(dot in opponent's frame)`. The mean is over 5v5 obs of kind
  `faceoff`, all seasons.
* **Skill model.** One row per 5v5 draw with both takers known:
  `logit P(home taker wins) = zone cell + β · (side_home − side_away) + f · (fringe_home − fringe_away)
  + (u + δ)_{home taker} − (u + δ)_{away taker}`, where u is persistent taker skill (by position group C /
  other), δ a season deviation, `side` = handedness × dot side, and f the fringe-group fixed effect. Fitted with
  the same penalized Laplace machinery as FIN, with SQUAREM-accelerated EM variance components.
* **Player value.** For each draw: `S × (P(win) − P(win with the taker's talent set to the reference))`, the
  reference being the league average (above average) or the fringe pool (above replacement).
  Uncertainty by the delta method on the posterior of `u + δ`.
* **No double count.** The skill part of the swing is claimed here; the luck part (who won given equal skill)
  is claimed by nobody. PM-EPA gives winners no start share and DPM excludes `start_faceoff`.

## 5. Penalties (PEN)

* **Scope.** Penalties called with the game at 5v5 (both goalies in). Minor (2), double minor (4) and major
  (5) penalties with a known committer and/or drawer. Misconducts (10 min), bench minors without a drawer
  contribute only on the side that is known, and coincident penalties (both teams penalized at the same time)
  are excluded (no advantage arises). Penalties called at other strengths belong to special teams.
* **Cost per penalty.** For each qualifying penalty, `net xG` for the non-penalized team from the whistle until
  strength returns to even (or the penalty length, capped), using the shared cross-fitted shot xG. Cost by class
  = the mean over penalties (with SE). This is the expected consequence; actual power-play performance is left
  to the special-teams component, so there is no overlap.
* **Player value.** Minor-equivalent counts drawn and taken (class cost ÷ minor cost), rates per 60 of 5v5 TOI,
  gamma-Poisson empirical-Bayes shrinkage by position × status; value = (drawn − taken) × cost per minor
  above the pooled fringe replacement rates.

## 6. Replacement, scale and integration

All three use the regular/fringe definition of the earlier components (top 12 F / 6 D per team) and enter
`player_value_season` as goals and wins above replacement; totals add them to the existing three. The team
reconciliation is repeated with every component and slopes are **reported, not used to rescale**.

## 7. Pre-registered gates (failures are reported, not tuned away)

| Gate | Threshold |
|---|---|
| TO-1 turnover accounting: −ΣT2 ÷ (DPM-excluded turnover-born + unobserved starts) | within [0.75, 1.25] |
| TO-2 team-level slope of T2 (joint regression with the other components) | within [0.5, 1.5] |
| TO-3 season-centred YOY of shrunk TO per 60 (≥ 500 min), F and D | ≥ 0.30 each |
| TO-4 shrunk TO predicts next-season TO per 60 better than raw (MSE) | lower, paired z > 2 |
| FO-1 shrunk taker skill beats no-taker on next-season draw log-loss (pooled test folds) | z > 2 |
| FO-2 shrunk beats raw win-rate log-odds | z > 2 |
| FO-3 calibration slope of realized next-season excess wins on predicted (≥ 300 test draws) | within [0.7, 1.4] |
| FO-4 draw swing S positive at every dot | all > 0 |
| PEN-1 cost per minor penalty | 95% CI above 0 |
| PEN-2 shrunk drawn and taken rates predict next-season counts better than raw (Poisson deviance) | z > 2 each |
| PEN-3 season-centred YOY of drawn per 60 and taken per 60 (≥ 500 min) | ≥ 0.15 each |

Forward folds: fit on seasons ≤ Y, predict Y + 1, variance components fixed from 2017–2021 seasons; test
folds 2023, 2024, 2025 (validation 2022), as for finishing.
**Reported, not gated:** rink/home diagnostic, direct/pooled split, top and bottom players, reconciliation
slopes with all six components, sensitivity to the fringe cutoff.

## 8. Data notes

* Faceoff events: `player1` = winner, `player2` = loser (every 5v5 draw has both). Penalty events: `player1` =
  committer (null for 3.1%), `player2` = drawer (null for 8.8%). Penalty minutes: 2 (72.6k), 5 (5.2k),
  10 (1.8k), 4 (0.8k), 0 (0.4k).
* Turnover credits use the λ = 1 companion run already produced by the pipeline; nothing is re-run.

## 8b. Role-adjusted turnover variant (TO-RA), added 2026-09-29 before it was fitted

**Why.** TO is right at team level but biased in allocation against puck-dominant players: recorded giveaways
are charged, while successful puck handling is mostly unobserved. Arena normalization cannot fix this
(§9, deviation 7). TO-RA answers a different question: *how does a player's charge compare with players who
have a similar amount of observed puck involvement?* It is a **variant beside TO** (`to_ra_gar`); TO stays the
default and the accounting-consistent team-level component. TO-RA is not accounting-consistent (residuals do not
sum to the team charge) and is meant for comparing players.

**Definition (fixed here).**

* **Involvement** `inv_per60`: per-60 5v5 count of a player's *successful observed* puck events, i.e. his direct
  credit rows in components carry, move_start, move_end and shot_attempt (giveaways are the outcome and are
  excluded; takeaways are excluded because their recording carries the same scorer bias as giveaways). Usable
  games only.
* **Expected charge.** Within each season and position group, a TOI-weighted least-squares fit of the
  arena-adjusted per-60 turnover charge on `inv` and `inv²`, using player-seasons with ≥ 200 5v5 minutes;
  predictions for every player.
* **Residual** = observed per-60 charge − expected. It is empirical-Bayes shrunk exactly like TO (prior mean =
  pooled residual rate of the status group; τ² from qualified regulars; the same game-block bootstrap SE), and
  measured above the pooled fringe residual, giving `to_ra_gar` in goals. The uncertainty of the fitted
  expectation is not propagated.

**Gates (fixed here; failures are reported, not tuned away).**

| Gate | Threshold |
|---|---|
| RA-1 the adjustment removes the involvement gradient: \|r\| between the residual per-60 charge and `inv_per60`, qualified players, by position | < 0.15 each |
| RA-2 the residual is still a stable signal: season-centred YOY of shrunk TO-RA per 60 (≥ 500 min) | ≥ 0.30, F and D |
| RA-3 shrunk TO-RA beats raw at predicting next-season residual per 60 (paired z) | z > 2 |

**Reported, not gated:** R² of the expectation by position; the stayer/changer YOY gap; Spearman rank correlation of
TO and TO-RA among qualified skaters; the mean change by involvement quintile; the most-charged and least-charged
skaters under each; the team-level reconciliation with TO-RA substituted for TO; the total value with TO-RA
substituted (`total_gar_to_ra`). **Caveat fixed in advance:** involvement is correlated with real puck-management
skill, so the adjustment also removes some genuine skill; TO-RA is a like-for-like comparison, not a truer measure.

## 9. Deviations log

*(any change to §3–§7 after results are seen is recorded here with its reason)*

1. **DPM valuation column now also excludes `start_unobserved`** (§2), decided from the identity check before
   any component was built. The DPM headline and published columns are unchanged; only the valuation refit
   (`src/valuation`) gains a fourth target. This changes the earlier integration numbers: the three-component
   joint slopes became 2.04 / 1.30 / 1.61 (from 1.68 / 1.18 / 1.46).
2. **Arena (scorer) adjustment added to turnover cost, after a diagnostic (2026-09-29).** With the unadjusted
   component built, the arena-level spread of the per-60 turnover charge was 0.03-0.045 in 2017-2023 (15-23% of the
   -0.20 league mean, comparable to the 0.039 between-player SD) and 0.008-0.009 in 2024-25, and team-changers were
   far less stable than stayers (0.35 vs 0.64). A first estimate mixed in the home team's own style, so the
   effect is now estimated from visiting players only (each visitor against his own away rate), empirical-Bayes
   shrunk by season, and subtracted from every game at that arena. Gates were unaffected (TO-3 F 0.62 → 0.60, D 0.55 →
   0.48) and the stayer/changer gap narrowed (0.29 → 0.22). The unadjusted version is kept (`to_gar_unadj`). The
   component now defaults to the adjusted version.
3. **Reconciliation targets.** The first six-component regression on 5v5 goal differential gave the penalty
   component a slope of -2.1 ± 0.8. Penalty value is realized on the power play and penalty kill, not at 5v5, and
   its univariate 5v5 slope was 0.34 with r = 0.02, so the regression used the wrong target. Penalties are now
   excluded from the 5v5 joint regression and checked against special-teams goal differential (slope 1.35 ± 0.51).
4. **TO-2 outcome.** The pre-registered band for the turnover team-level slope is 0.5-1.5. The shrunk joint slope
   is 1.51 ± 0.66 (band edge missed by 0.01) and the unshrunk one 1.18 ± 0.44. Recorded as a narrow failure of the
   shrunk regression; the definition was not changed.
5. **Shrinkage and team-level scale.** A comparison of shrunk and unshrunk team sums showed that a large part of
   the PM-EPA under-scaling at team level is shrinkage compression (possession slope 2.04 → 1.56 unshrunk), so
   the unshrunk regression is reported alongside the shrunk one and team-level slopes above 1 are read
   accordingly. Nothing was rescaled.
7. **Arena normalization variants tested, no change (2026-09-29, `src/components/arena_variants.py`).** Asked
   whether normalizing turnover counting by arena could be improved, two variants of the adopted additive
   adjustment were compared out of sample on visiting players (odd/even game ids): a count-proportional
   (multiplicative) arena factor, and separate effects for the direct (recorded giveaway) and pooled (unrecorded
   loss) parts. Additive and multiplicative reduce game-level squared error by 5.08% and 5.14% in 2017-2023 (and by
   nothing in 2024-25, correctly), so they are indistinguishable. The direct and pooled arena effects are
   positively related (r = 0.15; excess SD 0.028 and 0.017 per 60, total effect about their sum), not opposite, so
   arena recording does not move charge between the parts. Arena effects are highly persistent (r 0.75-0.79
   between consecutive seasons), which is why they inflate the stability of players who stay on one team. The
   adopted adjustment already captures this; the puck-dominance bias in allocation is a different problem that
   arena normalization does not address.
8. **Role-adjusted turnover variant added at the model owner's request (2026-09-29, §8b).** Design and gates were written
   before it was fitted. All three gates pass (RA-1 |r| 0.02 F / 0.01 D; RA-2 YOY 0.55 F / 0.45 D; RA-3 z = 15.7).
   Involvement explains 11% (F) and 6% (D) of the variance in the charge, so the effect is modest (rank correlation with
   TO 0.93 / 0.96; totals correlate 0.999; the mean change in total ranges from -0.06 goals for the least-involved quintile
   to +0.24 for the most-involved). With TO-RA substituted in the team-level regression R² is 0.473 (0.464 with TO) and
   the TO-RA slope is 2.19 ± 0.68. TO remains the default; `to_ra_gar` and `total_gar_to_ra` are variants.
6. **Sensitivity analysis scope.** The regular/fringe cutoff sensitivity covers the possession, defense and
   finishing components; the turnover, faceoff and penalty replacement stays at the default cutoff.
