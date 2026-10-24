# Finishing Talent (FIN) and Value Integration — Methodology and Pre-Registration

*Written 2026-09-29, before any finishing model was fitted. The rules in §7 and §8 are fixed here and
are not changed after results are seen; any later deviation is logged in §11.*

## 1. Goal and scope

PM-EPA values what a skater does **with** the puck (possession value); DPM values what the skater's
presence **prevents**; both are in xG. FIN values what the skater does **at the end of the chain**:
how much better or worse than an average shooter, on the same shots against the same goalies, the
skater turns unblocked attempts into goals. Output: goals above average and goals above replacement
per season, with 95% intervals.

**In scope:** unblocked, non-empty-net attempts by skaters, all strengths used for *estimation*
(talent is shared across situations); **5v5 shots enter the value total**. Power-play (and other
non-5v5) finishing is reported as memo columns reserved for the future special-teams component.
**Out of scope:** blocked attempts, empty-net shots, shootouts, goalies as shooters, goaltending value
(goalie effects are estimated only as an adjustment and kept as a by-product).

## 2. What public data does and does not support

The brief asks for pre-shot passes and lateral movement. Public play-by-play has **no pass events**.
We use what exists: the previous event's type, owner, location and elapsed time, and (derived)
lateral displacement from the shooter's own team's previous event, time since the last faceoff, skater
counts, and the number of same-team attempts in the previous 10 s. Tracking-grade passing features
would plug into the same xG design (`src/finishing/xg2.py`) when available.

## 3. Accounting: why FIN does not double count

For any credited shot:

```
goal − V_dec  =  (V_att − V_dec)  +  (xG − V_att)  +  (xG_f − xG)  +  (goal − xG_f)
                 └ PM-EPA ┘         └ excluded ┘       └ unclaimed ┘   └────── FIN ──────┘
```

* PM-EPA books `V_att − V_dec` (`shot_attempt`). Its terminal value is **xG, never the goal** (`G_p =
  xG(e_p)`), and `xG − V_att` is computed but excluded (`shot_quality`).
* DPM is built on the same xG ledger, so goals and shooter skill never enter it.
* FIN is defined as the residual `goal − xG_f` only, against a **finishing-only** xG offset `xG_f`.
  The shared xG that PM-EPA and DPM depend on is not modified, so their published numbers stand.
* `xG − V_att` (shot quality / selection) and `xG_f − xG` are reported in a memo column and are **not**
  in the total (decision 4). A unit test checks the four-term identity above.

## 4. xG_f (finishing offset)

Gradient-boosted classifier (same family and hyperparameters as the shared xG), trained on unblocked,
non-empty-net attempts. Two feature sets on identical rows and splits:

* **F0:** the shared xG's features.
* **F1:** F0 + previous-event ownership, lateral displacement and cross-ice proxy (own-team previous
  event only), time since faceoff, own/opponent skater counts, own-team attempts in the last 10 s.

Temporal protocol from `config.yaml`: train 2017–21, validate 2022, test 2023–25. Reported: log-loss,
AUC, Brier, calibration-in-the-large, ECE, reliability curves (overall, by strength, by shot type).
Final `xG_f` is cross-fitted by season (season Y scored by a model trained on the other seasons).
No shooter or goalie identity enters xG_f. Talent is estimated only in the finishing model.

## 5. Finishing model

```
logit P(goal_i) = logit(xG_f,i)  +  a[season, strength]  +  f_pos · fringe(shooter, season)
                  + u_p  +  δ_{p,season}  (+ w_{p,type})  +  v_g  +  ε_{g,season}
u_p ~ N(Z_p γ, σ²_{u,pos})    δ ~ N(0, σ²_{δ,pos})    w ~ N(0, σ²_w)    v ~ N(0, σ²_v)    ε ~ N(0, σ²_ε)
```

* `a` are unpenalized season × strength-group intercepts. They absorb league-level xG drift, so drift
  is not assigned to players.
* `f_pos` are unpenalized fringe-player effects by position group. They estimate the replacement group's
  mean directly, so replacement level is not biased by shrinkage toward the regulars' mean (§9).
* `u_p` is persistent shooter talent, `δ` a season deviation, `w` shot-type deviations (slap, backhand,
  tip/deflection relative to wrist/snap; variant B only), `v`/`ε` the goalie adjustment.
* `Z_p γ` is the **pluggable prior mean**: `PriorSpec` supplies player covariates (later: NHL EDGE
  skating and shot-speed metrics, z-scored, with missingness indicators). γ is estimated jointly and
  unpenalized. Default prior = zero mean with position-specific variances (`PositionPrior`).

**Estimation, in two tiers (decision 2).**

1. *Production and evaluation:* penalized (Laplace) logistic mixed model. Newton iterations on the
   penalized log-likelihood; variance components by empirical-Bayes EM (`σ²_k ← mean(b_k² +
   diag(H⁻¹)_k)`); intervals from the Laplace posterior covariance. Seconds to minutes per fit, which
   the many refits in §7 need.
2. *Check:* full NUTS (JAX/numpyro, separate optional environment) on the single-season model and a
   multi-season window; posterior means and standard deviations are compared with tier 1
   (`src/finishing/nuts_check.py`, `requirements-bayes.txt`).

Why not brms/PyMC-style full Bayes for everything: with 0.95M shots and ~15k random effects, the
evaluation protocol needs dozens of refits; and this machine has no C compiler, which PyMC's default
backend needs. Laplace is accurate here because each effect has many Bernoulli observations, which is
tested against NUTS.

## 6. Reliability and stability diagnostics (reported for every run)

* **Single-season independent fits** (each season fitted alone, no pooling across seasons) give
  honest year-over-year and split-half numbers. A pooled model's persistent effect `u_p` is shared
  across seasons, so a naive YOY of its estimates is circular.
* YOY r and split-half (odd/even `game_id`) r of the model estimate vs raw G − xG_f.
* Shots to stabilization: analytic `n₅₀ = 1 / (Ī · σ²_shooter)` with `Ī` the mean Bernoulli information
  per attempt, and empirical Spearman-Brown reliability vs attempts (random split of each player's
  shots).
* Forward prediction (§7).

## 7. Pre-registered selection and gates

**Splits.** Variance components (σ²) are estimated **once by EM on seasons 2017–2021** and held fixed
in every forward fold. Forward fold "predict season Y+1 from a fit on seasons ≤ Y".
Validation fold: Y+1 = 2022. Test folds: Y+1 = 2023, 2024, 2025.

**Season intercepts for the held-out season.** `a[Y+1, strength]` is set so that the held-out cell's
predicted goals equal its observed goals (league-level calibration only), identically for every model
compared, so the comparison isolates player information.

**Models compared on a fold (all include the same offset and cell intercepts).**

| Model | Player information |
|---|---|
| M0 | none |
| Mg | goalie persistent effects only |
| Mraw | raw smoothed shooter ratio: `log((G_p + 1)/(xG_p + 1))` over seasons ≤ Y, plus goalies |
| Mshr | model A (shrunk shooter and goalie effects) |
| MshrB | model B (A + shooter × shot-type) |

**D-xG.** Use F1 for `xG_f` iff on the validation season the paired per-shot log-loss improvement of F1
over F0 exceeds 2 paired SE **and** the pooled test-season log-loss of F1 is not worse than F0.
Otherwise `xG_f` = the shared cross-fitted xG (then the unclaimed term of §3 is zero).

**D-type.** Use model B iff on the validation fold its held-out log-likelihood beats model A by more
than 2 paired SE. Otherwise model A.

**Gates (must pass to ship v1; failures are reported, not tuned away).**

| Gate | Threshold |
|---|---|
| F1 Mshr beats M0 on pooled test folds | log-loss lower by > 2 SE |
| F2 Mshr beats Mraw on pooled test folds | log-loss lower by > 2 SE |
| F3 calibration slope of realized next-season excess goals on predicted (players ≥ 100 test attempts) | within [0.7, 1.4] |
| F4 xG_f cross-fitted calibration, pooled 5v5 | Σxg ÷ Σgoals within [0.97, 1.03] |
| F5 join integrity | every goal in `events` is in the finishing table or in the documented exclusions (empty-net, no coordinates, blocked-shot rows) |
| F6 four-term identity (§3) | max abs residual < 1e-9 (unit test) |

**Reported, not gated:** YOY and split-half (§6), stabilization, the effect of the goalie term,
NUTS agreement, biggest over/under-performers vs raw G − xG.

## 8. Integration (`src/valuation/`)

**Common estimation basis.** Every component enters as a posterior mean with an interval.

* *Possession (PM-EPA 5v5):* empirical-Bayes shrinkage of the published per-60 rate using its
  bootstrap SE: `μ̂ = m + B(x − m)`, `B = τ²/(τ² + se²)`. `m` is the pooled raw rate of the player's
  status group (regular / fringe) in the same season × position; `τ²` is method-of-moments on
  qualified (≥ 500 min) regulars. The published PM-EPA columns are not changed.
* *Defense (DPM 5v5):* valuation-ready `dpm_val_per60`, which drops turnover- and faceoff-born
  opponent value (§5 of the DPM methodology). Refit with an explicit replacement-group effect (§9).
* *Finishing:* §5, shots at 5v5.

**Scale.** All three are in goals (xG scale) = rate × 5v5 TOI ÷ 60 (PM-EPA's `toi_ev` for all three).
Wins = goals ÷ goals-per-win. A team-level reconciliation reports how Σ player components track team
5v5 goal differential. Slopes far from 1 are **reported, not silently rescaled**.

**Goals per win.** From the data: OLS through the origin of (`wins − GP/2`) on regular-season goal
differential (shootouts excluded, from events) across team-seasons 2017–2025. Goals per win = 1 / slope,
with a team-season bootstrap interval. A per-season table is reported as a cross-check.

## 9. Replacement level (applied uniformly)

* **Regulars / fringe.** Per season, per team (`primary_team`), rank forwards and defensemen separately
  by all-strength TOI: the top **12 F and 6 D** are regulars, all others (including players with very
  few games) are fringe. Sensitivity to N (11/13 F; 5/7 D) is reported.
* **Replacement rate** = the mean of the *fringe pool* for each component, by season × position:
  * PM-EPA: pooled raw rate `Σ credit / Σ TOI` of the fringe pool (large pool, unshrunk, unbiased).
  * FIN: the unpenalized `f_pos` group effect, so it is unbiased by shrinkage.
  * DPM: the published estimates of low-TOI players are shrunk toward zero, which would bias the
    fringe mean upward (about 0.05 xGA/60 ≈ 0.8 goals per 1,000 min, ~30% of the player SD). DPM is therefore refit
    once per season with unpenalized fringe-count covariates (fringe F and D among defenders and among
    attackers on ice), holding the DPM specification otherwise fixed (α from the DPM CV, ρ = 1 chain,
    same stints). The published DPM columns are unchanged; the valuation uses the refit values. The
    replacement effect itself is then **averaged over seasons per position** (§11 deviation 5); the
    per-season effect is kept as an alternative.
* Value above replacement = (component rate − replacement rate) × TOI ÷ 60.

## 10. Data flags found before modelling

1. `config.yaml` says `xg.model: logit` but the code compares logit and GBM and uses GBM. The config key is dead.
2. `shots_xg.parquet` omits the goalie ID and shooter handedness (joined here from `events` and `players`).
3. The `EN` strength label (38k attempts) includes 6v5 pulled-goalie play. Only 8.1k attempts are truly
   empty-net (`opp_goalie == 0`); the exclusion uses `opp_goalie`.
4. The temporally held-out shared xG over-predicts goals by 13% (2023–25) before isotonic recalibration,
   and cross-fitted calibration varies 0.97–1.08 by season. Season × strength intercepts handle this here.
5. Five goals have no coordinates and drop out of the shot table.
6. `player2` (the goalie) is null on 3,746 goals (mostly empty-net) and 2,601 missed shots.
7. PM-EPA's league mean is not zero (about +0.04 xG/60 pooled) and DPM's zero is approximate, so both are
   re-based on replacement, not on an assumed zero.
8. PM-EPA's TOI (`toi_ev`) and DPM's stint minutes cover slightly different games (match-rate filter).

## 11. Deviations log

*(any change to §4–§9 after results are seen is recorded here with its reason)*

**Implementation clarifications made before results were seen (2026-09-29):**

1. **Forward folds omit the fringe block.** Replacement status is a same-season, TOI-based label, so the
   evaluation folds (§7) fit without the fringe fixed effect; the production fit includes it. This shifts the
   fringe group's location only and does not affect the shrunk-vs-raw comparison.
2. **Goals-per-win unsigned-integer bug.** The first implementation computed goal differential from unsigned
   counts, which wraps for negative margins (result −6×10⁸). Fixed by using signed counts, and both the
   goals-per-win and the team-reconciliation code now have a regression test.
3. **EM iteration budget.** The first tuning run stopped at a cap of 12 plain EM iterations while the estimates
   were still moving; the cap was raised before any evaluation result was read.

**Changes made after results were seen:**

4. **EM stopping rule — variance components had not converged (2026-09-29).** With the cap raised to 120
   iterations and a "max relative change per step < 0.5%" rule, plain EM reported convergence at 74 iterations.
   An Aitken extrapolation of the stored traces showed a per-step rate of 0.98–0.99 and predicted that the
   season-deviation SDs would fall a further 15–20% (forward d_F 0.102 → ~0.084). Plain EM crawls because the
   persistent and season-deviation variances are weakly separated, so a small per-step change is not convergence.
   Replaced by **SQUAREM-accelerated EM with an absolute standard-deviation-scale tolerance** (1e-4 logit):
   a relative tolerance never triggers for a component whose true variance is ~0 (it approaches the boundary
   sub-linearly), as a simulation showed. The accelerated fit converged in 8 cycles and matched the
   extrapolation (d_F 0.082, u_F 0.150). **Every downstream result was recomputed** (tuning, D-type, forward
   folds, production fit, stability, integration, NUTS inputs); the pre-registered decisions and gates were
   re-applied mechanically to the new fit, and none changed (D-xG: F1; D-type: A; F1, F2, F3 pass). A unit
   test now checks that the fitted variances are a fixed point of the plain EM map.
5. **DPM replacement level pooled across seasons (default changed by the author, 2026-09-29).** The
   pre-registered per-season fringe-effect estimate for DPM proved noisy (SD across seasons 0.036 F / 0.029 D
   xGA/60, positive for forwards in 2021 and 2025), which distorts single seasons (defensemen 72% of the
   2025-26 top 25). Initially both versions were published with the per-season one as default; on the model
   author's decision the **season-pooled effect is now the default** (`def_gar`, `total_gar`, `dpm_rep_per60`),
   and the pre-registered per-season version is kept as `def_gar_perseason` / `total_gar_perseason`. The
   choice rests on the estimator's variance (an implausible sign flip), not on which players rank better.
   Effect on the team reconciliation: total slope 1.40 → 1.31; joint slopes 2.05/1.17/1.76 →
   1.68/1.18/1.46. Pooling assumes a stable replacement gap over 2017-2025.

**Known limitations recorded, not changed:**

6. Laplace intervals ignore uncertainty in the variance components. The NUTS check with free variance
   components (single season 2024) shows this is negligible there (Laplace/NUTS SD ratio median 1.00). An
   earlier comparison that showed ~7% narrower Laplace intervals used the unconverged variance components of
   deviation 4 and was superseded by the corrected run.
7. The sensitivity of finishing replacement to the regular/fringe cutoff re-references the fitted shifts
   without refitting; DPM uncertainty uses the published bootstrap SE of the un-refit headline; total-value
   intervals assume independent components.
8. 61 attempts by goalies as shooters (0 goals) and one attempt with a missing coordinate are excluded.
