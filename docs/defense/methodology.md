# Defensive PM-EPA (DPM) — Methodology and Pre-Registration

*Status: written 2026-09-28, before any defensive model was fitted. The decision rules in §6 are fixed
here and are not changed after results are seen; any later deviation is logged in §8.*

## 1. Goal and scope

PM-EPA (v1.0) values what a skater's team does **with** the puck at 5v5. DPM values what a skater
prevents the **opponent** from doing with it: expected goals against (xGA), per 60 minutes of 5v5
TOI, above an average skater, adjusted for teammates, opponents, zone starts, score and venue.

It is the second component of the planned all-in player valuation. The two are designed to sum
without double counting on the goal scale (§5).

**In scope:** 5v5, regular season, skaters. **Out of scope:** goaltending (xG is pre-save, so the
goalie never enters), penalty kill (later component), faceoffs, and the offensive-turnover cost of
giving the puck away (a separate component; see §3).

## 2. Why not just xGA RAPM?

Plain defensive xGA RAPM is the public standard, but it is noisy: shots against are rare, so a
season of 5v5 minutes gives year-over-year r of roughly 0.2–0.3. The PM-EPA ledger gives a denser
signal. Every opponent possession's xG G_p already splits **exactly** into (λ = 0 identity,
`reports/metrics/identity_check.json`):

```
G_p = V_start(where O gained the puck) + Σ Δ(movement) + terminal(attempt or loss) + shot quality
```

A defender who stops a zone entry or forces a dump-in reduces the movement term. That shows up on
many possessions that never produce a shot, so it should be measurable with less noise than xGA itself.

## 3. Ledger decomposition (from the defending team D's view)

For each opponent (O) possession, the λ = 0 `credits` rows are summed back to team level
(pooled rows re-summed) and classified:

| DPM component | Ledger rows of O's possession | Attributed to D's… |
|---|---|---|
| `start_turnover` | O's start value when O gained the puck from D: `takeaway`, `loose_opp_giveaway`, `loose_unobserved` | **offense / puck handling** (excluded from DPM; feeds the future turnover-cost component, and PM-EPA's λ = 1 column) |
| `start_faceoff` | O's start value after a faceoff win | **faceoffs** (excluded; future component) |
| `start_recovery` | O's start value after a loose puck: `loose_own_*`, `rebound`, `loose_opp_miss/sog/block` | **defense** (retrievals, rebound control) |
| `start_unobserved` | value at O's first observation when the start was not observed | defense (reported; tiny) |
| `move` | O's movement segments Δ (carry, move_*, rebound segments) | **defense** (entry/exit denial, gap control) |
| `term` | O's terminal decision value: shot attempt (V_att − V_dec), giveaway, unobserved loss, icing/offside, takeaway-against, other stoppages (−V) | **defense** (forcing losses, preventing attempts) |
| `quality` | O's shot quality xG − V_att | defense + shooter (reported; expected noisy) |

The seven components sum exactly to xGA. Because ridge regression is linear in its target, the
component coefficients at a common α sum exactly to the xGA RAPM coefficient (unit-tested).

## 4. Estimation

**Stints.** Maximal intervals within a period with constant on-ice skaters, 5v5 with both goalies
in, from shift charts. Only games with on-ice match rate ≥ 0.9 are used (same rule as the PM-EPA
RAPM validation). Each stint gives two rows, one per team on defense.

**Assigning ledger rows to stints.** Each ledger row carries the event where it was booked. It is
assigned to the stint containing that event under the on-ice convention (`start < t ≤ end`;
faceoffs `start ≤ t < end`). The assignment is checked against the event's on-ice lists and the
match rate is reported.

**Regression (weighted ridge, one per component, shared design).**

```
y_c,row = 60 · Σ(component c of O's possessions booked in the stint) / minutes      weight = minutes
X       = +1 for each attacking skater (offense block) | +1 for each defending skater (defense block)
          | home-attack | stint starts with a faceoff in O's OZ / NZ / DZ (on-the-fly = base)
          | O's score state (−2…+2, tied = base)
```

The intercept and covariates are unpenalized; player columns share one penalty α_c per component.
Solved in closed form from the normal equations. The **defense coefficient is sign-flipped** so that
positive = fewer xG against = good.

**α selection.** Per component, 5-fold game-grouped CV on the training seasons (2017–2021), grid
{100, 300, 1000, 3000, 10000, 30000}. Chosen once, then fixed for all fits.

**Fits produced.** (a) single season, (b) three-season windows, (c) prior-informed single season
(ρ ∈ {0, 0.5, 1} times last season's coefficient used as the ridge prior mean; ρ chosen per §6).

**Uncertainty.** Game-block Poisson bootstrap (B = 200) of each season fit; 95% interval from
percentiles.

## 5. Combining with PM-EPA without double counting

Team 5v5 net xG = Σ own possessions G_p − Σ opponent possessions G_p. PM-EPA partitions the first
sum across the team's skaters; DPM estimates each skater's effect on the second. The two ledgers
share no rows. The excluded rows of the second (`start_turnover`, `start_faceoff`) are left for the
turnover and faceoff components so that nothing is counted twice.

The possession-denial mechanism (a team that keeps the puck gives up fewer possessions) is real
defensive value and stays in DPM. §6 reports the DPM–PM-EPA correlation so the overlap in *signal*
is visible.

## 6. Pre-registered evaluation and decision rules

**Splits (seasons as consecutive pairs Y → Y+1).** Train pairs 2017→18 … 2020→21; validation pair
2021→22 and 2022→23; test pairs 2023→24 (crosses the scoring regime break; flagged) and 2024→25.

**Primary test: out-of-sample xGA prediction.** Fit on season Y. Predict every stint-row xGA/60 in
Y+1 as intercept + covariates (from Y) + Σ attacker offense coefficients + Σ defender defense
coefficients (players unseen in Y = 0). Score = minute-weighted MSE, with a game-bootstrap SE of
the difference from baseline.

| Model | Defense term |
|---|---|
| B0 | none (league average) |
| M0 | xGA RAPM (total target) |
| M1 | DPM-core: `start_recovery + start_unobserved + move + term` (quality excluded) |
| M2 | M1 + `quality` |
| M3 | stacked: every component coefficient with non-negative weights w_c, learned on the train pairs |

**Decision rule D1 (headline definition).** On the validation pairs, pick the model with the lowest
MSE. If another model is within 1 bootstrap SE of it, prefer the simpler, in the order M1 < M2 < M0 < M3.
The chosen model is then reported once on the test pairs and not changed.

**Decision rule D2 (prior ρ).** ρ is chosen by the same validation MSE rule (ties → ρ = 0).

**Gates (must pass to ship v1; failures are reported, not tuned away).**

| Gate | Threshold |
|---|---|
| G1: headline beats B0 on test pairs | MSE lower, difference > 2 SE |
| G2: season-centred YOY r, ≥ 500 5v5 min | F ≥ 0.25 and D ≥ 0.25 |
| G3: headline YOY ≥ plain xGA RAPM YOY | both positions |
| G4: identity (component coefficients at a common α sum to xGA RAPM) | max abs diff < 1e-8 |
| G5: ledger → stint assignment match rate | ≥ 0.98 of 5v5 ledger value |

**Reported, not gated:** split-half r; team-changer YOY; r with PM-EPA/60 (overlap); r with
on-ice xGA/60 relative to teammates; team-level sum vs team xGA; face validity (top/bottom 15 by
position, with CIs). Face validity is inspected but never used to tune.

## 7. Outputs

* `data/outputs/dpm_player_season.csv`: player-season DPM/60 (headline), components, xGA RAPM,
  95% CI, TOI, position percentile, qualification flag.
* `reports/defense/validation_report.md`, `reports/metrics/dpm_*.json`.

## 8. Deviations log

*(any change to §3–§6 after results are seen is recorded here with its reason)*

**Clarifications made while implementing, before any result was seen (2026-09-28):**

1. **Order of D1 and D2.** D1 is decided with independent season fits (ρ = 0). D2 then picks ρ for
   the D1 model.
2. **Stability gates use ρ = 0 fits.** A prior built from last season's coefficients would inflate
   YOY and split-half r mechanically. G2, G3 and the split-half figures are therefore always
   computed on independent single-season fits, whatever ρ D2 picks for the published values.
3. **Split-half** = odd vs even game_id within a season. A player needs ≥ 250 minutes in each half.
4. **G5 stint situation rule (fixed before any model was fitted).** The first build scored G5 = 0.976.
   The cause was taking a stint's situation code from the last event *before* it, which lags a
   situation change. The rule now uses the codes of the events inside the stint. G5 = 0.990 after
   the fix, with a regression test added.

**Consequences of the pre-registered selection (written after `dpm_selection.json` was seen):**

5. D1 selected **M0**, so the headline is xGA RAPM with the D2 prior (ρ = 1). The hypothesis in §2,
   that dropping components would denoise the estimate, was **not supported** on validation. The
   headline definition follows the rule; nothing was re-tuned.
6. With M0 as headline, the final fits use **one common α (α_xg) for every component**. The G4
   identity then makes the component columns an exact additive breakdown of the headline.
   `dpm_val_per60` = headline − `start_turnover` − `start_faceoff` is published as the
   valuation-ready column, per §5. These are output definitions only; no gate or model choice
   depends on them.
