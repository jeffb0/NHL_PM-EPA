# Special-teams components — methodology and pre-registration

*Status: written 2026-09-29, before any special-teams model was fitted. The decision rules in §6 are fixed here.
Changes after results are seen are logged in §9.*

## 1. What is being built

Four components on the same goals-above-replacement scale as the 5v5 components, one per game state, at the
author's request (2026-09-29):

| component | column | state (from the on-ice team's view) |
|---|---|---|
| power play | `st_pp_gar` | more skaters than the opponent, extra-attacker states included (5v4, 5v3, 4v3, 6v5, 6v4, 6v3) |
| penalty kill | `st_pk_gar` | fewer skaters than the opponent, extra-attacker states of the opponent included (4v5, 3v5, 3v4, 5v6, 4v6, 3v6) |
| four on four | `st_4v4_gar` | 4v4 |
| three on three | `st_3v3_gar` | 3v3 |

Goalies are not valued here (the goaltending component is separate and open). A stint in a mismatched state gives
value to both classes: the advantaged side's skaters are PP skaters and the disadvantaged side's are PK skaters.

## 2. Value definition

For a player in class *c* the component is, in goals,

`st_c_gar = (r_c - r_rep) x TOI_c / 60`,

where `r_c` is the player's net expected-goal rate per 60 in class *c* (offense minus defense, both from the
strength-specific ridge model of §3, plus the player's finishing above the fringe pool from the finishing model,
§4) and `r_rep` is the replacement rate (the fringe pool, §3). Expected goals are shot xG of unblocked attempts
(the same xG as the finishing component's `xg_v1`), which is calibrated in every state
(goals / xG: PP 1.00, PK 0.98, 4v4/3v3 1.00).

**No overlap with other components.**

* The 5v5 components use 5v5 stints only, and the ST states are disjoint from 5v5, so there is no overlap in time.
* The penalty component (PEN) values *drawing or taking* a penalty (the change in special-teams time). ST values
  performance *per special-teams minute above replacement*. A player who draws many penalties gets PEN credit for
  the extra PP time; ST values what he does with it relative to a replacement player on the same ice time.
* The finishing component's 5v5 value covers 5v5 shots only. Its non-5v5 finishing, held so far as the memo column
  `fin_gar_other_strengths_memo`, is added to the four ST components here by state (§4) and is not added to
  `total_gar` a second time (the memo column is not in the total).
* The PM-EPA and DPM ledgers are not used here. `pm_epa_pp/pk` exist in the possession pipeline but the value
  surfaces are 5v5-fitted, so the ST components are built directly on shot xG. This is a deliberate difference from
  the 5v5 possession components and is listed as a limitation.

## 3. Model: strength-specific ridge RAPM

Stints are maximal intervals with constant on-ice skaters and constant game state; faceoffs are stint boundaries
(zone start recorded), as in `docs/defense/methodology.md`. Each stint gives two rows, one per attacking side; the
response is the attacking side's xG per 60, weighted by minutes.

* **Roles.** In a mismatched state each player gets role-specific columns (`id x 10 + role`, role 1 = advantaged
  side, 2 = disadvantaged side); in 4v4 and 3v3 there is one role. Within a role, an *offense* coefficient is the
  player's effect on his own side's xG rate and a *defense* coefficient his effect on the opponent's. The PP
  component uses the advantaged-side pair, the PK component the disadvantaged-side pair.
* **Covariates (unpenalized).** Attacking-side-is-advantaged indicator and one indicator per ordered
  (attackers, defenders) skater count, home attack, zone start (OZ/NZ/DZ), score state (±1, ±2), and counts of
  fringe forwards and defensemen in each role and block (offense/defense). The fringe counts carry the
  replacement level.
* **Penalty.** Ridge with the chained prior of the defense model (prior mean = ρ × the player's coefficient the
  previous season for the same role; 0 for new players). α by game-grouped 5-fold CV within seasons, pooled over
  seasons, separately per class; ρ from {0, 0.5, 1} by held-out loss on validation pairs.
* **Replacement.** A regular's value over replacement in a block is his coefficient minus the pooled (over seasons)
  fringe effect of that block and position group, exactly as in the pooled DPM default. Fringe players carry their
  own shrunk deviation.
* **Regulars and fringe.** The status definition of `src/valuation/replacement.py` (top 12 F / 6 D per team by
  all-strength TOI) is reused unchanged.
* **Uncertainty.** Game-block Poisson bootstrap of the ridge fit (fixed α, ρ), 100 replicates.

## 4. Finishing by state

The finishing model (`src/finishing`) has one persistent shooter effect used in every state. The finishing part
of `st_c_gar` is the shot-by-shot difference between the shooter's fitted probability and the fringe pool's, summed
over the player's unblocked attempts in state *c*, computed from the saved shifts `fin_shot_shifts.parquet` in the
same way as the 5v5 finishing value. State assignment uses the shooter's team skaters: PP = more skaters than the
opponent (own-goalie-pulled 6v5 included), PK = fewer, 4v4, 3v3. Posterior draws for non-5v5 finishing were not
saved, so the finishing part carries no interval; the component's interval is the ridge bootstrap only (stated).

## 5. Integration

`player_value_season` gains `st_pp_gar`, `st_pk_gar`, `st_4v4_gar`, `st_3v3_gar` (with `_sd`, `_wins`) and the totals
`total_gar_all = total_gar + Σ st_*_gar` and `total_war_all`. `total_gar` keeps its 5v5 definition so that the
calibration of §`docs/calibration` is unchanged; folding the ST classes into the joint calibration (all-strength
goal differential) is the next step.

## 6. Gates (failures are reported, not tuned away)

| gate | rule | applies to |
|---|---|---|
| ST-1 coverage | (a) ≥ 90% of shot xG in the ST states (all-strength shots outside 5v5, by shooting team's skater counts) is assigned to a valid stint of the matching class; (b) non-5v5 skater-minutes in valid stints ≥ 85% of `toi_all − toi_5v5` summed over players (reported by class) | all classes |
| ST-2 out-of-sample | on test pairs (2023→24, 2024→25), the fitted model beats the covariates-only baseline at predicting next-season stint xG/60 (paired game-block z > 2) | PP, PK gated; 4v4 and 3v3 reported |
| ST-3 stability | season-centred year-over-year r of the shrunk net rate (≥ 100 minutes in class in both seasons; independent ρ = 0 fits) at least 0.20 | PP, PK gated; 4v4 and 3v3 reported |
| ST-4 team reconciliation | team-season class goal differential regressed on the team's summed class value (primary team): slope within [0.5, 2.0] (shrunk sums compress; the calibration report showed slopes above 1 are partly expected) | PP, PK gated; 4v4 and 3v3 reported |
| ST-5 no double counting | correlation of `st_pp_gar` and `st_pk_gar` per minute with the PEN component's drawn/taken rates is reported (no threshold) | reported |
| ST-6 finishing sanity | Σ over players of the state finishing value equals the finishing model's non-5v5 total to 1e-6 (`fin_gar_other`) | all classes together |

4v4 and 3v3 hold few minutes (about 2% of ice time each); their components are expected to be dominated by
shrinkage. They are published as requested, with their gate results, and are flagged as low-information in the
report.

## 7. Limitations stated in advance

* Shot xG only: no possession-value ledger (entries, passing, breakouts) in ST states.
* Teammate confounding is handled by ridge, not solved: PP units are highly collinear, so the RAPM is a
  regularized estimate, especially for PP1 forwards.
* Special-teams deployment (who is on PP1) is a coach's choice: value is per minute above replacement, and the
  minutes themselves are not credited or debited.
* Empty-net and extra-attacker states are folded into PP/PK by skater count and share the class fixed effects
  only through the (attackers, defenders) count indicators.
* Primary-team assignment for the team reconciliation misplaces traded players' value.

## 8. Deliverables

`src/specialteams/`, `config.yaml` section, tests, `reports/specialteams/validation_report.md`, model card, updated
valuation table, README.

## 9. Deviations log

*(any change to §2–§6 after results are seen is recorded here with its reason)*

1. **Minutes threshold for 4v4 and 3v3 (before their gate results were seen).** ST-3 asks for 100 minutes in the class in
   both seasons. Nobody reaches 100 minutes in 4v4 (max 56 per season) or 3v3 (max 44), so for those two classes,
   whose gates are reported and not applied, the threshold is 15 minutes (`min_minutes_thin`). PP and PK keep 100.
3. **Data-integrity fix after the first run (stint builder).** The first run failed ST-1 for 3v3 (xG coverage 79%) and its
   3v3 stints held 290 of about 1,360 goals. Cause: bookkeeping events (`period-end`, `game-end`, `shootout-complete`)
   carry placeholder skater counts (for example 4v4 at the end of a 3v3 overtime), which invalidated the interval that
   contains a sudden-death goal. These events are now excluded from the skater-count consistency check. Coverage is now
   97–98% in every group; the whole special-teams chain was rerun from the stints step. No threshold or rule was changed.
2. **Jitter on the ridge diagonal (1e-6) for the special-teams fits only.** Thin states can leave a fringe-count
   covariate column empty in a cross-validation fold, which makes the unpenalized block singular. The 5v5 models are
   untouched (`ridge.JITTER` is 0 there).
