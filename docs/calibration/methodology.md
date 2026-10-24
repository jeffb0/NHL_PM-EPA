# Joint Calibration of Component Scales — Methodology and Pre-Registration

*Written 2026-09-29 before any calibration model was fitted. Rules in §3-§5 are fixed here; later deviations
are logged in §7.*

## 1. The problem, and what the team-level slopes do and do not identify

The integrated table adds six components (PM-EPA, DPM, FIN, TO, FO; PEN is checked separately) that are each
defined in goals above replacement. Regressing team 5v5 goal differential on the summed components gives
slopes above 1 for the shrunk components (joint: possession 1.53, defense 1.14, finishing 1.80, turnover 1.51,
faceoff 1.99; `reports/valuation/integration_report.md`). Two very different things produce slopes above 1:

* **Scale error:** a component's unit is not one goal. This is what a calibration should correct.
* **Shrinkage compression:** an independent-player prior shrinks effects shared across a team (system, arena,
  pooled credit), so team sums of shrunk values understate team totals. The turnover work showed this is large
  (possession slope 2.04 → 1.56 unshrunk, → 1.16 with turnover and faceoffs added).

Rescaling player values by the shrunk team slopes would therefore inflate individual values for the second
reason. **The calibration must be estimated on quantities where shrinkage does not intervene**: the *unshrunk*
components. Two further identification limits are stated in advance:

* *Concurrent player-level regressions are biased by construction.* Regressing a player's on-ice outcome from the
  same season on his shrunk estimate gives a slope of 1/B (B = shrinkage factor), because the estimate was
  built from that outcome. Player-level validation therefore uses **next-season outcomes**.
* *Forward slopes conflate calibration with persistence.* They are used to check that calibration does not
  hurt prediction, not to set the scale.

## 2. Design

**Estimand.** A weight `w_c` per component such that `w_c × component_c` is in goals: the unit-scale factor.

**Estimation (team level, accounting identity).** For team-season k, with `S_ck` the sum over the team's skaters
(by primary team) of component c above replacement, using *unshrunk* versions where they are defined:

```
GD5_k = a + Σ_c w_c S_ck + e_k        GD5 = 5v5 goals for − against; 284 team-seasons
```

* Unshrunk components: possession (raw PM-EPA rate above the fringe rate), finishing (goals − xG_f above the fringe
  rate per attempt), turnover (raw arena-adjusted charge above the fringe rate), faceoffs (Σ swing × (win − league
  win probability at that dot) above the fringe rate per draw). **DPM has no unshrunk analogue** (a RAPM coefficient
  needs regularization), so the published valuation column is used. Penalties are realized on special teams and are
  excluded from the 5v5 identity (checked against special-teams goal differential).
* **Ridge toward 1:** `w = argmin Σ_k (GD5_k − a − Σ w_c S_ck)² + λ Σ_c (w_c − 1)²`, with the intercept unpenalized.
  Data that cannot distinguish a slope from 1 leave the weight at 1. λ is chosen by leave-one-season-out
  cross-validation (grid geometric in λ). Standard errors by a team-season bootstrap at the chosen λ.

**Application.** `w_c` multiplies the *shrunk* player component (`*_gar`), giving `*_gar_cal`, `total_gar_cal` and
`total_war_cal` (÷ goals per win). The assumption is that shrinkage handles noise and w handles unit scale.
The existing uncalibrated columns are kept.

**Player-level check (out of sample).** For player-season i in season t+1 with team-season k(i):

```
on-ice 5v5 GD per 60_{i,t+1} = α_k + b · own total rate_{i,t} + g · teammates' mean total rate_{i,t} + ε
```

weighted least squares by 5v5 minutes with team-season fixed effects (within transformation). Own and teammate
rates come from season t, teammates being the four skaters sharing the ice in t+1 stints (a player without a
prior-season value counts as 0). The comparison is between the uncalibrated total (all w = 1) and the calibrated
total.

## 3. Decision rule (pre-registered)

**Adopt the calibrated weights (D-cal)** iff the leave-one-season-out team-level prediction error of the
calibrated unshrunk sum is lower than that of the all-ones sum, with paired z > 2 over team-seasons. Otherwise the
weights are set to 1 and the result is reported as "no calibration justified". The applied columns follow this
decision mechanically.

## 4. Gates (failures are reported, not tuned away)

| Gate | Threshold |
|---|---|
| CAL-1 the unshrunk joint regression is reported with slope, SE and R² per component | reported |
| CAL-2 D-cal outcome (weights adopted or set to 1) | as §3 |
| CAL-3 no harm at player level: within-team forward squared error of the calibrated total is not worse than the uncalibrated total | paired z > −2 |
| CAL-4 with adopted weights, every weight within [0.5, 2] | all |
| CAL-5 the applied `total_gar_cal` differs from `total_gar` by a weighted mean of 0 goals per player-season only if weights are 1 | consistency check |

**Reported, not gated:** shrunk-versus-unshrunk slope table; team-level slope of GD on xG differential (the xG-to-goals
factor); forward-persistence slopes by component; the change in rankings under calibration; position-specific
weights (F vs D) as a diagnostic.

## 5. Scope

5v5 only. Special teams (PP/PK) is not yet a component, so PEN is calibrated against special-teams goal
differential only as a reported check. New components enter automatically when their `<name>_gar` (and an
unshrunk `<name>_raw_gar`) columns exist.

## 5b. All-strength extension (v2), written 2026-09-29 before it was fitted

The special-teams components (`docs/specialteams/methodology.md`) make an all-strength calibration possible. It is a
separate step beside the 5v5 calibration above, which is unchanged and stays the published `total_gar_cal`.

* **Identity.** `GD_k = a + Σ_c w_c S_ck + e_k` with `GD` = all goals for minus against in the team-season (every event
  strength except shootouts; empty-net goals included, since the extra-attacker states belong to PP and PK), 284
  team-seasons, ten components: possession, defense, finishing (5v5), turnover (TO-RA), faceoffs, **penalties**, and the four
  special-teams components. Penalties enter here because their value is realized through special-teams time, which the
  all-strength target counts (in v1 they were checked separately).
* **Unshrunk analogues.** Penalties: the class-cost-weighted drawn minus taken counts above the fringe rate, with no
  empirical-Bayes shrinkage (`pen_raw_gar`). Special-teams components: the RAPM part is used as published (like DPM, it
  has no unshrunk analogue) plus the finishing part recomputed unshrunk as goals minus xG_f above the fringe pool's rate
  per attempt in that state. Possession, defense, finishing, turnover and faceoff are as in §2.
* **Estimation, decision rule and gates.** As §2, §3 and §4 (ridge toward 1, λ by leave-one-season-out CV, adopt only if
  paired z > 2, CAL-3 to CAL-5 with the all-strength total). Applied columns: `<component>_gar_cal_all`,
  `total_gar_all_cal_all`, written to `player_value_calibrated_all` beside a ridge-weighted variant. If weights are set
  to 1, the applied all-strength total equals `total_gar_all` exactly.
* **Player-level check.** Same design as §2 on next-season on-ice goal differential per 60 over all modeled strengths
  (5v5 stints plus special-teams stints), predictors per all-strength minute, team-season fixed effects, teammate
  context (mean of the other skaters on the ice, whatever the state).
* **Stated in advance.** The 4v4 and 3v3 weights are weakly identified (team-season standard deviation of the summed value
  is below one goal), so the ridge will hold them near 1; PP and PK special-teams RAPM is shrunk, so shrinkage
  compression can inflate their team-level slopes (the ST-4 slopes were PP 1.11 and PK 0.52 on shrunk sums).

## 5c. Goalie in the joint calibration (v3), written 2026-09-29 before it was fitted

With the goaltending component (`docs/goaltending`), the all-strength identity of §5b gains an eleventh regressor, the team's goalie value. The v2
calibration stays as published; v3 is a separate step writing suffixed outputs.

* **Identity.** `GD_k = a + Σ_c w_c S_ck + w_gk S_gk,k + e_k`, all-strength goal differential, 284 team-seasons, the ten components of §5b plus
  `gk`. `S_gk,k` is the team's goalie value in goals above replacement, summed over the shots the team's goalies faced (assigned to the defending team,
  so a goalie who changes teams is split correctly; no primary-team assignment is involved).
* **Unshrunk analogue.** `Σ (p_replacement − goal)` over the team's unblocked shots faced: the goals a replacement goalie would have been expected to
  allow (shot xG, shooter shift and the replacement effect) minus the goals actually allowed. The shrunk value is `Σ (p_replacement − p_goalie)`
  from the mixed model. **Stated in advance:** the unshrunk goalie sum is mechanically part of team goal differential, so its slope is near 1 by
  construction; the goalie weight is a scale check, not an independent identification. The shrunk goalie slope is expected above 1 (shrinkage
  compression, as for the skater components).
* **Estimation, decision rule and gates.** As §5b (ridge toward 1, λ by leave-one-season-out CV, adopt only if paired z > 2 against all ones). The
  decision is on the joint eleven-weight vector against all ones, and additionally the v3 CV error is compared with v2's (reported). CAL-3 to CAL-5
  as in §5b for the skater totals (deployment-adjusted CAL-3 is the operative version, §7 deviation 4). CAL-4 includes the goalie weight.
* **Application.** Skater weights multiply the skater columns as in §5b (`total_gar_all_cal` in the v3 table); the goalie weight multiplies
  `goalie_gar` in a goalie table (`goalie_gar_cal`). Nothing else changes.
* **Reported, not gated:** the change in each skater weight from v2 to v3 (the goalie regressor removes goaltending variance from the residual, so skater
  weights are better identified); the team-level R² with and without the goalie.

## 5d. Position-specific weights (v4), written 2026-09-29 before it was fitted as a pre-registered spec

**Why.** A review by the author found a clear misfit: Josh Morrissey and Mason Appleton (2022) come out at the same total (2.73 goals above replacement)
although Morrissey is an elite defenseman and Appleton a depth forward. Diagnostics found that defensemen's values are structurally compressed
(possession value for defensemen has a standard deviation of 0.016 goals per 60 against 0.116 for defense; the D total tracks ice time and
scoring only weakly, correlations 0.20 and 0.16 against 0.37 and 0.60 for forwards), and that the position-split team-level regression gives very
different slopes by position (e.g. forwards' penalty-kill value −0.98 ± 0.35, defensemen's 3.16 ± 0.51). The pooled calibration cannot see this.

**What was tried before this pre-registration (disclosed).** Four specifications were run once on the same data: pooled weights (v3), possession
split by position, possession and defense split, and the seven main components split by position. Only the last improved held-out error
(z = 2.24 against pooled, 2.27 against all ones); the first two were worse than pooled. Choosing the best of four is a forking path, so this spec is
**exploratory**: it is registered here as the single v4 spec and judged by the rules below, and its evidence is weaker than that of a
single pre-specified test. A block-specific shrinkage change to the special-teams RAPM was also tried and found not supported (held-out error
flat, 357.19 against 357.22).

**Spec.** The all-strength identity of §5c with the components possession, defense, finishing, turnover, penalties, power play and penalty kill each
entering **twice** (forwards' team sum and defensemen's team sum), faceoffs, 4v4, 3v3 and the goalie pooled: 18 weights. Ridge toward 1, λ by
leave-one-season-out CV, bootstrap intervals, as §2 to §3.
* **Decision (D-cal-pos):** adopt the 18 weights iff the held-out error beats *both* the all-ones sum and the pooled ridge of §5c with paired z > 2.
* **Application:** each player's component (the shrunk `*_gar` column) is multiplied by the weight for his position; applied weights are
  clipped to [0, 3] (a component can be switched off, not reversed, and no more than tripled); unclipped weights are reported.
* **Gates:** CAL-3 (deployment-adjusted player-level next-season check, no harm, z > −2) for the position-weighted total; CAL-5 (identical to the
  uncalibrated total when weights are 1). Reported, not gated: the weights with intervals, the effect on defensemen relative to forwards, the
  correlation of defensemen's totals with ice time and scoring before and after (an external-validity look, not a criterion), and the two
  named players.
* **Stated in advance:** the position-specific weights at the team level are identified from team-sum variation only (284 team-seasons), the F and D
  sums are correlated, and a weight near 0 or above 2 says the component's *scale for that position* is off in the data, not that the component is
  wrong for every player. Replacement-level offsets (constant per minute) cannot be separated from weights here.

## 6. Limitations stated in advance

* 284 team-seasons give wide intervals; weights near 1 are the expected outcome unless the data disagree strongly.
* Team-level residual variance includes goaltending, coaching and depth, which no component measures.
* Primary-team assignment misplaces the share of value of players traded mid-season.
* DPM is calibrated on its shrunk values only.

## 7. Deviations log

*(any change to §2-§5 after results are seen is recorded here with its reason)*

1. **Ridge-weighted variant published (added after the team-level result, z = 1.69 < 2).** The rule keeps all weights at 1. So that the alternative is auditable, the ridge-weighted total is still published as `total_gar_ridge`, and the player-level forward check (§4) was run on it. This does not change the decision. CAL-3 fails for this variant (paired z = -5.60 on goals), which independently supports not adopting it.
2. **Post-hoc variant comparison (`src/calibration/variants.py`, not pre-registered).** After the forward component slopes showed a negative turnover slope, alternative totals (dropping or swapping single components) were compared with the current total on the same forward check. This is exploratory: it was run once, on the same data that motivated it, and it changes no default. Results are in `reports/metrics/cal_variants.json` and the narrative of the validation report.
3. **Default turnover component changed to TO-RA (user decision, 2026-09-29).** The first calibration run (original turnover charge) showed that a total with TO-RA predicted next-season results much better (paired z +8.9 goals, +7.4 xG) than the total with the original charge. The valuation layer now uses TO-RA as `to_gar`; the original is `to_gar_orig` and the total with it is `total_gar_to_orig`. The whole calibration was rerun on the new default; all numbers in the report are from that rerun. The decision rule was not changed (weights stay at 1; z = 1.50).
4. **All-strength player check adjusted for deployment (§5b, after seeing the raw result).** The pre-registered all-strength player
   check regressed next-season on-ice goal differential per 60 over all modeled strengths on the total rate. It failed CAL-3
   (paired z = −10.82). The raw outcome contains a deployment effect the design missed: power-play stints have a positive league
   mean goal differential and penalty-kill stints a negative one, independent of player skill (the penalty-kill component's
   forward slope was −2.98 on that outcome). The corrected check subtracts each stint's league mean for the side's state cell
   (5v5, power play, penalty kill, 4v4, 3v3) and adds the player's next-season minute shares by state as controls
   (`cal_player_all.json`, key `deployment_adjusted`). Both results are reported; the decision rule for adopting weights (§3,
   team level) is unchanged. Under the adjusted check CAL-3 passes (goals z = −1.03, xG z = +4.17).
5. **Thin adoption margin for the all-strength weights (recorded, no change made).** Adopted by the pre-registered rule with z = 2.17 (threshold 2). The margin is thin.
