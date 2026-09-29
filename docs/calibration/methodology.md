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
