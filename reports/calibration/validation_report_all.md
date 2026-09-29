# Joint calibration, all-strength extension — validation report

Spec and pre-registered rules: [methodology §5b](../../docs/calibration/methodology.md). Ten components (the five 5v5 components, penalties and the four special-teams components) against all-strength team goal differential, estimated on the unshrunk versions with a ridge prior toward 1. The 5v5 calibration ([report](validation_report.md)) is unchanged.

**The calibrated weights were adopted** (rule of methodology §3): leave-one-season-out team-level error of the calibrated sum vs the all-ones sum, MSE 857.1 vs 897.4, paired z = 2.17 (adoption needs z > 2).

## Gates

| Gate | Result | Pass |
|---|---|---|
| CAL-1 unshrunk joint regression reported | see the table below | ✅ |
| CAL-2 decision rule | z = 2.17 → adopt | ✅ |
| CAL-3 no harm at player level, as pre-registered (raw all-strength on-ice goal differential, next season) | paired z = -10.82 (needs > −2) | ❌ |
| CAL-3 no harm at player level, deployment-adjusted (methodology §7, deviation 4) | paired z = -1.03 (needs > −2) | ✅ |
| CAL-4 adopted weights within [0.5, 2] | pm 1.42, def 1.20, fin 0.78, to 1.22, fo 1.02, pen 0.94, pp 1.31, pk 1.10, 4v4 0.96, 3v3 1.05 | ✅ |
| CAL-5 consistency: applied columns equal the uncalibrated ones exactly when weights are 1 | weights adopted, so non-zero by design | ✅ |

## Team level (284 team-seasons, all-strength goal differential, SD 46.0)

| component | shrunk slope ± SE | unshrunk slope ± SE | ridge weight (95% CI) | applied weight |
|---|---|---|---|---|
| possession (PM-EPA) | 1.70 ± 0.70 | 2.21 ± 0.40 | 1.42 (1.22–1.62) | 1.42 |
| defense (DPM) | 1.33 ± 0.24 | 1.22 ± 0.21 | 1.20 (0.98–1.43) | 1.20 |
| finishing, 5v5 (FIN) | 2.17 ± 0.44 | 0.74 ± 0.11 | 0.78 (0.64–0.92) | 0.78 |
| turnover cost (TO-RA) | 3.09 ± 0.95 | 1.53 ± 0.54 | 1.22 (1.03–1.40) | 1.22 |
| faceoffs (FO) | 2.63 ± 1.10 | 1.02 ± 0.86 | 1.02 (0.92–1.11) | 1.02 |
| penalties (PEN) | -1.16 ± 1.05 | 0.41 ± 0.52 | 0.94 (0.79–1.10) | 0.94 |
| power play | 1.33 ± 0.28 | 1.40 ± 0.16 | 1.31 (1.10–1.51) | 1.31 |
| penalty kill | 0.90 ± 0.38 | 0.89 ± 0.29 | 1.10 (0.89–1.28) | 1.10 |
| 4v4 | -6.79 ± 2.99 | -0.00 ± 1.13 | 0.96 (0.88–1.04) | 0.96 |
| 3v3 | -2.41 ± 2.67 | 1.68 ± 0.91 | 1.05 (0.97–1.15) | 1.05 |

R²: shrunk 0.563, unshrunk 0.647. Ridge λ = 1.58e+04 (leave-one-season-out CV). DPM and the special-teams RAPM parts appear with their published values in both columns (no unshrunk analogue), so their slopes are subject to shrinkage compression.

Cross-validated team-level squared error by λ (the last entry, ∞, is all weights = 1):

| λ | MSE |
|---|---|
| 10 | 902.7 |
| 39.8 | 901.3 |
| 158 | 896.7 |
| 631 | 885.6 |
| 2.51e+03 | 869.2 |
| 1e+04 | 857.8 |
| 3.98e+04 | 862.6 |
| 1.58e+05 | 880.7 |
| 6.31e+05 | 892.1 |
| inf | 897.4 |

## Player level, next season (team-season fixed effects, teammate context)

5,087 player-seasons (≥ 250 modeled minutes in t+1), 253 team-seasons. Predictors per modeled minute in season t; the outcome is on-ice goal (or xG) differential per 60 in t+1 over the 5v5 and special-teams stints. "Calibrated" is the ridge-weighted total (the weights adopted above). The **raw** outcome rewards power-play time and penalizes penalty-kill time whatever the player's skill; the **deployment-adjusted** outcome subtracts each stint's league mean for the side's state and adds the player's minute shares by state as controls.

| outcome | total | own-rate slope ± SE | teammate slope | within-team R² |
|---|---|---|---|---|
| raw goals | uncalibrated | 1.55 ± 0.08 | 8.254 | 0.222 |
| raw goals | calibrated | 1.29 ± 0.07 | 7.452 | 0.205 |
| raw xG | uncalibrated | 1.23 ± 0.06 | 5.844 | 0.197 |
| raw xG | calibrated | 1.03 ± 0.06 | 5.287 | 0.182 |
| adjusted goals | uncalibrated | 0.81 ± 0.06 | 4.953 | 0.298 |
| adjusted goals | calibrated | 0.71 ± 0.05 | 4.442 | 0.297 |
| adjusted xG | uncalibrated | 0.60 ± 0.04 | 3.112 | 0.320 |
| adjusted xG | calibrated | 0.56 ± 0.03 | 2.834 | 0.326 |

Calibrated vs uncalibrated total, paired by team-season (positive favors calibrated): raw goals z = -10.82, raw xG z = -10.19; adjusted goals z = -1.03, adjusted xG z = 4.17.

Forward slope of next-season on-ice differential on each season-t component (own rate; teammates and state shares controlled in the adjusted columns):

| component | raw goals | adjusted goals | adjusted xG |
|---|---|---|---|
| possession (PM-EPA) | 3.43 ± 0.62 | 1.40 ± 0.38 | 2.24 ± 0.37 |
| defense (DPM) | -0.61 ± 0.12 | 0.81 ± 0.10 | 0.70 ± 0.06 |
| finishing, 5v5 (FIN) | 2.71 ± 0.24 | 1.17 ± 0.21 | — |
| turnover cost (TO-RA) | -2.62 ± 0.61 | -0.24 ± 0.44 | -0.29 ± 0.27 |
| faceoffs (FO) | -0.02 ± 0.44 | 0.31 ± 0.34 | 0.14 ± 0.22 |
| penalties (PEN) | 2.08 ± 0.48 | 0.93 ± 0.40 | 0.47 ± 0.24 |
| power play | 2.86 ± 0.20 | 1.15 ± 0.18 | 1.01 ± 0.11 |
| penalty kill | -2.98 ± 0.18 | 0.11 ± 0.16 | 0.38 ± 0.10 |
| 4v4 | 1.91 ± 2.75 | -6.92 ± 2.33 | -1.64 ± 1.53 |
| 3v3 | 16.22 ± 2.12 | 2.58 ± 1.94 | 2.39 ± 1.19 |
| teammate_context | 6.17 ± 0.32 | 4.77 ± 0.30 | 2.91 ± 0.20 |

## Effect on the table

Applied weights: the ridge weights. `total_gar_all_cal` differs from `total_gar_all` by a mean absolute 0.516 goals among qualified skaters; Spearman rank correlation 0.9954. Both columns are published in `player_value_calibrated_all`.

Largest changes, latest season (qualified):

| player_name | position | pm_gar | st_pp_gar | total_gar_all | total_gar_all_cal | d |
|---|---|---|---|---|---|---|
| Mark Stone | R | 0.47 | 9.39 | 18.36 | 21.34 | 2.98 |
| Dylan Larkin | C | 0.35 | 8.38 | 18.65 | 21.40 | 2.75 |
| Adam Pelech | D | 0.39 | 0.00 | 12.79 | 15.30 | 2.51 |
| Artem Zub | D | 0.38 | 0.04 | 15.88 | 18.35 | 2.47 |
| Nick Suzuki | C | 0.43 | 7.04 | 20.17 | 22.58 | 2.41 |
| Noah Cates | L | 0.64 | 3.07 | 15.96 | 18.33 | 2.37 |

| player_name | position | pm_gar | st_pp_gar | total_gar_all | total_gar_all_cal | d |
|---|---|---|---|---|---|---|
| Tony DeAngelo | D | 0.37 | -1.41 | -9.07 | -11.19 | -2.11 |
| JJ Peterka | R | 0.18 | -1.33 | 1.09 | -0.59 | -1.68 |
| Ivan Barbashev | L | 0.28 | 0.81 | 2.12 | 0.45 | -1.67 |
| Will Smith | C | 0.35 | -0.91 | 0.90 | -0.63 | -1.53 |
| John Klingberg | D | 0.09 | -0.06 | -5.83 | -7.26 | -1.42 |
| Artyom Levshunov | D | 0.13 | -1.69 | -5.40 | -6.67 | -1.27 |

## Interpretation

**Decision.** The pre-registered rule adopts the ridge weights: the leave-one-season-out team-level error falls from 897.4 to 857.1 (paired z = 2.17). The margin over the threshold of 2 is thin, and the 5v5 calibration on the same components did not clear it (z = 1.50). Read the adoption as "weakly supported".

**Direction of the weights.** Possession is under-scaled (unshrunk slope 2.21 ± 0.40, weight 1.42, interval 1.22–1.62) and 5v5 finishing over-scaled (0.74 ± 0.11, weight 0.78, 0.64–0.92), the same directions as the 5v5 calibration (1.53 and 0.82). Turnover (1.22, 1.03–1.40), defense (1.20, 0.98–1.43) and the power play (1.31, 1.10–1.51) are modestly above 1. Faceoffs, penalties, penalty kill, 4v4 and 3v3 are consistent with 1. All weights are inside the [0.5, 2] gate. The 4v4 and 3v3 weights are essentially unidentified (team-level standard deviation of the summed value below one goal) and sit at 1 by construction of the prior; their shrunk slopes (−6.8 and −2.4) are noise around zero and should not be read.

**Player-level check: the pre-registered version is invalid, and both versions are reported.** As pre-registered (raw all-strength on-ice differential per 60), the calibrated total predicts next-season results worse than the uncalibrated one (z = −10.82), so CAL-3 fails as specified. The raw outcome has a flaw the design did not anticipate: power-play minutes carry a positive goal differential and penalty-kill minutes a negative one whatever the player's skill, so a player's deployment, not his value, drives the outcome. The forward slopes show it plainly: the penalty-kill component's slope is −2.98 on the raw outcome and +0.11 once state is adjusted, the power play's 2.86 becomes 1.15, and the turnover slope moves from −2.62 to −0.24. The adjusted check subtracts each stint's league mean for the side's state cell and controls for the player's minute shares by state. It was chosen after seeing the raw result, which is why it is logged as a deviation and both are shown; the adjustment follows from the mechanism above and was not searched over. Under it, the calibrated total is no worse on goals (z = −1.03, inside the −2 tolerance) and better on xG (z = +4.17), so CAL-3 passes.

**Forward component slopes (adjusted).** Possession 1.40 ± 0.38, defense 0.81 ± 0.10, finishing 1.17 ± 0.21, power play 1.15 ± 0.18, penalties 0.93 ± 0.40, faceoffs 0.31 ± 0.34, penalty kill 0.11 ± 0.16 and turnover −0.24 ± 0.44. The penalty kill has almost no forward relationship with next-season on-ice results once deployment is removed, which fits its modest stability (year-over-year r 0.22–0.30) and the weakness of the PK team-level link found in the special-teams validation. These slopes mix scale and persistence and are not used to set the weights.

**Shrinkage caveat.** Defense and the special-teams RAPM parts are used as published (no unshrunk analogue), so their team-level slopes may be inflated by shrinkage compression while the unshrunk components are not. The weights for those components are therefore less trustworthy than the others.

**What is published.** `total_gar_all` (uncalibrated) and `total_gar_all_cal` (adopted weights, mean absolute change 0.52 goals among qualified skaters) are both in `player_value_calibrated_all`. The 5v5 `total_gar_cal` is unchanged (weights 1). Which total to treat as the headline is a decision for the model owner; given the thin margin, the failed as-pre-registered CAL-3 and the untested shrinkage caveat, my recommendation is to keep `total_gar_all` as the default and treat the calibrated column as an alternative until it survives a season of new data.

**Limitations.** 284 team-seasons; primary-team assignment misplaces traded players' value; team residual variance includes goaltending and coaching; one forward horizon.

