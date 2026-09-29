# Joint calibration of component scales — validation report

Spec and pre-registered rules: [methodology](../../docs/calibration/methodology.md). The calibration is estimated on the *unshrunk* components at team level (where shrinkage cannot masquerade as scale error) with a ridge prior toward 1, and applied to the shrunk player values.

**No calibration was justified: all weights are set to 1** (leave-one-season-out team-level error of the calibrated sum vs the all-ones sum: MSE 468.5 vs 485.2, paired z = 1.50; adoption needs z > 2).

## Gates

| Gate | Result | Pass |
|---|---|---|
| CAL-1 unshrunk joint regression reported | see the table below | ✅ |
| CAL-2 decision rule | z = 1.50 → weights = 1 | ✅ |
| CAL-3 no harm at player level (next-season within-team error, ridge-weighted variant vs uncalibrated; moot while weights stay 1) | paired z = -7.66 (needs > −2) | ❌ |
| CAL-4 adopted weights within [0.5, 2] | pm 1.00, def 1.00, fin 1.00, to 1.00, fo 1.00 | ✅ |
| CAL-5 consistency: applied columns equal the uncalibrated ones exactly when weights are 1 | max absolute change 0 | ✅ |

## Team-level scale (284 team-seasons, 5v5 goal differential)

| component | shrunk slope ± SE | unshrunk slope ± SE | ridge weight (95% CI) | applied weight |
|---|---|---|---|---|
| possession (PM-EPA) | 1.20 ± 0.48 | 1.79 ± 0.28 | 1.53 (1.21–1.82) | 1.00 |
| defense (DPM) | 1.07 ± 0.17 | 1.05 ± 0.16 | 1.09 (0.84–1.36) | 1.00 |
| finishing (FIN) | 1.87 ± 0.26 | 0.81 ± 0.08 | 0.82 (0.69–0.95) | 1.00 |
| turnover cost (TO) | 2.19 ± 0.68 | 1.18 ± 0.40 | 1.15 (0.77–1.52) | 1.00 |
| faceoffs (FO) | 2.07 ± 0.78 | 1.32 ± 0.63 | 1.10 (0.85–1.34) | 1.00 |

R²: shrunk 0.473, unshrunk 0.542. Ridge λ = 3.98e+03 (chosen by leave-one-season-out CV). DPM has no unshrunk analogue and appears with its published values in both columns. Team goal differential on team xG differential (context for the xG-scale components): slope 1.03 ± 0.05, R² 0.58.

Cross-validated team-level squared error by λ (the last entry, ∞, is all weights = 1):

| λ | MSE |
|---|---|
| 10 | 472.3 |
| 39.8 | 472.2 |
| 158 | 471.6 |
| 631 | 470.1 |
| 2.51e+03 | 468.5 |
| 1e+04 | 470.0 |
| 3.98e+04 | 475.2 |
| 1.58e+05 | 480.6 |
| 6.31e+05 | 483.7 |
| inf | 485.2 |

## Player level, next season (team-season fixed effects, teammate-context control)

5,168 player-seasons (≥ 200 5v5 minutes in t+1), 253 team-seasons. Outcome: on-ice 5v5 differential per 60 in t+1; predictors from season t.

| outcome | total | own-rate slope ± SE | teammate slope | within-team R² |
|---|---|---|---|---|
| goals | uncalibrated | 0.93 ± 0.07 | 6.650 | 0.125 |
| goals | calibrated | 0.81 ± 0.06 | 6.206 | 0.105 |
| xG | uncalibrated | 0.61 ± 0.04 | 3.591 | 0.161 |
| xG | calibrated | 0.59 ± 0.04 | 3.529 | 0.157 |

Slopes below 1 here combine shrinkage calibration and genuine season-to-season change in talent, so they are not used to set the scale. Calibrated vs uncalibrated total, paired by team-season: goals z = -7.66, xG z = -1.61 (positive favors the calibrated total).

Forward slope of next-season on-ice goal differential on each season-t component (own rate; teammate context controlled):

| component | goals: slope ± SE | xG: slope ± SE |
|---|---|---|
| possession (PM-EPA) | 1.99 ± 0.38 | 2.77 ± 0.36 |
| defense (DPM) | 0.50 ± 0.08 | 0.45 ± 0.04 |
| finishing (FIN) | 1.58 ± 0.11 | — |
| turnover cost (TO) | -0.66 ± 0.35 | -1.04 ± 0.24 |
| faceoffs (FO) | 0.02 ± 0.24 | 0.44 ± 0.14 |
| teammate_context | 6.06 ± 0.38 | 3.28 ± 0.24 |

## Effect on the table

Applied weights: all 1 (no calibration justified), so `total_gar_cal` equals `total_gar`. The ridge-weighted variant (`total_gar_ridge`, not adopted unless stated above) changes total goals above replacement by a mean absolute 0.285 among qualified skaters, with a Spearman rank correlation of 0.9894 against `total_gar`.

Largest changes under the ridge weights, latest season (qualified):

| player_name | position | pm_gar | fin_gar | total_gar | total_gar_ridge | d |
|---|---|---|---|---|---|---|
| Adam Pelech | D | 0.39 | -0.13 | 10.91 | 12.20 | 1.28 |
| Devon Toews | D | 1.44 | 0.41 | 6.34 | 7.45 | 1.11 |
| Lane Hutson | D | 1.48 | 0.55 | 6.98 | 8.08 | 1.09 |
| Artem Zub | D | 0.38 | 0.46 | 9.27 | 10.24 | 0.97 |
| Cale Makar | D | 1.79 | 2.28 | 8.79 | 9.72 | 0.93 |
| Moritz Seider | D | 0.07 | 0.50 | 11.21 | 12.11 | 0.90 |

| player_name | position | pm_gar | fin_gar | total_gar | total_gar_ridge | d |
|---|---|---|---|---|---|---|
| Morgan Geekie | C | -0.11 | 6.45 | 4.95 | 3.50 | -1.45 |
| Mark Scheifele | C | 0.76 | 6.17 | 1.36 | -0.06 | -1.43 |
| David Pastrnak | R | 0.06 | 6.01 | 3.75 | 2.35 | -1.41 |
| Nathan MacKinnon | C | 0.79 | 7.24 | 4.45 | 3.07 | -1.39 |
| Tage Thompson | C | 0.83 | 7.54 | 5.05 | 3.76 | -1.29 |
| Cole Caufield | R | 0.77 | 7.63 | 5.79 | 4.51 | -1.28 |

## Interpretation

This report is computed with the **role-adjusted turnover component (TO-RA) as the default turnover component** (user decision, 2026-09-29, made after the first version of this calibration compared the two). The original turnover charge is kept as `to_gar_orig` / `total_gar_to_orig`.

**Team level.** The unshrunk possession component runs at about 1.8 times its nominal scale (1.79 ± 0.28) and finishing at about 0.8 (0.81 ± 0.09). The ridge-weight intervals for both exclude 1 (possession 1.53, 1.21–1.82; finishing 0.82, 0.69–0.95). Defense (1.05 unshrunk), turnover (1.18 ± 0.40) and faceoffs (1.32 ± 0.63) are consistent with 1. The shrunk slopes above 1 for finishing (1.87), turnover (2.19) and faceoffs (2.07) sit against unshrunk slopes near 1 or below, which confirms the shrinkage-compression point in §1 of the methodology: independent-player priors shrink team-shared effects, so a slope above 1 on shrunk values does not by itself mean the scale is wrong.

**Decision.** The leave-one-season-out gain from the ridge weights (MSE 468.5 vs 485.2) gives paired z = 1.50, below the pre-registered threshold of 2, so all weights stay at 1. With only 284 team-seasons the honest reading is "suggestive, not established": possession may be under-scaled and finishing over-scaled, but the data did not clear the bar set in advance.

**Player level.** The ridge-weighted total predicts next-season within-team on-ice goal differential worse than the current total (paired z = -7.66; own-rate slope 0.81 vs 0.93). On xG the two are indistinguishable (z = -1.61). CAL-3 therefore fails for the ridge variant, which supports not adopting it; the gate is moot while weights stay at 1.

**Forward component slopes** (next-season goal differential on each season-t component, teammates controlled) are not calibration weights, because they mix scale error with year-to-year persistence and with how each component is observed. Possession (1.99) and finishing (1.58) are above 1, defense (0.50) is below, faceoffs are near zero for goals (0.02 ± 0.24), and **turnover cost is still negative (-0.66 ± 0.35 goals, -1.04 ± 0.24 xG)**, though less so than with the original charge (-0.99 and -1.19 in the first run). Players charged more turnover cost in one season tend to do better the next, once teammates are controlled. Likely contributors, not tested here: turnovers are recorded only for players who touch the puck, and the charge is partly a by-product of playing style that also drives shot volume. The role adjustment removes the involvement-driven part of the level but evidently not all of this relationship.

**Post-hoc variant comparison** (exploratory, `src/calibration/variants.py`; not pre-registered, and run on the same data that raised the question; z is against the current total with TO-RA, positive favors the variant):

| total | goals z | xG z |
|---|---|---|
| with the original (not role-adjusted) turnover | -8.90 | -7.39 |
| without turnover cost | +5.71 | +3.90 |
| without faceoffs | +2.15 | +0.68 |
| without penalties | -5.69 | -3.94 |
| without turnover and penalties | +0.61 | +0.37 |
| without turnover, faceoffs and penalties | +2.83 | +1.35 |

TO-RA predicts next-season results clearly better than the original charge, which is the basis for making it the default. Dropping the turnover component altogether still predicts better than including even TO-RA, so turnover cost remains the weakest component of the total for this purpose. Penalties help. Faceoffs are marginal. No component has been removed: the pre-registered structure stays, and the turnover question is deferred to the whole-model validation.

**Limitations.** 284 team-seasons; primary-team assignment misplaces the share of value of players traded mid-season; DPM is calibrated only on its shrunk values; team residual variance includes goaltending and coaching, which no component measures; and the forward check uses a single next-season horizon.

