# Joint calibration with the goalie (v3) — validation report

Spec and pre-registered rules: [methodology §5c](../../docs/calibration/methodology.md). Eleven components (the ten of the all-strength extension plus the team's goalie value, summed over shots faced by the defending team) against all-strength team goal differential. The v2 extension ([report](validation_report_all.md)) and the 5v5 calibration ([report](validation_report.md)) are unchanged.

**No calibration was justified: all weights are set to 1**: leave-one-season-out team-level error of the calibrated sum vs the all-ones sum, MSE 588.9 vs 609.8, paired z = 1.58 (adoption needs z > 2).

## Gates

| Gate | Result | Pass |
|---|---|---|
| CAL-1 unshrunk joint regression reported | see the table below | ✅ |
| CAL-2 decision rule | z = 1.58 → weights = 1 | ✅ |
| CAL-3 no harm at player level, as pre-registered (raw all-strength on-ice goal differential, next season) | paired z = -6.91 (needs > −2) | ❌ |
| CAL-3 no harm at player level, deployment-adjusted (methodology §7, deviation 4) | paired z = 0.02 (needs > −2) | ✅ |
| CAL-4 adopted weights within [0.5, 2] | pm 1.00, def 1.00, fin 1.00, to 1.00, fo 1.00, pen 1.00, pp 1.00, pk 1.00, 4v4 1.00, 3v3 1.00, gk 1.00 | ✅ |
| CAL-5 consistency: applied columns equal the uncalibrated ones exactly when weights are 1 | max absolute change 0 | ✅ |

## Team level (284 team-seasons, all-strength goal differential, SD 46.0)

| component | shrunk slope ± SE | unshrunk slope ± SE | ridge weight (95% CI) | applied weight |
|---|---|---|---|---|
| possession (PM-EPA) | 1.72 ± 0.69 | 2.04 ± 0.31 | 1.23 (1.12–1.34) | 1.00 |
| defense (DPM) | 1.52 ± 0.24 | 1.55 ± 0.17 | 1.26 (1.12–1.39) | 1.00 |
| finishing, 5v5 (FIN) | 1.93 ± 0.43 | 0.77 ± 0.09 | 0.83 (0.73–0.94) | 1.00 |
| turnover cost (TO-RA) | 2.92 ± 0.93 | 1.28 ± 0.42 | 1.14 (1.04–1.23) | 1.00 |
| faceoffs (FO) | 2.34 ± 1.08 | 0.11 ± 0.67 | 0.97 (0.92–1.02) | 1.00 |
| penalties (PEN) | -0.86 ± 1.03 | 1.00 ± 0.41 | 1.02 (0.93–1.11) | 1.00 |
| power play | 1.24 ± 0.28 | 1.41 ± 0.13 | 1.24 (1.12–1.35) | 1.00 |
| penalty kill | 0.93 ± 0.37 | 0.49 ± 0.23 | 0.95 (0.82–1.07) | 1.00 |
| 4v4 | -9.29 ± 3.02 | 0.14 ± 0.87 | 0.97 (0.94–1.01) | 1.00 |
| 3v3 | -2.86 ± 2.62 | 0.62 ± 0.71 | 1.00 (0.95–1.04) | 1.00 |
| goaltending (GK) | 0.73 ± 0.21 | 0.93 ± 0.07 | 0.93 (0.82–1.03) | 1.00 |

R²: shrunk 0.581, unshrunk 0.787. Ridge λ = 2.51e+04 (leave-one-season-out CV). DPM and the special-teams RAPM parts appear with their published values in both columns (no unshrunk analogue), so their slopes are subject to shrinkage compression.

Cross-validated team-level squared error by λ (the last entry, ∞, is all weights = 1):

| λ | MSE |
|---|---|
| 10 | 639.3 |
| 39.8 | 638.1 |
| 158 | 634.1 |
| 631 | 623.7 |
| 2.51e+03 | 606.0 |
| 1e+04 | 591.1 |
| 3.98e+04 | 590.6 |
| 1.58e+05 | 600.6 |
| 6.31e+05 | 606.9 |
| inf | 609.8 |

## Change from the all-strength extension without the goalie (v2)

Cross-validated team-level error with all weights at 1: 897.4 (v2) → 609.8 (with the goalie); calibrated: 857.1 → 588.9. Unshrunk R²: 0.647 → 0.787. Paired z for adopting weights: 2.17 → 1.58.

| component | v2 ridge weight | v3 ridge weight (95% CI) |
|---|---|---|
| possession (PM-EPA) | 1.42 | 1.23 (1.12–1.34) |
| defense (DPM) | 1.20 | 1.26 (1.12–1.39) |
| finishing, 5v5 (FIN) | 0.78 | 0.83 (0.73–0.94) |
| turnover cost (TO-RA) | 1.22 | 1.14 (1.04–1.23) |
| faceoffs (FO) | 1.02 | 0.97 (0.92–1.02) |
| penalties (PEN) | 0.94 | 1.02 (0.93–1.11) |
| power play | 1.31 | 1.24 (1.12–1.35) |
| penalty kill | 1.10 | 0.95 (0.82–1.07) |
| 4v4 | 0.96 | 0.97 (0.94–1.01) |
| 3v3 | 1.05 | 1.00 (0.95–1.04) |
| goaltending (GK) | — | 0.93 (0.82–1.03) |

## Player level, next season (team-season fixed effects, teammate context)

5,087 player-seasons (≥ 250 modeled minutes in t+1), 253 team-seasons. Predictors per modeled minute in season t; the outcome is on-ice goal (or xG) differential per 60 in t+1 over the 5v5 and special-teams stints. "Calibrated" is the ridge-weighted total (the weights adopted above). The **raw** outcome rewards power-play time and penalizes penalty-kill time whatever the player's skill; the **deployment-adjusted** outcome subtracts each stint's league mean for the side's state and adds the player's minute shares by state as controls.

| outcome | total | own-rate slope ± SE | teammate slope | within-team R² |
|---|---|---|---|---|
| raw goals | uncalibrated | 1.55 ± 0.08 | 8.254 | 0.222 |
| raw goals | calibrated | 1.34 ± 0.07 | 7.820 | 0.211 |
| raw xG | uncalibrated | 1.23 ± 0.06 | 5.844 | 0.197 |
| raw xG | calibrated | 1.06 ± 0.06 | 5.561 | 0.187 |
| adjusted goals | uncalibrated | 0.81 ± 0.06 | 4.953 | 0.298 |
| adjusted goals | calibrated | 0.74 ± 0.05 | 4.597 | 0.298 |
| adjusted xG | uncalibrated | 0.60 ± 0.04 | 3.112 | 0.320 |
| adjusted xG | calibrated | 0.57 ± 0.03 | 2.909 | 0.325 |

Calibrated vs uncalibrated total, paired by team-season (positive favors calibrated): raw goals z = -6.91, raw xG z = -6.25; adjusted goals z = 0.02, adjusted xG z = 3.55.

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

Applied weights: all 1. `total_gar_all_cal` differs from `total_gar_all` by a mean absolute 0.432 goals among qualified skaters; Spearman rank correlation 1.0000. Both columns are published in `player_value_calibrated_all`.

Largest changes, latest season (qualified):

| player_name | position | pm_gar | st_pp_gar | total_gar_all | total_gar_all_cal | d |
|---|---|---|---|---|---|---|
| Auston Matthews | C | 0.97 | 4.07 | 16.83 | 16.83 | 0.00 |
| Matt Boldy | L | 0.92 | 4.36 | 16.91 | 16.91 | 0.00 |
| Leon Draisaitl | C | 0.19 | 10.24 | 19.67 | 19.67 | 0.00 |
| Moritz Seider | D | 0.07 | 1.04 | 13.69 | 13.69 | 0.00 |
| Jordan Kyrou | R | 0.59 | -0.31 | 9.63 | 9.63 | 0.00 |
| Sam Malinski | D | 0.84 | 0.06 | 7.14 | 7.14 | 0.00 |

| player_name | position | pm_gar | st_pp_gar | total_gar_all | total_gar_all_cal | d |
|---|---|---|---|---|---|---|
| Jason Robertson | L | 0.81 | 11.31 | 25.28 | 25.28 | -0.00 |
| Dylan Larkin | C | 0.35 | 8.38 | 18.65 | 18.65 | -0.00 |
| Tim Stützle | C | 0.64 | 3.90 | 12.36 | 12.36 | -0.00 |
| Adam Pelech | D | 0.39 | 0.00 | 12.79 | 12.79 | -0.00 |
| Jonas Brodin | D | 0.07 | 0.02 | 10.20 | 10.20 | -0.00 |
| Mikey Anderson | D | 0.50 | 0.00 | 8.73 | 8.73 | -0.00 |

## Interpretation

**Including the goalie is a large improvement to the team-level identity.** Cross-validated team-level error with all weights at 1 falls from 897 to 610 goals² (−32%), and the unshrunk R² from 0.647 to 0.787. Goaltending was the biggest unmeasured piece of team goal differential; with it in the regression the skater components are better identified.

**The goalie component is on the goal scale.** Its unshrunk team slope is 0.93 ± 0.07 and its ridge weight 0.93 (95% interval 0.82 to 1.03). As stated in §5c that near-1 slope is largely mechanical (the unshrunk sum is part of goal differential), so it is a scale check and not independent evidence. The shrunk goalie slope is 0.73 ± 0.21; unlike the skater components it is not above 1, so there is no sign that shrinkage compression inflates the shrunk goalie values.

**The skater evidence for recalibration weakens once goaltending is accounted for, and the decision changes.** The all-strength extension without the goalie adopted its ridge weights at z = 2.17. With the goalie the same rule gives z = 1.58, so **all weights stay at 1**. The skater weights move toward 1: possession 1.42 → 1.23, defense 1.20 → 1.26, finishing 0.78 → 0.83, turnover 1.22 → 1.14, power play 1.31 → 1.24, penalty kill 1.10 → 0.95, faceoffs, penalties, 4v4 and 3v3 stay at or near 1. Several intervals still exclude 1 (possession 1.12–1.34, defense 1.12–1.39, finishing 0.73–0.94, turnover 1.04–1.23, power play 1.12–1.35), but the held-out gain from using the weights is not large enough to clear the pre-registered bar. So the previous, provisional `total_gar_all_cal` should be treated as superseded: it reflected variance that goaltending explains. The default uncalibrated total (`total_gar_all`) is consistent with this result, and `total_gar_all_cal` in the v3 table equals it exactly.

**Player-level check.** Under the deployment-adjusted version the ridge-weighted total is no worse than the uncalibrated one on goals (z = 0.02) and better on xG (z = +3.55); as pre-registered (raw outcome) it fails (z = −6.91) because of the power-play/penalty-kill deployment confound documented earlier. Since no weights are adopted, this only concerns the reported ridge variant.

**One table for every player.** `data/outputs/player_value_all_in.csv` stacks skaters (all-strength total) and goalies (`goalie_gar`) with usage (minutes or shots faced), goals above replacement, wins and an interval. A caution for comparing them: goalie totals depend heavily on the replacement convention (a replacement goalie is 0.88 goals per 100 shots worse than average, from goalie-seasons under 500 shots), which credits a full-workload goalie with roughly 20 goals before any skill difference; the twelve highest 2025 values in the table are all goalies. The calibration confirms the goalie's *scale* (weight 0.93), not its *level*; the team-level identity's intercept absorbs replacement offsets and cannot check them. Treat goalie-versus-skater rankings as sensitive to that convention.

**Limitations.** As in the earlier calibrations: 284 team-seasons, one horizon, team residuals that include coaching and luck; the goalie unshrunk regressor is mechanical; DPM and special-teams RAPM have no unshrunk analogue.

