# Integrated player value — goals and wins above replacement (5v5)

Components: possession (PM-EPA, shrunk), defense (DPM valuation rate, refit with a replacement group), finishing (FIN). Spec: [methodology §8-§9](../../docs/finishing/methodology.md). All values are 5v5, in goals unless labeled wins; intervals combine the component standard deviations assuming independence.

## Goals per win (derived from the data)

**5.90 goals per win** (95% CI 5.66–6.13; R² 0.91; 284 team-seasons, 2017-18 to 2025-26). OLS through the origin of (wins − GP/2) on goal differential (shootouts excluded).

| season | goals per win |
|---|---|
| 2017 | 4.81 |
| 2018 | 5.62 |
| 2019 | 6.15 |
| 2020 | 5.39 |
| 2021 | 5.85 |
| 2022 | 6.31 |
| 2023 | 6.12 |
| 2024 | 5.79 |
| 2025 | 6.73 |

## Replacement level (top 12 F / 6 D per team are regulars; the rest are the replacement pool)

Per-60 replacement rates by season and position (positive = better than the regulars' mean for defense and possession):

| season | PM-EPA F | PM-EPA D | DPM F | DPM D |
|---|---|---|---|---|
| 2017 | 0.0504 | -0.0199 | -0.0008 | -0.0737 |
| 2018 | 0.0465 | -0.0174 | -0.0222 | -0.0865 |
| 2019 | 0.0476 | -0.0235 | -0.0608 | -0.0670 |
| 2020 | 0.0544 | -0.0147 | -0.0455 | -0.1282 |
| 2021 | 0.0674 | 0.0024 | 0.0255 | -0.0563 |
| 2022 | 0.0812 | 0.0069 | -0.0188 | -0.0556 |
| 2023 | 0.0638 | 0.0027 | -0.0103 | -0.0290 |
| 2024 | 0.0370 | -0.0043 | -0.0651 | -0.0478 |
| 2025 | 0.0577 | 0.0085 | 0.0398 | -0.0907 |

The DPM rates come from the refit with unpenalized fringe-count covariates, so they are not biased toward zero by ridge shrinkage. Finishing replacement is the fringe pool's mean shooter shift from the fitted model.

**Caution: the per-season DPM replacement rate is noisy.** Across seasons it has an SD of 0.036 (F) and 0.029 (D) xGA/60, i.e. about 0.6 / 0.5 goals per 1,000 minutes of common shift for every player of that position, and it flips sign for forwards in some seasons. The pre-registered per-season definition is the default (`def_gar`, `total_gar`). An alternative with the replacement effect pooled over all seasons (F -0.018, D -0.071 per 60) is provided as `def_gar_pooledrep` / `total_gar_pooledrep`. Use the pooled columns for cross-season or forward-versus-defense comparisons until a decision is made.

## Scale checks

SD of goals above replacement among qualified skaters (≥ 500 5v5 min): pm_gar 0.37, def_gar 2.13, fin_gar 1.48, total_gar 2.42.

Correlation between components (qualified): pm_gar~def_gar 0.03, pm_gar~fin_gar 0.39, def_gar~fin_gar -0.23.

### Team reconciliation

284 team-seasons: 5v5 goal differential regressed on the sum of player goals above replacement (by primary team). A slope near 1 means the components are on the goal scale. **Slopes are reported, not used to rescale.**

| regressor | slope | r |
|---|---|---|
| total_gar | 1.404 | 0.698 |
| total_gar_pooledrep | 1.314 | 0.677 |
| pm_gar | 3.968 | 0.457 |
| def_gar | 1.401 | 0.537 |
| fin_gar | 2.471 | 0.468 |

Joint regression on the three components: pm_slope = 2.05, def_slope = 1.17, fin_slope = 1.76, r2 = 0.51. SD of team 5v5 goal differential 30.9; SD of summed player value 15.4.

## Sensitivity to the regular/fringe cutoff

| cutoff | corr with baseline total | rank corr | mean shift (goals per 1000 min) |
|---|---|---|---|
| 11F_5D | 0.990 | 0.987 | -0.520 |
| 13F_7D | 0.990 | 0.987 | 0.230 |

## Total value, 2025-26 (≥ 500 5v5 min)

**Top 15**

| player_name | team | position | toi_5v5 | pm_gar | def_gar | fin_gar | total_gar | total_gar_lo | total_gar_hi | total_war |
|---|---|---|---|---|---|---|---|---|---|---|
| Moritz Seider | DET | D | 1564.53 | 0.07 | 12.75 | 0.50 | 13.32 | 9.07 | 17.58 | 2.26 |
| Adam Pelech | NYI | D | 1389.98 | 0.39 | 12.21 | -0.13 | 12.47 | 9.45 | 15.49 | 2.11 |
| Artem Zub | OTT | D | 1343.18 | 0.38 | 11.03 | 0.46 | 11.86 | 8.70 | 15.03 | 2.01 |
| Jason Robertson | DAL | L | 1214.32 | 0.81 | 1.73 | 9.28 | 11.83 | 6.85 | 16.80 | 2.01 |
| Esa Lindell | DAL | D | 1417.27 | 0.36 | 9.66 | 0.35 | 10.37 | 7.40 | 13.34 | 1.76 |
| Colton Parayko | STL | D | 1412.45 | 0.40 | 8.54 | 0.53 | 9.48 | 6.33 | 12.64 | 1.61 |
| Damon Severson | CBJ | D | 1296.33 | 0.61 | 7.82 | 0.79 | 9.22 | 5.91 | 12.53 | 1.56 |
| Dylan DeMelo | WPG | D | 1485.25 | 0.24 | 8.57 | 0.06 | 8.87 | 5.78 | 11.96 | 1.50 |
| Jonas Brodin | MIN | D | 1078.12 | 0.07 | 8.26 | 0.32 | 8.65 | 6.06 | 11.24 | 1.47 |
| Jack Eichel | VGK | C | 1073.72 | 0.14 | 4.10 | 4.33 | 8.56 | 4.81 | 12.32 | 1.45 |
| Cale Makar | COL | D | 1289.12 | 1.79 | 4.29 | 2.28 | 8.36 | 4.94 | 11.78 | 1.42 |
| Jordan Kyrou | STL | R | 911.22 | 0.59 | 3.54 | 4.19 | 8.32 | 4.90 | 11.75 | 1.41 |
| Connor Murphy | CHI | D | 1144.32 | -0.12 | 7.96 | 0.41 | 8.26 | 5.53 | 10.99 | 1.40 |
| Zach Werenski | CBJ | D | 1539.38 | 1.42 | 4.38 | 2.36 | 8.16 | 3.52 | 12.79 | 1.38 |
| Pavel Dorofeyev | VGK | R | 1090.22 | 0.83 | 1.72 | 5.54 | 8.08 | 4.17 | 11.99 | 1.37 |

**Bottom 10**

| player_name | team | position | toi_5v5 | pm_gar | def_gar | fin_gar | total_gar | total_gar_lo | total_gar_hi | total_war |
|---|---|---|---|---|---|---|---|---|---|---|
| Josh Anderson | MTL | R | 794.73 | 0.41 | -4.13 | 0.70 | -3.01 | -5.60 | -0.42 | -0.51 |
| Mason McTavish | ANA | C | 916.38 | 0.38 | -4.85 | 1.40 | -3.07 | -6.49 | 0.34 | -0.52 |
| Anthony Duclair | NYI | L | 711.00 | -0.00 | -5.10 | 1.63 | -3.47 | -5.96 | -0.99 | -0.59 |
| Evander Kane | VAN | L | 988.98 | 0.53 | -5.66 | 1.64 | -3.50 | -7.53 | 0.53 | -0.59 |
| Artyom Levshunov | CHI | D | 1025.97 | 0.13 | -3.66 | 0.01 | -3.52 | -6.22 | -0.82 | -0.60 |
| Connor Bedard | CHI | C | 1090.98 | 0.63 | -7.19 | 3.03 | -3.53 | -8.31 | 1.25 | -0.60 |
| Max Domi | TOR | C | 1009.70 | -0.51 | -4.06 | 0.88 | -3.69 | -6.90 | -0.49 | -0.63 |
| John Klingberg | SJS | D | 921.27 | 0.09 | -6.40 | 0.44 | -5.87 | -8.74 | -2.99 | -0.99 |
| Chandler Stephenson | SEA | C | 1094.05 | -0.76 | -7.07 | 1.69 | -6.15 | -8.96 | -3.34 | -1.04 |
| Tony DeAngelo | NYI | D | 1242.42 | 0.37 | -7.53 | 0.49 | -6.67 | -10.08 | -3.26 | -1.13 |

## Finishing vs raw G−xG, last three seasons pooled (2023-24 to 2025-26; ≥ 300 5v5 attempts)

`raw_g_minus_xg` = goals − xG_f; `fin_gaa` = model goals above average (shrunk, goalie-adjusted). `model_minus_raw` is what shrinkage and the goalie adjustment removed.

**Best finishers (model)**

| player_name | position | attempts | raw_g_minus_xg | fin_gaa | fin_gar | model_minus_raw |
|---|---|---|---|---|---|---|
| Leon Draisaitl | C | 530 | 12.74 | 14.12 | 21.03 | 1.38 |
| Jason Robertson | L | 700 | 16.47 | 13.39 | 21.46 | -3.08 |
| David Pastrnak | R | 871 | 10.02 | 12.71 | 22.47 | 2.70 |
| Nikita Kucherov | R | 812 | 16.84 | 12.26 | 21.22 | -4.58 |
| Artemi Panarin | L | 742 | 31.61 | 12.10 | 19.90 | -19.52 |
| Auston Matthews | C | 833 | 3.97 | 11.91 | 23.07 | 7.93 |
| Tage Thompson | C | 740 | 30.30 | 11.81 | 20.03 | -18.49 |
| Morgan Geekie | C | 501 | 20.09 | 11.29 | 18.16 | -8.79 |
| Filip Forsberg | L | 904 | 10.75 | 10.22 | 19.59 | -0.53 |
| Brayden Point | C | 494 | 15.97 | 9.67 | 17.14 | -6.30 |
| Cole Caufield | R | 807 | 15.23 | 9.66 | 19.65 | -5.57 |
| Mark Scheifele | C | 521 | 9.58 | 9.42 | 17.28 | -0.16 |

**Worst finishers (model)**

| player_name | position | attempts | raw_g_minus_xg | fin_gaa | fin_gar | model_minus_raw |
|---|---|---|---|---|---|---|
| Brady Tkachuk | L | 804 | -8.09 | -7.95 | 0.89 | 0.14 |
| Mikael Backlund | C | 643 | -5.42 | -7.92 | -1.45 | -2.50 |
| Jordan Martinook | L | 569 | -8.34 | -7.77 | -1.02 | 0.57 |
| Barrett Hayton | C | 392 | -20.14 | -7.01 | -1.05 | 13.13 |
| Cole Sillinger | C | 489 | -12.30 | -6.77 | -1.15 | 5.53 |
| Zach Hyman | L | 697 | -6.69 | -6.48 | 5.79 | 0.20 |
| Drew O'Connor | L | 538 | -13.93 | -6.38 | 1.17 | 7.54 |
| Warren Foegele | L | 662 | -1.87 | -6.08 | 2.60 | -4.20 |
| Cole Smith | R | 314 | -12.24 | -5.45 | -0.92 | 6.79 |
| Zach Benson | L | 390 | -11.76 | -5.41 | 0.45 | 6.35 |
| Ross Colton | C | 575 | -16.69 | -5.29 | 1.92 | 11.40 |
| Adam Lowry | C | 430 | -12.57 | -5.17 | 1.10 | 7.40 |

**Largest raw over-performance that the model discounts most**

| player_name | position | attempts | raw_g_minus_xg | fin_gaa | fin_gar | model_minus_raw |
|---|---|---|---|---|---|---|
| Artemi Panarin | L | 742 | 31.61 | 12.10 | 19.90 | -19.52 |
| Tage Thompson | C | 740 | 30.30 | 11.81 | 20.03 | -18.49 |
| Anthony Mantha | R | 318 | 21.61 | 5.77 | 9.96 | -15.84 |
| Alex Tuch | R | 647 | 17.60 | 3.04 | 10.12 | -14.55 |
| JJ Peterka | R | 662 | 17.56 | 5.72 | 13.15 | -11.85 |
| Adrian Kempe | R | 697 | 17.86 | 6.29 | 13.85 | -11.57 |
| Egor Chinakhov | R | 398 | 13.97 | 3.19 | 7.44 | -10.78 |
| Casey Mittelstadt | C | 341 | 12.66 | 2.75 | 6.86 | -9.90 |
| Jakob Chychrun | D | 698 | 13.59 | 3.70 | 5.30 | -9.89 |
| Jonathan Marchessault | R | 539 | 9.62 | 0.09 | 6.11 | -9.53 |

**Largest raw under-performance that the model discounts most**

| player_name | position | attempts | raw_g_minus_xg | fin_gaa | fin_gar | model_minus_raw |
|---|---|---|---|---|---|---|
| Anders Lee | L | 714 | -18.90 | -4.19 | 5.99 | 14.71 |
| Tomas Hertl | C | 485 | -15.74 | -1.72 | 4.67 | 14.03 |
| Jake DeBrusk | L | 518 | -13.60 | 0.30 | 6.95 | 13.90 |
| Barrett Hayton | C | 392 | -20.14 | -7.01 | -1.05 | 13.13 |
| Yanni Gourde | C | 464 | -13.85 | -1.36 | 4.51 | 12.49 |
| Ross Colton | C | 575 | -16.69 | -5.29 | 1.92 | 11.40 |
| Timo Meier | R | 880 | -12.54 | -1.73 | 7.78 | 10.81 |
| Nino Niederreiter | R | 493 | -13.01 | -2.36 | 4.26 | 10.64 |
| Ryan Nugent-Hopkins | C | 461 | -11.17 | -0.72 | 5.74 | 10.44 |
| Michael Bunting | L | 520 | -11.96 | -1.56 | 4.67 | 10.40 |

## Interpretation, sanity checks and open decisions

**The three components are not yet on equal footing.** They are all in goals, but the team reconciliation shows they do not carry equal weight:

* Regressing team 5v5 goal differential on the components jointly gives slopes of **2.05 (possession), 1.17 (defense) and 1.76 (finishing)**, R² 0.51. The SD of summed player value (15.4 goals) is about half the SD of team goal differential (30.9). Team goal differential also contains goaltending, coaching and depth effects that are not player components, so a slope of exactly 1 is not expected. But slopes near 2 for possession and finishing say those two are being under-weighted relative to defense.
* Possession is the clearest case. PM-EPA credits only individually observed puck actions plus a 30% pooled share, excludes shot quality and start values, and is then shrunk, so its SD is only 0.37 goals per qualified player-season against 2.13 for defense and 1.48 for finishing. This is a scale property of the metric, not a bug in the integration. **No rescaling was applied.** The recommended next step is the joint calibration (regress team goal differential on the components, with shrinkage) as part of the combining framework.
* The reconciliation intercepts are large (about −57 goals) because the sum is measured above replacement, not above average.

**Replacement level.**

* The pre-registered per-season DPM replacement rate is noisy: its SD across seasons is 0.036 (F) and 0.029 (D) xGA/60, i.e. 0.5–0.6 goals per 1,000 minutes shifted for every skater of a position, and it turns positive for forwards in 2021 and 2025 (fringe forwards rated *better* than regulars, which is not credible). This visibly distorts single seasons: defensemen are **72% of the 2025-26 top 25** but 29% of the top 100 across all seasons (they are 36% of players).
* Pooling the replacement effect across seasons (`def_gar_pooledrep`, `total_gar_pooledrep`) reduces that outlier (2025-26: 44%) and moves the reconciliation slope from 1.40 to 1.31. **I recommend the pooled columns as the default.** The pre-registered columns remain the published default until you decide, and the change would be logged in the deviations section.
* Sensitivity to the regular/fringe cutoff is small for rankings (rank correlation 0.987 for both 11F/5D and 13F/7D) but not for levels: −0.5 and +0.2 goals per 1,000 minutes.
* The possession replacement rate is *higher* than the regulars' mean for forwards (about +0.05 xG/60 pooled). Depth forwards accrue PM-EPA quickly, because PM-EPA rewards forechecking, retrievals and takeaways. Regulars' possession value above replacement is therefore small (mean +0.29 goals; fringe pool mean 0.0).

**Other cautions.**

* Defense is the largest and least certain component. The median 95% interval on total value is ±2.5 goals (D) to ±2.8 (F). The top of the defense scale (Seider +12.8, Pelech +12.2 goals in 2025-26) reflects a chained prior plus a large DPM spread, and these are not comparable with typical WAR magnitudes.
* Total value intervals combine the component SDs assuming independence. The components are correlated (finishing ~ possession +0.39, finishing ~ defense −0.23), so the total intervals are approximate.
* DPM uncertainty uses the published bootstrap SE (of the un-refit headline). The refit changes point estimates slightly.
* Finishing is 5v5 only. Power-play and other-strength finishing is in `fin_gar_other_strengths_memo`, reserved for the special-teams component (its intervals are not computed).
* Skaters who changed teams are assigned to their primary team in the reconciliation.

