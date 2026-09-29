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

**The DPM replacement level is pooled over all seasons (the default).** The per-season estimate is noisy: across seasons it has an SD of 0.036 (F) and 0.029 (D) xGA/60, i.e. about 0.6 / 0.5 goals per 1,000 minutes of common shift for every player of that position, and it flips sign for forwards in some seasons. The default (`def_gar`, `total_gar`) therefore uses the pooled effect (F -0.018, D -0.071 per 60); the pre-registered per-season version is kept as `def_gar_perseason` / `total_gar_perseason`. The table above shows the per-season rates for reference only.

## Scale checks

SD of goals above replacement among qualified skaters (≥ 500 5v5 min): pm_gar 0.37, def_gar 2.10, fin_gar 1.48, total_gar 2.41.

Correlation between components (qualified): pm_gar~def_gar 0.05, pm_gar~fin_gar 0.39, def_gar~fin_gar -0.23.

### Team reconciliation

284 team-seasons: 5v5 goal differential regressed on the sum of player goals above replacement (by primary team). A slope near 1 means the components are on the goal scale. **Slopes are reported, not used to rescale.**

| regressor | slope | r |
|---|---|---|
| total_gar | 1.314 | 0.677 |
| total_gar_perseason | 1.404 | 0.698 |
| pm_gar | 3.968 | 0.457 |
| def_gar | 1.602 | 0.576 |
| fin_gar | 2.471 | 0.468 |

Joint regression on the three components: pm_slope = 1.68, def_slope = 1.17, fin_slope = 1.46, r2 = 0.46. SD of team 5v5 goal differential 30.9; SD of summed player value 15.9.

## Sensitivity to the regular/fringe cutoff

| cutoff | corr with baseline total | rank corr | mean shift (goals per 1000 min) |
|---|---|---|---|
| 11F_5D | 0.996 | 0.995 | -0.535 |
| 13F_7D | 0.999 | 0.998 | 0.233 |

## Total value, 2025-26 (≥ 500 5v5 min)

**Top 15**

| player_name | team | position | toi_5v5 | pm_gar | def_gar | fin_gar | total_gar | total_gar_lo | total_gar_hi | total_war |
|---|---|---|---|---|---|---|---|---|---|---|
| Jason Robertson | DAL | L | 1214.32 | 0.81 | 2.89 | 9.28 | 12.99 | 8.01 | 17.96 | 2.20 |
| Moritz Seider | DET | D | 1564.53 | 0.07 | 12.23 | 0.50 | 12.80 | 8.54 | 17.05 | 2.17 |
| Adam Pelech | NYI | D | 1389.98 | 0.39 | 11.75 | -0.13 | 12.00 | 8.98 | 15.02 | 2.04 |
| Artem Zub | OTT | D | 1343.18 | 0.38 | 10.58 | 0.46 | 11.41 | 8.25 | 14.58 | 1.94 |
| Esa Lindell | DAL | D | 1417.27 | 0.36 | 9.18 | 0.35 | 9.90 | 6.93 | 12.87 | 1.68 |
| Jack Eichel | VGK | C | 1073.72 | 0.14 | 5.12 | 4.33 | 9.59 | 5.83 | 13.34 | 1.63 |
| Jordan Kyrou | STL | R | 911.22 | 0.59 | 4.41 | 4.19 | 9.19 | 5.77 | 12.62 | 1.56 |
| Pavel Dorofeyev | VGK | R | 1090.22 | 0.83 | 2.76 | 5.54 | 9.12 | 5.21 | 13.03 | 1.55 |
| Kirill Kaprizov | MIN | L | 1291.72 | 0.84 | 1.77 | 6.48 | 9.09 | 4.53 | 13.64 | 1.54 |
| Colton Parayko | STL | D | 1412.45 | 0.40 | 8.07 | 0.53 | 9.01 | 5.85 | 12.16 | 1.53 |
| Damon Severson | CBJ | D | 1296.33 | 0.61 | 7.39 | 0.79 | 8.79 | 5.48 | 12.09 | 1.49 |
| Nick Suzuki | MTL | C | 1195.08 | 0.43 | 4.33 | 3.95 | 8.72 | 5.04 | 12.39 | 1.48 |
| Dylan DeMelo | WPG | D | 1485.25 | 0.24 | 8.07 | 0.06 | 8.37 | 5.28 | 11.47 | 1.42 |
| Jonas Brodin | MIN | D | 1078.12 | 0.07 | 7.90 | 0.32 | 8.29 | 5.70 | 10.88 | 1.41 |
| Sam Reinhart | FLA | C | 872.38 | 0.21 | 5.19 | 2.83 | 8.23 | 5.40 | 11.05 | 1.40 |

**Bottom 10**

| player_name | team | position | toi_5v5 | pm_gar | def_gar | fin_gar | total_gar | total_gar_lo | total_gar_hi | total_war |
|---|---|---|---|---|---|---|---|---|---|---|
| Morgan Rielly | TOR | D | 1310.98 | 0.31 | -3.53 | 0.69 | -2.53 | -5.46 | 0.40 | -0.43 |
| Evander Kane | VAN | L | 988.98 | 0.53 | -4.72 | 1.64 | -2.55 | -6.58 | 1.48 | -0.43 |
| Tanner Pearson | WPG | L | 528.18 | -0.10 | -2.39 | -0.20 | -2.69 | -4.36 | -1.02 | -0.46 |
| Max Domi | TOR | C | 1009.70 | -0.51 | -3.10 | 0.88 | -2.73 | -5.93 | 0.48 | -0.46 |
| Anthony Duclair | NYI | L | 711.00 | -0.00 | -4.42 | 1.63 | -2.79 | -5.28 | -0.31 | -0.47 |
| Ben Chiarot | DET | D | 1462.78 | -0.03 | -3.54 | 0.49 | -3.09 | -7.36 | 1.19 | -0.52 |
| Artyom Levshunov | CHI | D | 1025.97 | 0.13 | -4.00 | 0.01 | -3.87 | -6.57 | -1.17 | -0.66 |
| Chandler Stephenson | SEA | C | 1094.05 | -0.76 | -6.03 | 1.69 | -5.10 | -7.91 | -2.29 | -0.87 |
| John Klingberg | SJS | D | 921.27 | 0.09 | -6.71 | 0.44 | -6.18 | -9.05 | -3.30 | -1.05 |
| Tony DeAngelo | NYI | D | 1242.42 | 0.37 | -7.95 | 0.49 | -7.09 | -10.50 | -3.67 | -1.20 |

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

* Regressing team 5v5 goal differential on the components jointly gives slopes of **1.68 (possession), 1.18 (defense) and 1.46 (finishing)**, R² 0.46. The SD of summed player value (15.9 goals) is about half the SD of team goal differential (30.9). Team goal differential also contains goaltending, coaching and depth effects that are not player components, so a slope of exactly 1 is not expected. But slopes well above 1 for possession and finishing say those two are being under-weighted relative to defense.
* Possession is the clearest case. PM-EPA credits only individually observed puck actions plus a 30% pooled share, excludes shot quality and start values, and is then shrunk, so its SD is only 0.37 goals per qualified player-season against 2.10 for defense and 1.48 for finishing. This is a scale property of the metric, not a bug in the integration. **No rescaling was applied.** The recommended next step is the joint calibration (regress team goal differential on the components, with shrinkage) as part of the combining framework.
* The reconciliation intercepts are large (about −54 goals) because the sum is measured above replacement, not above average.

**Replacement level.**

* **The DPM replacement level is pooled over all seasons (the default, chosen 2026-09-29).** The pre-registered per-season estimate is noisy: its SD across seasons is 0.036 (F) and 0.029 (D) xGA/60, i.e. 0.5–0.6 goals per 1,000 minutes shifted for every skater of a position, and it turns positive for forwards in 2021 and 2025 (fringe forwards rated *better* than regulars, which is not credible). Under the per-season version defensemen were 72% of the 2025-26 top 25. With the pooled level they are 44%, and 35% of the top 100 across all seasons (they are 36% of players). The per-season version remains available as `def_gar_perseason` / `total_gar_perseason`.
* Pooling assumes the true replacement gap is stable across 2017-2025. If it drifted, pooled values carry a small bias in the early and late seasons.
* Sensitivity to the regular/fringe cutoff is small for rankings (rank correlation 0.995 for 11F/5D and 0.998 for 13F/7D) but not for levels: −0.5 and +0.2 goals per 1,000 minutes.
* The possession replacement rate is *higher* than the regulars' mean for forwards (about +0.05 xG/60 pooled). Depth forwards accrue PM-EPA quickly, because PM-EPA rewards forechecking, retrievals and takeaways. Regulars' possession value above replacement is therefore small (mean +0.29 goals; fringe pool mean 0.0).

**Other cautions.**

* Defense is the largest and least certain component. The median 95% interval on total value is ±2.5 goals (D) to ±2.8 (F). The top of the defense scale (Seider +12.2, Pelech +11.8 goals in 2025-26) reflects a chained prior plus a large DPM spread, and these are not comparable with typical WAR magnitudes.
* Total value intervals combine the component SDs assuming independence. The components are correlated (finishing ~ possession +0.39, finishing ~ defense −0.23), so the total intervals are approximate.
* DPM uncertainty uses the published bootstrap SE (of the un-refit headline). The refit changes point estimates slightly, and the pooled replacement level's own uncertainty is not included.
* Finishing is 5v5 only. Power-play and other-strength finishing is in `fin_gar_other_strengths_memo`, reserved for the special-teams component (its intervals are not computed).
* Skaters who changed teams are assigned to their primary team in the reconciliation.

