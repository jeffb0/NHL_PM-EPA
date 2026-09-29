# Integrated player value — goals and wins above replacement (5v5)

Components: possession (PM-EPA, shrunk), defense (DPM valuation rate, refit with a replacement group), finishing (FIN), turnover cost (TO), faceoffs (FO) and penalties (PEN). Spec: [finishing methodology §8-§9](../../docs/finishing/methodology.md) and [components methodology](../../docs/components/methodology.md). All values are 5v5, in goals unless labeled wins; intervals combine the component standard deviations assuming independence.

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
| 2017 | 0.0504 | -0.0199 | 0.0181 | -0.0451 |
| 2018 | 0.0465 | -0.0174 | -0.0098 | -0.0502 |
| 2019 | 0.0476 | -0.0235 | -0.0494 | -0.0448 |
| 2020 | 0.0544 | -0.0147 | -0.0387 | -0.0927 |
| 2021 | 0.0674 | 0.0024 | 0.0287 | -0.0436 |
| 2022 | 0.0812 | 0.0069 | -0.0024 | -0.0346 |
| 2023 | 0.0638 | 0.0027 | 0.0003 | -0.0111 |
| 2024 | 0.0370 | -0.0043 | -0.0474 | -0.0217 |
| 2025 | 0.0577 | 0.0085 | 0.0507 | -0.0767 |

The DPM rates come from the refit with unpenalized fringe-count covariates, so they are not biased toward zero by ridge shrinkage. Finishing replacement is the fringe pool's mean shooter shift from the fitted model.

**The DPM replacement level is pooled over all seasons (the default).** The per-season estimate is noisy: across seasons it has an SD of 0.035 (F) and 0.025 (D) xGA/60, i.e. about 0.6 / 0.4 goals per 1,000 minutes of common shift for every player of that position, and it flips sign for forwards in some seasons. The default (`def_gar`, `total_gar`) therefore uses the pooled effect (F -0.006, D -0.047 per 60); the pre-registered per-season version is kept as `def_gar_perseason` / `total_gar_perseason`. The table above shows the per-season rates for reference only.

## Scale checks

SD of goals above replacement among qualified skaters (≥ 500 5v5 min): pm_gar 0.37, def_gar 1.88, fin_gar 1.48, to_gar 0.39, fo_gar 0.52, pen_gar 0.49, total_gar 2.44.

Correlation between components (qualified): pm_gar~def_gar -0.02, pm_gar~fin_gar 0.39, pm_gar~to_gar -0.03, pm_gar~fo_gar 0.10, pm_gar~pen_gar 0.25, def_gar~fin_gar -0.28, def_gar~to_gar 0.27, def_gar~fo_gar 0.03, def_gar~pen_gar 0.01, fin_gar~to_gar -0.47, fin_gar~fo_gar 0.15, fin_gar~pen_gar 0.25, to_gar~fo_gar -0.13, to_gar~pen_gar -0.12, fo_gar~pen_gar -0.02.

### Team reconciliation

284 team-seasons: 5v5 goal differential regressed on the sum of player goals above replacement (by primary team). A slope near 1 means the components are on the goal scale. **Slopes are reported, not used to rescale.**

| regressor | slope | r |
|---|---|---|
| total_gar | 1.330 | 0.657 |
| total_gar_perseason | 1.465 | 0.688 |
| pm_gar | 3.968 | 0.457 |
| def_gar | 1.745 | 0.515 |
| fin_gar | 2.471 | 0.468 |
| to_gar | 2.691 | 0.222 |
| fo_gar | 3.195 | 0.192 |

Joint regressions (slope ± SE), 5v5 components only (penalties are realized on special teams and are checked against special-teams goal differential below):

| regression | R² | pm | def | fin | to | fo |
|---|---|---|---|---|---|---|
| three components (PM, DPM, FIN) | 0.448 | 2.04 ± 0.43 | 1.30 ± 0.16 | 1.61 ± 0.25 | — | — |
| five 5v5 components, shrunk | 0.464 | 1.53 ± 0.47 | 1.14 ± 0.17 | 1.80 ± 0.26 | 1.51 ± 0.66 | 1.99 ± 0.80 |
| five 5v5 components, possession and turnover unshrunk | 0.479 | 1.16 ± 0.31 | 1.11 ± 0.16 | 1.79 ± 0.25 | 1.18 ± 0.44 | 1.88 ± 0.79 |
| five 5v5 components, role-adjusted turnover (TO-RA) instead of TO | 0.473 | 1.20 ± 0.48 | 1.07 ± 0.17 | 1.87 ± 0.26 | 2.19 ± 0.68 | 2.07 ± 0.78 |

SD of team 5v5 goal differential 30.9; SD of summed player value 15.3.

Penalty check against special-teams goal differential (non-5v5 goals): slope 1.35 ± 0.51, R² 0.025 (special-teams GD SD 15.9).

## Sensitivity to the regular/fringe cutoff

| cutoff | corr with baseline total | rank corr | mean shift (goals per 1000 min) |
|---|---|---|---|
| 11F_5D | 0.997 | 0.996 | -0.508 |
| 13F_7D | 0.999 | 0.999 | 0.188 |

## Total value, 2025-26 (≥ 500 5v5 min)

**Top 15**

| player_name | team | position | toi_5v5 | pm_gar | def_gar | fin_gar | to_gar | fo_gar | pen_gar | total_gar | total_gar_lo | total_gar_hi | total_war |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Moritz Seider | DET | D | 1564.53 | 0.07 | 9.87 | 0.50 | 0.07 | 0.00 | 0.57 | 11.08 | 6.64 | 15.53 | 1.88 |
| Jason Robertson | DAL | L | 1214.32 | 0.81 | 1.43 | 9.28 | -0.98 | -0.00 | 0.52 | 11.07 | 5.91 | 16.23 | 1.88 |
| Adam Pelech | NYI | D | 1389.98 | 0.39 | 10.12 | -0.13 | 0.55 | 0.00 | -0.18 | 10.74 | 7.46 | 14.03 | 1.82 |
| Ryan O'Reilly | NSH | C | 1140.72 | 0.95 | 3.34 | 3.10 | 0.07 | 1.79 | 0.46 | 9.72 | 5.63 | 13.80 | 1.65 |
| Anze Kopitar | LAK | C | 947.27 | 0.54 | 3.63 | 3.03 | -0.11 | 2.00 | 0.25 | 9.33 | 6.23 | 12.44 | 1.58 |
| Artem Zub | OTT | D | 1343.18 | 0.38 | 8.65 | 0.46 | 0.29 | 0.00 | -0.56 | 9.21 | 5.83 | 12.60 | 1.56 |
| Jordan Kyrou | STL | R | 911.22 | 0.59 | 3.75 | 4.19 | -0.18 | -0.02 | 0.60 | 8.93 | 5.36 | 12.50 | 1.51 |
| Jonas Brodin | MIN | D | 1078.12 | 0.07 | 7.22 | 0.32 | 0.64 | 0.00 | 0.56 | 8.81 | 6.04 | 11.58 | 1.49 |
| Jack Eichel | VGK | C | 1073.72 | 0.14 | 4.28 | 4.33 | -1.12 | 0.26 | 0.79 | 8.67 | 4.66 | 12.68 | 1.47 |
| Cale Makar | COL | D | 1289.12 | 1.79 | 3.38 | 2.28 | 0.32 | 0.00 | 0.85 | 8.62 | 5.00 | 12.23 | 1.46 |
| Nick Suzuki | MTL | C | 1195.08 | 0.43 | 3.26 | 3.95 | -0.12 | 0.93 | 0.09 | 8.54 | 4.65 | 12.42 | 1.45 |
| Esa Lindell | DAL | D | 1417.27 | 0.36 | 7.78 | 0.35 | -0.43 | 0.00 | 0.44 | 8.50 | 5.33 | 11.67 | 1.44 |
| Kirill Kaprizov | MIN | L | 1291.72 | 0.84 | 1.42 | 6.48 | -0.93 | -0.00 | 0.61 | 8.42 | 3.69 | 13.16 | 1.43 |
| Jared Spurgeon | MIN | D | 1229.12 | 0.39 | 5.99 | 0.75 | 0.53 | 0.00 | 0.66 | 8.32 | 5.63 | 11.00 | 1.41 |
| Colton Parayko | STL | D | 1412.45 | 0.40 | 5.81 | 0.53 | 0.07 | 0.00 | 1.49 | 8.31 | 4.96 | 11.65 | 1.41 |

**Bottom 10**

| player_name | team | position | toi_5v5 | pm_gar | def_gar | fin_gar | to_gar | fo_gar | pen_gar | total_gar | total_gar_lo | total_gar_hi | total_war |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ben Chiarot | DET | D | 1462.78 | -0.03 | -3.06 | 0.49 | 0.31 | 0.00 | -0.31 | -2.60 | -7.06 | 1.85 | -0.44 |
| Logan Mailloux | STL | D | 989.28 | 0.12 | -2.80 | 0.39 | -0.10 | 0.00 | -0.24 | -2.63 | -5.65 | 0.39 | -0.45 |
| Andrei Svechnikov | CAR | R | 1024.47 | 0.61 | -3.69 | 1.79 | -0.71 | -0.02 | -0.68 | -2.71 | -6.89 | 1.48 | -0.46 |
| Josh Anderson | MTL | R | 794.73 | 0.41 | -3.77 | 0.70 | -0.18 | 0.02 | -0.06 | -2.86 | -5.65 | -0.08 | -0.49 |
| Evander Kane | VAN | L | 988.98 | 0.53 | -4.89 | 1.64 | -0.76 | -0.02 | 0.04 | -3.46 | -7.68 | 0.76 | -0.59 |
| Connor Bedard | CHI | C | 1090.98 | 0.63 | -6.48 | 3.03 | -0.56 | -0.66 | 0.52 | -3.52 | -8.51 | 1.47 | -0.60 |
| Max Domi | TOR | C | 1009.70 | -0.51 | -3.30 | 0.88 | -1.03 | 0.13 | -0.01 | -3.83 | -7.24 | -0.42 | -0.65 |
| Artyom Levshunov | CHI | D | 1025.97 | 0.13 | -3.83 | 0.01 | -0.34 | 0.00 | -0.09 | -4.13 | -7.02 | -1.24 | -0.70 |
| John Klingberg | SJS | D | 921.27 | 0.09 | -6.58 | 0.44 | -0.27 | 0.00 | 0.24 | -6.08 | -9.10 | -3.06 | -1.03 |
| Tony DeAngelo | NYI | D | 1242.42 | 0.37 | -8.70 | 0.49 | -0.22 | 0.00 | -0.05 | -8.10 | -11.68 | -4.53 | -1.37 |

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

**Scale: turnover and faceoffs explain much of PM-EPA's under-scaling, and shrinkage explains more.** The team reconciliation regresses team 5v5 goal differential on the summed player components:

* **Three components only** (with the updated defense column, which now also excludes unobserved starts): PM-EPA 2.04 ± 0.43, DPM 1.30 ± 0.16, FIN 1.61 ± 0.25 (R² 0.45).
* **Adding turnover cost and faceoffs** (all shrunk): PM-EPA 1.53, DPM 1.14, FIN 1.80, TO 1.51 ± 0.66, FO 1.99 ± 0.80 (R² 0.46).
* **With possession and turnover unshrunk:** PM-EPA 1.16 ± 0.31, DPM 1.11, TO 1.18 ± 0.44, FO 1.88, FIN 1.79 (R² 0.48). Possession, defense and turnover are then close to unit scale.
* **Two separate effects are at work.** The missing turnover and faceoff value was real (the possession slope falls from 2.04 to 1.53). The rest is shrinkage compression: an independent-player prior shrinks effects shared across a team, so team sums of shrunk values understate team totals. Slopes above 1 at team level are therefore expected for every shrunk component and do not mean player values are wrong.
* **Finishing (1.8) and faceoffs (1.9) remain the least well scaled at team level.** Whether finishing would fall to unit scale unshrunk was not tested. **No rescaling was applied.** The joint calibration is the next step and should be built with this shrinkage effect in mind.
* The SD of summed player value (15.3 goals) is still about half the SD of team goal differential (30.9), which also contains goaltending, coaching and depth effects that are not player components.
* Penalties are excluded from the 5v5 regression because they are realized on special teams. Against special-teams goal differential their slope is 1.35 ± 0.51.

**Replacement level.**

* **The DPM replacement level is pooled over all seasons (the default).** The per-season estimate was noisy (SD 0.036 for forwards, 0.029 for defensemen, xGA/60, about 0.5–0.6 goals per 1,000 minutes, and positive for forwards in 2021 and 2025). The per-season version remains available as `def_gar_perseason` / `total_gar_perseason`. Pooling assumes a stable gap between regulars and replacement over 2017-2025.
* **Sensitivity to the regular/fringe cutoff** is small for rankings (rank correlation 0.996 for 11F/5D and 0.999 for 13F/7D) but not for levels (−0.5 and +0.2 goals per 1,000 minutes). In the sensitivity runs the turnover, faceoff and penalty replacement stays at the default cutoff.
* **The possession replacement rate is higher than the regulars' mean for forwards** (about +0.05 xG/60 pooled), because depth forwards accrue PM-EPA quickly. Regulars' possession value above replacement is small (mean +0.29 goals), and their turnover value is negative (−0.20) for the reason given in the components report (the replacement pool handles the puck less).

**Face validity of the combined table (2025-26, top of the list).** Seider 11.1, Robertson 11.1, Pelech 10.7, O'Reilly 9.7, Kopitar 9.3, Zub 9.2, Kyrou 8.9, Brodin 8.8 goals above replacement. O'Reilly and Kopitar are there through faceoffs (+1.8 and +2.0) on top of solid defense and finishing. Defensemen are 36% of the top 25 (36% of players). The bottom is DeAngelo (−8.1), Klingberg (−6.1), Levshunov and Domi.

**Other cautions.**

* **The turnover component moves value away from puck-dominant players** (see the components report): it charges observed giveaways but cannot credit unobserved successful plays. It is right at team level and biased in allocation. The role-adjusted variant (`to_ra_gar`, `total_gar_to_ra`) compares each player with others of similar observed puck involvement; it changes totals by up to about ±0.5 goals a season (mean +0.24 for the most-involved quintile, −0.06 for the least), and the team-level regression fit is slightly better with it (R² 0.473 vs 0.464).
* **Defense is the largest and least certain component.** The median 95% interval on total value is ±2.7 goals (D) to ±3.0 (F). Its top values (Seider +9.9, Pelech +10.1 goals from defense alone in 2025-26) are not comparable with typical WAR magnitudes.
* **Total-value intervals combine component SDs assuming independence.** The components are correlated (finishing ~ turnover −0.47, finishing ~ possession +0.39, defense ~ turnover +0.27), so the total intervals are approximate.
* **DPM uncertainty** uses the published bootstrap SE, and the pooled replacement level's own uncertainty is not included.
* **Finishing is 5v5 only**; power-play finishing is in a memo column reserved for special teams.
* **Skaters who changed teams** are assigned to their primary team in the reconciliation.

