# Finishing talent (FIN) — validation report

Model variant **A** (no shot-type effects), offset = xG_f (F1). Spec and pre-registered rules: [methodology](../../docs/finishing/methodology.md). Units: goals above a league-average shooter (or above the replacement pool) on the same shots against the same goalies.

## Gates

| Gate | Result | Pass |
|---|---|---|
| F1 shrunk model beats no-player model on next-season log-loss (pooled 2023-25) | z = 7.2 (Δ 0.00033/shot) | ✅ |
| F2 shrunk model beats raw smoothed G/xG | z = 8.1 (Δ 0.00053/shot) | ✅ |
| F3 next-season calibration slope in [0.7, 1.4] | 0.93 (95% CI 0.75–1.10, 1449 players) | ✅ |
| F4 xG_f cross-fit calibration, 5v5 | 1.0117 | ✅ |
| F5 every goal is in the finishing table or a documented exclusion | 63,239 + 3,732 empty-net + 5 no-coordinates = 66,976; unaccounted 0 | ✅ |
| F6 four-term identity | `tests/test_finishing.py::test_four_term_identity_…` | ✅ (in the unit-test suite) |

## Variance components (logit-scale standard deviations)

| component | train seasons 2017-21, model A | train seasons, model B | production (all seasons) |
|---|---|---|---|
| d_D | 0.1024 | 0.1013 | 0.1260 |
| d_F | 0.0818 | 0.0818 | 0.0908 |
| e | 0.0495 | 0.0494 | 0.0568 |
| u_D | 0.0888 | 0.0805 | 0.1142 |
| u_F | 0.1496 | 0.1477 | 0.1438 |
| v | 0.0717 | 0.0717 | 0.0590 |
| w | — | 0.1429 | — |

u = persistent shooter talent (F/D), d = shooter season deviation, v/e = goalie persistent/season, w = shot-type deviation. Fringe-player finishing effect (logit, relative to regulars): F -0.189 (±0.020), D -0.088 (±0.038).

## Forward prediction (fit on seasons ≤ Y, predict Y+1; variance components fixed from 2017-21)

Held-out log-loss per shot; season × strength intercepts of the held-out season are set to league-level calibration for every model.

| model | validation 2022 | test 2023 | test 2024 | test 2025 | pooled test |
|---|---|---|---|---|---|
| M0 | 0.22164 | 0.21323 | 0.21363 | 0.22076 | 0.21585 |
| Mg | 0.22163 | 0.21326 | 0.21357 | 0.22074 | 0.21583 |
| Mraw | 0.22217 | 0.21375 | 0.21352 | 0.22093 | 0.21605 |
| MshrA | 0.22141 | 0.21302 | 0.21322 | 0.22038 | 0.21552 |
| MshrB | 0.22141 | 0.21304 | 0.21323 | 0.22041 | 0.21553 |

Paired differences on the pooled test folds (positive = first model better):

| comparison | Δ log-loss/shot | SE | z |
|---|---|---|---|
| Mg vs M0 | 0.00002 | 0.00002 | 0.85191 |
| Mraw vs M0 | -0.00020 | 0.00009 | -2.13729 |
| MshrA vs M0 | 0.00033 | 0.00005 | 7.16502 |
| MshrA vs Mraw | 0.00053 | 0.00007 | 8.10340 |
| MshrA vs Mg | 0.00031 | 0.00004 | 7.52509 |
| MshrB vs M0 | 0.00031 | 0.00005 | 6.63800 |
| MshrB vs Mraw | 0.00051 | 0.00007 | 7.74878 |
| MshrB vs MshrA | -0.00002 | 0.00001 | -1.48330 |

D-type: variant A (validation MshrB vs MshrA: Δ 0.00000, z = 0.17; B needs z > 2.0).

Player-level calibration: realized next-season excess goals regressed on predicted (players with ≥ 100 test attempts; slope 1 = well calibrated, < 1 = over-dispersed):

| model | slope | 95% lo | 95% hi | corr | players |
|---|---|---|---|---|---|
| Mraw | 0.543 | 0.439 | 0.657 | 0.320 | 1449 |
| MshrA | 0.927 | 0.748 | 1.097 | 0.346 | 1449 |
| MshrB | 0.920 | 0.749 | 1.092 | 0.343 | 1449 |

## Reliability and stability (single-season independent fits, no pooling across seasons)

| metric | model estimate | raw G−xG per attempt | raw total G−xG |
|---|---|---|---|
| YOY r (≥200 attempts both seasons) | 0.240 | 0.252 | 0.270 |
|   forwards | 0.269 | 0.274 | — |
|   defensemen | 0.092 | 0.021 | — |
| split-half r (odd/even games) | 0.206 | 0.172 | — |
| split-half, Spearman-Brown (full season) | 0.342 | 0.293 | — |

YOY pairs: 929; split-half player-seasons: 1,592.

### Shots needed to stabilize (reliability 0.5)

Mean Bernoulli information per attempt: 0.0585.

| basis | sd (logit) | attempts for r = 0.5 |
|---|---|---|
| single_season_F | 0.2 | 652.5 |
| total_F | 0.2 | 591.3 |
| single_season_D | 0.2 | 574.2 |
| total_D | 0.2 | 591.4 |

Empirical (random half-splits of each player's career attempts, Spearman-Brown):

| attempts | players | reliability |
|---|---|---|
| 100 | 1155 | 0.124 |
| 200 | 935 | 0.198 |
| 400 | 724 | 0.349 |
| 800 | 435 | 0.514 |
| 1600 | 175 | 0.600 |

Empirical attempts for reliability 0.5: 754.

## Full-Bayes (NUTS) check of the Laplace estimates

* **season_2024, fixed sigma** — shooters: corr of means 0.9988, Laplace/NUTS sd ratio median 1.001, NUTS mean inside Laplace 95%: 1.000; goalies: corr 0.9995, sd ratio 1.005. max R-hat 1.004, min n_eff 1315, divergences 0.
* **season_2024, free sigma** — shooters: corr of means 0.9974, Laplace/NUTS sd ratio median 1.002, NUTS mean inside Laplace 95%: 1.000; goalies: corr 0.9994, sd ratio 0.979. max R-hat 1.003, min n_eff 349, divergences 0.
  * sigma posterior (free-sigma): sd_u_D=0.154±0.066, sd_u_F=0.164±0.022, sd_v=0.089±0.021
* **window_2023_2025, fixed sigma** — shooters: corr of means 0.9994, Laplace/NUTS sd ratio median 1.000, NUTS mean inside Laplace 95%: 1.000; goalies: corr 0.9988, sd ratio 0.999. max R-hat 1.002, min n_eff 1793, divergences 0.
* **window_2023_2025, free sigma** — shooters: corr of means 0.9961, Laplace/NUTS sd ratio median 0.999, NUTS mean inside Laplace 95%: 1.000; goalies: corr 0.9972, sd ratio 0.979. max R-hat 1.092, min n_eff 66, divergences 6.
  * sigma posterior (free-sigma): sd_d_D=0.184±0.053, sd_d_F=0.050±0.026, sd_e=0.061±0.016, sd_u_D=0.138±0.055, sd_u_F=0.168±0.012, sd_v=0.032±0.018

## Finishing leaders, 2025-26 (5v5, ≥ 200 attempts; goals above average with 95% interval)

**Top 12**

| player_name | team | position | attempts_5v5 | goals_5v5 | xg_f_5v5 | raw_g_minus_xg_5v5 | fin_gaa_5v5 | fin_gaa_lo | fin_gaa_hi | fin_gar_5v5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Jason Robertson | DAL | L | 286 | 25 | 21.63 | 3.37 | 5.75 | 2.09 | 10.22 | 9.28 |
| Nikita Kucherov | TBL | R | 250 | 25 | 17.26 | 7.74 | 4.35 | 1.30 | 8.36 | 7.18 |
| Cole Caufield | MTL | R | 252 | 33 | 20.42 | 12.58 | 4.30 | 0.48 | 8.54 | 7.63 |
| Tage Thompson | BUF | C | 272 | 31 | 20.06 | 10.94 | 4.26 | 0.95 | 8.69 | 7.54 |
| Dylan Guenther | UTA | R | 274 | 28 | 19.92 | 8.08 | 3.70 | -0.26 | 8.34 | 6.98 |
| Filip Forsberg | NSH | L | 271 | 23 | 16.77 | 6.23 | 3.61 | 0.33 | 6.46 | 6.34 |
| Kirill Kaprizov | MIN | L | 245 | 18 | 17.93 | 0.07 | 3.57 | 0.35 | 7.58 | 6.48 |
| Nathan MacKinnon | COL | C | 296 | 31 | 23.48 | 7.52 | 3.42 | -0.44 | 7.59 | 7.24 |
| David Pastrnak | BOS | R | 250 | 16 | 16.59 | -0.59 | 3.27 | 0.51 | 6.44 | 6.01 |
| Kyle Connor | WPG | L | 279 | 24 | 20.66 | 3.34 | 3.04 | -0.35 | 6.55 | 6.44 |
| Pavel Dorofeyev | VGK | R | 210 | 15 | 15.39 | -0.39 | 2.99 | 0.10 | 6.85 | 5.54 |
| Macklin Celebrini | SJS | C | 243 | 26 | 17.99 | 8.01 | 2.94 | -1.14 | 7.44 | 5.89 |

**Bottom 12**

| player_name | team | position | attempts_5v5 | goals_5v5 | xg_f_5v5 | raw_g_minus_xg_5v5 | fin_gaa_5v5 | fin_gaa_lo | fin_gaa_hi | fin_gar_5v5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Andrei Svechnikov | CAR | R | 221 | 15 | 18.31 | -3.31 | -1.15 | -3.82 | 2.15 | 1.79 |
| Sam Bennett | FLA | C | 211 | 18 | 16.63 | 1.37 | -1.31 | -3.91 | 1.58 | 1.36 |
| Alex Laferriere | LAK | R | 270 | 16 | 17.15 | -1.15 | -1.40 | -4.17 | 1.67 | 1.47 |
| Timo Meier | NJD | R | 326 | 13 | 22.08 | -9.08 | -1.44 | -4.57 | 2.30 | 2.17 |
| Nazem Kadri | CGY | C | 225 | 11 | 16.44 | -5.44 | -1.62 | -3.76 | 1.28 | 1.09 |
| Logan Stankoven | CAR | C | 206 | 15 | 17.61 | -2.61 | -1.69 | -4.52 | 1.95 | 1.14 |
| Anthony Beauvillier | WSH | R | 202 | 14 | 18.00 | -4.00 | -1.73 | -4.26 | 1.68 | 1.17 |
| Beckett Sennecke | ANA | R | 227 | 20 | 21.66 | -1.66 | -1.86 | -6.10 | 2.99 | 1.62 |
| Jackson Blake | CAR | R | 209 | 14 | 20.56 | -6.56 | -1.98 | -5.20 | 1.82 | 1.28 |
| Anders Lee | NYI | L | 239 | 9 | 22.57 | -13.57 | -2.00 | -4.69 | 1.55 | 1.57 |
| Ross Colton | COL | C | 222 | 7 | 16.26 | -9.26 | -2.27 | -4.41 | 0.62 | 0.36 |
| Mikael Backlund | CGY | C | 209 | 11 | 12.59 | -1.59 | -2.46 | -4.13 | -0.39 | -0.38 |

## Interpretation and sanity checks

**What the evidence supports.**

* **Finishing talent is real but small, and raw G − xG is mostly noise.** Persistent shooter talent has an SD of about 0.14 logit (forwards) and 0.11 (defensemen). One SD of talent is roughly 15% higher scoring odds, about one goal per 100 unblocked attempts at the league's 6% average. Season-to-season deviations around that talent are of similar size (SD 0.09 for forwards, 0.13 for defensemen).
* **Shrinkage is not optional.** Using raw smoothed G/xG history to predict next season's goals is *worse* than using no player information at all (held-out log-loss z = −2.1). The shrunk estimates beat both (z = +7.2 vs no-player, +8.1 vs raw), and their calibration slope on realized next-season excess goals is 0.93 (raw: 0.54). Raw estimates over-state talent by nearly a factor of two.
* **About 570–650 unblocked attempts are needed before an estimate is half signal** (analytic 574–653; empirical 754; reliability is 0.20 at 200 attempts and 0.51 at 800). The median shooter-season has 121 attempts, so this is roughly three to five seasons for a regular forward. A single season's finishing should not be over-read: the median 95% interval for a regular is about ±2.9 goals and only 14.9% of intervals exclude zero.
* **Stability is modest.** Year-over-year r among players with ≥ 200 attempts is 0.24 (forwards 0.27; defensemen 0.09 on 183 pairs), essentially the same as raw G − xG (0.25). Shrinkage does not re-rank players who all have similar sample sizes. Its benefit is *scale*: it cuts the SD of pooled three-season totals from 7.2 to 3.6 goals while keeping r = 0.79 with the raw ordering. The split-half correlation is slightly higher for the model (0.21 vs 0.17; Spearman-Brown 0.34 vs 0.29).
* **The goalie adjustment does not help prediction** (held-out gain z = 0.85) and the goalie SD is small (0.06 logit). It is kept because it is cheap and correct in principle, but goalie effects from this pre-shot-xG design should not be read as goaltending value. Shooter × shot-type effects were tested and rejected (validation z = 0.17).

**Sanity check on names.** The model's best 5v5 finishers over the last three seasons are Draisaitl, Robertson, Pastrnak, Kucherov, Panarin, Matthews, Thompson, Geekie, Forsberg, Point, Caufield and Scheifele: recognizable elite shooters. The largest raw over-performances that the model discounts most are Panarin (raw +31.6 → +12.1), Thompson (+30.3 → +11.8), Mantha (+21.6 → +5.8), Tuch, Peterka and Kempe: single hot stretches that regress. The largest raw under-performances it discounts most are Anders Lee, Hertl, DeBrusk, Hayton and Gourde. Matthews goes the other way (raw +4.0 → +11.9): his multi-season history is strong and the model pools it. Note that this is *retrospective smoothing*, which also uses later seasons. For forecasting, use forward fits (as in the evaluation above).

**Where this can mislead.**

* **Finishing is whatever xG_f cannot see.** Screens, traffic, goalie movement and pre-shot passing are not in public data, so net-front specialists whose shots are worth less than location and type imply (Brady Tkachuk, Backlund, Martinook and Zach Hyman are all in the bottom ten) appear as poor finishers. Read the low end as "converts below what location and type predict", not necessarily as poor shooting skill.
* **Interval accuracy was checked against full Bayes.** In the NUTS check on 2024 the Laplace posterior SDs match NUTS both with the variance components fixed and with them free (median ratio 1.00; 5th-95th percentile 0.95-1.08 for shooters), and the NUTS shooter means lie inside every Laplace 95% interval. On the 2023-25 hierarchical window the fixed-variance posterior is essentially identical (shooter SD ratio 1.00, correlation of means 0.9994, no divergences). The free-variance run on that window agrees on shooter means and SDs (median ratio 1.00) but its variance hyperparameters mixed poorly (max R-hat 1.09, minimum effective sample size 66, 6 divergences), because the persistent/season split is weakly identified, so its sigma posteriors are indicative only. Uncertainty in the variance components therefore adds little to the player intervals here. The interval is still only as good as the xG_f offset and the model form.
* **The xG_f offset's calibration slope drifts by season** (0.93 in 2025-26 to 1.09 in 2020-21; pooled 1.005). Season × strength intercepts absorb level drift but not slope drift, so in seasons where xG_f over-rates high-danger shots the players who take them are pushed toward poor finishing. A per-season slope on the offset is the first v1.1 candidate.
* **Defensemen are thin.** Their shot counts are small and their finishing YOY is not distinguishable from noise.
* **The fringe (replacement) group finishes worse than regulars:** −0.19 logit for forwards and −0.09 for defensemen. That difference is what makes regular players' finishing value above replacement larger than above average.


## Data flags

See [methodology §10](../../docs/finishing/methodology.md) and the deviations log (§11).
