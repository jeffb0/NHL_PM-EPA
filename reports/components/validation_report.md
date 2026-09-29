# Turnover cost, faceoffs and penalties (5v5) — validation report

Spec and pre-registered gates: [methodology](../../docs/components/methodology.md). Values are goals (xG scale) above the fringe replacement pool, empirical-Bayes shrunk where the data are counts or rates.

## Gates

| Gate | Result | Pass |
|---|---|---|
| TO-1 turnover accounting: −ΣT2 ÷ (DPM-excluded turnover-born + unobserved starts) | 0.89 (against turnover-born only: 1.77) | ✅ |
| TO-2 team-level slope of the turnover component (joint regression, shrunk) | 1.51 ± 0.66 (unshrunk: 1.18) | ❌ |
| TO-3 season-centred YOY of shrunk TO per 60 (≥ 500 min) | F 0.60 · D 0.48 | ✅ |
| TO-4 shrunk beats raw at predicting next-season TO per 60 | z = 14.5 | ✅ |
| FO-1 shrunk taker skill beats no-taker on next-season draws | z = 24.5 | ✅ |
| FO-2 shrunk beats raw win-rate log-odds | z = 12.4 | ✅ |
| FO-3 calibration slope of realized next-season excess wins | 0.84 (0.77–0.91) | ✅ |
| FO-4 draw swing positive at every dot | min dot value 0.0094 | ✅ |
| PEN-1 cost per minor above 0 | 0.175 xG (0.173–0.177) | ✅ |
| PEN-2 shrunk beats raw at predicting next-season counts (Poisson deviance) | drawn z = 7.7 · taken z = 7.2 | ✅ |
| PEN-3 season-centred YOY (shrunk), drawn and taken per 60 | drawn 0.48 · taken 0.52 | ✅ |
| RA-1 role adjustment removes the involvement gradient: absolute r (residual, involvement) | F 0.02 · D 0.01 | ✅ |
| RA-2 season-centred YOY of shrunk TO-RA per 60 | F 0.55 · D 0.45 | ✅ |
| RA-3 shrunk TO-RA beats raw at predicting next-season residual | z = 15.7 | ✅ |

Gate thresholds use z > 2.0. TO-2 is evaluated after integration (`python -m src.valuation.run`).

## Turnover cost

Accounting by slice (5v5, usable games): T2 = -17,117 xG charged; DPM-excluded turnover-born starts 9,688; unobserved starts 9,524. T2 also charges giveaways and losses whose opponent start was not observed after the turnover, so the valuation defense column excludes `start_unobserved` as well.

### Arena (scorer) effect on the per-60 turnover charge, visitors only

League mean charge is about −0.20 per 60. Excess arena SD = spread of arena means beyond sampling noise:

| season | excess arena SD | noise SD |
|---|---|---|
| 2017 | 0.0382 | 0.0058 |
| 2018 | 0.0452 | 0.0060 |
| 2019 | 0.0348 | 0.0066 |
| 2020 | 0.0337 | 0.0067 |
| 2021 | 0.0388 | 0.0055 |
| 2022 | 0.0296 | 0.0053 |
| 2023 | 0.0303 | 0.0051 |
| 2024 | 0.0081 | 0.0062 |
| 2025 | 0.0093 | 0.0056 |

Arena effects of 15–20% of the mean charge in 2017–2023 disappear in 2024-25, when the NHL's recording changed. They are subtracted (shrunk) from every game at that arena. Stayer/changer stability, season-centred YOY r:

| version | stayers | changers | gap |
|---|---|---|---|
| adjusted raw | 0.601 | 0.368 | 0.233 |
| unadjusted raw | 0.642 | 0.348 | 0.294 |
| adjusted shrunk | 0.607 | 0.387 | 0.220 |
| unadjusted shrunk | 0.651 | 0.360 | 0.291 |

Direct (giveaway) share of the absolute charge: 0.52. SD of goals above replacement among qualified skaters: 0.39. The remaining stayer/changer gap is consistent with real style and system persistence; it is not fully arena bias.

### Role-adjusted turnover variant (TO-RA)

Compares each player's arena-adjusted charge with players of similar *observed successful* puck involvement (carry, move_start, move_end, shot_attempt credit rows per 60) in the same season and position, then shrinks and measures above the fringe residual. A variant beside TO (`to_ra_gar`), not a replacement. Involvement explains only 11% (forwards) and 6% (defensemen) of the variance in charge, so rankings move little (Spearman with TO: F 0.93, D 0.96; SD of goals 0.39 → 0.35).

Correlation of the raw per-60 charge with involvement is F -0.29, D -0.24 (more involvement, larger charge: the puck-dominance bias) and of the residual F 0.02, D 0.01. YOY shrunk: TO-RA F 0.55 · D 0.45 vs TO F 0.60 · D 0.48.

Mean goals above replacement by involvement quintile (Q5 = most involved):

| involvement quintile | TO | TO-RA | player-seasons |
|---|---|---|---|
| Q1 | -0.051 | -0.106 | 1013 |
| Q2 | -0.105 | -0.090 | 1004 |
| Q3 | -0.178 | -0.109 | 1002 |
| Q4 | -0.253 | -0.123 | 1004 |
| Q5 | -0.350 | -0.111 | 1011 |

Most charged skaters, pooled over qualifying seasons (≥ 3,000 5v5 minutes; goals above replacement):

| player | inv60 | to_gar | to_ra_gar |
|---|---|---|---|
| Leon Draisaitl | 22.4 | -11.7 | -10.2 |
| David Pastrnak | 36.1 | -11.5 | -7.8 |
| Nikita Kucherov | 31.7 | -9.7 | -6.6 |
| Mathew Barzal | 23.2 | -9.4 | -7.9 |
| Nathan MacKinnon | 34.0 | -8.8 | -5.0 |
| Mitch Marner | 27.4 | -8.8 | -6.5 |
| Evgeni Malkin | 24.0 | -8.7 | -7.3 |
| Artemi Panarin | 26.4 | -8.6 | -6.4 |
| Johnny Gaudreau | 24.8 | -8.6 | -7.0 |
| Dylan Larkin | 28.1 | -8.3 | -5.9 |

Most charged under TO-RA:

| player | inv60 | to_gar | to_ra_gar |
|---|---|---|---|
| Leon Draisaitl | 22.4 | -11.7 | -10.2 |
| Mathew Barzal | 23.2 | -9.4 | -7.9 |
| David Pastrnak | 36.1 | -11.5 | -7.8 |
| Evgeni Malkin | 24.0 | -8.7 | -7.3 |
| Johnny Gaudreau | 24.8 | -8.6 | -7.0 |
| Jonathan Huberdeau | 21.3 | -7.8 | -6.7 |
| J.T. Miller | 20.0 | -7.3 | -6.6 |
| Nikita Kucherov | 31.7 | -9.7 | -6.6 |

Least charged under TO-RA:

| player | inv60 | to_gar | to_ra_gar |
|---|---|---|---|
| Brett Pesce | 20.9 | 6.6 | 7.4 |
| Jared Spurgeon | 17.7 | 4.9 | 5.0 |
| Jaccob Slavin | 22.4 | 3.6 | 4.8 |
| Jonas Brodin | 16.3 | 4.9 | 4.7 |
| Adam Larsson | 17.8 | 4.1 | 4.4 |
| Gustav Forsling | 24.3 | 2.5 | 3.7 |

## Faceoffs

490,394 five-on-five draws with both takers known; home taker wins 0.509. Mean swing S (value of winning vs losing a draw) = 0.0245 xG. Start value V by dot (winner's frame): (+0,+0) 0.0111, (-20,-22) 0.0107, (-20,+22) 0.0108, (+20,-22) 0.0109, (+20,+22) 0.0110, (-69,-22) 0.0094, (-69,+22) 0.0095, (+69,-22) 0.0165, (+69,+22) 0.0164.

Taker skill SD (logit): centers 0.24 persistent + 0.07 season; others 0.29 + 0.05. Side (handedness × dot side) coefficient 0.116; fringe-taker effect -0.051 logit.

Forward prediction (fit ≤ Y, predict Y+1; variance components from 2017-21):

| model | test 2023 | test 2024 | test 2025 | pooled |
|---|---|---|---|---|
| M0 | 0.68978 | 0.68917 | 0.68895 | 0.68930 |
| Mraw | 0.68304 | 0.68518 | 0.68184 | 0.68337 |
| Mshr | 0.68150 | 0.68122 | 0.68075 | 0.68116 |

| comparison | Δ log-loss/draw | z |
|---|---|---|
| Mshr vs M0 | 0.00814 | 24.46404 |
| Mshr vs Mraw | 0.00221 | 12.40234 |
| Mraw vs M0 | 0.00593 | 15.08678 |

Calibration slopes of realized next-season excess wins on predicted (taker-seasons with ≥ 300 draws): Mraw 0.78 (0.68–0.87, n = 430); Mshr 0.84 (0.77–0.91, n = 430).

Best and worst taker-seasons (≥ 500 draws; goals above replacement):

| player_name | season | fo_draws | win_pct | fo_gaa | fo_gar | fo_gar_sd |
|---|---|---|---|---|---|---|
| Ryan O'Reilly | 2017 | 1470 | 0.602 | 3.627 | 4.807 | 0.394 |
| Patrice Bergeron | 2021 | 1049 | 0.639 | 3.597 | 4.449 | 0.314 |
| Patrice Bergeron | 2022 | 1063 | 0.626 | 3.375 | 4.252 | 0.322 |
| Jonathan Toews | 2018 | 1251 | 0.565 | 2.747 | 3.742 | 0.360 |
| Ryan O'Reilly | 2018 | 1296 | 0.569 | 2.704 | 3.738 | 0.364 |
| Sidney Crosby | 2023 | 1389 | 0.580 | 2.396 | 3.505 | 0.378 |
| Anze Kopitar | 2018 | 1324 | 0.554 | 2.330 | 3.386 | 0.369 |
| Jonathan Toews | 2017 | 1070 | 0.584 | 2.492 | 3.343 | 0.320 |

| player_name | season | fo_draws | win_pct | fo_gaa | fo_gar | fo_gar_sd |
|---|---|---|---|---|---|---|
| Jack Hughes | 2024 | 592 | 0.377 | -1.948 | -1.475 | 0.206 |
| Jack Hughes | 2020 | 556 | 0.356 | -1.845 | -1.403 | 0.193 |
| Danila Yurov | 2025 | 702 | 0.402 | -1.838 | -1.278 | 0.308 |
| Logan Cooley | 2023 | 617 | 0.384 | -1.698 | -1.206 | 0.233 |
| Connor Bedard | 2024 | 601 | 0.381 | -1.565 | -1.084 | 0.229 |
| Dawson Mercer | 2021 | 527 | 0.362 | -1.444 | -1.021 | 0.207 |

## Penalties

| class (min) | penalties | net xG for the advantaged team | SE | mean advantage (s) |
|---|---|---|---|---|
| 2 | 46906 | 0.175 | 0.001 | 98.503 |
| 4 | 612 | 0.316 | 0.012 | 187.708 |
| 5 | 10 | 0.279 | 0.198 | 232.100 |

Majors are almost all coincident (fighting) and are excluded, so the 5-minute class rests on very few penalties and has almost no effect. League rates per 60 (minor-equivalents): drawn 0.489, taken 0.519. SD of goals above replacement among qualified skaters: 0.49.

| rate | YOY raw | YOY shrunk | next-season deviance raw | deviance shrunk |
|---|---|---|---|---|
| drawn | 0.459 | 0.477 | 3.360 | 1.571 |
| taken | 0.511 | 0.519 | 2.843 | 1.490 |

Team-level check: penalty value is realized on special teams, not at 5v5. Regressing team special-teams goal differential (all non-5v5 goals) on the summed penalty value gives a slope of 1.35 ± 0.51 (R² 0.025; special-teams GD SD 15.9, dominated by power-play and penalty-kill skill and luck). Against 5v5 goal differential the penalty component has no relationship, as expected.

## Interpretation and sanity checks

**Ten of the eleven pre-registered gates pass; TO-2 fails narrowly.** TO-2 asks that the turnover component's team-level slope (joint regression with the other components) lie in 0.5–1.5. It is 1.51 ± 0.66 with shrunk components (band edge missed by 0.01) and 1.18 ± 0.44 with the possession and turnover values unshrunk. The interval includes 1 either way. It is reported as computed, not tuned.

**Turnover cost.**

* **It closes most of the gap PM-EPA showed at team level.** With the turnover and faceoff components in the regression, the possession slope on team 5v5 goal differential falls from 2.04 to 1.53 (shrunk) and 1.16 (unshrunk). The part of PM-EPA's apparent under-scaling that survives is shrinkage compression: an independent-player prior shrinks effects shared across a team (system, arena, pooled credit), so team sums of shrunk values understate team totals.
* **The charge is real and stable.** Season-centred YOY of the shrunk per-60 charge is 0.60 (forwards) and 0.48 (defensemen). Shrinking beats raw at predicting next season (z = 14.5).
* **Arena (scorer) bias was large before 2024 and is gone since.** Excess arena SD was 0.03–0.045 per 60 in 2017–2023, about 15–23% of the −0.20 league mean charge, and 0.008–0.009 in 2024-25 (when NHL recording changed). Visitors-only estimates rule out home-team style as the cause. The adjustment lowers the gap between stayers' and team-changers' stability from 0.29 to 0.22 (stayers 0.65 → 0.61, changers 0.36 → 0.39). The rest of that gap is consistent with real style and system persistence. Out of sample (visiting players, odd/even games) the arena adjustment cuts game-level error by 5.1% in 2017–2023 and by nothing in 2024-25; a count-proportional (multiplicative) version does the same (5.1%), and separate effects for the direct and pooled parts add nothing (they are positively related, r = 0.15). Arena effects persist from season to season (r 0.75–0.79), which is why they flatter the stability of players who stay on one team.
* **A role-adjusted variant (TO-RA, `to_ra_gar`) mitigates part of this and passes its three gates** (residual-vs-involvement |r| 0.02 and 0.01; shrunk YOY 0.55 and 0.45; z = 15.7). It compares each player's charge with players of similar observed successful puck involvement. Raw charge does rise with involvement (r = −0.29 for forwards, −0.24 for defensemen; more involvement, larger charge), and the adjustment flattens that gradient: mean goals above replacement by involvement quintile go from −0.05 (least involved) to −0.35 (most involved) under TO, and are level at about −0.10 under TO-RA. **The effect is modest**, because observed involvement explains only 11% (forwards) and 6% (defensemen) of the variance in charge: the rank correlation with TO is 0.93 and 0.96, and totals move by up to about ±0.5 goals a season (correlation of totals 0.999; the 2025-26 top 15 changes by one name, Matthews in and Spurgeon out). It helps the high-involvement stars most: Pastrnak −11.5 → −7.8, Kucherov −9.7 → −6.6, MacKinnon −8.8 → −5.0 (pooled goals). It barely helps Draisaitl (−11.7 → −10.2), whose observed involvement is low relative to his charge, so his puck handling is not captured by the recorded events. TO-RA also removes some genuine puck-management skill, and its residuals do not sum to the team charge. Use it to compare players, and TO for team accounting.
* **The player-level allocation of TO is biased against puck-dominant players.** Giveaways and takeaways are recorded, but successful passes and carries are not. The most-charged skaters (pooled over 2017-2025) are Draisaitl (−11.7 goals), Pastrnak (−11.5), Kucherov (−9.7), Barzal (−9.4) and MacKinnon (−8.8), and the least-charged are shutdown defensemen (Pesce +6.6, Spurgeon +4.9, Brodin +4.9, Larsson, Dillon, Slavin). Team totals are right; the split between a star's charged giveaways and his uncredited good plays is not. The replacement pool also touches the puck less, so regulars average −0.20 goals from turnovers against +0.03 for the fringe pool.

**Faceoffs.**

* **A strong, repeatable skill.** Persistent taker SD is 0.24 logit for centers. Shrunk skill beats no-taker on next-season draws by z = 24.5 and raw win rate by z = 12.4; the calibration slope is 0.84 (0.77–0.91), so even shrunk estimates are slightly over-dispersed.
* **The value of a draw is small.** A win is worth 0.0245 xG on average (start value only, one possession), so the best takers are worth about +4 to +5 goals in a season at 60–64% win rates. Pooled over 2017-2025 the best are O'Reilly (+24.5), Kopitar (+23.1), Staal (+22.4), Crosby (+21.8), Tavares (+21.8) and Bergeron (+20.9); the worst are Jack Hughes (−7.6), Barzal, Namestnikov, Schmaltz and Zegras. The SD among takers with 300+ draws is 0.86 goals.
* **Defensemen and most wingers take almost no draws**, so the component is effectively a center component (SD among defensemen 0.0).
* **Only the skill part of the swing is claimed.** The luck part (who won given equal skill) is claimed by nobody, and swing values cover the immediate possession only.

**Penalties.**

* **A minor penalty at 5v5 costs 0.175 xG** (95% CI 0.173–0.177; double minors 0.316). Majors are nearly all coincident (fighting), so only 10 qualify and they carry no weight.
* **Drawing and taking are both repeatable** (YOY 0.48 and 0.52), and shrinkage beats raw at predicting next-season counts (z = 7.7 and 7.2). The best net drawers, pooled, are McDavid (+15.8), MacKinnon (+11.6), Pettersson, Ehlers, Kadri and Gaudreau; the worst are Ian Cole, Strome, Zadorov, Ritchie and Backlund.
* **Penalty value is realized on special teams, not at 5v5.** Against 5v5 goal differential it has no relationship, as it should not (regressing 5v5 goal differential on all six components first produced a misleading negative penalty slope, −2.1, which is why the check uses the right target). Against special-teams goal differential the slope is 1.35 ± 0.51.
* **Drawer and committer coverage differ** (the drawer is missing for 8.8% of penalties, the committer for 3.1%), so league net drawn minus taken is negative. The replacement rates absorb this because they use the same coverage.

**Limits.**

* **Only turnovers carry an arena adjustment.** Referee and home bias in penalty calls is not modelled, and neither are rink effects on faceoff outcomes.
* **The replacement pool and shrinkage priors for the three components** use the same regular/fringe definition as the earlier components, but the sensitivity analysis varies the cutoff only for the possession, defense and finishing components.
* **The three components are small next to defense and finishing** (SD 0.39, 0.52 and 0.49 goals per qualified skater against 1.88 and 1.48), but they are precisely estimated and change individual players' totals by one to two goals.

