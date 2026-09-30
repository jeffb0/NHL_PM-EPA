# NHL EDGE tracking features — report

Spec and pre-registered gates: [methodology](../../docs/edge/methodology.md). EDGE season summaries (max skating speed, bursts over 20 mph, distance, top shot speed, zone time) exist for 2021–2025 only, so the projection test has two test seasons (2024, 2025) and a training window of target seasons 2022 onward.

## Gates

| Gate | Result | Pass |
|---|---|---|
| ED-1 EDGE coverage of skater-seasons with ≥ 300 minutes (2021–2025) ≥ 90% | 100.0% (3,395 of 3,395) | ✅ |
| ED-2 max-speed age curve declines after 27 (27→33 interval excludes 0) | forwards -0.37 mph (-0.47 to -0.26); defensemen -0.21 (-0.34 to -0.03) | ✅ |
| ED-3 EDGE features improve the horizon-1 projection on 2024–2025 (z > 2) | z = -0.63 (MSE 0.0149 vs 0.0148; 1,348 player-seasons) | ❌ |
| ED-4 80% interval coverage with EDGE features in [70%, 90%] | 82.9% | ✅ |

## Feature sets

| features | MSE | MSE without | z vs without | players 30+ (n = 451) | MSE 30+ | MSE 30+ without | 80% coverage |
|---|---|---|---|---|---|---|---|
| all_five | 0.0149 | 0.0148 | -0.6321 | 0.3007 | 0.0143 | 0.0143 | 0.8294 |
| speed_only | 0.0149 | 0.0148 | -0.1091 | -0.0729 | 0.0143 | 0.0143 | 0.8264 |
| speed_distance | 0.0149 | 0.0148 | -0.1640 | -0.0194 | 0.0143 | 0.0143 | 0.8264 |

## Does speed explain aging? Age effect on the projected rate, 27 to 34 (goals per 60), EDGE-era data

| position | without EDGE | with EDGE features |
|---|---|---|
| F | -0.050 | -0.050 |
| D | -0.011 | -0.009 |

## Exploratory: is a change in speed a leading indicator of next season's value? (next-season total rate on this season's rate, EDGE level and EDGE change, age terms; player-cluster t)

| feature | position | rows | level coefficient (t) | change coefficient (t) |
|---|---|---|---|---|
| max skating speed (mph) | F | 1066 | -0.0074 (-1.2) | 0.0054 (0.9) |
| bursts over 20 mph per 60 | F | 1066 | -0.0041 (-0.8) | 0.0153 (1.6) |
| distance skated per 60 (miles) | F | 1066 | 0.0054 (1.1) | 0.0079 (1.2) |
| max skating speed (mph) | D | 581 | -0.0042 (-0.6) | -0.0008 (-0.1) |
| bursts over 20 mph per 60 | D | 581 | -0.0001 (-0.0) | 0.0029 (0.5) |
| distance skated per 60 (miles) | D | 581 | -0.0092 (-1.6) | 0.0128 (2.1) |

## Physical aging curves (season-centred; change from age 27 to 33, minute-weighted, 95% player-cluster interval)

| feature | position | 27 → 33 | 95% interval | rows |
|---|---|---|---|---|
| max skating speed (mph) | F | -0.365 | -0.471 to -0.256 | 2193 |
| max skating speed (mph) | D | -0.207 | -0.344 to -0.025 | 1200 |
| bursts over 20 mph per 60 | F | -2.484 | -3.320 to -1.884 | 2193 |
| bursts over 20 mph per 60 | D | -0.553 | -0.817 to -0.243 | 1200 |
| distance skated per 60 (miles) | F | -0.142 | -0.240 to -0.055 | 2193 |
| distance skated per 60 (miles) | D | -0.119 | -0.221 to -0.019 | 1200 |
| top shot speed (mph) | F | -0.422 | -1.202 to 0.481 | 2193 |
| top shot speed (mph) | D | 0.293 | -0.932 to 1.596 | 1200 |
| offensive-zone time share | F | -0.002 | -0.009 to 0.004 | 2193 |
| offensive-zone time share | D | -0.004 | -0.011 to 0.001 | 1200 |

## Standardized EDGE coefficients in the projection ridge (per unit of the season-and-position z-score; goals per 60)

| feature | forwards | defensemen |
|---|---|---|
| z_max_speed | -0.0054 | 0.0034 |
| z_bursts60 | 0.0046 | 0.0078 |
| z_dist60 | 0.0022 | -0.0102 |
| z_shot_speed | 0.0041 | -0.0019 |
| z_oz_share | 0.0113 | 0.0075 |
| edge_missing | -0.0046 | -0.0488 |

## Interpretation

**Physical aging is real and clearly visible in EDGE.** Between 27 and 33, forwards lose about 0.37 mph of maximum skating speed (95% interval 0.26 to 0.47) and defensemen about 0.21 mph (0.03 to 0.34). Forwards' bursts over 20 mph drop by about 2.5 per 60 minutes (1.9 to 3.3), defensemen's by 0.55, and both lose distance covered per 60 (0.14 and 0.12 miles). Top shot speed and offensive-zone share show no clear age pattern. Forwards age faster physically than defensemen, mirroring the value aging curves from the projection layer (forwards lose about 0.05 goals per 60 between 27 and 34, defensemen about 0.01).

**But it does not improve the projections.** Adding last season's EDGE features (all five, speed only, or speed plus distance) to the projection leaves horizon-1 error unchanged on the 2024 and 2025 test seasons (MSE 0.0149 against 0.0148, z = −0.63; speed only z = −0.11; speed plus distance z = −0.16), including for players aged 30 and over (z = 0.30 on 451 player-seasons). Interval coverage stays sound (82.9%). The size of the age effect on the projected rate is the same with and without EDGE (forwards −0.050 in both cases, defensemen −0.011 against −0.009), so speed and workload do not explain the aging the model already estimates: the value history, which reflects declining speed through results, already carries that information. ED-3 fails; ED-1, ED-2 and ED-4 pass.

**Is a speed drop a leading indicator?** Exploratory, 12 regressions on forwards and defensemen: the coefficient on the change in max speed, bursts or distance between last season and this one, predicting next season's value beyond this season's value, is not significant in 11 of 12 cases (the exception, a defenseman's change in distance, has t = 2.1, about what one significant result in twelve would produce by chance). There is no sign that a fall in speed foreshadows a fall in value that the value history has not already shown.

**What this can and cannot say.** The test is small: two test seasons, 1,348 player-seasons, five EDGE seasons in all, and season summaries (peak speed, counts) rather than game-by-game series. A null result here means tracking summaries add nothing detectable to the value history at this sample size, not that physical decline is irrelevant, and a game-level speed trend (available only if per-game EDGE data were ingested) could still carry information that a season peak hides. The ingestion, the feature construction and the `Spec.extra` hook are in place; the test should be repeated as seasons accumulate. No default changes.

**Where the accuracy gains are not.** Year-over-year accuracy already improved substantially through the projection layer (and team context helped movers); EDGE does not add to that. The remaining projection error is dominated by the noise in a single season's value estimate and by role, minutes and injury changes, which season-level tracking does not address.

