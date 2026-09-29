# Goaltending component — validation report

Spec and pre-registered rules: [methodology](../../docs/goaltending/methodology.md). Goals above replacement for goalies, all strengths: a logistic mixed model on unblocked shots with the xG offset, the shooter's fitted shift, a season × strength effect, a persistent and a season-deviation goalie effect, plus an unpenalized replacement effect for goalie-seasons with fewer than 500 shots.

## Gates

| Gate | Result | Pass |
|---|---|---|
| GK-1 goals on non-empty-net shots attributed to a goalie ≥ 99.5% | 99.98% | ✅ |
| GK-2 year-over-year r of single-season shrunk effects (≥ 500 shots both seasons) ≥ 0.15 | 0.13 (n = 439); raw goals minus xG: 0.00 | ❌ |
| GK-3 beats the goalie-free baseline on test folds [2023, 2024, 2025] (paired z > 2.0) | z = 2.75; with prior fringe status added (tried): z = -0.69 | ✅ |
| GK-4 forward slope of realized goals saved per shot on the prior estimate in [0.7, 1.4] | 0.91 ± 0.17 (n = 562); with prior fringe status added (tried): 0.30 ± 0.08 | ✅ |
| GK-5 rolling goalie term beats the crude proxy at predicting next-season team goal differential (paired z > 2) | z = 1.02 (RMSE 48.2 vs 48.8) | ❌ |

## Model choice and variance components

Variant chosen by the D-type rule on the 2022 fold: **G1** (G2 vs G1 held-out log-loss z = 1.43, threshold 2.0). Standard deviations (logit units): u 0.059, d 0.053 (u persistent, d season deviation). The replacement goalie saves -0.88 goals per 100 shots relative to an average one. Correlation of goalie-season values between G1 and G2: 0.99.

## Forward folds (each refit on data through the training season)

| trained through | test season | log-loss baseline | log-loss model | paired z | z, with fringe status |
|---|---|---|---|---|---|
| 2017 | 2018 | 0.2111 | 0.2112 | -0.0163 | -0.1415 |
| 2018 | 2019 | 0.2110 | 0.2110 | 1.4137 | -2.3345 |
| 2019 | 2020 | 0.2129 | 0.2129 | 1.9775 | -1.0361 |
| 2020 | 2021 | 0.2185 | 0.2185 | 2.5922 | -0.4886 |
| 2021 | 2022 | 0.2203 | 0.2203 | 2.0130 | -2.4693 |
| 2022 | 2023 | 0.2120 | 0.2121 | -0.8170 | -1.1963 |
| 2023 | 2024 | 0.2122 | 0.2122 | 2.0667 | 1.3300 |
| 2024 | 2025 | 0.2196 | 0.2195 | 3.5453 | -1.0950 |

## Scale (goalies with at least 1500 shots)

| goalie-seasons | mean goals above replacement | SD goals | SD wins | mean posterior SD (goals) |
|---|---|---|---|---|
| 273 | 20.95 | 8.56 | 1.45 | 6.33 |

## Leaders, 2025 (≥ 1500 shots)

| player_name | team | shots | goals | expected_goals | gsaa | goalie_gar | gsaa_sd |
|---|---|---|---|---|---|---|---|
| Logan Thompson | WSH | 2462 | 140.0 | 178.0 | 18.1 | 41.1 | 7.5 |
| Ilya Sorokin | NYI | 2325 | 144.0 | 174.4 | 18.7 | 41.1 | 7.1 |
| Jeremy Swayman | BOS | 2427 | 146.0 | 175.6 | 15.2 | 37.9 | 7.4 |
| Igor Shesterkin | NYR | 2149 | 126.0 | 146.3 | 15.9 | 35.0 | 6.1 |
| Andrei Vasilevskiy | TBL | 2227 | 132.0 | 155.5 | 13.0 | 33.2 | 6.4 |
| Connor Hellebuyck | WPG | 2364 | 162.0 | 165.3 | 10.8 | 32.4 | 6.8 |
| Jet Greaves | CBJ | 2380 | 142.0 | 157.1 | 11.2 | 31.6 | 7.4 |
| John Gibson | DET | 2281 | 144.0 | 157.7 | 8.1 | 28.7 | 6.6 |

Lowest:

| player_name | team | shots | goals | expected_goals | gsaa | goalie_gar | gsaa_sd |
|---|---|---|---|---|---|---|---|
| Alex Nedeljkovic | SJS | 1561 | 109.0 | 107.6 | -0.4 | 13.6 | 5.3 |
| Sergei Bobrovsky | FLA | 1948 | 154.0 | 136.6 | -4.2 | 13.5 | 6.2 |
| Yaroslav Askarov | SJS | 1988 | 157.0 | 139.0 | -6.4 | 11.6 | 7.3 |
| Kevin Lankinen | VAN | 1936 | 160.0 | 142.3 | -7.2 | 11.1 | 6.8 |
| Arturs Silovs | PIT | 1572 | 115.0 | 107.0 | -3.8 | 10.1 | 5.9 |

## Limitations

* Expected goals omit screens, traffic and pre-shot movement; the team-season effect absorbs part of it.
* Goalie and defense effects are weakly separated: starters face most of their team's shots.
* Save quality is rare-event data: values are strongly shrunk and intervals wide; the delta-method interval does not include the replacement level's own uncertainty.
* No rest, back-to-back or workload adjustment; the value is per shot faced.

## Interpretation

**What passed.** Goal attribution is complete (99.98% of goals on non-empty-net shots have a goalie). The persistent goalie effect predicts next-season shot outcomes better than the goalie-free baseline on the test folds (paired z = 2.75), and the forward calibration slope is 0.91 ± 0.17: a goalie's estimated advantage carries over to the next season at close to face value.

**What failed.** Stability: the year-over-year correlation of single-season shrunk effects is 0.13, under the 0.15 gate (the raw goals-minus-xG correlation is 0.00, so the shrinkage model extracts real signal that the raw rate does not). Whole-model use (GK-5): the rolling goalie term improves next-season team goal differential over the crude proxy only slightly (RMSE 48.2 against 48.8, paired z = 1.02, threshold 2), so GK-5 fails.

**Model choice.** The D-type rule selects the no-team-effect variant (G1): adding a defending team-season effect improves held-out log-loss on the 2022 fold by z = 1.43 over G1, under the threshold of 2 (both beat the goalie-free baseline). The two variants' goalie-season values correlate 0.99, so the choice hardly matters for the published values. The persistent goalie sd is 0.059 logit and the season-deviation sd 0.053, so about half of a goalie's season-to-season variation in results is temporary.

**How much is known about any one goalie.** The average posterior sd of a starter's season value is 6.3 goals against a spread of 8.6 goals across starters (the standard deviation in wins is 1.45): goalies are strongly shrunk and their intervals are wide. This is the honest state of goalie evaluation from shot data: save quality is rare-event data and the persistent part is small.

**Replacement level.** The unpenalized fringe effect puts a replacement goalie (a goalie-season with fewer than 500 shots) 0.88 goals per 100 shots below an average one, which is what makes starters' mean value 21 goals above replacement. The convention matters: it is large relative to the differences between starters, it comes from goalies with few shots (injury call-ups and backups), and it credits any goalie who plays a full season with roughly 15 goals before any skill difference. Treat goalie totals as valuable for ranking goalies against each other and less well-anchored in absolute terms than the skater components.

**Deviations worth knowing.** The first run's replacement definition (mean of shrunk fringe effects) was near zero by construction and was replaced (deviation 1). A tried change that added a goalie's prior fringe status to the forecast made the forecast worse (test z = −0.69, slope 0.30) and was reverted; the persistent-effect forecast pre-registered in §3 is what is reported.

**Whole model.** Replacing the crude goalie proxy with the rolling component leaves the earlier whole-model conclusions unchanged: the skater total plus the goalie term has RMSE 40.1 against 38.4 for the naive roster-weighted on-ice xG baseline with the same goalie term (z = −1.13), and the total still adds to prior team results only at z = 1.61. A stronger goalie component would probably require modeling shot quality beyond xG, rest, and team context.

