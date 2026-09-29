# Projection and aging layer — validation report

Spec and pre-registered gates: [methodology](../../docs/projection/methodology.md). Weighted ridge projections of each skater's all-strength rate (goals above replacement per 60 modeled minutes) from shrunk component histories, age and body size; finishing and faceoffs enter as **rolling-refit** values so no forecast uses later information. Expanding-window evaluation on target seasons 2020–2025 (horizon 1), players with at least 300 target-season minutes, minute-weighted error, player-cluster bootstrap z.

Each cell shows the result **as pre-registered** (training rows need ≥ 200 target-season minutes) → **revised** (≥ 20; methodology §9, deviation 1). Everything below the gate table is from the revised run.

## Gates

| Gate | As pre-registered → revised | Pass |
|---|---|---|
| PR-1 beats the position-mean baseline (z > 2) | z = 10.84 → z = 10.63 | ✅ → ✅ |
| PR-2 beats Marcel (weights 5/4/3, 1,000-minute ballast, no age) (z > 2) | z = 5.48 → z = 5.23 | ✅ → ✅ |
| PR-3 the age terms improve the projection (z > 2) | z = 4.49 → z = 4.74 | ✅ → ✅ |
| PR-4 calibration slope of actual on projected rate in [0.8, 1.2] | 0.94 ± 0.01 → 0.93 ± 0.01 | ✅ → ✅ |
| PR-5 80% interval coverage in [70%, 90%] | 79.0% → 78.9% | ✅ → ✅ |
| PR-6 team level: projected rates beat last season's rolling-refit rates (z > 2) | z = -2.92 → z = -2.75 (RMSE 48.8 vs 44.9 → 48.4 vs 44.9) | ❌ → ❌ |

## Player-level accuracy by horizon (revised run)

| horizon (seasons ahead) | test seasons | rows | MSE projection | MSE position mean | MSE Marcel | MSE last season | z vs mean | z vs Marcel | z vs last |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2020–2025 | 4018 | 0.0164 | 0.0378 | 0.0178 | 0.0179 | 10.6348 | 5.2252 | 5.1252 |
| 2 | 2022–2025 | 2708 | 0.0225 | 0.0388 | 0.0234 | 0.0262 | 8.4794 | 1.7272 | 7.4908 |
| 3 | 2024–2025 | 1348 | 0.0249 | 0.0355 | 0.0248 | 0.0284 | 5.9101 | -0.1125 | 5.8895 |

## Variants (horizon 1; `z` positive means the main model is better)

| variant | MSE | main model better, z |
|---|---|---|
| total_only | 0.0167 | 2.6584 |
| no_age | 0.0169 | 4.7398 |
| no_hw | 0.0164 | -0.5016 |
| m0_500 | 0.0160 | -6.3477 |
| m0_2000 | 0.0168 | 8.0099 |

## Team level (goalie proxy and intercept, plus the row's term, fitted on earlier seasons; test seasons 2021, 2022, 2023, 2024, 2025)

| model | RMSE (goals) |
|---|---|
| projected | 48.38 |
| rolling last season | 44.89 |
| projected plus prior gd | 41.93 |
| rolling plus prior gd | 41.91 |
| goalie only | 50.89 |
| prior team gd | 41.42 |
| projected fixed slope | 46.96 |
| rolling fixed slope | 43.35 |

| comparison | mean difference | z |
|---|---|---|
| projected vs rolling | -325.78 | -2.75 |
| projected vs goalie only | 248.76 | 1.12 |
| projected vs prior team gd | -625.03 | -2.54 |
| projected plus gd vs rolling plus gd | -1.85 | -0.03 |
| fixed slope projected vs rolling | -325.81 | -3.82 |

## What persists: fitted weights on the shrunk component histories (horizon 1, per unit of shrunk rate; revised run)

| feature | forwards | defensemen |
|---|---|---|
| rt_pm | 1.121 | 2.448 |
| rt_def | 1.523 | 1.318 |
| rt_fin | 0.979 | 0.180 |
| rt_to | 0.467 | 0.461 |
| rt_fo | 1.962 | 130.471 |
| rt_pen | 1.007 | 0.286 |
| rt_pp | 1.096 | 0.774 |
| rt_pk | 0.825 | 0.832 |
| rt_4v4 | 0.818 | -0.023 |
| rt_3v3 | 2.359 | 0.438 |
| no_history | -0.027 | -0.014 |

Minutes model: mean absolute error 293 against 303 for last season's minutes (3972 player-seasons).

## Aging curves (effect on projected rate per 60 relative to age 27, 95% player-cluster bootstrap interval)

| age | position | effect | lo95 | hi95 |
|---|---|---|---|---|
| 20.000 | D | 0.002 | -0.016 | 0.018 |
| 22.000 | D | 0.001 | -0.012 | 0.013 |
| 24.000 | D | 0.001 | -0.007 | 0.008 |
| 26.000 | D | -0.005 | -0.010 | 0.002 |
| 28.000 | D | -0.010 | -0.019 | -0.001 |
| 30.000 | D | -0.017 | -0.027 | -0.005 |
| 32.000 | D | -0.023 | -0.038 | -0.006 |
| 34.000 | D | -0.025 | -0.037 | -0.011 |
| 36.000 | D | -0.027 | -0.044 | -0.010 |
| 38.000 | D | -0.028 | -0.052 | -0.005 |
| 20.000 | F | 0.019 | -0.002 | 0.037 |
| 22.000 | F | 0.014 | -0.002 | 0.026 |
| 24.000 | F | 0.008 | -0.001 | 0.016 |
| 26.000 | F | -0.013 | -0.023 | -0.002 |
| 28.000 | F | -0.035 | -0.049 | -0.019 |
| 30.000 | F | -0.053 | -0.068 | -0.036 |
| 32.000 | F | -0.070 | -0.088 | -0.049 |
| 34.000 | F | -0.079 | -0.095 | -0.059 |
| 36.000 | F | -0.088 | -0.109 | -0.065 |
| 38.000 | F | -0.096 | -0.120 | -0.068 |

## Limitations

* About nine seasons and shortened 2020 and 2021 seasons: few transitions, especially at the young and old ends; the aging curves are conditional on staying in the league (survivorship).
* Injuries, role changes and trades are not modeled; minutes are projected conditional on playing.
* Skaters only. No EDGE tracking features are used (none are in the repository); `Spec.extra` accepts any extra feature columns.

## Interpretation

**What works.** At the player level the projection is a clear improvement on simple approaches for next season: minute-weighted error 0.0164 against 0.0378 for the position mean (z = 10.6), 0.0178 for the Marcel-style weighted history (z = 5.2) and 0.0179 for last season's rate (z = 5.1). The age terms matter (z = 4.7), the calibration slope of actual on projected rate is 0.93 ± 0.01, and the 80% intervals cover 78.9% of test player-seasons. Position-specific aging curves are estimated: forwards lose about 0.035 goals per 60 by age 28 and 0.07 by 32 relative to 27, defensemen decline more slowly (about 0.010 at 28 and 0.023 at 32), and the intervals exclude zero from the late twenties on. Height and weight add nothing (removing them changes the error by z = −0.5).

**What does not work.** At the team level, summing projected rates does **worse** than summing last season's rolling-refit rates (RMSE 48.4 against 44.9 goals, z = −2.75, over 2021–2025), so PR-6 fails, as it did before the training-floor revision (RMSE 48.8, z = −2.92). Projections also lose their edge over Marcel at longer horizons (z = 1.7 at two seasons and −0.1 at three), and a shorter Marcel ballast of 500 minutes forecasts better than the pre-registered 1,000 (z = −6.3; not adopted, reported).

**Why the two disagree (updated).** The first explanation offered here, that the defensive and special-teams components carry chained priors that make their estimates persistent by construction, was tested by re-estimating them without the chain ([chain-free report](../chainfree/validation_report.md)): the chain does create persistence (defensive year-over-year correlation 0.78 against 0.30) but removing it leaves the team-level result unchanged (PR-6 z = −2.77), so that explanation is **not supported**. Exploratory diagnostics point instead to team context: at the team level last season's rates alone forecast best (RMSE 44.9) and adding older seasons makes it worse (47.5 for two seasons), so talent-only smoothing helps forecast a player but not the team he plays for. The exploratory results-target variant (the same ridge trained on next-season deployment-adjusted on-ice goal differential, state shares as usage controls) predicts results better than a total-only model (z = 4.0) and at the team level reaches RMSE 42.4 against 44.9 for last season's rolling-refit rates (z = 1.2) and 50.9 for the goalie-only baseline (z = 2.9); it still does not beat the prior season's team goal differential (41.4; z = −0.4). This variant was not pre-registered and is a diagnostic, not a gate.

**What the published projections are.** `player_projection.csv` gives rate projections, minutes, goals above replacement and wins for the next season with 80% intervals, and rates for two and three seasons out. They forecast the *value estimates* (target 1). Use them for ranking, aging and contract-length questions; do not read them as validated forecasts of team results, where PR-6 failed. Minutes are projected conditional on playing (mean absolute error 293 against 303 for last season's minutes) and there is no injury or retirement model.

**Limitations.** Nine seasons, two shortened; aging curves are conditional on staying in the league; features are the component estimates themselves (errors in variables); the fitted weight on faceoffs for defensemen is large only because the feature has almost no variance. No EDGE features were used (none in the repository); `Spec.extra` accepts them when they are ingested.

