# Whole-model validation — report

Spec and pre-registered gates: [methodology](../../docs/wholemodel/methodology.md). The question: does the published total (`total_gar_all`, weights 1) predict next-season team goal differential, and does it add information beyond simple baselines? 189 out-of-sample team-seasons (2020–2025), expanding-window OLS, every model including the same goalie proxy.

## Gates

| Gate | Result | Pass |
|---|---|---|
| WM-1 M beats B0 (paired z > 2) | z = 4.57; out-of-sample R² vs B0 0.33 | ✅ |
| WM-2 M beats the naive roster-weighted on-ice xG baseline B3 (z > 2) | z = -0.64 (RMSE 39.9 vs 38.9) | ❌ |
| WM-3 M+B1 beats B1: the total adds information beyond the team's prior results (z > 2) | z = 1.94 (RMSE 37.7 vs 39.4) | ❌ |
| WM-4 calibration slope of actual on the fixed-slope prediction in [0.7, 1.3] | 1.16 ± 0.10 | ✅ |
| WM-5 leakage-clean total beats B0 (z > 2) and differs from the headline by |z| < 2 | clean vs B0 z = 3.11; headline minus clean z = 2.62 | ❌ |
| WM-6 M better than B0 in ≥ 4 of 6 test seasons | 6 of 6 | ✅ |

## Models (out of sample)

| model | RMSE (goals) | R² vs B0 |
|---|---|---|
| B0 goalie proxy only | 48.82 | 0.00 |
| B1 + prior team goal differential | 39.38 | 0.35 |
| B2 + prior team xG differential | 39.75 | 0.34 |
| B3 + naive roster-weighted on-ice xG | 38.91 | 0.36 |
| M + total_gar_all (headline) | 39.94 | 0.33 |
| M+B1 + total and prior team goal differential | 37.68 | 0.40 |
| M with slope fixed at 1 | 39.67 | 0.34 |
| M, previous team and minutes (no oracle roster) | 44.49 | 0.17 |
| intercept only | 49.47 | -0.03 |

Paired comparisons (squared-error difference, positive favors the first-named model):

| comparison | mean difference | z |
|---|---|---|
| M vs B0 | 788.14 | 4.57 |
| M vs B3 | -80.73 | -0.64 |
| M+B1 vs B1 | 131.18 | 1.94 |
| M vs B1 | -43.99 | -0.32 |
| M vs B2 | -15.25 | -0.12 |
| B0 vs constant | 63.84 | 1.73 |
| M fixed vs B0 | 809.84 | 5.23 |

## By test season

| season | MSE M | MSE B0 | MSE B1 |
|---|---|---|---|
| 2020 | 1054 | 1343 | 879 |
| 2021 | 2122 | 3401 | 1664 |
| 2022 | 1981 | 3434 | 2571 |
| 2023 | 1367 | 2462 | 1212 |
| 2024 | 1913 | 1918 | 1347 |
| 2025 | 1144 | 1726 | 1611 |

## Variants and drop-one (each replaces the total in M; `z vs headline` positive means the headline is better)

| total | RMSE | z vs headline | z vs B0 |
|---|---|---|---|
| calibrated weights (total_gar_all_cal) | 39.99 | 0.38 | 4.51 |
| 5v5 total only | 41.60 | 1.73 | 3.72 |
| original (not role-adjusted) turnover | 40.09 | 3.00 | 4.53 |
| leakage-clean total (unshrunk finishing, faceoffs, ST finishing) | 43.52 | 2.62 | 3.11 |
| no persistent-effect components (exploratory) | 41.10 | 1.91 | 4.33 |
| without pm | 39.96 | 0.09 | 4.60 |
| without def | 40.85 | 1.31 | 4.72 |
| without fin | 40.23 | 0.78 | 4.53 |
| without to | 39.67 | -1.47 | 4.70 |
| without fo | 40.25 | 1.98 | 4.41 |
| without pen | 40.02 | 0.50 | 4.50 |
| without pp | 40.98 | 1.79 | 4.33 |
| without pk | 40.26 | 0.67 | 4.21 |
| without 4v4 | 39.88 | -1.23 | 4.59 |
| without 3v3 | 39.92 | -0.31 | 4.55 |
