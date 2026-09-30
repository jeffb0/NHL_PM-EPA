# Standings backtest — validation report

Test seasons 2021–2025 (160 team-seasons); every model and baseline is fitted on earlier seasons only. Error is in points per 82 games. Best history baseline: **B2 prior-season goal differential**.

## Gates

| gate | rule | statistic | result |
|---|---|---|---|
| ST1 | M1 beats B0 (z > 2) | z = 4.31 | PASS |
| ST2 | M1 not worse than best history baseline (z > −2) | z = -1.46 | PASS |
| ST3 | blend M3 beats best history baseline (z > 2) | z = -0.82 | FAIL |
| ST4 | calibration slope in [0.8, 1.2] | slope 1.22 ± 0.16 | FAIL |
| ST5 | 80% coverage in [70%, 90%] | 80% | PASS |

## Error by model

| model | RMSE | MAE | Spearman (mean) | top-16 overlap |
|---|---|---|---|---|
| B0 league mean | 16.91 | 13.61 | | |
| B1 prior-season points | 13.79 | 11.05 | 0.52 | 69% |
| B2 prior-season goal differential | 13.70 | 11.08 | 0.52 | 68% |
| B3 prior-season xG differential | 14.13 | 11.42 | | |
| M1 roster model (no fitted slope) | 14.58 | 11.85 | 0.48 | 68% |
| M2 roster model, fitted slopes | 15.12 | 12.14 | | |
| M3 roster model + prior goal differential | 13.94 | 11.20 | 0.52 | 66% |
| variant R: last season's rates (no projection) | 14.23 | 11.53 | 0.47 | 64% |
| variant R + prior goal differential | 14.03 | 11.31 | | |
| variant P-res: results-target projection | 14.97 | 12.14 | 0.51 | 65% |
| M1 with the actual season roster and minutes (diagnostic) | 13.92 | 11.28 | | |

## Paired comparisons (z > 0 favors the first model)

| comparison | z |
|---|---|
| M1 vs B0 | 4.31 |
| M1 vs best history | -1.46 |
| M3 vs best history | -0.82 |
| M1 vs M1 R | -1.29 |
| M1 oracle vs M1 | 2.99 |
| M3 R vs best history | -1.05 |
| P res vs best history | -1.71 |
| M2 vs best history | -1.95 |
| M1 vs M3 | -1.10 |

## RMSE by season

| season | M1 | M3 | B2 |
|---|---|---|---|
| 2021 | 17.8 | 14.8 | 14.5 |
| 2022 | 15.0 | 15.6 | 16.1 |
| 2023 | 12.4 | 10.8 | 11.3 |
| 2024 | 13.7 | 13.5 | 12.0 |
| 2025 | 13.5 | 14.5 | 14.1 |
