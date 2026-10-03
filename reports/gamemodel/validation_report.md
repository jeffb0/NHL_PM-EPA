# Game-level model — validation report

Test seasons 2021–2025, 6502 games; hyperparameters tuned on 2019–2020 only and frozen; the logistic step is fitted on earlier seasons. Best team-rating benchmark: **G2 team rating on blended margin**.

## Gates

| gate | rule | statistic | result |
|---|---|---|---|
| GM1 | P beats G0 (z > 2) | z = 9.58 | PASS |
| GM2 | P beats the better team rating (z > 2) | z = 1.99 | FAIL |
| GM3 | in-season updating helps: P beats P-pre (z > 2) | z = 5.74 | PASS |
| GM4 | player attribution helps: P beats P-team (z > 2) | z = 2.40 | PASS |
| GM5 | calibration slope in [0.8, 1.2] | slope 0.91 | PASS |

## Scores (lower log loss and Brier are better)

| model | log loss | Brier | accuracy | calibration slope |
|---|---|---|---|---|
| G0 home-ice constant | 0.6905 | 0.2487 | 53.7% |  |
| G1 team rating on goals | 0.6655 | 0.2365 | 59.5% | 0.83 |
| G2 team rating on blended margin | 0.6613 | 0.2345 | 60.0% | 0.84 |
| P-pre preseason player prior only | 0.6692 | 0.2383 | 59.0% | 0.90 |
| P-team preseason prior + team-level update | 0.6599 | 0.2339 | 60.1% | 0.90 |
| P-prev previous game's lineup | 0.6586 | 0.2334 | 60.4% | 0.95 |
| P player ratings, updated in season | 0.6585 | 0.2333 | 60.4% | 0.91 |

## Paired comparisons (z > 0 favors the first model; second z clusters by date)

| comparison | log-loss difference | z | z (date clusters) |
|---|---|---|---|
| P vs G0 | 0.0320 | 9.58 | 9.35 |
| P vs G1 | 0.0069 | 3.39 | 3.49 |
| P vs G2 | 0.0028 | 1.99 | 1.92 |
| P vs P-pre | 0.0107 | 5.74 | 5.76 |
| P vs P-team | 0.0013 | 2.40 | 2.46 |
| P vs P-prev | 0.0001 | 0.06 | 0.06 |
| P-pre vs best-team | -0.0079 | -3.72 | -3.71 |
| P-team vs best-team | 0.0014 | 1.02 | 0.99 |

## Tuned hyperparameters (2019–2020)

| model | values |
|---|---|
| P | lam = 0.04, alpha = 0, n0 = 3000 |
| P_team | lam = 0.02, alpha = 0, n0 = 3000 |
| G1 | k = 0.02, rho = 1, alpha = 1 |
| G2 | k = 0.05, rho = 1, alpha = 0.25 |

## Log loss by season

| season | G0 | G1 | G2 | P-pre | P-team | P |
|---|---|---|---|---|---|---|
| 2021 | 0.6906 | 0.6465 | 0.6441 | 0.6556 | 0.6440 | 0.6435 |
| 2022 | 0.6922 | 0.6614 | 0.6544 | 0.6707 | 0.6573 | 0.6558 |
| 2023 | 0.6900 | 0.6595 | 0.6605 | 0.6578 | 0.6537 | 0.6532 |
| 2024 | 0.6871 | 0.6643 | 0.6625 | 0.6656 | 0.6588 | 0.6569 |
| 2025 | 0.6926 | 0.6957 | 0.6851 | 0.6962 | 0.6853 | 0.6832 |

## Market comparison

No odds file supplied (`data/external/odds.csv`); no claim about the betting market is made.
