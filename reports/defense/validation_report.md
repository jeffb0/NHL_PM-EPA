# Defensive PM-EPA (DPM) — Validation Report

Headline model **M0**, prior ρ = **1.0**. Spec and pre-registered rules: [docs/defense/methodology.md](../../docs/defense/methodology.md). Units: xG against per 60 5v5 minutes; positive = fewer xG against than an average skater.

## Gates

| Gate | Result | Pass |
|---|---|---|
| G1 headline beats B0 on test pairs | ΔMSE -0.0480 (SE 0.0071) | ✅ |
| G2 YOY ≥ 0.25 (season-centred, ≥500 min) | F 0.41 · D 0.38 | ✅ |
| G3 YOY ≥ plain xGA RAPM | trivially met: with M0 the ρ = 0 headline is xGA RAPM (F 0.41 · D 0.38) | ➖ |
| G4 component coefficients sum to xGA RAPM | max abs diff 3.3e-13 | ✅ |
| G5 ledger → stint assignment | 0.9901 of 5v5 ledger value | ✅ |

## Data

| season | stints | hours | xg | goals |
|---|---|---|---|---|
| 2017 | 290199 | 1006.632 | 4783.633 | 4893.000 |
| 2018 | 289438 | 1011.522 | 4781.749 | 5079.000 |
| 2019 | 227275 | 783.209 | 3725.553 | 3909.000 |
| 2020 | 198963 | 682.682 | 3150.076 | 3348.000 |
| 2021 | 356106 | 1042.893 | 5234.954 | 5443.000 |
| 2022 | 442311 | 1035.482 | 5404.538 | 5432.000 |
| 2023 | 442927 | 1037.852 | 5055.867 | 5312.000 |
| 2024 | 433299 | 1010.812 | 4887.652 | 4954.000 |
| 2025 | 444368 | 1044.743 | 5436.834 | 5353.000 |

## Ridge penalties (game-grouped CV, train seasons)

| target | alpha |
|---|---|
| start_turnover | 1000.000 |
| start_faceoff | 1000.000 |
| start_recovery | 1000.000 |
| start_unobserved | 3000.000 |
| move | 3000.000 |
| term | 1000.000 |
| quality | 3000.000 |
| xg | 1000.000 |

## Model selection (D1, D2) and held-out test (G1)

Minute-weighted MSE of next-season stint xGA/60. Lower is better. B0 = league average defense.

| model | validation | test | test Δ vs B0 | SE | test goals |
|---|---|---|---|---|---|
| B0 | 133.103 | 132.072 | None | None | 1247.460 |
| M0 | 133.056 | 132.024 | -0.048 | 0.007 | 1247.420 |
| M1 | 133.070 | 132.038 | -0.034 | 0.006 | 1247.421 |
| M2 | 133.063 | 132.020 | -0.053 | 0.005 | 1247.405 |
| M3 | 133.058 | 132.019 | -0.053 | 0.007 | 1247.409 |

M3 stacking weights: start_turnover 1.65, start_faceoff 2.49, start_recovery 0.53, start_unobserved 1.70, move 1.68, term 1.98, quality 1.52

D2 (prior ρ, validation MSE): ρ=0.0: 133.05591, ρ=0.5: 133.03565, ρ=1.0: 133.01908

Test pairs separately: 2023-2024: B0 126.5438, M0 126.4695, M1 126.4890, M2 126.4743, M3 126.4676; 2024-2025: B0 137.4210, M0 137.3986, M1 137.4075, M2 137.3848, M3 137.3908

## Stability (independent ρ = 0 fits)

Season-centred YOY r:

| metric | F | D |
|---|---|---|
| DPM (headline) | 0.408 | 0.378 |
| xGA RAPM | 0.408 | 0.378 |
| DPM, team changers | 0.318 | 0.307 |
| xGA RAPM, team changers | 0.318 | 0.307 |
| xGF RAPM (offense, same fit) | 0.542 | 0.346 |
| component: start_turnover | 0.423 | 0.316 |
| component: start_faceoff | 0.418 | 0.126 |
| component: start_recovery | 0.305 | 0.318 |
| component: start_unobserved | 0.188 | 0.199 |
| component: move | 0.140 | 0.158 |
| component: term | 0.297 | 0.278 |
| component: quality | 0.247 | 0.343 |

Split-half r (odd vs even games):

| season | F DPM | F xGA RAPM | D DPM | D xGA RAPM |
|---|---|---|---|---|
| 2017 | 0.410 | 0.410 | 0.358 | 0.358 |
| 2018 | 0.392 | 0.392 | 0.343 | 0.343 |
| 2019 | 0.347 | 0.347 | 0.314 | 0.314 |
| 2020 | 0.188 | 0.188 | 0.304 | 0.304 |
| 2021 | 0.429 | 0.429 | 0.387 | 0.387 |
| 2022 | 0.405 | 0.405 | 0.407 | 0.407 |
| 2023 | 0.393 | 0.393 | 0.449 | 0.449 |
| 2024 | 0.405 | 0.405 | 0.356 | 0.356 |
| 2025 | 0.341 | 0.341 | 0.407 | 0.407 |

## Reported (not gated)

| check | F | D |
|---|---|---|
| r with PM-EPA/60 (overlap) | 0.06 | 0.14 |
| r with xGF RAPM | -0.06 | 0.01 |
| r with −(raw on-ice xGA/60) | 0.67 | 0.61 |

Team level (season-centred): r(Σ TOI-weighted skater DPM, −team 5v5 xGA/60) = 0.98

## Face validity, 2025-26 (≥500 5v5 min; inspected, never tuned)

**Defense: top 15**

| player_name | team | toi_5v5 | dpm_per60 | dpm_lo | dpm_hi | xga_rapm_per60 |
|---|---|---|---|---|---|---|
| Adam Pelech | NYI | 1392.900 | 0.490 | 0.373 | 0.578 | 0.180 |
| Moritz Seider | DET | 1555.083 | 0.457 | 0.309 | 0.606 | 0.300 |
| Jonas Brodin | MIN | 1074.517 | 0.442 | 0.307 | 0.600 | 0.253 |
| Artem Zub | OTT | 1342.200 | 0.423 | 0.302 | 0.577 | 0.160 |
| Jaccob Slavin | CAR | 650.550 | 0.411 | 0.261 | 0.539 | 0.201 |
| Connor Murphy | CHI | 1139.150 | 0.369 | 0.240 | 0.501 | 0.251 |
| Esa Lindell | DAL | 1405.300 | 0.348 | 0.220 | 0.458 | 0.182 |
| Brayden McNabb | VGK | 1075.617 | 0.345 | 0.230 | 0.455 | 0.123 |
| Trevor van Riemsdyk | WSH | 951.800 | 0.343 | 0.174 | 0.491 | 0.084 |
| Joel Hanley | CGY | 895.083 | 0.339 | 0.187 | 0.489 | 0.137 |
| Colton Parayko | STL | 1412.433 | 0.322 | 0.200 | 0.440 | 0.190 |
| Dylan Samberg | WPG | 1200.467 | 0.319 | 0.223 | 0.446 | 0.173 |
| Vincent Desharnais | SJS | 752.083 | 0.311 | 0.149 | 0.430 | 0.238 |
| Jonas Siegenthaler | NJD | 1367.683 | 0.304 | 0.179 | 0.438 | 0.008 |
| Olli Maatta | CGY | 656.250 | 0.285 | 0.145 | 0.425 | 0.042 |

**Defense: bottom 10**

| player_name | team | toi_5v5 | dpm_per60 | dpm_lo | dpm_hi | xga_rapm_per60 |
|---|---|---|---|---|---|---|
| Erik Karlsson | PIT | 1250.700 | -0.197 | -0.337 | -0.060 | -0.018 |
| Mason Lohrei | BOS | 1058.367 | -0.199 | -0.328 | -0.072 | -0.099 |
| Logan Mailloux | STL | 984.400 | -0.202 | -0.354 | -0.041 | -0.086 |
| Elias Pettersson | VAN | 938.933 | -0.203 | -0.364 | -0.030 | -0.215 |
| Simon Nemec | NJD | 1163.000 | -0.240 | -0.436 | -0.089 | -0.114 |
| Ben Chiarot | DET | 1457.733 | -0.245 | -0.402 | -0.081 | -0.233 |
| Morgan Rielly | TOR | 1306.367 | -0.251 | -0.364 | -0.133 | -0.159 |
| Artyom Levshunov | CHI | 1023.917 | -0.293 | -0.437 | -0.155 | -0.262 |
| Tony DeAngelo | NYI | 1238.533 | -0.471 | -0.638 | -0.348 | -0.205 |
| John Klingberg | SJS | 916.933 | -0.527 | -0.723 | -0.354 | -0.189 |

**Forwards: top 15**

| player_name | team | toi_5v5 | dpm_per60 | dpm_lo | dpm_hi | xga_rapm_per60 |
|---|---|---|---|---|---|---|
| Mark Jankowski | CAR | 643.967 | 0.435 | 0.321 | 0.554 | 0.171 |
| Garnet Hathaway | PHI | 584.650 | 0.378 | 0.268 | 0.500 | 0.164 |
| Sam Reinhart | FLA | 858.683 | 0.347 | 0.204 | 0.466 | 0.103 |
| Jack Drury | COL | 954.817 | 0.338 | 0.208 | 0.466 | 0.277 |
| Noah Cates | PHI | 953.900 | 0.331 | 0.215 | 0.441 | 0.166 |
| Pontus Holmberg | TBL | 881.000 | 0.328 | 0.226 | 0.450 | 0.237 |
| Alexander Wennberg | SJS | 1080.083 | 0.314 | 0.183 | 0.465 | 0.142 |
| Mark Stone | VGK | 791.667 | 0.294 | 0.144 | 0.412 | 0.142 |
| Casey Cizikas | NYI | 801.500 | 0.277 | 0.170 | 0.392 | 0.151 |
| Seth Jarvis | CAR | 906.950 | 0.274 | 0.146 | 0.411 | 0.028 |
| Jason Dickinson | CHI | 817.483 | 0.273 | 0.136 | 0.439 | 0.157 |
| Marcus Foligno | MIN | 608.700 | 0.270 | 0.157 | 0.383 | 0.071 |
| Oskar Bäck | DAL | 703.367 | 0.268 | 0.130 | 0.384 | 0.186 |
| Eetu Luostarinen | FLA | 932.317 | 0.262 | 0.101 | 0.404 | 0.116 |
| Noel Acciari | PIT | 707.400 | 0.259 | 0.132 | 0.388 | 0.167 |

**Forwards: bottom 10**

| player_name | team | toi_5v5 | dpm_per60 | dpm_lo | dpm_hi | xga_rapm_per60 |
|---|---|---|---|---|---|---|
| Brad Marchand | FLA | 608.400 | -0.315 | -0.482 | -0.134 | -0.292 |
| Patrick Kane | DET | 903.833 | -0.316 | -0.442 | -0.188 | -0.040 |
| Oliver Kapanen | MTL | 919.617 | -0.316 | -0.454 | -0.143 | -0.244 |
| Evander Kane | VAN | 983.100 | -0.318 | -0.508 | -0.133 | -0.301 |
| Vinnie Hinostroza | MIN | 657.533 | -0.334 | -0.477 | -0.159 | -0.293 |
| Miles Wood | CBJ | 573.217 | -0.364 | -0.500 | -0.213 | -0.102 |
| Chandler Stephenson | SEA | 1081.150 | -0.370 | -0.489 | -0.228 | -0.257 |
| Connor Bedard | CHI | 1083.817 | -0.425 | -0.601 | -0.256 | -0.283 |
| Anthony Duclair | NYI | 708.300 | -0.451 | -0.628 | -0.290 | -0.268 |
| Alex Ovechkin | WSH | 973.900 | -0.571 | -0.723 | -0.424 | -0.131 |
