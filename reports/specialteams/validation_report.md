# Special-teams components — validation report

Spec and pre-registered rules: [methodology](../../docs/specialteams/methodology.md). Four components (power play, penalty kill, 4v4, 3v3) on the goals-above-replacement scale: strength-specific ridge RAPM on shot xG plus finishing above the fringe pool. PP and PK are gated; 4v4 and 3v3 are reported (few minutes; see the limitations).

## Gates

| Gate | Result | Pass |
|---|---|---|
| ST-1 coverage: shot xG assigned to a valid stint (≥ 90%) and non-5v5 skater-minutes in valid stints (≥ 85%) | xG: PP/PK 97.8%, 4v4 97.4%, 3v3 98.3%; minutes 90.7% | PP/PK ✅ · 4v4 ✅ · 3v3 ✅ · minutes ✅ |
| ST-2 beats the covariates-only baseline on test pairs (paired game-bootstrap z > 2) | power play z = 6.64, penalty kill z = 6.64, 4v4 z = 0.55, 3v3 z = 2.77 | power play ✅ · penalty kill ✅ · 4v4 ❌ · 3v3 ✅ |
| ST-3 year-over-year r of the shrunk net rate ≥ 0.20 for forwards and defensemen | power play F 0.28 / D 0.18; penalty kill F 0.30 / D 0.22; 4v4 F 0.04 / D 0.08; 3v3 F 0.17 / D 0.15 | power play ❌ · penalty kill ✅ · 4v4 ❌ · 3v3 ❌ |
| ST-4 team reconciliation slope in [0.5, 2.0] | power play 1.11 ± 0.06; penalty kill 0.52 ± 0.11; 4v4 0.64 ± 0.18; 3v3 0.59 ± 0.24 | power play ✅ · penalty kill ✅ · 4v4 ✅ · 3v3 ✅ |
| ST-6 state finishing values sum to the finishing model's non-5v5 total | max abs diff 3.6e-15 | ✅ |

**Gated failures (reported, not tuned away): power play ST-3.**

## Fit choices (pre-registered rules)

| model | ridge α (CV) | prior ρ | test MSE model | test MSE baseline | test games |
|---|---|---|---|---|---|
| PP and PK (mismatched states) | 300.000 | 0.500 | 397.302 | 398.043 | 2564 |
| 4v4 | 300.000 | 0.000 | 164.110 | 164.140 | 1079 |
| 3v3 | 300.000 | 1.000 | 330.541 | 331.108 | 559 |

## Coverage

| group | stints | hours | shot xG | goals |
|---|---|---|---|---|
| 3v3 | 29502 | 109.9 | 1327.1 | 1359.0 |
| adv | 467208 | 1891.4 | 19317.0 | 19205.0 |
| 4v4 | 39925 | 134.9 | 844.5 | 810.0 |

## Year-over-year stability (independent single-season fits)

| component | F r | F n | D r | D n |
|---|---|---|---|---|
| power play | 0.28 | 1289 | 0.18 | 375 |
| penalty kill | 0.30 | 564 | 0.22 | 669 |
| 4v4 | 0.04 | 380 | 0.08 | 555 |
| 3v3 | 0.17 | 248 | 0.15 | 133 |

## Team reconciliation

Team-season goal differential in the state regressed on the team's summed component value (primary team). Slopes compress toward 1 for shrunk sums, and special-teams goals are rare, so the intervals are wide.

| component | slope | SE | r | team-seasons | SD team GD (goals) | SD summed value |
|---|---|---|---|---|---|---|
| power play | 1.11 | 0.06 | 0.73 | 284 | 12.92 | 8.56 |
| penalty kill | 0.52 | 0.11 | 0.27 | 284 | 11.98 | 6.37 |
| 4v4 | 0.64 | 0.18 | 0.20 | 284 | 2.39 | 0.76 |
| 3v3 | 0.59 | 0.24 | 0.14 | 284 | 3.13 | 0.77 |

## Relation to the penalty component (reported)

Correlation of a player's component value per minute with the penalty component's drawn and taken rates (players with ≥ 100 minutes in the state). The penalty component values the extra special-teams time; these values are per minute, so a positive correlation with *drawn* reflects that the players who draw penalties are also strong in the state, not that time is counted twice.

| component | drawn/60 | taken/60 | players |
|---|---|---|---|
| power play | 0.44 | -0.03 | 2421 |
| penalty kill | 0.60 | -0.03 | 2052 |

## Scale

| component | qualified player-seasons | median minutes | SD goals | SD RAPM part | SD finishing part |
|---|---|---|---|---|---|
| power play | 2421 | 186.55 | 2.38 | 1.62 | 1.09 |
| penalty kill | 2052 | 151.73 | 1.49 | 1.43 | 0.12 |
| 4v4 | 1751 | 21.38 | 0.15 | 0.10 | 0.07 |
| 3v3 | 936 | 19.67 | 0.25 | 0.14 | 0.15 |

## Leaders, power play, 2025 (≥ 100 minutes)

| player_name | team | position | st_pp_min | st_pp_rapm_gar | st_pp_fin_gar | st_pp_gar |
|---|---|---|---|---|---|---|
| Jason Robertson | DAL | L | 352.97 | 6.43 | 4.89 | 11.31 |
| Leon Draisaitl | EDM | C | 263.98 | 4.30 | 5.94 | 10.24 |
| Connor McDavid | EDM | C | 329.75 | 6.45 | 3.38 | 9.83 |
| Kirill Kaprizov | MIN | L | 326.90 | 4.10 | 5.66 | 9.76 |
| Mark Stone | VGK | R | 230.17 | 6.64 | 2.75 | 9.39 |
| Nathan MacKinnon | COL | C | 369.85 | 5.48 | 3.72 | 9.20 |
| Jake Guentzel | TBL | C | 358.35 | 6.27 | 2.82 | 9.09 |
| Alex DeBrincat | DET | R | 304.18 | 5.08 | 3.99 | 9.07 |

Lowest:

| player_name | team | position | st_pp_min | st_pp_rapm_gar | st_pp_fin_gar | st_pp_gar |
|---|---|---|---|---|---|---|
| Artyom Levshunov | CHI | D | 202.82 | -1.70 | 0.00 | -1.69 |
| Brandon Montour | SEA | D | 150.65 | -2.15 | 0.35 | -1.79 |
| Alexis Lafrenière | NYR | L | 233.63 | -3.53 | 1.54 | -1.99 |
| Vincent Trocheck | NYR | C | 238.78 | -2.60 | 0.43 | -2.17 |
| Rasmus Dahlin | BUF | D | 279.08 | -3.18 | 0.81 | -2.37 |

## Leaders, penalty kill, 2025 (≥ 100 minutes)

| player_name | team | position | st_pk_min | st_pk_rapm_gar | st_pk_fin_gar | st_pk_gar |
|---|---|---|---|---|---|---|
| Brandon Hagel | TBL | L | 179.82 | 7.30 | 0.21 | 7.51 |
| Ilya Mikheyev | CHI | R | 211.35 | 7.17 | 0.32 | 7.49 |
| Anthony Cirelli | TBL | C | 203.32 | 7.03 | 0.21 | 7.24 |
| Noah Cates | PHI | L | 150.83 | 6.75 | 0.07 | 6.81 |
| Artem Zub | OTT | D | 263.72 | 6.36 | 0.00 | 6.36 |
| Ridly Greig | OTT | C | 181.73 | 6.18 | 0.05 | 6.23 |
| Ryan McLeod | BUF | C | 211.65 | 5.92 | 0.28 | 6.20 |
| Connor Dewar | PIT | C | 220.62 | 5.69 | 0.13 | 5.83 |

Lowest:

| player_name | team | position | st_pk_min | st_pk_rapm_gar | st_pk_fin_gar | st_pk_gar |
|---|---|---|---|---|---|---|
| Jared Spurgeon | MIN | D | 146.55 | -0.43 | 0.00 | -0.43 |
| Jamie Oleksiak | SEA | D | 137.77 | -0.75 | 0.00 | -0.75 |
| Martin Fehérváry | WSH | D | 249.57 | -0.87 | 0.01 | -0.86 |
| Calle Jarnkrok | TOR | C | 122.32 | -0.95 | 0.04 | -0.91 |
| Ryan Pulock | NYI | D | 233.00 | -2.22 | 0.00 | -2.22 |

## Limitations

* Shot xG only; no possession-value ledger in special-teams states.
* PP units are collinear (the same five skaters share ice time), so RAPM is a regularized estimate; PP1 defensemen are the weakest-identified group (stability r below the gate).
* 4v4 and 3v3 have about 7 median minutes per player-season; their values are dominated by shrinkage and finishing, their gates are not met, and they are published as requested with a low-information flag.
* The finishing part carries no interval (non-5v5 posterior draws were not saved); the component's interval is the ridge bootstrap only.
* Deployment is not valued: the component is per special-teams minute above replacement.

## Interpretation

**What passed.** After the stint fix logged in the methodology (§9, item 3), coverage is 97–98% of shot xG in every group and 91% of non-5v5 skater-minutes. The strength-specific model beats the covariates-only baseline out of sample for PP and PK (z = 6.64) and for 3v3 (z = 2.77, reported); it does not for 4v4 (z = 0.55). The state finishing values reproduce the finishing model's non-5v5 total to rounding error. The penalty-kill component clears the stability gate (year-over-year r 0.30 forwards, 0.22 defensemen), and the team-level slopes are inside the pre-registered band for all four components (PP 1.11 ± 0.06, PK 0.52 ± 0.11, 4v4 0.64 ± 0.18, 3v3 0.59 ± 0.24).

**What failed.** One gated check: **power-play stability for defensemen** (year-over-year r 0.18 against the 0.20 gate; forwards 0.28). It is reported and not tuned away. 4v4 and 3v3 stability (r 0.04–0.17) and the 4v4 out-of-sample test are also below their reporting thresholds; those two components are low-information by construction (median 21 and 20 minutes per player-season) and their values are mostly shrinkage and finishing.

**How large the improvement is.** The out-of-sample gain is statistically clear but small in size: test MSE 397.30 against 398.04 for the covariates-only baseline (0.19%). Stint-level xG per 60 is very noisy, and the model's player terms explain a small share of it. The stability correlations (0.2–0.3) say the same: special-teams talent is measurable but weakly. The PK slope of 0.52 sits at the low edge of its band and its team-level correlation is only 0.27, so PK goal differential is mostly not explained by summed player value (goaltending and luck dominate it).

**Scale and level.** Among skaters with at least 500 5v5 minutes, mean goals above replacement are 1.11 (PP), 1.30 (PK), 0.15 (4v4) and 0.13 (3v3) per player-season, so adding the four components moves the mean total from 2.28 to 4.96 goals (Spearman rank correlation between `total_gar` and `total_gar_all` is 0.85). The PP and PK levels are driven partly by the pooled fringe effects (a fringe forward on the PK allows 0.37 more xG per 60 and generates 0.88 fewer than the pool of regulars), which are estimated from few fringe minutes. That level has been reconciled only through the team slopes, not against an independent replacement-level benchmark, and should be read with that in mind.

**Overlap with penalties.** Value per minute correlates 0.44 (PP) and 0.60 (PK) with the penalty component's drawn rate and about −0.03 with the taken rate. The drawn-rate correlation is what one expects if good players both draw penalties and perform in the state; the per-minute construction means the extra time itself is not counted here. It is not evidence of double counting, and it is not evidence against it either: a state-by-state check with drawn penalties held fixed would be needed.

**What this changes in the table.** `total_gar` (5v5 plus penalties) is unchanged and the joint calibration still applies to it. `total_gar_all` adds the four components; it has not been calibrated, and the calibration's all-strength extension is the next step.

