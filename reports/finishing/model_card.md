# Model Card — Finishing Talent (FIN) v1.0 and Value Integration

*Format after Mitchell et al. (2019). Companion to the [PM-EPA](../model_card.md) and [DPM](../defense/model_card.md) model cards.*

## Model details

| | |
|---|---|
| Name | FIN: finishing talent above a finishing-only xG offset; plus the integration layer (`src/valuation/`) |
| Version | 1.0 (2026-09-29) |
| Offset | xG_f: gradient-boosted classifier, feature set F1 (shared xG features + previous-event ownership, lateral movement from the shooter's own team's previous event, time since faceoff, skater counts, recent own-team attempts). Cross-fitted by season. |
| Finishing model | Hierarchical logistic model: `logit P(goal) = logit(xG_f) + season × strength intercepts + fringe-group effect + shooter persistent effect + shooter-season deviation + goalie persistent + goalie-season`. Penalized (Laplace) mixed model; variance components by SQUAREM-accelerated empirical-Bayes EM; intervals from the Laplace posterior; full-Bayes NUTS check in an optional environment. |
| Unit | Goals (5v5) above a league-average shooter, and above the replacement pool; wins = goals ÷ 5.90 |
| Code | `python -m src.finishing.run` then `python -m src.valuation.run` (`make finishing valuation`) |
| Spec | [docs/finishing/methodology.md](../../docs/finishing/methodology.md): pre-registered rules and a deviations log |

## Intended use

* The **finishing axis** of the planned all-in valuation: how much better or worse than an average shooter, on the same shots against the same goalies, a skater converts unblocked attempts.
* A player-level **prior hook**: `PriorSpec` / `CovariatePrior` plug external per-player covariates (planned: NHL EDGE shot-speed and skating metrics) into the prior mean of the persistent shooter effect with no model change.
* The combined table `data/outputs/player_value_season.csv` (possession, defense, finishing, total; goals and wins above replacement).

## Out-of-scope uses

* **Single-season finishing verdicts.** The median 95% interval for a regular is about ±2.9 goals; 15% exclude zero.
* **Goaltending value.** xG is pre-shot; the goalie term is an adjustment only.
* **Power-play or other non-5v5 value** (memo columns only, no intervals).
* **Forecasts from the published per-season estimates.** They use later seasons too (retrospective smoothing); use forward fits for forecasting.
* **Summing the three components as if they were equally calibrated** (see Caveats).

## Training data

947,091 unblocked, non-empty-net skater attempts, 2017-18 → 2025-26 (63,239 goals; 1,892 shooters; about 100 goalies per season). Every one of the 66,976 goals in the event data is accounted for: 63,239 in the table, 3,732 empty-net, 5 without coordinates. Excluded attempts: empty-net (`opp_goalie == 0`), blocked, shootouts, 61 attempts by goalies as shooters, 1 with a missing coordinate.

## Performance

All figures from [validation_report.md](validation_report.md) and [xg_validation.md](xg_validation.md).

| Component | Result |
|---|---|
| xG_f vs shared-feature xG, held-out log-loss (paired) | validation z = 4.7, test z = 9.9 (small absolute gain: cross-fit log-loss 0.2154 vs 0.2163) |
| xG_f cross-fit, 5v5 Σxg ÷ Σgoals | 1.012 (gate 0.97–1.03) |
| Shrunk model vs no-player model, next-season log-loss (pooled 2023–25) | z = +7.2 |
| Shrunk vs raw smoothed G/xG | z = +8.1; raw is *worse* than no player information (z = −2.1) |
| Calibration slope of realized next-season excess goals on prediction | 0.93 (95% CI 0.75–1.10); raw 0.54 |
| Persistent talent SD (logit) | forwards 0.14, defensemen 0.11 |
| YOY r (≥ 200 attempts) | model 0.24, raw 0.25 (forwards 0.27) |
| Split-half r (Spearman-Brown) | model 0.34, raw 0.29 |
| Attempts to reliability 0.5 | analytic 570–650, empirical 754 (3–5 seasons for a regular) |
| Laplace vs full Bayes (NUTS), shooter SDs | median ratio 1.00 (5–95%: 0.95–1.08) |
| Goals per win (data-derived) | 5.90 (95% CI 5.66–6.13), R² 0.91 |

## Combining with PM-EPA and DPM

The three ledgers are disjoint by construction (`goal − V_dec` splits into PM-EPA, an excluded shot-quality term, an unclaimed context term, and FIN; a unit test checks it). Replacement level is one definition applied to all three: top 12 forwards and 6 defensemen per team by TOI are regulars, the rest are the replacement pool, and each component's replacement rate is estimated from that pool without shrinkage bias (see methodology §9).

## Ethical considerations

* Named athletes are rated; publish intervals with any ranking.
* Finishing is the residual of a model that cannot see screens, traffic or passing. Players in roles that create shots xG undervalues can look like poor finishers.
* Replacement level is defined by TOI rank, which correlates with quality, and by team depth; players on deep teams are more often labeled fringe.

## Caveats and recommendations

* **Scale mismatch.** A team-level check gives joint slopes of 1.68 (possession), 1.18 (defense) and 1.46 (finishing) on team 5v5 goal differential. PM-EPA in particular is under-scaled relative to goals. Do not add the columns as if equally calibrated until the joint calibration step is done. No rescaling has been applied.
* **DPM replacement level is pooled across seasons** (the default), because the per-season estimate is noisy; the per-season version is `total_gar_perseason`.
* Prefer multi-season windows for finishing decisions.
* Re-validate xG_f when event definitions change (as in 2024-25).
