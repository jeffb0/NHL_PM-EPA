# Model Card — PM-EPA v1.0

*Format after Mitchell et al. (2019), "Model Cards for Model Reporting".*

## Model details

| | |
|---|---|
| Name | PM-EPA: Puck-Movement Expected Goals Added |
| Version | 1.0 (2026-09-27) |
| Type | Possession-value (EPA-style) player metric. Three value functions: V_start (state value after gaining the puck), V_dec (decision-point value), V_att (shot-attempt value). All are histogram gradient-boosted regressors (Poisson loss) on location, state kind, strength and score, cross-fitted by season within scoring regime. A shot xG model (gradient-boosted classifier) supplies the reward. |
| Unit | Expected goals (xG); rates per 60 minutes of 5v5 time on ice |
| Code | this repository; `python -m src.pipeline` |
| Config | `config.yaml` (effective config of each run in `reports/metrics/run_log.json`) |
| License / data | Code MIT; data © NHL via public APIs, research use |

## Intended use

* Describing how much offensive **transition and possession value** a skater's observable puck actions add at 5v5, per 60 minutes, with uncertainty.
* Comparing skaters **within a season, position and scoring regime**, alongside other evidence (on-ice xG, RAPM, video, tracking where available).
* Research on possession value from event data; a baseline to beat once tracking-grade events are available.

## Out-of-scope uses

* **All-in-one player value or WAR.** PM-EPA excludes finishing, defense without the puck, penalties, faceoffs and goaltending.
* **Contract or salary decisions on PM-EPA alone.**
* **Fine-grained rankings inside the confidence interval.** Adjacent ranks are usually statistically indistinguishable.
* **Comparisons across the 2024-25 scorekeeping change**, or across rinks without adjustment.
* **Evaluating passers.** Passes are unobserved in public data.
* **Goaltenders**, which are excluded, and playoffs, which are not validated.

## Training data

* NHL regular seasons 2017-18 → 2025-26: 11,052 games, 3.44M events, 8.39M shifts, 1.93M inferred possessions.
* **Splits:** train 2017-21, validate 2022, test 2023-25 for model selection. Final models are cross-fitted, so season Y is valued by models trained on other seasons in the same scoring regime.
* **Known data issues:**
  * Scorer judgment and rink bias in giveaways, takeaways and hits.
  * A recording regime change in 2024-25 (giveaways ×2, takeaways −35%).
  * 151 games lack usable shift data; they are excluded from rates.

## Performance

<!-- BEGIN PERFORMANCE -->
All figures from `reports/validation_report.md` (full tables, CIs, per-season breakdowns).

| Component | Metric | Value |
|---|---|---|
| Shot xG, temporal test 2023-25 | log-loss / AUC / Σxg÷Σgoals | 0.229 / 0.751 / 1.13 (1.03 after isotonic) |
| Shot xG, cross-fitted | log-loss / AUC / Σxg÷Σgoals | 0.219 / 0.773 / 1.007 |
| V_start, test | R² vs constant / calibration slope | 0.226 / 1.004 |
| V_dec, test | R² vs constant | 0.42 |
| PM-EPA/60 YOY (season-centred) | F / D | 0.53 / 0.52 |
| PM-EPA/60 YOY, team changers (season-centred) | F / D | 0.37 / 0.30 |
| Split-half (≥800 min, half-season r) | F / D | 0.56 / 0.63 |
| Convergent r (G+A1/60; on-ice xGF/60) | F | 0.40; 0.63 |
| Discriminant r (individual SH%; G − ixG) | F | 0.15; −0.004 |
| r with 5v5 offensive xG RAPM (3 windows) | F / D | 0.59–0.71 / 0.42–0.62 |
| Team PM-EPA vs team xGF | r | 0.69 |

**Performance across groups.** Defense rankings are more face-valid than forward rankings. Special-teams rates rest on thinner surfaces (PP V_dec has 5.6% radial-monotonicity violations). All within-regime pairs have YOY in 0.43–0.61.
<!-- END PERFORMANCE -->

## Ethical considerations and bias assessment

* **People are being rated.** Public rankings of named athletes can affect reputations and careers. Every rate ships with a bootstrap 95% CI. We recommend never publishing ranks without intervals, or without the direct vs pooled split.
* **Measurement bias against puck-dominant players.** Their observable failures (giveaways) are fully charged, while much of their positive value (passes) is unobserved and diluted. The first build was strongly biased this way. The production design (V_dec, attempt/quality split, λ = 0) reduces the bias but cannot remove it with PBP data.
* **Scorer / rink bias.** Home-rink recording tendencies persist across seasons (validation report §9.3). Players on some teams are systematically advantaged or disadvantaged.
* **Role and deployment confounding.** Pooled components share credit with linemates. Stability partly reflects team and system, which is why team-changer stability is reported separately.
* **Position differences.** Forwards and defensemen have different baselines, so compare within position (percentiles are within position).
* **No protected attributes** are used or inferred.

## Caveats and recommendations

* Use `pm_direct_per60_ev` to see what is individually observed; `pm_pooled_per60_ev` behaves like an on-ice stat.
* The headline (λ = 0) barely penalizes defensive-zone giveaways; `pm_epa_per60_ev_lam1` is the risk-adjusted companion.
* Require ≥500 5v5 minutes, and prefer multi-season pooling for decisions.
* Re-validate whenever the NHL changes event definitions (as in 2024-25).
* Upgrade path: tracking-grade pass/carry events plug into the same engine and remove most pooled credit (`docs/pm_epa/limitations.md`, R1).
