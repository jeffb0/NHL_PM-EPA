# Whole-model validation — methodology and pre-registration

*Written 2026-09-29 before any whole-model validation was run. The gates in §5 are fixed here; changes after results
are seen are logged in §8.*

## 1. Question

Skater components (possession, defense, finishing, turnover, faceoffs, penalties, and the four special-teams states) have
each passed their own checks and been calibrated together. This validation asks the question a user of the total cares
about: **does the total of player values, as published (`total_gar_all`, weights 1), predict how teams actually do the
next season, and does it add information beyond simpler things a person could compute?**

Headline model: `total_gar_all` (uncalibrated, the default chosen 2026-09-29). Alternatives reported: `total_gar_all_cal`
(calibrated weights), 5v5-only `total_gar`, and the variants of §4.

## 2. Design

**Unit.** Team-season T (target season) with season T−1 as the information set. Test seasons T = 2020…2025 (six); the
first two pairs (T = 2018, 2019) only train. Target: the team's all-strength goal differential in T (all goals for minus
against at every event strength, shootouts excluded), in goals.

**Roster-weighted sum.** For a team in season T, `S = Σ_i r_{i,T−1} × m_{i,T}` over the skaters whose primary team in T is
that team: `r` is the player's total value in T−1 per modeled minute (5v5 plus special-teams minutes), `m` his modeled minutes
in T. This uses the actual season-T roster and usage ("oracle roster"), the standard way to test player values, and isolates
the value estimates from roster forecasting. Players with no T−1 value carry `r = 0` (replacement). A second version uses the
T−1 team and T−1 minutes (no oracle), reported.

**Goalie proxy (control, not part of the model).** No goaltending component exists yet, and goaltending dominates team
goal differential. Every regression therefore includes `G = Σ_g shots_faced_{g,T} × rate_{g,T−1}`, with `rate` the goalie's
prior-season (xG − goals) per shot faced shrunk with k = 1,000 shots (`rate = sum / (n + k)`; new goalies 0). All models and
baselines get the same `G`, so it cannot favor the model.

**Fitting.** OLS with intercept on all earlier target seasons (expanding window), predicting one test season at a time.
A second version fixes the model's slope at 1 (predicted difference = `S` plus an intercept from the training mean and the
fitted goalie term): if the components are in goals, no fitted slope is needed.

**Baselines (same protocol, same `G`).**

| id | predictors |
|---|---|
| B0 | goalie proxy only (and the intercept) |
| B1 | + team goal differential in T−1 (per game × games in T) |
| B2 | + team all-strength xG differential in T−1 (same scaling) |
| B3 | + naive roster-weighted on-ice xG differential per 60 in T−1 (the same sum with each player's raw on-ice all-strength xGD/60) |
| M | + `S` (headline) |
| M+B1 | + `S` and the T−1 team goal differential |

## 3. Variants (each replaces `S` in M; paired against the headline)

* calibrated weights (`total_gar_all_cal`); 5v5 total only (`total_gar`); original (non-role-adjusted) turnover;
* drop-one: the total without each of the ten components;
* **leakage-clean total:** the finishing and faceoff components and the finishing part of the special-teams components come
  from models that use every season (a persistent shooter or taker effect), so their season-(T−1) values contain information
  from later seasons. The clean total replaces them with their season-only unshrunk versions (`fin_raw_gar`, `fo_raw_gar`,
  `st_*_raw_gar`). Everything else in the total uses only data up to the season, apart from cross-fitted shot xG and
  pooled hyperparameters (no player identity). A full rolling refit of the whole pipeline is not done (limitation).

## 4. Statistics

Out-of-sample MSE over the 6 test seasons × teams; paired z over team-seasons for squared-error differences (positive favors
the first-named model); out-of-sample R² against B0; the calibration slope of actual on predicted (fixed-slope predictions);
season-by-season improvement over B0.

## 5. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| WM-1 | M beats B0 out of sample, paired z > 2 |
| WM-2 | M beats B3 (the naive roster-weighted on-ice xG differential), paired z > 2 |
| WM-3 | M+B1 beats B1 (the model adds information beyond the team's prior results), paired z > 2 |
| WM-4 | calibration slope of actual on the fixed-slope prediction within [0.7, 1.3] |
| WM-5 | the leakage-clean total beats B0 with z > 2, and its difference from the headline is not significant (\|z\| < 2); otherwise the headline is flagged leakage-sensitive |
| WM-6 | M improves on B0 in at least 4 of the 6 test seasons |

Reported, not gated: M versus B1 and B2; the variants; the no-oracle roster version; the fixed-slope MSE; ranked
component contributions from the drop-one table.

## 6. Limitations stated in advance

* Team-season goal differential is noisy (about 250 team-seasons in total, 6 seasons in test); modest R² gains are expected.
* Primary-team assignment misplaces the value of players traded mid-season.
* No goaltending component; the proxy `G` is a control and a crude one.
* The leakage handling is a sensitivity analysis, not a rolling refit; hyperparameters chosen on full data (for example the
  DPM ridge α and the finishing variance components) are shared across seasons.
* The validation cannot separate a player's talent from his team context beyond what the ridge components already do.

## 7. Deliverables

`src/wholemodel/`, `reports/wholemodel/validation_report.md` and metrics, tests, README section.

## 8. Deviations log

*(any change to §2–§5 after results are seen is recorded here with its reason)*

1. **Exploratory variant added after the WM-5 result.** WM-5 (the leakage-clean total) failed, but it swaps shrunk components for
   unshrunk ones, so it confounds leakage with noise. A `no_persistent_effects` total (no finishing, faceoffs or special-teams
   finishing) was added as an exploratory row. It does not change any gate's verdict. Whether the gate design was flawed is noted
   in the report; a rolling refit of the finishing and faceoff models is the proper resolution and is not done here.
