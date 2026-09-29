# Projection and aging layer — methodology and pre-registration

*Written 2026-09-29 before any projection model was fitted. The gates in §6 are fixed here; changes after results are
seen are logged in §9.*

## 1. What is being built

For every skater, a forecast of his value in the next one to three seasons: a rate (goals above replacement per 60 modeled
minutes, all strengths) with a prediction interval, an expected-minutes projection, and the expected goals above replacement
(rate × minutes). The layer also produces position-specific **aging curves**. Skaters only; goalies are excluded (their aging is a
separate question).

The whole-model validation (docs/wholemodel) found that the total does not forecast next-season teams better than simple baselines,
and that finishing and faceoffs carry no forward information once leakage is removed. A projection layer that learns how much of
each component persists is the natural response, and this validation asks whether it helps.

## 2. Inputs (no future information)

* **Component rates** per season: the ten components of the all-strength total (possession, defense, finishing, turnover, faceoffs,
  penalties, PP, PK, 4v4, 3v3), each divided by the player's modeled minutes (5v5 plus special-teams stints) × 60. The finishing,
  faceoff and special-teams-finishing parts are the **rolling-refit** values (`wm_rolling_values`, refit on data up to each
  season), never the full-data values. Everything else uses only data up to its season.
* **Age** at 1 October of the season start (birth dates from the roster files), **position group** (F/D), height and weight.
* **History features** for target season T = t + k (horizon k = 1, 2, 3), from seasons ≤ t only. Marcel-style weights 5/4/3 on
  the last three seasons times minutes give the weighted component rates `r̄_c`; `M_eff = Σ_s m_s · w_s / 5` (minutes with older
  seasons discounted); the shrunk feature is `r̃_c = r̄_c × M_eff / (M_eff + M0)` with M0 = 1,000 minutes (sensitivity 500, 2,000).
  A player with no history has zero features and an indicator `no_history`.
* **Optional prior covariates** (`PriorSpec`-style hook for future EDGE tracking features): any (player_id, season) feature table joins
  the design. **No EDGE data is in this repository**, so none is used here; the hook is tested with a synthetic covariate.

## 3. Model

Separately for forwards and defensemen and for each horizon: weighted ridge regression of the target rate on
`[r̃_c (ten), no_history, age hinge basis (age − 27, (age − 24)+, (age − 28)+, (age − 32)+ at the target season), height, weight]`, weights
= target-season minutes, rows with ≥ 200 target-season minutes, features standardized, intercept unpenalized, λ by leave-one-season-out
cross-validation over training target seasons. Residual variance model `σ²(m) = a + b/m` fitted on training residuals gives 80% prediction
intervals. **Minutes:** a ridge regression of target-season minutes on last-season minutes, the two-season mean, the age basis and position, on
players with ≥ 200 minutes in the target season (conditional on playing; retirement and leaving the league are not modeled).

## 4. Evaluation

Expanding window: for target season T = 2020…2025 the model is fitted on transitions whose target season is before T and predicts T
(horizon 1; horizon 2 uses targets whose history ends before T − 1). Player-level squared error weighted by target minutes over rows with
≥ 300 target-season minutes; paired z from a player-cluster bootstrap of the difference. Team level: the whole-model M protocol with the
projected rates in place of last season's rates (`S_proj = Σ r̂_{i,T} m_{i,T}`, oracle roster as in docs/wholemodel).

**Baselines:** B-mean (position mean of the training target rates); B-last (last season's total rate); B-marcel (Marcel weights 5/4/3 over
three seasons times minutes, regressed to the position mean with a 1,000-minute ballast, no age term).
**Variants (each replaces M):** total rate only (no component decomposition), no age terms, no height/weight, M0 = 500 and 2,000.

## 5. Reported, not gated

Aging curves with player-cluster bootstrap intervals; the fitted component weights per position (which components persist);
horizon 2 and 3 accuracy; minutes-model error against last season's minutes; the no-oracle team forecast (projected minutes and previous team).

## 6. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| PR-1 | M beats B-mean at horizon 1 (paired z > 2) |
| PR-2 | M beats B-marcel at horizon 1 (paired z > 2) |
| PR-3 | M with the age terms beats M without them (paired z > 2) |
| PR-4 | calibration slope of actual on projected rate (minute-weighted, horizon 1, test seasons) in [0.8, 1.2] |
| PR-5 | 80% prediction-interval coverage on horizon-1 test rows in [70%, 90%] |
| PR-6 | team level: M with projected rates beats M with the rolling-refit last-season rates (paired z > 2) |

## 7. Limitations stated in advance

* Nine seasons and a shortened 2020 and 2021 give few transitions; young-player and old-player age effects are weakly identified.
* Survivorship: only players who play in the target season are in the data, so aging curves are conditional on staying in the league.
* Injuries, role changes, trades and linemate effects are not modeled beyond what the component rates already contain.
* Component rates are themselves estimates with error; the ridge and shrunk features address but do not remove errors-in-variables.
* Goalies are not projected; no EDGE tracking features are used (none available here).

## 8. Deliverables

`src/projection/`, `data/outputs/player_projection.csv`, `reports/projection/aging_curves.csv` and validation report, tests, README section.

## 9. Deviations log

*(any change to §2–§6 after results are seen is recorded here with its reason)*

1. **Training floor 200 → 20 target-season minutes (after PR-6 failed).** §3 trains on rows with ≥ 200 target-season minutes. Conditioning
   training rows on a variable that follows quality (minutes) can bias projections upward for low-usage players, and the team level sums
   projections over every roster player, so a floor of 20 minutes was tried as a revision. It did not rescue PR-6 (z −2.92 → −2.75). Both
   are reported side by side; the published projections use the revised floor. Minutes are still modeled conditional on ≥ 200 minutes.
2. **PR-6 implementation corrected.** PR-6 was first computed with the projected total's slope fixed at 1 (z = −4.0); §4 specifies the M protocol
   (slope fitted on earlier seasons), which is what the gate now uses (z = −2.75 revised). The fixed-slope figure stays in the report.
3. **Results-target variant added after PR-6 failed (exploratory).** The same ridge trained on next season's deployment-adjusted on-ice goal
   differential per 60, with the season's state shares as usage controls (the calibration extension's outcome). It tests whether the
   projection's team-level failure comes from predicting value *estimates* that are persistent partly through chained priors. It has no gate.
4. **Reproducibility fix.** The cluster bootstrap resampled players in an unstable group order; players are now sorted, so z values are
   reproducible (values changed in the second decimal).
