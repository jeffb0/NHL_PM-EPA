# Team standings projection from opening-night rosters — methodology and pre-registration

*Written 2026-09-30 before any standings forecast was built or scored. The gates in §6 are fixed here; changes after results are seen are
logged in §9. Purpose: decide whether the player-level value model plus the projection layer can forecast a season's standings from each team's
opening-night roster (target: the 2026-27 season), and how much to trust it.*

## 1. Question

For season T, using only information available before the season starts, forecast each team's points (scaled to 82 games), and from it a
ranking and playoff-line classification. The forecast is built bottom-up: project every rostered player's rate and minutes, add the goalies, sum to a
team goal differential, convert to points. The backtest asks whether this pipeline, run honestly on past seasons, beats what a team's own recent
results already say.

## 2. What earlier validation already tells us (stated in advance, so the result is not a surprise)

* Whole-model validation: given the *actual* roster and usage, last season's summed player values predict next-season team goal differential
  (RMSE 41.2 goals against 48.8 for a goalie-only model), but **do not beat the prior season's team goal differential** (z = −0.95) and add little to it
  (z = 1.25). Without the oracle roster (previous team, previous minutes) RMSE is 44.5.
* Projection layer: projections beat simple baselines at the player level, but **summed to teams they did worse than last season's rates** (PR-6 failed:
  RMSE 48.4 against 44.9, z = −2.75). An exploratory variant trained on next-season on-ice results instead of value estimates did better at team level
  (42.4; not gated, used usage controls that are unknown before a season).
* Expectation: this pipeline will roughly match, not clearly beat, a prior-results baseline. The gates are written so that a finding of parity
  is reported as parity. A forecast that only matches history is still useful if it adds information to history (the blend gate) and explains
  *why* a team is expected to move (aging, departures, arrivals).

## 3. Information rules (no future information)

* **Opening-night roster.** Players who dressed for the team in its **first two games** of season T (skaters and goalies). The skaters are ranked by
  projected minutes within position and the top 13 forwards and 7 defensemen are kept. A player who was injured or scratched in both of those games is missing;
  a player recalled for them is included. This is the closest reconstruction of an opening-night lineup the game data supports; the real 2026-27 forecast
  will use the actual published rosters. Everything else about a player (his history, age, size) is from before T.
* **Player features** from seasons ≤ T − 1 only, exactly as the projection layer (docs/projection): rolling-refit finishing and faceoff
  components, Marcel 5/4/3 shrunk history, age at 1 October of T, height and weight; players with no history enter with the no-history indicator and
  position-median age and size. Models for season T are trained on transitions whose target season is ≤ T − 1.
* **Goalies.** Rolling-refit persistent save rates fitted on data through T − 1 (`gk_rolling_rates`; a goalie with no prior estimate gets the fold's replacement rate).
* **Conversion from goal differential to points** and all baseline regressions use seasons < T only.
* Shortened seasons (2019-20 ended early, 2020-21 had 56 games) are scaled to 82 games. They are used in training where the model is fitted, never as test seasons.
* Known leakage carried over: the player values themselves use hyperparameters and component estimates fitted on all seasons, except the rolling-refit parts
  (finishing, faceoffs, special-teams finishing). The defense and special-teams RAPM estimates for season s are fitted with priors chained from earlier seasons
  only, but with hyperparameters chosen on the full panel. The standings backtest inherits this and cannot remove it.

## 4. The forecast

1. **Player rates.** Ridge projection (docs/projection, training floor 20 target-season minutes as published) of each rostered skater's all-strength rate
   (goals above replacement per 60 modeled minutes) for T.
2. **Minutes.** The projection layer's minutes model (conditional on playing) gives each kept skater's expected modeled minutes; within team and position the
   minutes are rescaled so each team's forward total and defense total equal the league-mean team totals of season T − 1 (per 82 games). No injury or games-missed model.
3. **Skater value** S = Σ rate × minutes / 60 (goals per 82 games).
4. **Goalie value** G = Σ over roster goalies of (expected share of the team's shots) × (league-mean team shots against per 82, season T − 1) × (rolling save rate).
   Shares are proportional to prior shots faced (season T − 1 plus 0.5 × season T − 2); a goalie with no history counts as 600 shots.
5. **Goal differential** M1: ĝd = (S + G) − league mean of (S + G) over the season's teams (no fitted coefficient; league goal differential sums to zero, so there is no intercept to fit).
6. **Points** p̂ = a + b ĝd, with (a, b) from the regression of actual points per 82 on actual goal differential per 82 in seasons < T (all 2017 onward).
7. **Uncertainty.** The standard deviation of the training-season errors of the same forecast gives an 80% interval; a season simulation (games, ties, overtime, schedule)
   is a second stage, built only if this stage is not rejected.

## 5. Models and baselines (all fitted on seasons < T; expanding window; test seasons 2021-2025, 160 team-seasons)

* **M1** (above, primary). **M2:** fitted slopes on S and G (expanding OLS of points per 82 on the two sums). **M3:** blend of M1's ĝd and the team's prior-season goal differential per 82 (expanding OLS).
* **Declared variants** (reported, not gated; declared now because of §2): **R**, the same pipeline with last season's rolling rate in place of the projected rate (no projection, no aging);
  **P-res**, the projection trained on next-season results (deployment-adjusted on-ice goal differential per 60) with last season's state shares as usage controls.
* **Baselines:** **B0** league mean; **B1** prior-season points per 82 (regressed to the mean); **B2** prior-season goal differential per 82 (regressed); **B3** prior-season expected goal differential per 82 (regressed).
  A team with no prior season (expansion) is given the league mean.
* **Diagnostics:** oracle-roster version (actual season-T minutes and the players who actually played) to isolate roster-uncertainty cost; Spearman rank correlation; mean absolute rank error;
  accuracy of the top-16 call (playoff line, league-wide); error by team over time.

Scoring: root mean squared error and mean absolute error of points per 82, paired z from the mean and standard error of the team-season difference in squared error.

## 6. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| ST-1 | M1 beats B0 (paired z > 2) |
| ST-2 | M1 is not worse than the best history baseline among B1–B3 (paired z > −2) |
| ST-3 | the blend M3 beats the best history baseline among B1–B3 (paired z > 2) |
| ST-4 | calibration slope of actual points on M1's predicted points in [0.8, 1.2] |
| ST-5 | 80% interval coverage of M1 in [70%, 90%] |

The 2026-27 forecast is published either way, **labelled with each gate's outcome**. If ST-2 fails the forecast is presented as weaker than last season's results alone; if ST-3 fails
it is not presented as adding information to them.

## 7. Limitations stated in advance

* Opening-night rosters are reconstructed from the first two games played; the real 2026-27 rosters will be published, so the backtest slightly understates roster knowledge and
  ignores preseason injuries it could not see.
* No games-missed or injury model, so forecasts assume a healthy roster; no trade-deadline or in-season roster change.
* The player values use the projection layer's total (`total_gar_all` with rolling-refit persistent components), **not** the owner-preferred position-weighted-plus-TR total (docs/transition §9), because
  the projection layer and its leakage controls are built on the former. Adding TR to the projection is a separate step.
* Coaching, systems and schedule strength are not modeled; points include overtime-loss points, which are partly luck.
* Five test seasons; a team-season SD of about 12 points means differences between forecasts of 1–2 points RMSE are not resolvable.

## 8. Deliverables

`src/standings/`, `data/outputs/standings_backtest.csv`, `reports/standings/validation_report.md`, tests, README; then, conditional on §6, the 2026-27 projection.

## 9. Deviations log

*(any change to §3–§6 after results are seen is recorded here with its reason)*

1. **Primary forecast changed from M1 to M1 + C1 + C3 (2026-09-30).** After a separate pre-registered test of three improvements (docs/standings/improvements.md), the portable team residual (C1) and
   deployment-aware minutes (C3) were implemented under that document's rules; the published 2026-27 forecast uses them (backtest RMSE 13.96 against 14.58 for M1; not a reliable improvement, z = 1.42; still not better than B2, z = −0.58).
   M1 stays in the output for reference. The §6 gates were not re-run for the new primary; its calibration slope is 0.84 (inside the ST-4 band) and it still fails the spirit of ST-3 (it does not beat last season's goal differential).

## 10. The 2026-27 forecast procedure (added 2026-09-30, after the backtest result, before any forecast was examined)

* **Data.** `python -m src.ingestion.rosters_next 2026` snapshots the published NHL rosters (dated folders under `data/raw/rosters_next/2026/`, never overwritten). The forecast is rerun from the latest snapshot;
  it should be rerun once after opening night with the dressed lineups, and each published forecast records which snapshot it used. The snapshot has no injury information.
* **Model.** The primary forecast is M1 exactly as in §4, with the goal-differential-to-points map fitted on all seasons through 2025 and the interval from the standard deviation of M1's 2019–2025 errors. M3 and B2 are shown for reference, fitted on all earlier seasons.
* **Differences from the backtest roster rule (implementation, logged):** (1) a published camp roster lists 22–33 players, not the players who dressed in two games, so the top 13 forwards and 7 defensemen by projected minutes are kept from the whole list;
  (2) only each team's two goalies with the most prior shots (new goalies count as 600) are kept, because the list includes prospects and AHL goalies. Goalie rolling save rates for 2026 come from the goaltending model refitted on shots through 2025 (same settings as the forward step).
* **Presentation rule from §6:** ST-3 failed, so the forecast is labelled as not adding validated information to last season's results; ST-4 slightly failed, so the forecast is slightly compressed (strong teams are under-predicted, weak teams over-predicted).
* **Outputs:** `data/outputs/standings_2026_forecast.csv`, `standings_2026_players.csv` (each team's projected players, minutes and goals above replacement), `reports/standings/forecast_2026.md`.
2. **Pre-NHL priors added to the rate model behind the forecast (2026-10-01).** docs/prospects: they improve player projections overall (z = 2.62) but not team forecasts (game level z = −0.25; standings 13.96 → 13.99, z = −1.39).
   Implemented under that document's pre-registered rule for sound methods; the effect on any team's 2026-27 forecast is at most about 0.7 points. The interval uses 13.99.
