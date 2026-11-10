# Game-level model with player ratings that update during the season — methodology and pre-registration

*Written 2026-10-01 before any game-level model was built or scored. The design, the tuning protocol and the gates in §7 are fixed here; changes after results are
seen are logged in §10. Motivation: the standings backtest has 160 team-seasons and cannot resolve realistic gains (docs/standings/improvements.md); the same seasons hold about 6,500 games.*

## 1. Question

Predict the home team's win probability for every regular-season game from the players who dress, using only information available before puck drop. Does a model built from
player ratings that carry over from the player-value work and **update game by game** predict better than a well-tuned team rating (the standard public benchmark)?
And how much comes from (a) the preseason player prior, (b) in-season updating, (c) attributing updates to players rather than teams, (d) knowing the lineup?

## 2. Information rules

* **Known before a game:** all earlier games; the skaters who dress and the starting goalie (public shortly before the game; a closing betting line also knows them); the schedule (rest).
  A variant that uses the *previous* game's lineup instead measures what lineup news is worth.
* **Not used:** anything from the game itself or later; the season's own player values (they are estimated on the whole season). Ice time in the game being predicted is *projected* from the player's earlier games.
* **Preseason priors** for season T come from models trained on seasons < T (the projection layer and the portable team residual of docs/standings/improvements.md, coefficient fixed at 1; the goalie rolling rates fitted through T − 1).
* **Inherited leakage (declared):** the season < T player values behind the priors use hyperparameters fitted on all seasons (rolling-refit persistent components excepted); the shot xG model is cross-fitted. Same as the standings backtest.

## 3. The model (P)

* **State.** Each skater has a rating θ_i in goals above replacement per 60 minutes of his ice time. At the start of season T, θ_i = projected rate + 60 × (last season's team residual per skater-minute of his team). Each goalie has a
  save rating per unblocked shot faced: (n0 × prior rate + goals saved above expected so far this season) / (n0 + shots faced so far this season), prior = rolling rate through T − 1 (replacement rate if none).
* **Team strength for a game.** S = Σ over dressed skaters of θ_i × m_i / 60, where m_i is the skater's projected minutes: the mean of his ice time in his previous up to 10 games this season (else his mean per game last season, else the position median),
  rescaled so the team's skater minutes sum to 300. G = starting goalie's rating × the league-mean unblocked shots faced per team-game last season.
* **Expected goal differential.** μ = (S + G)_home − (S + G)_away + h, with h the mean home margin in 2017–2018.
* **Win probability.** P(home win) = logistic(b0 + b1 μ + b2 × home on a back-to-back + b3 × away on a back-to-back), coefficients fitted on earlier seasons' games (expanding; seasons 2019 onward).
* **Update after the game.** Observed margin y = α × goal differential + (1 − α) × expected goal differential (shootouts count as 0 goals). Error e = y − μ. Each dressed skater's rating moves in proportion to his actual minutes x_i = minutes / 60:
  θ_i ← θ_i ± λ e x_i / Σ_j x_j² (plus for the home team, minus for the away team), so each team's strength with that lineup moves by λ e. Ratings follow players across teams.
* **Hyperparameters** λ ∈ {0, 0.01, 0.02, 0.04, 0.06, 0.10}, α ∈ {0, 0.25, 0.5, 0.75, 1}, n0 ∈ {500, 1500, 3000}: chosen once by log loss on **2019 and 2020 only**, then frozen for the test seasons.

## 4. Benchmarks and ablations (same tuning protocol, same logistic step)

* **G0** home-ice constant. **G1** team rating on goals: R ← R ± k e / 2 with e = goal differential − (R_home − R_away + h), ratings multiplied by ρ between seasons, started at 0 in 2017; k ∈ {0.01, 0.02, 0.03, 0.05, 0.08}, ρ ∈ {0.5, 0.7, 0.85, 1}.
  **G2** the same with the blended margin (α from the same grid). G1 and G2 are the standard to beat.
* **Ablations of P:** **P-pre** (no in-season update: λ = 0 and the goalie fixed at its prior); **P-team** (the same preseason lineup prior, but the in-season error updates a team-level state instead of the players);
  **P-prev** (P with the previous game's lineup and goalie instead of the actual ones).

## 5. Evaluation

Test seasons 2021–2025, all regular-season games with complete data. Primary score: mean log loss of the home-win outcome (overtime and shootout results count as wins and losses). Also Brier score, accuracy,
calibration slope (logistic regression of the outcome on the predicted logit) and log loss by season. Paired z: mean over its standard error of the per-game difference in log loss (games treated as independent; a cluster-by-date check is reported).

## 6. Market comparison (built in, run when odds are supplied)

If `data/external/odds.csv` exists (columns `game_date, home, away, p_home_close`, the de-vigged closing probability), the report adds the market's log loss on the same games and the paired comparison.
No odds data are in the repository, so no claim about the market is made until then. Expectation stated in advance: closing lines are very efficient; matching them within about 0.005 log loss would be a strong result.

## 7. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| GM-1 | P beats G0 (paired z > 2) |
| GM-2 | P beats the better of G1 and G2 (paired z > 2) |
| GM-3 | in-season updating helps: P beats P-pre (paired z > 2) |
| GM-4 | player attribution helps: P beats P-team (paired z > 2) |
| GM-5 | calibration slope of P in [0.8, 1.2] |

Reported, not gated: P against P-prev (the value of lineup news); the tuned hyperparameters; a season-points summary (sum of each team's predicted win probabilities, preseason-only version, against actual points).

## 8. What the result can say

GM-2 is the claim that matters: a pass means player-level structure beats a tuned team rating on about 6,500 games. GM-3 and GM-4 say where it comes from. A GM-2 failure would mean the team rating already contains
what the player ratings offer at game level, and the player model's use is explanation, not prediction.

## 9. Limitations stated in advance

Tuning uses two unusual seasons (2019-20 cut short, 2020-21 with 56 intra-division games); a game's dressed lineup is taken from the game record (who played), which matches what is known at puck drop but not hours earlier;
no travel, altitude, referee or motivation effects; the update treats all strengths alike and credits every dressed skater by ice time (no line or matchup structure); empty-net goals are in the margin; score effects are not removed from expected goals.

## 10. Deviations log

*(any change to §3–§7 after results are seen is recorded here with its reason)*

1. **Reproducibility fix (2026-10-03).** When two goalies faced the same number of shots in a game the "starter" was chosen by an unstable sort, so log loss varied by about 0.00003 between runs (comparisons within a run were unaffected).
   Ties are now broken by goalie id and the model was rerun: every gate outcome and every z to two decimals is unchanged (GM-2 z = 1.99); log losses moved in the fifth decimal. The test has 6,502 games.
2. **Goalies without a rolling rate (2026-11-10; live forecasts only, the backtest above is not rerun).** §3 gives a goalie with no rolling rate the replacement rate. A review of the season's largest gaps to the market found two groups there who are not replacement-level (`python -m src.goaltending.unknown`, `reports/metrics/gk_unknown.json`; forward folds, fit on seasons ≤ T, look at T + 1):
   * **Returning goalies** (a fitted career effect but no shots in season T, so no rolling rate; 40 goalie-seasons, 13,758 shots): they saved −0.0011 goals per shot against average. Their career effect predicted −0.0006; the replacement rate assumed −0.0091 (squared error of the season totals 1,103 against 2,177). This was an implementation gap: the rolling rates were written only for goalies who played in T.
   * **Debut goalies** (no NHL shots before the season; 126 goalie-seasons, 37,540 shots): −0.0021 per shot (95% interval −0.0047 to +0.0003) against a replacement rate of −0.0092. Those who faced under 300 shots were at −0.0103, those with 300 or more at +0.0002; a forecast cannot tell them apart beforehand.
   
   In the live forecasts a returning goalie now gets his career effect and a goalie with no history gets the mean of past debut seasons (−0.0021 per shot, about −0.09 goals a game where the replacement rate was −0.38). Game-model backtest with both, hyperparameters frozen: log loss 0.65854 → 0.65838, z = 0.56 (returning alone 0.39, debut alone 0.54), so no gain in game prediction is claimed.
   The historical tables (`gk_rolling_rates.parquet`) and every backtest in this repository keep the original rule. This season it had applied to Semyon Varlamov (11,414 career shots, none in 2025-26) and Devon Levi, and produced the four largest gaps to the market in the log. Tested alongside and **not adopted**: leaving the goalie term out of what the expected-goal update target is compared with (z = 1.19, and late-season residuals did not support it).
