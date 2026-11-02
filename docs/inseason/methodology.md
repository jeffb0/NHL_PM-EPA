# In-season player updates: finishing and individual results — methodology and pre-registration

*Written 2026-11-02 before anything here was built, fitted or scored. Owner's request: let a player's rating within the season respond to (a) how he is finishing, for proven finishers whose goals regularly exceed the expected goals of their attempts, and (b) his own results, so a player in decline (or a breakout) is found individually rather than through his team.
The design, tests and rules below are fixed here; changes after results are seen are logged in §9.*

## 1. What the game model does now, and the two gaps

The game model (docs/gamemodel/methodology.md §3) starts each skater at a preseason rating (the projection: the last three seasons weighted 5/4/3, shrunk by minutes, with position-specific aging; finishing enters through its rolling-refit estimate) and after every game moves every dressed skater by the team's surprise in expected-goal differential (λ = 0.04, α = 0), shared in proportion to ice time. Thin-history players also get an empirical-Bayes adjustment from their own early games (docs/prospects/hybrid.md).

* **Gap F (finishing).** A skater's finishing estimate is fixed at its preseason value. Shots taken this season, and whether they went in, never change it. The finishing model's own variance components say part of finishing is season-specific (shooter-season deviation SD 0.09 logit for forwards, 0.13 for defensemen, against persistent talent 0.14 and 0.11; docs/finishing), so a season's shots carry some information about the rest of that season.
* **Gap P (individual results).** Outside the thin-history group, nothing a player does individually moves his rating: the update sees only the team's result and credits it by ice time (the declared limitation "no line or matchup structure"). A declining veteran on a line that is doing well moves up with it.

A third question is checked first because both updates depend on it:

* **Check D0 (target consistency).** Ratings include finishing (goals above what the attempts were worth), but the in-season update target is expected-goal differential, which by construction excludes finishing. If this is a material mismatch, a team with good finishers is expected to beat its expected-goal differential by its finishing, never does on that target, and the update steadily removes the finishing from its players' ratings. D0 measures this on the test seasons: the mean update error e per game regressed on the dressed lineup's summed preseason finishing (goals per game), and the fraction of each skater's preseason finishing component remaining at game 82 on average. Reported, not gated; it decides whether U-F needs the finishing-adjusted target of §2.3.

## 2. Update F: in-season finishing

### 2.1 Estimate

For each skater and each completed game, the finishing model (docs/finishing §5, model A: logit P(goal) = logit(xG_f) + season × strength intercept + shooter effect + goalie effect) is updated with his unblocked, non-empty-net attempts so far this season:

* **Prior** for his effect this season: the forward-fit estimate through the previous season (mean m₀, SD s₀; the same fit that produces the rolling-refit finishing value in his preseason rating), widened by the season deviation: variance s₀² + σ_d² (σ_d by position from the production variance components).
* **Posterior** after his attempts this season (one Laplace step around m₀; goalie effects and intercepts fixed at their preseason values; the current season's intercepts set by the league-level calibration of docs/finishing §7): `m₁ = m₀ + Σ(goal − p) / (1 / (s₀² + σ_d²) + Σ p(1 − p))`, with p each attempt's probability at m₀.
* **Rating change:** `Δθ_F = (m₁ − m₀) × π̄ × A`, where π̄ = his mean p(1 − p) per attempt and A = his attempts per 60 (this season's and last season's pooled, weighted by minutes), giving goals per 60. Added to his rating after each game, replacing the previous game's Δθ_F (it is a re-estimate, not an accumulation).

All strengths are pooled (the model's strength intercepts carry the level differences). Defensemen are included; their small samples shrink them almost to zero on their own.

### 2.2 Expectation stated in advance

Small. Information per attempt averages 0.0585; a forward with 60 attempts has ≈ 3.5 units against a prior precision of ≈ 55 (s₀ = 0.10, σ_d = 0.09), so his estimate moves about 6% of the way from prior toward his raw season. By 250 attempts it moves about 20%. Its effect on a team's game forecast is expected to be a few hundredths of a goal at most; a game-level gain is not expected to reach z = 2.

### 2.3 Finishing-adjusted update target (variant, used only if D0 shows a mismatch)

If D0's slope is clearly negative (z < −2), U-F is also run with the update target `xgd_F` = expected-goal differential + Σ over the game's attempts of (p at the shooter's current finishing estimate − xG_f): the goals the model expected the shooters to add. Both versions are reported; the gates apply to the one D0 selects.

## 3. Update P: individual results for every skater

The thin-history adjustment (docs/prospects/hybrid.md) extended to every skater, with its design and fitting kept as close as possible so the weights are learned from history rather than chosen:

* **Who:** all skaters (the thin-history models stay as they are for m_eff < 1,000; this adds models for the rest).
* **Signals** (cumulative over his games so far this season, per 60): the five hybrid signals (ice time per game, puck value, individual expected goals, points, 5v5 on-ice expected-goal differential), plus **5v5 on-ice expected-goal differential relative to his team with him off the ice** (separates him from the team, partly from his line), plus **ice time per game minus last season's** (a coach's verdict on decline or promotion).
* **Target:** his full-season value rate minus his preseason projection.
* **Fit:** ridge by position and checkpoint n ∈ {5, 10, 20, 40}, λ by leave-one-season-out, rows weighted by season minutes, using seasons before the one forecast; separate models for ages ≥ 31 at season start and younger (decline is the case the owner raised).
* **Use:** after his g-th game his rating is prior + the fitted adjustment at the largest checkpoint ≤ g. The team-surprise update continues unchanged on top.

**Variant P-recent (reported):** the same signals over his most recent 20 games instead of the whole season, at checkpoints n ≥ 20, to test whether recent form says more than the season so far.

## 4. Data and protocol

Historical seasons 2017–2025 from data/processed (attempts with xG_f from fin_shots; player-game lines from player_game). Hyperparameters of the game model stay frozen (λ = 0.04, α = 0, n0 = 3,000). Update F has no tuned parameters. Update P's ridge penalties are chosen by leave-one-season-out within the training seasons. Test seasons 2021–2025 (6,502 games), every model for season T fitted on seasons < T. The logistic step is refitted on earlier seasons for each variant, as in the game model.

## 5. Tests

1. **Player level.** Rest-of-season value rate (games after the checkpoint), prior against prior + update, minute-weighted squared error, paired z by player-cluster bootstrap, at each checkpoint. For U-F the target is rest-of-season goals minus xG_f per attempt (log loss of the rest-of-season attempts at m₀ against m₁). Reported subgroups: forwards aged 31+; the top decile of preseason finishers; this season's Jets and Senators.
2. **Game level.** The game model with U-F, with U-P, and with both, against the current model, paired z on per-game log loss over the test games; calibration slope.
3. **Reported, not gated.** D0; the fitted U-P coefficients by checkpoint and age group; the live 2026 adjustments for named players the owner raised (Kyle Connor, Nino Niederreiter); against the market on this season's logged games (too few for any claim).

## 6. Gates and rules

| gate | rule |
|---|---|
| IS-1 | U-F improves rest-of-season finishing prediction at 100 and 200 attempts (paired z > 2 at both) |
| IS-2 | U-P improves rest-of-season value prediction at n = 10 and n = 20 (paired z > 2 at both) |
| IS-3 | the combined model beats the current model at game level (paired z > 2) |
| IS-4 | no harm: each update alone, game-level paired z > −2; calibration slope in [0.8, 1.2] |

**Adoption rule (mechanical).** Each update is implemented in the live forecasts if its player-level gate (IS-1 or IS-2) passes and its game-level z is above 0 with IS-4 holding. Failing that, it is not used. An improvement in game prediction is claimed only if IS-3 passes. Because U-P changes the rating of every skater, the no-harm bar of the thin-history rule (z > −2 with player-level z > −1) is not enough here.

## 7. What the result can say

IS-1 says whether a season's finishing tells us anything about the rest of that season beyond a player's history; the finishing variance components predict "a little". IS-2 says whether individual in-season results find decline or improvement that the team-surprise update misses. IS-3 is the claim that matters for forecasts. Passing IS-1 or IS-2 with no game-level gain is a plausible outcome: player-level information can be real and still too small, spread over 18 skaters, to change who wins.

## 8. Limitations stated in advance

The U-P target is the model's own season value estimate, and at n = 40 the signals overlap half of it, which flatters the fit. On-ice and relative signals still reflect linemates and deployment. Finishing is whatever xG_f cannot see (screens, pre-shot passes), and its xG_f slope drifts by season (docs/finishing). Survivorship: players who lose their place mid-season leave the target. Nine seasons, two shortened.

## 9. Deviations log

*(any change to §2–§6 after results are seen is recorded here with its reason)*

1. **D0 outcome, applied mechanically (2026-11-02; `python -m src.inseason.d0`, `reports/metrics/inseason_d0.json`).** No mismatch: the finishing slope is not negative. On the 6,502 test games the update error's slope on lineup finishing is +0.84 (SE 0.27), against −0.18 (SE 0.03) on the rest of the rating; gap +1.02, z = +3.5. The gap is positive in all five test seasons (+0.15 to +2.02; z above 2 in 2021 and 2023, below 1 in 2024 and 2025) and is concentrated after a team's first 20 games (first 20: +0.32, z = 0.5; after: +1.22, z = 3.8).
   Lineups with more finishing history do better on expected-goal differential than their rating predicts, rather than worse. Per player-season, the finishing part of the preseason rating is fully kept through the season (share left 1.06 ± 0.05; the rest of the rating 0.95 ± 0.01). The finishing part is the projection's fitted coefficient on the finishing history (about 0.9 for forwards, 0.2 for defensemen) times the player's history relative to the training mean: SD 0.034 goals per 60, 90th percentile +0.047, Kyle Connor (2025) +0.108. Under §2.3, U-F runs with the plain expected-goal target; the finishing-adjusted variant is not run.
   Reported, not acted on: the positive gap suggests the finishing history also carries information about creating chances, beyond what the projection's other components capture. It is outside this specification and is not used here.
2. **Update F, implementation notes (2026-11-02, before results were seen).** In `src/inseason/finishing.py`, the forward fit for season T is model A with the fringe block, trained on seasons < T, with the variance components of `fin_tune.json` (EM on 2017–2021, so the 2021 test season is inside the window that set them, as in the finishing model's own forward folds).
   s₀² is the persistent effect's posterior variance with the season deviations integrated out, from each training season's information: 1/s₀² = 1/σ_u² + Σ_s I_s / (1 + σ_d² I_s). The fit's dense posterior covariance (about 12,000 columns per fold) is not formed. Shooters with no history get the population prior.
   The fringe term is left out of both arms (it is a same-season label). Rest-of-season log loss is clustered by player-season. Attempts are ordered by game id, which follows the schedule.
3. **Update F outcome, applied mechanically (2026-11-02; `python -m src.inseason.finishing`, `reports/metrics/inseason_finishing.json`).** IS-1 fails narrowly. Rest-of-season finishing prediction improves at 100 attempts (z = 2.37, 2,418 player-seasons) but not by the margin required at 200 (z = 1.94, 1,001 player-seasons). The update's size is about right: the calibration slope of realized rest-of-season excess goals on the update's prediction is 0.91 at 100 attempts and 1.15 at 200; for defensemen alone it is 2.6, too few goals to read.
   At game level the update helps slightly but not reliably: log loss 0.65854 → 0.65838, z = 1.08 (clustered by date 1.02), calibration slope 0.91. Forecasts move little (SD 0.027 goals, largest 0.11). IS-4 holds for F. Under §6 **update F is not implemented**: its player-level gate did not pass. Nothing is claimed for game prediction.
4. **Exploratory, after the outcome (2026-11-02; owner's question, not pre-registered, not used for any decision).** Does the update matter more for players with thin history? IS-1's player-level test was split by career NHL games before the season and by attempts in the finishing model's history (which starts in 2017). z at 100 / 200 attempts:
   * rookies, 59 / 14 player-seasons: 0.36 / 0.10;
   * seasons 2–4 (1–245 games), 777 / 233: 1.68 / 1.70;
   * established (246+), 1,582 / 754: 1.73 / 1.25.
   By prior attempts: under 300, 0.37 / 0.63 (calibration slope 0.37 at 100); 300–799, 2.19 / 1.21; 800+, 1.62 / 1.42.
   No subgroup clears 2 at both checkpoints. The thinnest-history groups are the weakest: their updates are the largest (SD of the effect change 0.053–0.10 against 0.035–0.049 for the established) but hold up least. Their early finishing regresses more than the population prior allows. The signal is spread across experience levels, not concentrated in young players. Twelve subgroup tests were run on results already seen; a single z near 2 among them would be expected by chance.
5. **Update P, implementation choices (2026-11-02, before any P result was computed).**
   * **Player-level target.** §5 asks for the rest-of-season value rate, which does not exist: the components (defense, special teams, finishing) are estimated on whole seasons. The target is therefore the full-season value rate, as in docs/prospects/hybrid.md, which flatters later checkpoints (§8). The game-level test is the clean out-of-sample test.
   * **"The current model."** The baseline for both tests is the model the live forecasts use: priors from rate model v3 (with the pre-NHL features), plus the thin-history adjustment for m_eff < 1,000. U-P adds models for m_eff ≥ 1,000 only.
   * **Signal definitions.** Relative on-ice: 5v5 on-ice xGD per 60 minus the team's 5v5 xGD per 60 with him off the ice. The team's 5v5 totals are its skaters' on-ice sums divided by five, as in the weekly report. Ice-time change: minutes per game so far minus last season's mean (0 with no last season). The age group is set by age on 1 October of the season.
   * **Other choices.** Ridge models need at least 40 training rows per position × age group × checkpoint, as in hybrid; otherwise there is no adjustment. P-recent is reported at n = 40 only, because at n = 20 the last 20 games are the season so far. IS-3's combined model adds U-F's adjustment to U-P's.
6. **Update P outcome, applied mechanically (2026-11-02; `python -m src.inseason.individual`, `reports/metrics/inseason_individual.json`).**
   **IS-2 passes.** The full-season value prediction improves at every checkpoint (test seasons 2021–2025, established skaters with 200+ minutes):

   | n | player-seasons | z | share of the prior's error removed |
   |---|---|---|---|
   | 5 | 2,687 | 2.14 | 2.3% |
   | 10 | 2,687 | 4.67 | 6.0% |
   | 20 | 2,655 | 6.63 | 10.4% |
   | 40 | 2,451 | 7.88 | 13.9% |

   The gain holds for forwards and defensemen and for both age groups. Ages 31+: z = 2.19 at n = 10, 3.10 at n = 20. Forwards 31+: 1.71 and 2.33. At n = 40 part of the gain is overlap with the target (§9 item 5); at n = 10 the first games are about an eighth of the season.
   **P-recent is worse than the season so far.** At n = 40 the last 20 games predict less well than all 40 (z = −3.49), so the season-so-far signals are kept.
   **Game level.** U-P against the current model: log loss 0.65854 → 0.65835, z = 0.77 (by date 0.78); calibration slope 0.89 (current 0.91); forecasts move by SD 0.062 goals.
   F and P together: 0.65822, z = 1.07. **IS-3 fails**, so no improvement in game prediction is claimed. IS-4 holds for P.
   Under §6 **update P is to be implemented**: IS-2 passes, its game-level z is above 0, and IS-4 holds. Update F stays out, its player-level gate having failed.
7. **Update P in the live forecasts (2026-11-02).** `src/live/predict.py` reads the decision from `reports/metrics/inseason_individual.json` and fits the models on every completed season (`data/processed/inseason_training.parquet`). The established group is m_eff ≥ 1,000 entering 2026 with age on 1 October, and the season-so-far lines come from the live tables.
   As with the thin-history adjustment, the logistic step and margin calibration are fitted on the historical model without it. Market-test record: docs/market/methodology.md §8 item 15.
   First live values (games through 2026-11-01): 568 established skaters; adjustments SD 0.036 goals per 60, range −0.12 to +0.10. Kyle Connor −0.031 (prior +0.421), Nino Niederreiter −0.047 (prior +0.138), Mark Scheifele −0.040, Cole Perfetti +0.029; Josh Morrissey none yet (4 games, below the first checkpoint).
