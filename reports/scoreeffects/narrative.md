# Score-effects adjustment — results

Pre-registration: [docs/scoreeffects/methodology.md](../../docs/scoreeffects/methodology.md). Metrics: `reports/metrics/score_effects.json`; weights: `data/outputs/score_effect_weights.csv`.

## The factors (all seasons, 5v5 expected goals for per 60 minutes by the shooting team's score state)

| period | down 2+ | down 1 | tied | up 1 | up 2+ |
|---|---|---|---|---|---|
| 1 | 2.45 | 2.52 | 2.51 | 2.56 | 2.56 |
| 2 | 2.77 | 2.88 | 2.79 | 2.90 | 2.85 |
| 3 | 2.55 | 2.58 | 2.32 | 2.36 | 2.45 |

Weights (tied rate divided by the state's rate): third period 0.91 when down two or more, 0.90 when down one, 0.98 when up one, 0.95 when up two or more; first and second periods between 0.96 and 1.02.

## Gates

| gate | rule | statistic | result |
|---|---|---|---|
| SE-1 | the score-adjusted update improves the game model (z > 2) | z = 0.39 (0.65854 → 0.65852) | FAIL |
| SE-2 | no harm (z > −2) | z = 0.39 | PASS |

**Implemented by the pre-registered rule (sound method, no harm): in the forward forecasts from the 2026-10-05 run, and shown in every game report.**

## Interpretation

**Score effects in expected goals are real but small at 5v5.** The one large effect is the one the author described: a team trailing in the third period generates 10–11% more expected goals per minute than it would at even score. In the first two periods the differences are a few percent.
The familiar large score effect in hockey is in shot *counts*; expected goals already discount much of it, because a trailing team's extra shots are mostly low-quality, while a leading team's fewer shots include more rush chances.
(Leading teams' rates are also slightly above the tied rate, partly because better teams are the ones that lead.)

**So the correction moves a game very little.** Game-level expected-goal differentials change by 0.03 goals on average and correlate 0.9995 with the raw ones. For WPG at DET on 2026-10-04 (Winnipeg led for 47 minutes) Detroit's 5v5 figure drops from 3.01 to 2.91 and Winnipeg's from 1.78 to 1.75 (all situations: 4.78 to 3.56, from 4.88 to 3.59): Detroit still clearly had the better of 5v5 play.
The methodology expected a leading team's shots to be marked *up*; in the data leading teams' rates are slightly above the tied rate, so their shots are marked down a little too (weights 0.95–0.98), and the adjustment works almost entirely through the trailing team's third-period discount.

**It does not improve the player-rating model** (z = 0.39), and re-tuning the update speed and the weight on goals with adjusted inputs selects the same values. **It does improve the team-rating benchmark** (z = 2.84, a gain of 0.0001): a pure team rating has nothing else to separate "led and coasted" from "was outplayed", while the player model's ratings move less per game.
With both adjusted, the player model's margin over the benchmark is 0.0027 (z = 1.93), essentially as before (0.0028, z = 1.99).

**What is not covered.** Special teams and pulled-goalie play are left raw. In the Detroit game the largest "chasing" effect was the pulled-goalie minutes, which the reports already show separately under "other". A time-remaining adjustment finer than the period, and team-specific tendencies when leading, are not modelled.
