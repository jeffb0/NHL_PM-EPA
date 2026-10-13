# Information quality on the day of a forecast — method

*Written 2026-10-12 before the analysis was run. Requested by the model owner ("investigate opportunities to improve information quality on the day of prediction"). This is a measurement study, not a model change: nothing is adopted by it and there is no gate.*

## 1. Questions

1. **How much is day-of information worth to this model?** In the backtest the model is told who actually dressed and who actually started in goal. What does it lose when it is told less?
2. **How good is the information the live run actually has at 8:00 Eastern?** How often is the logged starting goalie the one who starts, by where the name came from?
3. **Which improvements would move forecasts, and by how much at most?**

## 2. Part A: the value of information in the backtest (6,502 games, 2021–2025)

The model is re-run with the same ratings and updates (ratings always update on who actually played) but with degraded knowledge of tonight's game:

| variant | skaters | goalie |
|---|---|---|
| P (reference) | actual | actual |
| A1 | team's previous game | actual |
| A2 | actual | **rule**: the team's most-used starter so far this season; if the team played yesterday, its most-used other goalie (a team's first game of a season uses the actual starter) |
| A3 | actual | the previous game's starter |
| A4 | team's previous game | rule |
| P_prev (already in the game-model report) | team's previous game | the previous game's starter |

Reported for each: log loss and paired z against P; the mean absolute change in win probability and the share of games moved by more than two points. Also: how often the rule and the previous-starter guess name the actual starter (overall, on a back-to-back, otherwise);
how much of tonight's lineup the previous game's lineup contains.

A2 is the live forecast's fallback when the feed has no confirmed starter, so P against A2 bounds what better goalie information can add; P against A1 bounds what better skater information can add.

## 3. Part B: the live record (2026-27 games completed so far)

For each completed game, the first forecast logged on the game's date and, where there is one, the last forecast before puck drop: was the logged starter the goalie who started? By source (confirmed, expected, back-to-back rule, most prior work) and by the hour of the run.
The sample is about ten days of games, so this is a first reading, not an estimate.

## 4. Part C: options

Each candidate improvement is sized against the bounds from Part A and the error rates from Part B, with what it would cost to run.

## 5. Results

See reports/infoquality/narrative.md.

## 6. Follow-up test added after Part A was seen (2026-10-12), written before it was run

**What prompted it.** Part A showed that telling the model the wrong starting goalie moves its forecast by about two points on average, yet costs almost nothing in accuracy (the previous game's starter, right 44% of the time, does as well as the true starter).
If the model's goalie ratings were as informative as their spread implies, knowing the starter would be worth several times what was measured. The goalie prior strength n0 was also tuned at the top of its grid (3,000 shots). Hypothesis: **the model's goalie differences are too wide.**

**Test.** Two parameters, tuned on 2019–2020 only and then frozen, as in docs/gamemodel:

* `gscale` ∈ {0, 0.25, 0.5, 0.75, 1}: a goalie's rating is replaced by replacement + `gscale` × (rating − replacement);
* n0 ∈ {3,000, 6,000, 12,000}: prior shots before in-season results move a goalie's rating.

Primary: paired z on log loss over the 6,502 test games, tuned configuration against P. If tuning selects (1, 3,000) the hypothesis is rejected there.

**Rule.** This follow-up was prompted by a look at the test seasons, though its parameters are chosen on the tuning seasons only. Adopt in the forward forecasts if tuning selects a non-null configuration and z > 2; if z is between 0 and 2, flag for a prospective check; otherwise not adopted.
Reported: the grid; log loss by season; how far the tuned configuration moves forecasts.
