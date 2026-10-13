# Information quality on the day of a forecast — results

*Run 2026-10-12 under docs/infoquality/methodology.md. Numbers are in reports/metrics/infoquality.json. Part A and B are measurements; the goalie-scale test (§6 of the method) was added after Part A and written down before it was run.*

## Summary

**Day-of information is worth very little to this model as it is built.** Told nothing about tonight except what happened in each team's previous game, it forecasts as well as when told exactly who dressed and who started.
The reason is not that the live information is good (the 8:00 run names the right starting goalie about three times in four) but that the model's goalie term, which is what day-of news mostly changes, carries almost no predictive value.
Goalie identity nevertheless moves the forecast by more than two points in a third of games, which matters for picks made on a three-point edge.

## Part A: what the model loses when it knows less (6,502 games, 2021–2025)

| what the model is told | log loss | P better by (z) | mean change in win probability | games moved more than 2 pts |
|---|---|---|---|---|
| actual skaters, actual starter (P) | 0.65854 | | | |
| previous game's skaters, actual starter | 0.65867 | +0.30 | 1.2 pts | 17% |
| actual skaters, starter by rule | 0.65925 | +0.86 | 2.2 pts | 36% |
| actual skaters, previous game's starter | 0.65842 | −0.11 | 3.0 pts | 51% |
| previous skaters, starter by rule (close to the live 8:00 run) | 0.65937 | +0.89 | 2.6 pts | 46% |
| previous skaters, previous starter | 0.65860 | +0.06 | 3.4 pts | 56% |

* **Skaters.** A team's previous lineup contains 94% of tonight's skaters and 95% of tonight's minutes; it is identical 32% of the time and differs by three or more players 7% of the time. Knowing the true lineup is worth 0.0001 of log loss (z = 0.3).
* **Goalie.** The rule (most-used starter; the other goalie on a back-to-back) names the actual starter 62% of the time (69% on a back-to-back, 61% otherwise); "same as last game" is right 44% of the time (11% on a back-to-back).
  Knowing the true starter instead of the rule's is worth 0.0007 (z = 0.9, not significant). Guessing last game's starter, wrong more often than right, costs nothing.
* **When the rule's goalie is wrong** (at least one team in 60% of games) the forecast is off by 3.0 points on average and by 6.8 points one time in ten.

## Follow-up: are the model's goalie differences too wide?

| goalie differences scaled by | 0 (no goalie term) | 0.25 | 0.5 | 0.75 | 1 (published) |
|---|---|---|---|---|---|
| test log loss, 2021–2025 | 0.65905 | 0.65857 | 0.65832 | 0.65831 | 0.65854 |

Tuning on 2019–2020 chose a scale of 0.25 (and kept n0 at 3,000), a large gain on those seasons (0.66752 against 0.66906). On the test seasons that configuration is level with the published model (0.65857, z = −0.05). **Not adopted.**

What the table does show: **the whole goalie term is worth 0.0005 of log loss.** Removing goalies from the model entirely costs that much; halving their differences costs nothing (0.65832, slightly better). In the two tuning seasons the goalie term made forecasts worse.
The model is close to indifferent to who is in goal, so no improvement in goalie information can add much to it.

## Part B: the live record (65 completed games, 130 team-games, 2026-10-03 to 2026-10-12)

Was the logged starter the goalie who started?

| forecast | right | by source |
|---|---|---|
| first forecast on the game's date | 75% | feed "confirmed" 13 of 13; feed "expected" 61 of 80 (76%); back-to-back rule 8 of 8; most prior work 13 of 26 (50%); manual override 2 of 3 |
| last forecast before puck drop | 82% | feed "confirmed" 34 of 34; feed "expected" 60 of 82 (73%); back-to-back rule 12 of 12; override 1 of 2 |

At 8:00 Eastern almost every starter is "expected", not "confirmed", and one in four is wrong. Confirmed starters have been right every time, and the back-to-back rule has been right every time (12 of 12). Ten days of games is a first reading only.

## Part C: options, sized

| option | what it fixes | most it can add to accuracy | cost |
|---|---|---|---|
| A second run late in the afternoon (about 5:00 Eastern) | starters move from "expected" (76% right) to "confirmed" (100% so far) | under 0.0007 log loss, not distinguishable from zero | a second scheduled task; picks and the email would need a rule for which run counts |
| Flag goalie-dependent picks in the email | shows when a pick's three-point edge would vanish with the other goalie in net | none to accuracy; avoids staking on a guess | small code change |
| A second lineup source to cross-check the feed | feed errors and players the feed does not list (Dylan Larkin) | under 0.0001 for skaters | a new scraper to maintain |
| Narrow the goalie term | goalie news moving forecasts more than its value warrants | none shown (z = −0.05) | not supported by the test |

**Recommendation.** For forecast accuracy, none of these is worth much; the measurements say the model's accuracy does not depend on day-of news. For the picks, the cheap and useful one is the flag: with a quarter of morning starters wrong and goalie identity worth three points when it is,
some three-point edges are artefacts of a goalie guess. A late-afternoon run is the complete fix for that, at the cost of a second daily task.

## Correction to an earlier statement

The previous write-up said better information on the day "is worth more than any term" tested. The game-model backtest already contained the evidence against that (previous lineup and starter against actual: z = 0.06), and Part A confirms it. The statement was wrong.

## Limitations

The backtest's rule is a stand-in for the live feed, which is better than the rule (76% against 62%); a team's first game of a season uses the true starter; the live sample is ten days;
"worth little to this model" is a statement about this model's goalie and lineup terms, not about hockey: a model with a better goalie rating would value the news more.
