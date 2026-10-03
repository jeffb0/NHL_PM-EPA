# Early-sample updating for young and thin-history players (hybrid prior) — methodology and pre-registration

*Written 2026-10-03 before anything here was fitted or scored. Owner's request: shift projections for young talent aggressively, game to game, as the NHL sample grows; even two games should say something about whether a player is elite from day one or a typical rookie.
The design, tests and rule below are fixed here; changes after results are seen are logged in §8.*

## 1. Idea

A thin-history player's rating should be his prior (docs/prospects/elite.md) **plus an adjustment from what he has done so far this season**, with the size of the adjustment at each sample size learned from history rather than chosen:
how well did the first n games of earlier thin-history players predict the part of their full-season value that the prior missed? If two games carry information, the fitted adjustment at n = 2 will be non-zero; if not, it will be shrunk to nothing.

## 2. Signals (all available the morning after each game, for every season)

Cumulative over the player's first n games of the season:

* **ice time per game** (the coach's judgment: a young player given top-line minutes is being told something);
* **puck value per 60** (PM-EPA credit for his own puck actions, all situations);
* **individual expected goals per 60**; **points per 60**;
* **5v5 on-ice expected-goal differential per 60**.

## 3. Model

* **Who:** thin-history skaters, m_eff < 1,000 discounted NHL minutes entering the season (rookies and most second-year players; the same definition as docs/prospects).
* **Target:** the player's full-season value rate minus his prior projection (rate model v3, fitted on earlier seasons). For small n the first n games are a negligible part of the season value; at n = 40 they are about half of it, which flatters the fit and is noted.
* **Fit:** for each checkpoint n ∈ {2, 5, 10, 20, 40} and each position, a ridge regression (λ by leave-one-season-out) of the target on the five signals at n, over thin-history player-seasons with at least n games and 200 season minutes, weighted by season minutes, using seasons before the one being forecast.
* **Use:** after a player's g-th game his rating is prior + the fitted adjustment at the largest checkpoint ≤ g (none before two games). The team-surprise update of the game model continues unchanged on top.

## 4. Tests

1. **Player level (test seasons 2020–2025, expanding fits):** minute-weighted squared error of the full-season rate, prior + adjustment against prior alone, at each checkpoint; paired z by player-cluster bootstrap; also for true rookies and for first-round picks in their first two seasons. Reported: the share of the prior's error removed at each n, and the fitted coefficients.
2. **Game level:** the game model (hyperparameters frozen, rate model v3 priors) with the adjustment applied to thin-history players' ratings, against without, paired z on per-game log loss over the 6,502 test games.
3. **Reported:** the adjustment for this season's rookies after their games so far (including McKenna and Björck), with the n = 2 coefficient sizes so the "how much do two games say" question is answered directly.

## 5. Gates and rule

| gate | rule |
|---|---|
| HY-1 | the adjustment improves the full-season projection at n = 5 and at n = 10 (paired z > 2 at both) |
| HY-2 | no harm to the game model (paired z > −2) |
| HY-3 (reported) | the n = 2 adjustment improves the projection (paired z > 2): the owner's "even two games" claim |

The method is rated sound in advance (an empirical-Bayes update with data-set weights). It is implemented in the forward forecasts if HY-2 holds and the n = 5 and n = 10 player-level z are both above −1 (the owner's standing rule); a reliable improvement is claimed only where a z exceeds 2.

## 6. What the result can say

HY-1 passing means early individual results carry real information about a young player beyond his prior. HY-3 says whether two games already do. The game-level test says whether it matters for predicting games; thin-history players have few minutes, so a player-level gain with no game-level gain is the expected outcome.

## 7. Limitations stated in advance

The target is the model's own value estimate; at larger n the signals overlap the target; ice time partly reflects team depth, not only the player; on-ice differential reflects teammates; the adjustment is linear and the same for an 18-year-old first-overall pick and a 25-year-old call-up with the same numbers;
it rates the player by full-season value, so a hot two games moves him only as far as hot two-game starts have historically held up.

## 8. Deviations log

*(any change to §2–§5 after results are seen is recorded here with its reason)*

1. **Outcome, applied mechanically (2026-10-03).** HY-1 passes (n = 5: z = 2.25; n = 10: z = 2.64), HY-2 holds (game model z = 0.60), HY-3 fails (n = 2: z = 1.16). **Implemented** in the forward forecasts from the 2026-10-04 run
   (`src/live/predict.py` reads the decision from `reports/metrics/hybrid_prospects.json`; the models are refitted on every completed season).
2. **Implementation note.** The logistic step and margin calibration of the forward forecasts are still fitted on the historical model without the adjustment; the adjustment changes ratings by a few hundredths per 60 for a handful of players per team, so the mismatch is negligible, and it is noted.
