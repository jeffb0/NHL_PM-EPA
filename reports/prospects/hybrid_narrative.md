# Early-sample updating for young and thin-history players — results

Pre-registration: [docs/prospects/hybrid.md](../../docs/prospects/hybrid.md). Metrics: `reports/metrics/hybrid_prospects.json`. 1,105 thin-history player-seasons in the test seasons 2020–2025; fits use earlier seasons only.

## Gates

| gate | rule | statistic | result |
|---|---|---|---|
| HY-1 | improves the full-season projection at 5 and at 10 games (z > 2 at both) | z = 2.25 and 2.64 | PASS |
| HY-2 | no harm to the game model (z > −2) | z = 0.60 (0.65861 → 0.65854) | PASS |
| HY-3 | two games already improve the projection (z > 2) | z = 1.16 | FAIL |

**Implemented in the forward forecasts.**

## How much a young player's first games say

| games played | share of the prior's error removed | z | typical size of the adjustment (1 SD, per 60) | true rookies only: share removed (z) |
|---|---|---|---|---|
| 2 | 2% | 1.16 | 0.03 | 0% (−0.03) |
| 5 | 6% | 2.25 | 0.04 | 8% (1.19) |
| 10 | 8% | 2.64 | 0.05 | 9% (1.27) |
| 20 | 12% | 3.91 | 0.05 | 17% (1.92) |
| 40 | 18% | 4.57 | 0.05 | 30% (3.08) |

(Value per 60 is on the scale where a generic rookie is about 0.13, a good top-six forward 0.35, a star 0.60 or more.)

## Interpretation

**Early games carry real information, and it builds steadily, not suddenly.** By five games the adjustment reliably improves the full-season projection of a thin-history player, and it keeps improving: 6% of the prior's error removed at five games, 12% at twenty, 18% at forty (30% for true rookies).
This is the first change to the player projections in this project that passes a reliability gate at the player level and does not hurt the game model.

**Two games say very little.** The n = 2 adjustment removes 2% of the error over all thin-history players (not reliable) and nothing for true rookies. The "elite from day one" case cannot be recognised from two games in these data.

**What the early signal is.** In standardised terms the weight is on **ice time per game** and **5v5 on-ice expected-goal differential**; early points and individual expected goals carry almost nothing until twenty games or more. For defensemen it is mostly on-ice differential.
A young forward given top-line minutes whose team out-chances opponents while he is on the ice moves up; one sheltered on a fourth line, or caved in, moves down.

**The shifts are modest by design of the data, not by choice.** A one-standard-deviation adjustment is 0.03–0.05 per 60, and the largest in the current season after two games are about ±0.04 to ±0.08. So a rookie rated 0.17 who starts brilliantly reaches roughly 0.25 after a few weeks, not 0.60:
history says hot starts by young players have held up only that far. The model will keep moving him as the games accumulate, and next season his own NHL record takes over.

**This season so far.** Gavin McKenna: two games, 15.4 minutes a game, no points, no adjustment yet (−0.001; 0.59 expected goals for and 1.68 against at 5v5 while he was on the ice, ordinary ice time). Easton Cowan has the largest positive adjustment (+0.04). Viggo Björck has played one game, below the first checkpoint.
Toronto's win probability against Ottawa moves from 42.8% to 42.9%.

**Limitations.** The target is the model's own value measure, which the elite-prospect analysis suggests may under-credit young scorers; at 40 games the signals overlap the target; ice time partly reflects team depth; the adjustment is linear and identical for a first-overall pick and a call-up with the same numbers.
