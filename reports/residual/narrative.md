# Decomposing the portable team residual — results

Pre-registration: [docs/residual/methodology.md](../../docs/residual/methodology.md). Metrics: `reports/metrics/residual_decomp.json`; team table: `data/outputs/team_residual_buckets.csv`.
284 team-seasons (2017–2025). The accounting identity holds exactly (largest discrepancy 7e-14 goals). Team goal differential has a standard deviation of 46 goals; the residual, 22.

## What the residual is made of

| bucket | observed SD | explained SD | correlation | slope (observed on explained) | share of residual variance | same-team year-over-year | carry with players (± se) |
|---|---|---|---|---|---|---|---|
| **5v5 play** (expected goals for − against) | 22.9 | 12.1 | 0.85 | **1.62** | **45%** | **0.65** | **0.94 ± 0.15** |
| 5v5 finishing (goals − expected goals for) | 14.0 | 4.6 | 0.58 | 1.76 | 19% | 0.23 | 0.24 ± 0.15 |
| special teams | 12.9 | 11.2 | 0.71 | 0.82 | 14% | 0.07 | 0.29 ± 0.18 |
| goaltending | 18.7 | 18.7 | 1.00 | 1.00 | 0% | — | (zero by construction) |
| other (4v4, 3v3, pulled goalie, empty net) | 8.2 | 1.0 | 0.30 | 2.55 | 22% | 0.27 | 0.40 ± 0.25 |

(Goals per team-season. "Carry" is the coefficient of a team's residual on last season's bucket residual brought in by its current players, by minutes; the lump's coefficient in the same regression is 0.56 ± 0.08.)

## Interpretation

**Nearly half of the residual is 5v5 play that the components rank correctly but under-scale, and that half is the part that persists.** The play-driving components (possession, defense, turnovers, faceoffs) correlate 0.85 with teams' actual 5v5 expected-goal differential
but have about half its spread: the slope of observed on explained is 1.62. What they leave over is highly stable for a team (0.65 year over year) and travels with the players at essentially full value (0.94 ± 0.15).
This is a **measurement finding**: the regularised player estimates (ridge RAPM for defense, shrunk possession value) are compressed toward zero, a strong team's players are each shrunk, and the sum is far too small. It is not noise and it is not luck.
It is broader than the possession component alone, which is why rescaling possession by itself did little (docs/possession): the compression is in the whole 5v5 block, defense included.

**The rest is mostly not portable.** Shooting above expected at 5v5 (19% of the variance) carries at 0.24 ± 0.15, special teams (14%) at 0.29 ± 0.18 with almost no year-over-year stability (0.07), and the "other" bucket (22%, dominated by empty-net goals, which the player values do not cover at all) at 0.40 ± 0.25.
The published lump carries all of these at the same rate as the 5v5 play part, so it mixes one persistent piece with three largely transient ones. That is why its coefficient in this regression is 0.56: an average of about 0.9 and about 0.3.

**Who carries it (weakly identified).** Splitting the lump: forwards 0.70 ± 0.40, defensemen −0.36 ± 0.60, and the share of departed players that stays with the team 0.76 ± 0.29. The forward and defense figures are too noisy to separate.
The last one is notable: when players leave, most of "their" residual stays behind. Either part of the residual is organisational (coaching, system), or the equal per-minute split gives departing depth players credit that belonged to the stars who stayed. The data cannot tell these apart.

**Forecast test: better, not reliably, so not adopted.** Replacing the lump in the game model's preseason ratings by the bucket split (coefficients fitted on earlier seasons) improves the preseason-only model from 0.66920 to 0.66777 log loss.
That is the largest gain to the preseason prior of anything tested (about 18% of its gap to a team rating) but z = 1.80, short of the pre-registered 2; against a one-coefficient fitted lump it is z = 1.34; and with in-season updating the difference disappears (z = 0.14), because the updates learn the same thing within weeks.
Standings RMSE moves from 13.99 to 13.90 (z = 0.33). By the rule fixed in advance the split is **not implemented**.

**What it suggests next (new hypotheses, not tested here).** (1) Fix the scale where it arises: the 5v5 play components need a team-consistent de-shrinkage (for example, estimating the regularised models so that player sums reproduce team totals), rather than a carried patch.
(2) Carry only the 5v5 play bucket. Both come from looking at these seasons, so neither can be fairly tested on them again; the clean test is prospective, on 2026-27 games as they are played.

**Limitations.** About 30 teams a season; bucket coefficients are collinear for stable rosters and their early expanding estimates are unstable (the 2020 and 2021 folds have 5v5-play coefficients of 1.5–1.7 and negative "other" coefficients); traded players are assigned to one team;
5v5 expected goals include score effects; classical standard errors.
