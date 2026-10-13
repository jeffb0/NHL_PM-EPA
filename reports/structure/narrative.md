# Structural additions to the game model — results

*Run 2026-10-12 under docs/structure/methodology.md (pre-registered the same day, before the run). Numbers are in reports/metrics/structure.json. One exploratory check was added after the results and is logged in §7 of the methodology.*

## Verdict

**Nothing is adopted.** Neither change to the rating update helps, and none of the seven added terms improves the forecasts out of sample. Two terms pass the false-discovery test on the full sample and are flagged for follow-up by the rule:
the team-rating term and the low-scoring interaction. Neither helped in the out-of-sample test, so neither changes the model.

Baseline: log loss 0.65854 on 6,502 test games; the model's favourite won 60.4% of them.

## Changes to the rating update (tuned on 2019–2020, frozen, tested on 2021–2025)

| candidate | tuned value | log loss | paired z against P | verdict |
|---|---|---|---|---|
| U1, team layer | λ = 0.02, `lam_team` = 0.01 | 0.65896 | −1.72 | not adopted |
| U2, shot volume | w = 0 (λ = 0.04) | 0.65854 | — | not adopted: tuning selects the null value |

* **Team layer.** Tuning liked a small team term (tuning-season log loss 0.66883 against 0.66906), but in the five test seasons it was worse in every one.
* **Shot volume.** Every blend of shot-attempt differential into the update target was worse than expected goals alone in the tuning seasons, at every step size. Expected goals already carry what volume knows (the two differentials correlate 0.79).

## Terms added to the logistic step (8,359 games for the fit; 6,502 for the out-of-sample column)

| term | coefficient (log-odds) | p | sign as expected | passes false-discovery test | out-of-sample z | verdict |
|---|---|---|---|---|---|---|
| `team_rating` (benchmark G2's margin beside the model's) | +0.27 ± 0.09 | 0.003 | yes | yes | −0.23 | follow up |
| `low_scoring` (margin × expected total, per SD) | −0.08 ± 0.04 | 0.028 | yes | yes | −0.59 | follow up |
| `early_season` (margin × fewer than 15 games played) | −0.16 ± 0.09 | 0.08 | yes | no | +0.23 | not adopted |
| `stars` (top three ratings, per SD) | +0.05 ± 0.03 | 0.16 | yes | no | −0.67 | not adopted |
| `gk_tired` (starter also played last night) | −0.12 ± 0.13 | 0.34 | yes | no | −0.26 | not adopted |
| `gk_rusty` (starter idle 10+ days) | +0.05 ± 0.06 | 0.36 | no | no | −0.58 | not adopted |
| `home_ice_team` (team's own home edge) | −0.03 ± 0.40 | 0.93 | no | no | −1.65 | not adopted |

All seven together: log loss 0.65992, paired z −1.31.

## Reading

* **A team-level component exists but cannot be used reliably.** With every season in the fit, the team rating adds to the player model (p = 0.003): something about teams is not captured by summing their players.
  Out of sample it helped in 2021 and 2022 and hurt in 2023 and 2024. The two margins correlate 0.91, so their relative weight is poorly determined and moves from one training window to the next.
  The exploratory check (a fixed blend weight tuned on 2019–2020) chose 0.5, which was slightly worse than the player model in the test seasons (0.65876, z = −0.31); the best weight in hindsight, 0.2, gains 0.0002 (z = 0.8).
* **Low-scoring games.** A given margin is worth slightly more when the expected total is low, as theory says, but the effect is too small to help (and hurt in three of five test seasons).
* **The tired goalie.** Teams on a back-to-back started the same goalie both nights only 323 times in 2,642 (12%). Those teams won 42.1% against 44.5% expected: the right direction, about three points, and far from significant at that sample.
  The practical point is that coaches have already removed most of this effect by starting the rested goalie, which is what the live forecast assumes.
* **Stars and overtime.** Among the 1,864 games that went past regulation (22%), the star-heavy team was no more likely to win (t = 0.3). The model's own margin still predicts those games, weakly.
* **Team-specific home ice** is indistinguishable from noise.

## What three null studies say together

Thirty-nine hypotheses have now been tested against this model (11 rest and travel features, 19 situational angles, 9 structural candidates), each pre-registered. None improves it out of sample.
The model's favourite wins 60.4% of games, in the range public models report as the practical ceiling for single NHL games. The likelier remaining gains are not further terms in the game step:
they are better player ratings going into the season (docs/residual found the 5v5 play block under-scaled) and better information on the day (who is actually playing).
