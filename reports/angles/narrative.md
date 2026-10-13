# Situational angles — results

*Run 2026-10-12 under docs/angles/methodology.md (pre-registered the same day, before the run). Numbers are in reports/metrics/angles.json. No deviations.*

## Verdict

**None of the 19 angles is adopted and none is flagged for follow-up.** No angle passes the false-discovery test, none improves the forecasts out of sample, and the 19 together are no better than chance (joint likelihood-ratio p = 0.94).
Adding all 19 at once makes the forecasts clearly worse (paired z = −4.0). Teams in each situation won almost exactly as often as the model already expected.

## Table (8,359 games, 2019–2025; out-of-sample column on the 6,502 games of 2021–2025)

"Effect" is the change in win probability, in percentage points, for the team in the situation (continuous angles: per standard deviation), with its standard error. "Record" pools home and away teams in the situation; "expected" is the win rate the baseline model gave those same teams.

| angle | the saying | effect (pts) | p | record | win % | expected % | out-of-sample z |
|---|---|---|---|---|---|---|---|
| won previous game by 3+ | letdown | −1.9 ± 1.0 | 0.06 | 1688–1609 | 51.2 | 52.6 | −0.47 |
| scored 5+ in previous game | rolling | −1.2 ± 1.0 | 0.22 | 1742–1680 | 50.9 | 51.8 | −1.05 |
| head-to-head over last 6 meetings | "they own them" | +0.6 ± 0.6 | 0.36 | | | | +0.17 |
| starter's last 5 starts | hot goalie | −0.5 ± 0.6 | 0.36 | | | | −1.49 |
| lost 3+ straight | cold / due | −1.0 ± 1.2 | 0.41 | 1006–1234 | 44.9 | 45.7 | −0.38 |
| last-10 record | hot team | +0.5 ± 0.6 | 0.44 | | | | −0.46 |
| shut out in previous game | bounce-back | −1.4 ± 1.8 | 0.46 | 395–464 | 46.0 | 47.2 | −0.63 |
| lost previous game by 3+ | bounce-back | +0.6 ± 1.0 | 0.53 | 1567–1731 | 47.5 | 47.0 | −1.78 |
| goals above expected, last 10 | hot team | +0.3 ± 0.6 | 0.56 | | | | −0.33 |
| back-to-back after overtime | tired | −1.3 ± 2.6 | 0.62 | 258–336 | 43.4 | 44.3 | −0.89 |
| lost to this opponent last game | rematch | +1.0 ± 2.2 | 0.66 | 266–298 | 47.2 | 46.3 | −1.83 |
| last game of a 3+ game road trip | getaway game | +0.6 ± 1.5 | 0.69 | 626–700 | 47.2 | 46.7 | −2.05 |
| first home game after 3+ on the road | trap game | +0.4 ± 1.6 | 0.78 | 714–604 | 54.2 | 53.8 | −0.79 |
| plays again tomorrow | look-ahead | +0.3 ± 1.3 | 0.79 | 1308–1319 | 49.8 | 49.6 | −0.96 |
| lost last meeting by 3+ | revenge | −0.3 ± 1.2 | 0.81 | 949–1132 | 45.6 | 45.9 | −2.53 |
| lost last meeting | revenge | −0.1 ± 0.7 | 0.90 | 2537–2816 | 47.4 | 47.5 | −1.75 |
| previous game went past regulation | tired | +0.1 ± 1.0 | 0.91 | 1859–1833 | 50.4 | 50.3 | −1.10 |
| back-to-back after losing by 3+ | the owner's example | +0.2 ± 2.5 | 0.93 | 229–322 | 41.6 | 41.4 | −1.05 |
| won 3+ straight | hot | +0.1 ± 1.2 | 0.95 | 1194–1010 | 54.2 | 54.1 | +0.05 |

Ten of the 19 estimates point the way the saying claims and nine point the other way.

## Reading

* **The owner's example.** Road teams on the second night of a back-to-back after losing the first by three or more went 139–229 (37.8%) over seven seasons, against 39.3% expected. Home teams in the same spot went 90–93 (49.2% against 45.6% expected, z = 1.0).
  A quoted "5–2" is seven games out of several hundred like them; a run of 5–2 or better turns up constantly in a 368-game sequence of 38% propositions.
* **Why the records sound meaningful and are not.** Teams on winning streaks do win more (54.2%), and teams on losing streaks do win less (44.9%). The model already expected 54.1% and 45.7%, because good teams are the ones that go on streaks.
  The raw record measures team quality, home ice and rest; once those are priced, nothing is left.
* **The nearest thing to a signal is the letdown after a blowout win** (−1.9 points, p = 0.06), with a related lean after scoring five or more (−1.2). Both point toward regression after a big night, both are within what 19 tests produce by chance, and neither helps out of sample.
* **Results beyond expected goals carry no extra information.** The model ignores actual goals in-season (the tuned weight was zero). Last-10 record, goals above expected over ten games and the hot goalie were the fairest chances to show that is a mistake; all three are near zero.
* **Revenge, rematch, trap and getaway games:** every estimate is under one point with a standard error larger than the estimate.

## What the null means

Each estimate has a standard error of about 1 to 2.5 points of win probability, so an effect of two or three points in a rare situation cannot be excluded. What is excluded is the size of effect the sayings imply.
A model that already prices team strength, home ice and back-to-backs has nothing to gain from these, and adding them costs accuracy.
