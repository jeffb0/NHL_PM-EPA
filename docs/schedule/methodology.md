# Rest, travel and schedule effects in the game model — methodology and pre-registration

*Written 2026-10-12 before any feature was computed or any test run. Motivation (model owner): the game model knows only home ice and the back-to-back flag. Three hypotheses about schedule effects are tested here.
The features, tests and rule below are fixed in advance; changes after results are seen are logged in §7.*

## 1. Hypotheses

* **H-R, rest:** rest beyond the back-to-back flag matters (two days against four; a rest advantage over the opponent).
* **H-T, travel:** distance travelled, time-zone change and its direction matter.
* **H-D, trip and density:** the length of a road trip or homestand and schedule density (three games in four nights) matter.

## 2. Features

All are computed from the schedule alone (regular-season game dates and venues), so each is known before the game. Each team's games are ordered by date within a season. The venue is taken to be the home team's arena
(neutral-site and outdoor games are not distinguished). Arena coordinates and standard-time UTC offsets are in `src/schedule/features.py`; daylight-saving differences (Arizona) are ignored.

| block | feature | definition | predicted sign on home win |
|---|---|---|---|
| R | `rest_adv` | home rest minus away rest, each = days since the team's previous game capped at 3 (season opener = 3) | + |
| R | `layoff_h`, `layoff_a` | 1 if the team's previous game was 5 or more days ago (opener = 0) | none (rest against rust) |
| T | `km_h`, `km_a` | great-circle distance in 1,000 km from the venue of the team's previous game to tonight's venue (opener: from its home arena) | − , + |
| T | `east_a`, `west_a` | hours by which tonight's venue is east / west of the away team's home time zone | + , + |
| D | `trip_a` | the away team's consecutive road game number (1 = first game of a trip), capped at 6 | + |
| D | `stand_h` | the home team's consecutive home game number, capped at 6 | none |
| D | `dense_h`, `dense_a` | 1 if the team played at least 2 games in the previous 3 days (tonight is its third in four nights) | − , + |

## 3. Test

The game model P of docs/gamemodel as currently configured (hyperparameters frozen; raw expected-goal differentials, as in that backtest). Its expected margin for every game from 2019 is computed once.
The baseline logistic step uses that margin and the two back-to-back flags. Each candidate adds one block (R, T, D) or all three (ALL) to the logistic step.
As in the game-model backtest, the logistic step for test season T is fitted on seasons 2019 … T − 1; the test set is 2021–2025 (6,502 games).

* **Primary, per block:** paired z on per-game log loss, candidate against baseline.
* **Reported:** the same for ALL; log loss by season; coefficients with standard errors from one fit on all of 2019–2025 and a likelihood-ratio test of each block;
  the same features regressed on goal differential and on expected-goal differential (controlling the model's expected margin and the back-to-back flags), which have more power than win/loss;
  the back-to-back coefficients for scale; how often each feature is non-zero.

## 4. Gates and rule

| gate | rule |
|---|---|
| SC-1 (per block) | the block improves the game model (paired z > 2) |
| SC-2 (per block) | no harm (paired z > −2) |

These are fitted empirical effects, not a mechanical correction, so the method is not rated sound in advance. By the owner's standing rule:

* a block is **adopted** if its paired z is above 2;
* if its paired z is between −1 and 2 it is adopted only if the evidence that the effect exists is otherwise clear: its likelihood-ratio test on 2019–2025 wins has p < 0.05 **and** every coefficient with a predicted sign and |t| > 2 has that sign;
* otherwise it is **not adopted**.

Four comparisons are made (three blocks and ALL); with no true effect the chance that at least one passes z > 2 is about 9%. ALL is reported, not a candidate for adoption on its own.
Adoption would change the forward forecasts, so it would be logged in docs/market/methodology.md as a model change with its date.

## 5. Limitations known in advance

* Win/loss carries little information per game: an effect worth one percentage point of win probability is at the edge of what 6,502 games can detect. A null result bounds the effect; it does not show there is none.
* The 2020 season (short, division-only, series of consecutive games against one opponent) is in the training window and has unusual travel.
* Rest and travel are correlated with each other and with the back-to-back flag; the blocks are tested separately and together for that reason.
* The market prices the schedule, so an effect found here need not be an edge.

## 6. Results

See reports/schedule/narrative.md. Run 2026-10-12: no block adopted (paired z: rest −1.59, travel +0.18, trip and density −1.63).

## 7. Deviations

None yet.
