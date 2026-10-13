# Structural additions to the game model — methodology and pre-registration

*Written 2026-10-12 before any candidate was computed or any test run. Motivation (model owner): the rest/travel test and the situational-angles test were both null; find something the model is genuinely missing.
Those tests added outside facts to a model that already prices them. This one starts from the model's own structure (src/gamemodel/model.py) and asks what it leaves out by construction.
The candidates, tests and rule below are fixed in advance; changes after results are seen are logged in §7.*

## 1. What the model leaves out by construction

A team's strength is the minutes-weighted sum of its dressed skaters' ratings plus its starter's rating times a league-average shot count, plus one league-wide home advantage. After each game, the expected-goal surprise is shared among the dressed skaters by ice time.
A logistic step turns the expected margin into a win probability using the margin and two back-to-back flags. Therefore:

1. **There is no team-level term.** Everything a team is beyond its players (system, coaching, deployment) can only be learned by moving the players' ratings, and leaves with them.
2. **The back-to-back penalty ignores who is in goal.** The tired-goalie effect is the best-documented fatigue effect in hockey, and the model applies the same penalty whether the starter played last night or the rested goalie starts.
3. **The update listens only to expected goals.** Shot volume is a less noisy signal than shot quality over short samples and is not used.
4. **Home advantage is the same for every team.**
5. **The margin-to-win mapping is the same for every game,** though a given margin is worth more in a low-scoring game and less when ratings are uncertain early in a season.
6. **Roster shape is ignored.** About a fifth of games are decided at three-on-three or in a shootout, where a team's best few players matter and its depth does not.

## 2. Candidates

**Changes to the rating update** (hyperparameters tuned on 2019–2020 only, then frozen):

| # | candidate | definition |
|---|---|---|
| U1 | team layer | besides the skater update, a team-level term moves by `lam_team` × surprise after each game and is added to the team's strength (starts at zero each season). Grid: λ ∈ {0.02, 0.04, 0.06} × `lam_team` ∈ {0, 0.005, 0.01, 0.02, 0.04}. |
| U2 | shot-volume signal | the update target becomes (1 − w) × expected-goal differential + w × unblocked-shot-attempt differential × the league's expected goals per unblocked attempt (that season's prior seasons). Grid: λ ∈ {0.02, 0.04, 0.06} × w ∈ {0, 0.25, 0.5, 0.75, 1}. |

**Terms added to the logistic step** (each on its own; team-level terms as home minus away):

| # | term | definition | expected sign |
|---|---|---|---|
| A1 | `team_rating` | the expected margin of the team-rating benchmark G2 (carried across seasons), alongside the player model's margin | + |
| A2 | `gk_tired` | the team played yesterday and tonight's starter was its main goalie in that game | − |
| A3 | `gk_rusty` | tonight's starter's previous start this season was 10 or more days ago (or he has none and the team has played 5 or more games) | − |
| A4 | `home_ice_team` | the home team's own home advantage from earlier seasons: half its home-minus-road goal differential per game, less the league's, shrunk by n / (n + 400) games | + |
| A5 | `low_scoring` | the model margin × (the game's expected total expected goals, standardised), the total from both teams' totals in earlier games (20 games of last season's average as a prior) | − |
| A6 | `early_season` | the model margin × 1 if either team has played fewer than 15 games | − (ratings less certain) |
| A7 | `stars` | the sum of the three highest preseason ratings among the dressed skaters, standardised | + |

## 3. Tests

The game model P of docs/gamemodel as configured for its backtest (raw expected-goal differentials), 6,502 test games of 2021–2025; the baseline logistic step is the model margin plus the two back-to-back flags, fitted for season T on 2019 … T − 1.

* **Update changes:** paired z on per-game log loss of the changed model (tuned values frozen) against P. If tuning selects the null value (`lam_team` = 0 or w = 0) the candidate is rejected there.
* **Added terms:** the coefficient and two-sided p in one fit on 2019–2025, Benjamini–Hochberg across the seven terms at a false-discovery rate of 10%, and the out-of-sample paired z against the baseline.
* **Reported:** for A7, the same term among games that went past regulation (does the star-heavy team win them?); for A2, the win record of teams on a back-to-back split by tired and rested goalie against expectation; log loss of every variant; accuracy of the baseline.

## 4. Rule

Nine hypotheses are tested. With no true effect the chance that at least one shows a paired z above 2 is about one in five, so, as in docs/angles:

* an **added term** is adopted only if it passes Benjamini–Hochberg at 10% **and** its out-of-sample paired z is above 2; one of the two means follow-up (a prospective check), not adoption;
* an **update change** is adopted if tuning selects a non-null value **and** its paired z against P is above 2; a non-null value with z between 0 and 2 means follow-up;
* otherwise not adopted.

Adoption changes the forward forecasts and is logged in docs/market/methodology.md with its date.

## 5. Limitations known in advance

* The model's log loss (0.6585) is close to what public models report as the practical ceiling for single NHL games; any gain will be small in absolute terms.
* A1 and U1 address the same gap in two ways and are not independent.
* The backtest knows who actually started in goal; A2 and A3 are usable before the game only when the starter is known.
* A4 uses seasons from 2017, so early test seasons have few prior games per team.

## 6. Results

See reports/structure/narrative.md. Run 2026-10-12: nothing adopted; `team_rating` and `low_scoring` flagged for follow-up (pass the false-discovery test, no out-of-sample gain).

## 7. Deviations

1. **Exploratory check added after the results (2026-10-12), not part of the pre-registered test and not eligible for adoption.** Because `team_rating` was significant on the full sample but did not help out of sample, a fixed blend
   (1 − w) × model margin + w × team-rating margin was tried with w chosen on 2019–2020 only. Tuning chose w = 0.5: test log loss 0.65876 against 0.65854 (z = −0.31). The best w in hindsight was 0.2 (0.65831, z = 0.76).