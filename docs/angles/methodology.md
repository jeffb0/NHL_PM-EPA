# Situational "angles" from commentary and old-school handicapping — methodology and pre-registration

*Written 2026-10-12 before any angle was computed or any test run. Motivation (model owner): broadcasters and trend bettors cite situational records ("road teams on the second night of a back-to-back after losing the first by three or more are 5–2").
Most such records are small samples picked after the fact, but some may describe something real that the model does not see. The angles, tests and rule below are fixed in advance; changes after results are seen are logged in §7.
Sources for the list: the owner's example, and a web search of trend-betting articles (bounce-back after a blowout, revenge after a blowout, hot and cold streaks, first home game after a road trip, the zig-zag idea).*

## 1. Why the model might miss these

The game model's in-season update uses expected goals only (the tuned weight on actual goals is zero) and has no memory of the order of results, of who the opponent was last time, or of what happens tomorrow.
So streaks, blowouts, shutouts, revenge spots and look-ahead spots are all invisible to it. If any carries information beyond expected goals, this test can find it.

## 2. Angles

Each is known before the game. A team's games are ordered by date within a season (the head-to-head angle looks back across seasons). Scores are final scores (a shootout counts as a one-goal game); a "loss" includes overtime and shootout losses.
For angles defined per team, the tested variable is **the home team's value minus the away team's**, so one coefficient measures the angle for both sides.
"Folk direction" is what the saying claims for the team in the situation.

| # | angle | a team is in the situation when … | folk direction |
|---|---|---|---|
| 1 | `streak_w3` | it has won its last 3 or more games | better (hot) |
| 2 | `streak_l3` | it has lost its last 3 or more games | worse (cold); also said: "due" |
| 3 | `last10` | share of its last 10 games won, minus one half (0 with fewer than 5 games played) | better |
| 4 | `luck10` | goals minus expected goals, per game, over its last 10 games (0 with fewer than 5) | better (hot) |
| 5 | `blowout_loss` | it lost its previous game by 3 or more | better (bounce-back) |
| 6 | `blowout_win` | it won its previous game by 3 or more | worse (letdown) |
| 7 | `shutout_loss` | it was shut out in its previous game | better (bounce-back) |
| 8 | `scored5` | it scored 5 or more in its previous game | better (rolling) |
| 9 | `prev_ot` | its previous game went past regulation | worse (tired) |
| 10 | `b2b_after_blowout` | it played yesterday and lost by 3 or more (the owner's example) | better |
| 11 | `b2b_after_ot` | it played yesterday and the game went past regulation | worse |
| 12 | `front_b2b` | it plays again tomorrow | worse (look-ahead, saving legs) |
| 13 | `revenge` | it lost its most recent meeting with tonight's opponent this season | better |
| 14 | `revenge_blowout` | it lost that meeting by 3 or more | better |
| 15 | `rematch_lost` | its previous game was against tonight's opponent and it lost | better |
| 16 | `h2h` | wins minus losses, per game, over the last 6 meetings with tonight's opponent in any season (0 with fewer than 3) | better ("they own them") |
| 17 | `hot_goalie` | tonight's starter's goals saved above expected over his previous 5 starts this season (0 with fewer than 3) | better |
| 18 | `home_after_trip` | (home team only) its first home game after 3 or more straight road games | worse (trap game) |
| 19 | `trip_finale` | (away team only) the last game of a road trip of 3 or more games | worse (getaway game) |

Not tested for lack of data in the tables: afternoon starts, day of week, division rivalry, playoff-race desperation, special-teams streaks.

## 3. Tests

The game model P of docs/gamemodel as configured for its backtest; the baseline logistic step is the model's expected margin plus the two back-to-back flags. Each angle is added to the baseline **on its own**.

* **Evidence that the effect exists:** the angle's coefficient in one logistic fit on all games of 2019–2025, its two-sided p-value, and the Benjamini–Hochberg procedure across the 19 angles at a false-discovery rate of 10%.
* **Predictive value:** paired z on per-game log loss against the baseline over the 2021–2025 test games, with the logistic step for season T fitted on seasons 2019 … T − 1.
* **Reported, in the style the sayings are quoted:** for each yes/no angle, the win–loss record of teams in the situation (2019–2025), the wins the baseline model expected, and the z of the difference; the same coefficient on goal differential;
  a joint likelihood-ratio test of all 19; continuous angles (3, 4, 16, 17) are standardised.

## 4. Rule

Nineteen angles are tested, so about one would pass |t| > 2 by chance and the chance that at least one shows a paired z above 2 is about one in three. The owner's standing rule (adopt at z > 2) is therefore tightened for this family:

* an angle is **adopted** only if it passes Benjamini–Hochberg at 10% on the full-sample fit **and** its out-of-sample paired z is above 2;
* an angle that passes one of the two is **flagged for follow-up** (a prospective check on this season's games), not adopted;
* everything else is **not adopted**. These are fitted empirical effects, so the "sound method, adopt anyway" branch of the standing rule does not apply.

Adoption would change the forward forecasts and would be logged in docs/market/methodology.md.

## 5. Limitations known in advance

* Each yes/no situation covers a few hundred to a couple of thousand games; an effect under about three percentage points of win probability cannot be detected.
* The angles overlap (a blowout loss is often also a shutout loss and part of a losing streak); testing one at a time does not separate them.
* The backtest uses the goalie who actually started, which is known before the game only when the starter is confirmed.
* The market knows these sayings. An effect found here is a statement about hockey, not yet about prices.

## 6. Results

See reports/angles/narrative.md. Run 2026-10-12: no angle adopted or flagged (smallest p 0.06; joint likelihood-ratio p 0.94; all 19 together paired z −4.0).

## 7. Deviations

None yet.
