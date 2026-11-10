# Prospective test against betting markets — methodology and pre-registration

*Written 2026-10-03, after one forecast (WPG at DET, 2026-10-04) had been compared with lines by hand and before any logged game was scored. The protocol, metrics and thresholds below are fixed here; changes are logged in §8.*

## 1. Question

Does the game model's win probability, made before a game from information available then, contain information that the betting market's price does not? This is the only fair test of "alpha": forecasts and prices are recorded **before** the game and scored after it.
Backtests cannot answer it (no historical odds are in the repository, and every design choice so far has seen 2019–2025).

## 2. What is logged

`python -m src.live.daily` (run once a day, before the day's first game):

1. refreshes the season's completed games (`src.live.update`);
2. forecasts every game on the slate that has not started (today and tomorrow) with the frozen game model (docs/gamemodel: hyperparameters fixed, original rate model, in-season updating through the last completed game), each team's previous lineup,
   and the starting goalie with the most prior work unless `data/external/roster_overrides.json` names one;
3. captures moneyline and total prices for each of those games from scoresandodds.com for FanDuel, DraftKings and BetMGM (other listed books are stored too);
4. appends to `forecast_log.csv` and `odds_log.csv` (in `data/private/logs/` from 2026-10-18; item 14) with the capture time. Rows are never edited; a game forecast on two days has two rows.

**The scored forecast for a game is the last one logged before its start time, with the prices captured in the same run.** These are morning prices, not closing lines; closing lines are sharper, so beating morning prices is a weaker claim than beating the close, and that is stated with every result.

## 3. Model version

The forecasting model is frozen as of this document (game model P, hyperparameters in `reports/metrics/game_model.json`, logistic step refitted on all completed seasons). If the model is changed, the change and its date are logged in §8 and results are reported separately before and after.

## 4. Metrics (computed by `python -m src.live.score`)

* **Market probability:** for each book the two moneyline prices are converted to probabilities and normalised to sum to one (margin removed proportionally); the market consensus is the mean over FanDuel, DraftKings and BetMGM (books missing a price are skipped).
* **Primary:** mean log loss of the home-win outcome, model against market consensus, on games with both; paired z (mean over standard error of the per-game difference).
* **Blend test:** logistic regression of the outcome on the market's logit and the model's logit (reported once 300 games are scored): a positive, significant model coefficient means the model adds information to the price.
* **Paper bets (accounting only):** one unit on a side when the model's probability exceeds that side's break-even probability at the best of the three books by at least **3 percentage points**; return on those units, with a bootstrap interval. No real stakes are implied.
* Also reported: Brier score, accuracy, calibration of both, the number of games, and results by month.

## 5. Thresholds for any claim

* Nothing is claimed before **300 scored games**. Before that the report shows running numbers labelled "too few games".
* "The model matches the market": log-loss difference within ±0.005 at 300+ games.
* "The model adds information": blend coefficient with z > 2 at 300+ games.
* "The paper bets are profitable": return on units positive with a 95% bootstrap interval excluding zero, and at least 100 paper bets.

## 6. Expectation, stated in advance

Closing moneylines are very efficient and morning lines nearly so. The backtest edge of the model over a tuned team rating was 0.003 log loss and not established (z = 1.99). The expected result is that the market is at least as good as the model; a finding of parity would be a good outcome.

## 7. Limitations stated in advance

Morning prices, not closing; one odds source (a scraped comparison page, which can be late or wrong for a book); starting goalies and lineups are assumed, not confirmed; games the logger misses (a day not run) are simply absent, which is a selection risk if runs are skipped on particular days;
this is analysis of a model, not betting advice.

## 8. Change log

*(model changes, protocol changes and missed days are recorded here with dates)*

1. **2026-10-03, before any game was scored: source-error guard.** The first capture showed one book (BetMGM, FLA at ANA) with the sides apparently reversed (+154 / −190 against −127 to −146 / +108 to +122 at six other books).
   A book whose margin-free home probability is more than 8 percentage points from the median of all listed books for that game is dropped from the consensus and from the paper-bet prices. The raw rows stay in the log.
2. **2026-10-03: first capture was made in the afternoon**, a few hours before the first games, not in the morning.
3. **2026-10-03, before any game was scored: inputs v2 (model unchanged).** At the author's request each run now (a) snapshots the published rosters, (b) reads the RotoWire lineups page for each game's listed starting goalie (confirmed or expected)
   and injury list, (c) removes from a team's previous lineup anyone listed out, injured-reserve or suspended, anyone in `roster_overrides.json`, and anyone no longer on the published roster, filling to 12 forwards and 6 defensemen from the roster by last season's ice time,
   and (d) uses the listed starter unless a manual override names one. Day-to-day and game-time-decision players stay in. The forecast log records the goalie source, the players removed and added, and names that could not be matched.
   The rating model and its hyperparameters are unchanged; the `model` column marks rows made with these inputs. The 18 forecasts logged at 20:52 UTC on 2026-10-03 used the earlier inputs and are superseded by the 20:59 UTC run (the last before puck drop is the one scored).
4. **Schedule:** from 2026-10-04 the run is scheduled daily at 12:00 Eastern, so prices are midday, not morning.
5. **2026-10-04: rate model v3 (a model change).** Preseason ratings come from the projection with elite-prospect columns (draft slot bins, debut timing, NHL-equivalent production; docs/prospects/elite.md), adopted under that document's pre-registered rule at the author's request.
   Its backtest effect on the game model is nil to slightly negative (z = −1.08 preseason-only, −1.52 full); ratings change by a few hundredths of a goal per 60 for rookies and young players. Forecasts logged on 2026-10-03 used the original rate model;
   the `model` column separates them, and results are reported for both periods. Each run now also fetches career pages for players new to the rosters.
6. **2026-10-04: early-sample adjustment (a model change, same date).** Thin-history players' ratings carry an adjustment from their own games so far this season (ice time, puck value, individual expected goals, points, 5v5 on-ice expected goals), with weights learned at 2, 5, 10, 20 and 40 games
   (docs/prospects/hybrid.md; adopted under its pre-registered rule: full-season projections improve at 5 and 10 games, z = 2.25 and 2.64; game model z = 0.60). Changes 5 and 6 take effect together.
7. **2026-10-04 (after the day's first capture): back-to-back goalie rule (inputs v3; rating model unchanged).** At the author's instruction, a team on the second night of a back-to-back is assumed to start its backup
   (the named backup in `roster_overrides.json`, else the roster goalie with the most prior work other than the previous night's starter) unless the lineup feed lists a *confirmed* starter. A confirmed starter now outranks a manual starter override.
   For next-day forecasts the back-to-back is read from the schedule (the team plays today), since today's game is not completed. The rule was not backtested (history uses the goalies who actually played); it applies from the 2026-10-05 run.
8. **2026-10-05: score-adjusted differential in the in-season update (a model change).** 5v5 expected goals are weighted by score state before the game's surprise updates ratings (docs/scoreeffects; adopted under its pre-registered rule: no reliable gain, z = 0.39 on 6,502 games,
   no harm; the team-rating benchmark improves, z = 2.84). Game-level differentials change by 0.03 goals on average (correlation 0.9995 with the raw figure), so forecasts move very little.
9. **2026-10-06: the daily run moves to 8:00 Eastern** (at the author's request, with a morning email). Prices are now captured in the morning; starting goalies are more often "expected" than "confirmed" at that hour, and the back-to-back rule applies when no starter is confirmed.
   The run on 2026-10-06 was made at 8:07. Runs on 2026-10-03 to 2026-10-05 were made between 12:14 and 17:12 Eastern.
10. **2026-10-10: manual roster overrides are switched off** (at the author's request; `"active": false` in `roster_overrides.json`). From the 2026-10-10 run the live forecast takes lineup news only from the morning lineup feed and the published rosters:
    a player sits out only if the feed lists him out, injured-reserve or suspended, or he is off the published roster, and the starting goalie is the feed's confirmed starter, else the rested goalie on the second night of a back-to-back
    (the roster goalie with the most prior work other than the previous night's starter), else the feed's expected starter, else the goalie with the most prior work. Known gap: a player sitting out whom the feed does not list (Dylan Larkin, DET, on 2026-10-10) is eligible to fill a lineup spot.
11. **2026-10-13: an afternoon goalie update is added** (at the author's request; `python -m src.live.afternoon`, about 17:00 Eastern). For today's games whose starter on either side was not confirmed at the morning run and that have not started,
    the lineup feed and prices are fetched again and a new forecast and price capture are appended. By §2 the last forecast before puck drop is the one scored, so from this date those games are scored on the afternoon forecast and the afternoon prices; games confirmed in the morning are scored as before.
    The model is unchanged. Every forecast row now also records, for a side whose starter is not confirmed, the team's other goalie and the range of the home win probability over who starts (`p_home_min`, `p_home_max`).
    The picks ledger is not changed by the afternoon run (docs/market/picks.md).
12. **2026-10-12: a de-shrinkage of the 5v5 play block was tested and left off** by the author's decision (docs/deshrink). No change to the forecasts.
13. **2026-10-16: more books are logged, for the picks only** (author's request). Each run also captures Hard Rock Bet and theScore Bet moneyline and total prices from Action Network's scoreboard feed (New Jersey lines; one request per date); Caesars was already captured from the comparison page.
    The author's picks now take the best price among FanDuel, DraftKings, BetMGM, Caesars, Hard Rock Bet and theScore Bet (docs/market/picks.md §2). **The market test is unchanged:** its consensus, paper bets and scoring still use FanDuel, DraftKings and BetMGM only,
    and the source-error guard's median is still taken over the comparison page's books (the two new books are checked against it but do not move it).
14. **2026-10-18: the logs move to a private repository; fingerprints are published** (author's request, before making this repository public). The forecast, odds, scoring and picks logs and the emailed digests contain sportsbook prices and lineup-feed data,
    so they now live in `data/private/` (a separate private git repository, ignored here) and were removed from this repository's history. Nothing about how they are written changed. To keep the evidence checkable, every daily and afternoon run appends the SHA-256, size and row count
    of each private log to the public `data/live/public/fingerprints.csv`; a private log shown later can be checked against the fingerprint committed at the time. The public record is `reports/live/market_summary.md` (the test's totals),
    `data/live/public/picks_record.csv` and `reports/live/picks.md` (the picks in units and decimal odds). The history rewrite also replaced the commit author's e-mail with GitHub's no-reply address; commit dates and contents are otherwise unchanged.
15. **2026-11-02: individual results for established skaters (a model change; author's request).** The thin-history adjustment of change 6 is extended to skaters with m_eff ≥ 1,000. Their ratings carry an adjustment from their own games so far this season: the five signals of change 6, plus 5v5 on-ice xGD relative to the team with them off the ice, plus ice time against last season's. Weights are learned by position and age group (31+ or younger) at 5, 10, 20 and 40 games.
    It was adopted under the pre-registered rule of docs/inseason: full-season value improves at 10 and 20 games (z = 4.67 and 6.63), with game model z = 0.77, so no gain in game prediction is claimed. The in-season finishing update tested alongside it failed its gate and is not used. Forecasts move by about 0.06 goals of expected margin (SD in the backtest).
    It takes effect with the 2026-11-02 afternoon run, for the games that run re-forecasts, and for every game from the 2026-11-03 morning run. The `model` column marks the rows. As with change 6, the logistic step and margin calibration are still fitted on the historical model without the adjustment.
16. **2026-11-10: two corrections found by reviewing the largest gaps to the market (model changes; author's request).** The ten largest gaps in the log were decomposed into each team's skater and goalie terms.
    * **Goalies without a rolling rate.** A returning goalie now gets his career effect and a goalie with no NHL history the mean of past debut seasons, instead of the replacement rate (docs/gamemodel/methodology.md §10 item 2). The four largest gaps in the log (15 to 18 points, all against the Islanders with Semyon Varlamov in goal) came from this.
    * **The logistic step now matches the live model.** Since changes 5, 6 and 15 the win-probability step and the margin calibration were still fitted on the original model, whose expected margins are less spread (SD 0.60 goals against 0.65), so live probabilities were too extreme: the coefficient on the expected margin was 0.829 where the matching fit gives 0.779. On the 6,502 backtest games the live configuration scored 0.65887 with the mismatched step (calibration slope 0.85) and 0.65835 with the matching one (0.89), paired z = 3.2. Both are now fitted on the historical games run in the live configuration, each season's adjustment models fitted on earlier seasons (`historical_mu` in `src/live/predict.py`).
    
    Replaying the 260 logged games with both corrections: the largest gap falls from 17.6 to 12.0 points and the mean gap from 4.5 to 4.1. Checked and left alone: unmatched injury names (those players were off the published roster, which removes them anyway) and the team ratings (correlation 0.94 with each team's expected-goal differential per game).
    Known gap, not fixed: the lineup is the previous game's, so a regular returning from a missed game is not restored (Nick Suzuki and Kirby Dach, MTL, 2026-11-05: about 5 points of that game's 11-point gap). The feed's out lists are not stored, so an alternative rule cannot be tested yet.
    Takes effect with the 2026-11-10 afternoon run; the `model` column marks the rows. Forecasts logged before it are not changed.
