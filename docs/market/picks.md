# Picks ledger and unit sizing — rules

*Written 2026-10-04 at the model owner's request, before any pick was locked under these rules. This is a record-keeping and sizing tool for the owner's own small wagers. It is separate from the pre-registered market test
(docs/market/methodology.md), whose flat one-unit paper entries remain the test metric. Nothing here is betting or financial advice; the model has no demonstrated edge over the market (see §5).*

## 1. Units

One unit = one dollar. Stakes are whole units, between 1 and 3 per pick. Sizing uses a fixed reference bankroll of 100 units; it does not grow or shrink with results (so a losing run does not change stake sizes, and a winning run does not raise them).

## 2. When a pick is made

For each game on the day's slate, at the daily run (scheduled 12:00 Eastern), on the moneyline only:

1. Take the model's win probability for each side and the best available price among FanDuel, DraftKings and BetMGM (after the source-error guard of docs/market §8.1).
   **From 2026-10-16 (owner's request): the best price among FanDuel, DraftKings, BetMGM, Caesars, Hard Rock Bet and theScore Bet.** The market probability used for sizing is still the three-book consensus of the market test.
   Hard Rock Bet and theScore Bet come from a second source (Action Network, New Jersey lines; docs/market §8 item 13).
2. A side qualifies when the model's probability is at least **3 percentage points** above that price's break-even probability.
3. At most one pick per game. A game is locked once; later runs never change or add to it.
4. **Goalie flag (from 2026-10-13; the daily run has been at 8:00 Eastern since 2026-10-06).** Most starting goalies are not confirmed at that hour. Each pick is shown with whether its edge survives the team's other goalie starting, for any side not yet confirmed:
   a pick whose edge over break-even would fall below 3 points in the worst case is marked **goalie-dependent**. The flag is information for the owner; it does not change what is locked or the stake.
   An afternoon update (about 17:00 Eastern, docs/market/methodology.md §8 item 11) re-forecasts those games and says whether each locked pick still clears 3 points at its locked price. It never adds, removes or resizes a pick in the ledger.

## 3. How many units

The model is not trusted at face value. The probability used for sizing is **halfway between the market and the model**:

`p = market + 0.5 × (model − market)` (market = the three-book average with the margin removed).

The stake is **a quarter of the Kelly fraction** at that probability and the locked price, on the 100-unit reference bankroll, rounded to whole units:

`f = (p × b − (1 − p)) / b`, with b the profit per unit staked; `units = round(0.25 × f × 100)`, limited to 1–3.

If the Kelly fraction at the blended probability is not positive (the gap does not survive the halving and the bookmaker's margin), **no pick is made**.

Roughly, at near-even prices: a model gap of 3 to about 8 points over break-even is 1 unit, about 8 to 11 points is 2 units, and more than about 11 points is 3 units; for the same gap an underdog gets fewer units than a favourite. (The gap is halved from the market's margin-free probability, which sits about 2 points below the break-even of the price, so small gaps shrink to almost nothing.)

## 4. Why these choices

* **Halving the gap:** the model's edge over even a simple team rating was not established in the backtest, and large gaps to the market have so far been more likely to mean missing information than a real edge.
* **Quarter Kelly:** full Kelly sizing assumes the probability is right; a quarter cuts the variance sharply at a small cost in growth if it is right, and limits the damage if it is wrong.
* **Cap of 3 units:** the largest gaps are the least trustworthy.
* **Fixed bankroll:** keeps the ledger simple and the stakes small and predictable.

## 5. What to expect

Sportsbook prices include a margin of roughly 4–5% per game. A model with no real edge loses about half of that per unit staked on average. With stakes of 1–3 dollars and a few picks a day, the season's result will be dominated by chance for a long time:
several hundred picks are needed before profit says anything about the model. The ledger reports profit, return per unit, the record, and the closing comparison only as bookkeeping; the test of the model remains docs/market/methodology.md.

## 6. The ledger

`data/private/logs/picks_log.csv` (append-only; private from 2026-10-18, its fingerprint is public): when locked, the game, the side, the book and price, units, the model and market probabilities. `python -m src.live.picks` (also run by the daily job) locks the day's picks, settles finished games, and writes
`reports/live/picks.md`, `reports/live/picks_profit.png` (cumulative profit through the season) and the public record `data/live/public/picks_record.csv` (each pick in units and decimal odds with the model's probability and the result). A pick whose price is no longer available when the owner goes to place it is the owner's call; the ledger records the price at lock time.

## 7. Change log

*(any change to the sizing or locking rules is recorded here with its date; settled picks are never re-sized)*

1. **2026-10-04: display in decimal odds.** At the owner's request prices are shown as decimal odds (total return per unit staked: 1.80 means a one-unit pick returns 1.80, a profit of 0.80) in the ledger report, the market report and the daily summary. The logs keep the American price as captured (`american`), and `picks_settled.csv` carries both. Sizing and settlement are unchanged.
