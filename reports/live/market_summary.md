# Model against the betting market — summary

Protocol: [docs/market/methodology.md](../../docs/market/methodology.md). Forecasts are logged before each game and scored against the margin-free consensus of FanDuel, DraftKings and BetMGM. The per-game log contains sportsbook prices and is kept private; its SHA-256 at every run is in [data/live/public/fingerprints.csv](../../data/live/public/fingerprints.csv). **Analysis of a model, not betting advice.**

Games logged: 146. Scored: 132. Pending: 14.

## Results so far (too few games for any claim: 132 of 300)

| | model | market |
|---|---|---|
| log loss | 0.6198 | 0.6396 |
| Brier | 0.2157 | 0.2242 |
| picked the winner | 67.4% | 63.6% |

Log-loss difference (market − model; positive favors the model): +0.0198, z = 2.16.

Paper bets (one unit when the model is at least 3% above break-even at the best of the three books): 53 bets, +6.05 units, return +11.4% per unit (95% interval -10.9% to +35.2%); too few bets for any claim.
