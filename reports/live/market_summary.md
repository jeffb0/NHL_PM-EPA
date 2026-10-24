# Model against the betting market — summary

Protocol: [docs/market/methodology.md](../../docs/market/methodology.md). Forecasts are logged before each game and scored against the margin-free consensus of FanDuel, DraftKings and BetMGM. The per-game log contains sportsbook prices and is kept private; its SHA-256 at every run is in [data/live/public/fingerprints.csv](../../data/live/public/fingerprints.csv). **Analysis of a model, not betting advice.**

Games logged: 158. Scored: 144. Pending: 14.

## Results so far (too few games for any claim: 144 of 300)

| | model | market |
|---|---|---|
| log loss | 0.6290 | 0.6442 |
| Brier | 0.2200 | 0.2264 |
| picked the winner | 65.3% | 61.8% |

Log-loss difference (market − model; positive favors the model): +0.0152, z = 1.64.

Paper bets (one unit when the model is at least 3% above break-even at the best of the three books): 61 bets, +3.52 units, return +5.8% per unit (95% interval -16.8% to +27.9%); too few bets for any claim.
