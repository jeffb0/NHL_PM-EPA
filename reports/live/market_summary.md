# Model against the betting market — summary

Protocol: [docs/market/methodology.md](../../docs/market/methodology.md). Forecasts are logged before each game and scored against the margin-free consensus of FanDuel, DraftKings and BetMGM. The per-game log contains sportsbook prices and is kept private; its SHA-256 at every run is in [data/live/public/fingerprints.csv](../../data/live/public/fingerprints.csv). **Analysis of a model, not betting advice.**

Games logged: 119. Scored: 113. Pending: 6.

## Results so far (too few games for any claim: 113 of 300)

| | model | market |
|---|---|---|
| log loss | 0.6087 | 0.6277 |
| Brier | 0.2105 | 0.2185 |
| picked the winner | 68.1% | 64.6% |

Log-loss difference (market − model; positive favors the model): +0.0189, z = 1.94.

Paper bets (one unit when the model is at least 3% above break-even at the best of the three books): 47 bets, +5.62 units, return +12.0% per unit (95% interval -13.3% to +35.8%); too few bets for any claim.
