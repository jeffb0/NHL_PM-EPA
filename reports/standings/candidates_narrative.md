# Improving the player-to-team forecast — results

Pre-registration: [docs/standings/improvements.md](../../docs/standings/improvements.md). Backtest 2021–2025, 160 team-seasons, points per 82 games; every fitted quantity uses earlier seasons only.
The reference M1 is reproduced exactly (largest difference 0.0 points). Metrics: `reports/metrics/standings_candidates.json`; predictions: `data/outputs/standings_candidates_backtest.csv`.

| model | RMSE | z vs M1 | z vs B2 | calibration slope | seasons better than M1 | decision |
|---|---|---|---|---|---|---|
| M1 (original roster model) | 14.58 | — | −1.46 | 1.22 | — | reference |
| **C1 portable team residual** | **14.00** | 1.33 | −0.68 | 0.86 | 3 of 5 | **implemented** (sound, no reliable change) |
| C2 aggregation-consistent fine-tuning | 14.58 | 0.01 | −1.38 | 0.72 | 3 of 5 | not implemented (less certain, no gain) |
| **C3 deployment-aware minutes** | **14.48** | 1.81 | −1.30 | 1.16 | 3 of 5 | **implemented** (sound, no reliable change) |
| **C1 + C3 (published)** | **13.96** | 1.42 | −0.58 | 0.84 | 3 of 5 | used (vs C1 z = 0.80, vs C3 z = 1.21) |
| B2 last season's goal differential | 13.70 | 1.46 | — | 0.83 | 3 of 5 | reference |
| M3 team-level blend | 13.94 | 1.10 | −0.82 | 0.73 | 3 of 5 | reference |

## Interpretation

**None of the three is a reliable improvement** (no z above 2), so all three are, strictly, parity results. Two were implemented under the pre-registered rule for methodologically sound
candidates, and the combination is the new published forecast; the model owner's instruction and the rule made in advance, not the size of the gain, is the basis for that.

**C1 is the informative one.** Carrying last season's unexplained team result with the players closes about two thirds of the gap between the roster model and last season's goal differential
(14.58 → 14.00 against 13.70) and fixes the compression (calibration slope 1.22 → 0.86). The fitted coefficient is close to 1 in every fold (0.92–1.17), which means that, on average, a player's
share of last season's team residual predicts next season's team result about as well as it predicted last season's: most of what the player values miss **is portable with the players** (consistent
with the team-context finding that about three quarters of persistence travels with them). C1 ends up statistically indistinguishable from the team-level blend M3 (z = −0.15), which is expected
since both use last season's results; C1's advantage is that it keeps the decomposition by player and follows roster moves. It still does not beat last season's goal differential (z = −0.68).

**C3 helps slightly and consistently in minutes.** Projected shares of team minutes are closer to actual shares in all seven seasons (mean absolute share error 0.0285 → 0.0280), and team error drops by 0.1 point (z = 1.81).
Small, but in the expected direction, and the mechanism (coaches play better players more, relative to the roster) is standard.

**C2 found nothing.** Re-weighting the projection's components and age terms on team outcomes did not help (z = 0.01) and made the forecast over-dispersed (slope 0.72): with 30 features and at most 190 training
team-seasons the correction is mostly noise. Not implemented.

**What changes in the 2026-27 forecast.** Teams whose results last season exceeded what their players' values explain, and who kept those players, move up (Carolina +32 goals carried, Ottawa +28, Buffalo +20, Colorado +17);
teams that under-performed their player values move down (Vancouver −28, Calgary −20, Los Angeles −15, New York Rangers −14). Uncertainty is ±18 points (80%) per team.

**Limitations.** Five test seasons; the first has only two unusual training seasons; C1 uses last season's results, so it overlaps the history baselines; carried residuals include last season's luck (for example,
shooting and save percentages beyond what the model credits), which the coefficient near 1 does not separate from real unmodelled skill.
