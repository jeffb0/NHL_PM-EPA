# Chain-free defense and special teams — validation report

Spec and pre-registered gates: [methodology](../../docs/chainfree/methodology.md). The defensive component and the special-teams RAPM parts re-estimated with ρ = 0 (each season fitted independently) in place of the chained prior, then evaluated against real results. The published defaults are unchanged.

## Gates

| Gate | Result | Pass |
|---|---|---|
| CF-1 chain-free rolling total not significantly worse than the chained one at the team level (z > −2) | z = -0.71 (RMSE 41.4 vs 41.2) | ✅ |
| CF-2 with chain-free components PR-6 passes (projected beat last season's rates, z > 2) | z = -2.77 (RMSE 49.3 vs 45.2) | ❌ |
| CF-3 projected rates still beat Marcel at the player level (z > 2) | z = 3.92 | ✅ |

## Persistence by construction (year-over-year correlation, season-centred)

| component | forwards, chained ρ | forwards, ρ = 0 | defensemen, chained ρ | defensemen, ρ = 0 |
|---|---|---|---|---|
| defense | 0.78 | 0.30 | 0.78 | 0.33 |
| pp | 0.55 | 0.28 | 0.47 | 0.18 |
| pk | 0.57 | 0.29 | 0.50 | 0.23 |

Correlation between the chained and chain-free values among qualified skaters: defense 0.87, power play 0.97, penalty kill 0.97. Standard deviation of goals above replacement: defense 1.88 → 1.39, power play 1.62 → 1.50, penalty kill 1.43 → 1.36.

## Forward slopes (next-season deployment-adjusted on-ice goal differential per 60 on the component's rate; state shares controlled)

| component | slope | SE |
|---|---|---|
| defense_chained | 1.61 | 0.21 |
| defense_rho0 | 2.08 | 0.24 |
| pp_chained | 2.86 | 0.39 |
| pp_rho0 | 2.94 | 0.40 |
| pk_chained | 0.60 | 0.31 |
| pk_rho0 | 0.46 | 0.31 |

## Whole model (M protocol, test seasons 2020–2025)

| total | RMSE (goals) |
|---|---|
| chain-free rolling-refit | 41.43 |
| chained rolling-refit | 41.21 |
| chain-free rolling-refit + prior team GD | 38.32 |
| chained rolling-refit + prior team GD | 38.15 |

| comparison | mean difference | z |
|---|---|---|
| rho0 vs chained rolling | -18.12 | -0.71 |
| rho0 vs B0 | 667.00 | 3.77 |
| rho0 vs B1 | -165.13 | -1.09 |
| rho0 vs B3 | -201.87 | -1.40 |
| rho0 plus B1 vs B1 | 82.71 | 1.29 |
| headline rho0 vs headline | 27.39 | 1.13 |

## Projection layer on chain-free component rates

| gate | chained | chain-free |
|---|---|---|
| PR1 | z = 10.63 | z = 8.17 |
| PR2 | z = 5.23 | z = 3.92 |
| PR3 | z = 4.74 | z = 5.65 |
| PR4 | 0.93 | 0.96 |
| PR5 | 78.9% | 79.5% |
| PR6 | z = -2.75 | z = -2.77 |

Results-target variant (exploratory): team RMSE 43.2 against 45.2 for last season's chain-free rates (z = 0.86).

## Why projections lose to last season's values at the team level (exploratory diagnostics)

Team-level M protocol, test seasons 2021–2025; each row replaces the roster-weighted sum.

| sum of | team RMSE | z vs last season |
|---|---|---|
| s_last | 44.89 | -0.04 |
| s_blend_70_30 | 45.84 | -1.57 |
| s_two_seasons | 47.53 | -2.63 |
| s_marcel_5_4_3 | 47.46 | -2.48 |

| extra ballast M0 (minutes) in the projection features | team RMSE | z vs last season |
|---|---|---|
| 0 | 51.34 | -3.69 |
| 250 | 50.10 | -3.41 |
| 500 | 49.27 | -3.15 |
| 1000 | 48.38 | -2.75 |

Within-season correlation of the summed values with team goal differential: projected 0.62, last season 0.68; SD of the sums 16.4 against 23.3 goals.

## Interpretation

**The chain does create persistence.** Year-over-year correlation of the defensive values is 0.78 with the chained prior (forwards and defensemen alike) and 0.30 to 0.33 without it; for the power play it falls from 0.55 (forwards) and 0.47 (defensemen) to 0.28 and 0.18, and for the penalty kill from 0.57 and 0.50 to 0.29 and 0.23. Most of the apparent stability of the published defensive value is carried forward by the prior, not re-observed each season. The chain-free values are strongly related to the chained ones (correlation 0.87 for defense, 0.97 for both special-teams states) but less dispersed for defense (SD 1.39 against 1.88 goals).

**The chain is not why projections fail at the team level.** Chain-free components leave the whole-model result where it was: the chain-free rolling-refit total forecasts next-season team goal differential with RMSE 41.4 against 41.2 for the chained one (z = −0.71, not significant), so CF-1 passes only in the sense of "not significantly worse", and there is no improvement. The projection layer on chain-free rates behaves the same way: it still beats Marcel at the player level (z = 3.9, CF-3 passes) and still loses to last season's rates at the team level (RMSE 49.3 against 45.2, z = −2.77, so CF-2 fails). The chained-prior explanation offered in the projection report is therefore **not supported**.

**What the diagnostics point to instead (exploratory, not a gate).** At the team level, last season's rates alone give the best forecast (RMSE 44.9), and every way of adding older seasons makes it worse: a 70/30 blend 45.8, two seasons 47.5, the 5/4/3 Marcel weights 47.5. Heavier shrinkage inside the projection helps at the team level (48.4 with a 1,000-minute ballast against 51.3 with none), and the projected sums are less dispersed and rank teams less well within a season (correlation 0.62 with team goal differential against 0.68 for last season's sums). One reading is that a player's value estimate carries the context of his team in that season (shared credit for system, linemates and arena), so the sum of last season's values for a roster that mostly stays put is largely a measure of last season's team, and smoothing over older seasons dilutes that. Talent-only smoothing helps forecast a *player*, not the team he is on. This reading fits the whole-model finding that the total forecasts about as well as the prior team goal differential; it has not been tested directly (for example by measuring the share of a player's value that moves with him when he changes teams).

**Effect on the published values.** None: the chained defaults stay. The chain-free values are in `chainfree_values.parquet` as variants (`def_gar_rho0`, `st_<state>_rapm_gar_rho0`). They are the better choice for questions about persistence and forecasting a player's own future value, since they do not build in persistence; the chained values remain better within season, where the prior helps held-out prediction (per the defense and special-teams validations).

**Forward slopes.** Against next-season deployment-adjusted on-ice results, defense carries slopes of 1.61 ± 0.21 (chained) and 2.08 ± 0.24 (chain-free), power play 2.86 and 2.94, penalty kill 0.60 and 0.46 (standard errors 0.3 to 0.4): the chain-free values are as predictive of results as the chained ones for defense (higher, though within a few standard errors) and the power play, and slightly lower for the penalty kill (within noise), the same conclusion as the team-level result.

**Limitations.** ρ = 0 estimates are noisier for thin samples; the chained and chain-free intervals are not comparable (no new intervals were computed); the special-teams α is the one chosen under the chained fits.

