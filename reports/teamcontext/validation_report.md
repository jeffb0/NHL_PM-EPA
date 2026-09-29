# Team context ("system / chemistry") — report

Spec and pre-registered gates: [methodology](../../docs/teamcontext/methodology.md). No coach data exists in the repository, so context is the team-season: coaching, system, linemates and arena are bundled and not separable here.

## Gates

| Gate | Result | Pass |
|---|---|---|
| TC-1 value travels less than it stays (stayer minus mover slope, total rate; 95% interval excludes 0) | 0.07 (0.01 to 0.13) | ✅ |
| TC-2 movers: destination-context coefficient positive, t > 2 | 0.21, t = 3.56 | ✅ |
| TC-3 context features improve the projection for movers (z > 2) and do not hurt overall (z > −2) | movers z = 5.14; all z = 3.64 | ✅ |
| TC-4 team level: projections with context beat those without (z > 2) | z = 3.11 | ✅ |

## A. How much of a player's value travels (4442 consecutive-season pairs with ≥ 300 minutes in both; 1047 movers)

| component | stayer slope | mover slope | stayer r | mover r | share that travels (mover / stayer slope) | 95% interval |
|---|---|---|---|---|---|---|
| total | 0.80 | 0.72 | 0.76 | 0.67 | 0.91 | 0.84 to 0.98 |
| possession | 0.56 | 0.36 | 0.54 | 0.34 | 0.65 | 0.55 to 0.77 |
| defense | 0.85 | 0.85 | 0.78 | 0.78 | 1.01 | 0.94 to 1.08 |
| finishing | 0.81 | 0.68 | 0.75 | 0.64 | 0.84 | 0.76 to 0.94 |
| turnover | 0.53 | 0.34 | 0.54 | 0.34 | 0.64 | 0.50 to 0.78 |
| penalties | 0.42 | 0.42 | 0.40 | 0.40 | 0.99 | 0.85 to 1.19 |
| power play | 0.82 | 0.70 | 0.78 | 0.66 | 0.85 | 0.73 to 0.97 |
| penalty kill | 0.72 | 0.61 | 0.69 | 0.58 | 0.85 | 0.76 to 0.93 |

### A, chain-free components (defense and special-teams RAPM re-estimated with ρ = 0; the chained prior carries a player's own coefficient across teams, which inflates how much of the published defensive value 'travels')

| component | stayer slope | mover slope | share that travels | 95% interval |
|---|---|---|---|---|
| total | 0.67 | 0.50 | 0.74 | 0.65 to 0.85 |
| defense | 0.32 | 0.27 | 0.82 | 0.57 to 1.09 |
| power play | 0.74 | 0.59 | 0.79 | 0.66 to 0.93 |
| penalty kill | 0.67 | 0.57 | 0.86 | 0.77 to 0.95 |

## B. Context regression (next-season total rate on the shrunk history, age terms and context; player-cluster SEs)

| group | rows | rt_total coefficient (t) | ctx_dest coefficient (t) | ctx_origin coefficient (t) |
|---|---|---|---|---|
| movers | 1189 | 1.09 (26.1) | 0.21 (3.6) | -0.18 (-3.0) |
| stayers | 3851 | 1.21 (65.8) | -0.00 (-0.1) | — |
| movers_F | 782 | 1.03 (19.8) | 0.07 (0.9) | -0.17 (-2.1) |
| movers_D | 407 | 1.10 (14.6) | 0.39 (4.4) | -0.16 (-2.2) |

## C. Do context features improve the projections? (test seasons 2020–2025, players with ≥ 300 target minutes)

| rows | MSE base | MSE + context | MSE + context + prior team GD | context z | context + GD z |
|---|---|---|---|---|---|
| all | 0.0164 | 0.0161 | 0.0160 | 3.6362 | 4.3957 |
| movers | 0.0178 | 0.0165 | 0.0165 | 5.1355 | 4.3891 |
| stayers | 0.0160 | 0.0160 | 0.0158 | -0.6520 | 1.6141 |

Team level (M protocol, test seasons 2021, 2022, 2023, 2024, 2025):

| projection | team RMSE (goals) |
|---|---|
| base | 48.38 |
| ctx | 47.14 |
| ctx gd | 45.92 |
| rolling refit | 44.89 |
| B1 | 41.42 |

| comparison | mean difference | z |
|---|---|---|
| ctx vs base | 118.33 | 3.11 |
| ctx gd vs base | 231.81 | 3.74 |
| ctx vs rolling last season | -207.45 | -2.20 |
| ctx gd vs rolling last season | -93.97 | -1.09 |
| ctx gd vs prior team gd | -393.22 | -1.92 |

## Interpretation

**Most of a player's value travels; a measurable minority does not.** Players who change teams keep 91% of the persistence that stayers show in total value (mover slope 0.72 against 0.80; 95% interval for the share 0.84 to 0.98). That figure flatters the published defense and special teams, because the chained prior carries a player's own coefficient across teams. With those components re-estimated without the chain, the share that travels is **74%** for the total (0.65 to 0.85), 82% for defense (0.57 to 1.09, wide), 79% for the power play and 86% for the penalty kill. By component, the parts most tied to the team are possession (65% travels) and turnover (64%); finishing, the power play and the penalty kill sit near 85%; penalties travel fully. Roughly a fifth to a quarter of what makes a player's estimate persistent is therefore team-specific: system, coaching, linemates and arena together.

**A mover's new context shifts his next-season value, and his old context drags in the opposite direction.** For movers, next-season value rises by 0.21 goals per 60 for each unit of the destination team's prior-season mean rate (t = 3.6) and falls by 0.18 per unit of the origin team's (t = −3.0), holding his own shrunk history fixed. For stayers the context coefficient is zero (t = −0.1), as it should be: their own history already reflects it. The effect is about a fifth of the difference between two teams' contexts, not a full pass-through. By position the pattern differs: **defensemen** are strongly destination-dependent (0.39, t = 4.4) while **forwards** show mainly the origin drag (−0.17, t = −2.1) and no significant destination effect (0.07, t = 0.9). That fits the intuition that defensemen depend more on partners and system than forwards do, but it rests on 407 defense and 782 forward movers and should be read as suggestive.

**Adding context improves projections where it should.** With the destination and origin contexts and a mover flag as extra features, the projection improves for movers (error 0.0178 to 0.0165, z = 5.1), improves overall (0.0164 to 0.0161, z = 3.6) and leaves stayers unchanged (z = −0.7). At the team level (roster known) the context-aware projections beat the plain ones (RMSE 47.1 against 48.4, z = 3.1), and adding the destination's prior goal differential per game helps more (45.9, z = 3.7 against the plain projection). They still do not beat last season's rolling-refit rates (44.9) or the prior season's team goal differential (41.4; z = −1.9), so context recovers part of the gap the earlier projection test found, not all of it. All four pre-registered gates pass.

**What this does not settle.** The unit of context is the team-season, so coaching cannot be separated from teammates, deployment or arena, and no coach data exists in the repository. It also cannot separate system fit from the ridge models' remaining teammate confounding (the very issue you raised): a defenseman who is better with one partner is partly a chemistry effect and partly a partner effect. What the results do support is that team context is a real, modest part of an estimate and that using it improves projections for players who change teams. Movers are a selected group (contracts, trades, age) and primary-team assignment misplaces traded players, so the mover results describe association, not a clean causal effect of moving.

**Using it.** `python -m src.teamcontext.move --player "Name" --team XXX` projects a player under a chosen destination and splits the change into a context effect and an average move effect (the mover-flag coefficient, the same for every destination). It is a convenience built on the validated projection ridge, not a validated forecast of any single move.

**Next step if this is worth pursuing.** Ingest head coaches by game (the NHL API game pages list them). That gives a cleaner test than moves: same team, coaching change, does value change? It would isolate a coaching-and-system component from a teammate component, which team-season context cannot.

