# Pre-NHL priors for young and thin-history players — results

Pre-registration: [docs/prospects/methodology.md](../../docs/prospects/methodology.md). Metrics: `reports/metrics/prospects.json`. Career pages for all 1,887 skaters in the panel (and the 2026-27 roster) were fetched; 2,242 league-to-NHL transitions.

## Gates

| gate | rule | statistic | result |
|---|---|---|---|
| YP-1 | better on thin-history player-seasons (z > 2) | z = 1.36 (1,105 player-seasons) | FAIL |
| YP-2 | improves the preseason-only game model (z > 2) | z = −0.25 (6,502 games) | FAIL |
| YP-3 | no harm: all players z > −2 and full game model z > −2 | z = 2.62 and z = −0.54 | PASS |

**Decision by the pre-registered rule: implemented** (sound method, no gate at or below −1). Not a reliable team-level improvement.

## Detail

| test | without | with pre-NHL priors | z |
|---|---|---|---|
| player level, thin history (minute-weighted squared error) | 0.01777 | 0.01758 | 1.36 |
| thin history, forwards | 0.02016 | 0.01981 | 1.48 |
| thin history, defensemen | 0.01384 | 0.01390 | −0.55 |
| no NHL history at all (238 player-seasons) | 0.01634 | 0.01635 | −0.04 |
| all players (4,018 player-seasons) | 0.01637 | 0.01625 | 2.62 |
| game model, preseason-only (log loss) | 0.66920 | 0.66922 | −0.25 |
| game model, full | 0.65852 | 0.65855 | −0.54 |
| standings, published model (RMSE, points) | 13.96 | 13.99 | −1.39 |

Translation factors estimated from the data (points per game, league to NHL the next season): AHL 0.46 (1,503 transitions), KHL 0.61, SHL 0.45, Liiga 0.45, NCAA 0.37, QMJHL 0.31, WHL 0.30, OHL 0.29; pooled 0.43. These are close to published equivalencies, which supports the data and the method.

## Interpretation

**Player projections improve; team forecasts do not.** The added information is real at the player level: over all players the projection error falls with z = 2.62, and the fitted coefficient on NHL-equivalent production for forwards is positive and sizeable (0.17 goals per 60 per NHL-equivalent point per game).
The gain is concentrated in forwards with some NHL history (the faded columns still carry weight for a player with one season); for **true rookies it is nil** (z = −0.04) and for defensemen slightly negative, as §8 expected for a points-based measure.

**At the team level nothing moves.** On 6,502 games the preseason-only game model is unchanged (log-loss difference 0.00002) and the standings error is unchanged to slightly worse (0.03 points). Thin-history players get few minutes, the per-player changes are small
(the largest among 2026-27 roster players is about 0.7 goals over a season), and they largely cancel within a team (the largest team change is about 2 goals, under one point). This was the outcome §7 named as plausible.

**What it says about where the team-level gap is.** The preseason prior's weakness is not about young players. It loses to a team rating by about 0.008 log loss; pre-NHL information moves it by 0.00002. The missing information is something shared across a team's established players
(the portable-residual result points the same way), which is what the chemistry model is meant to test.

**Why it was still implemented.** The rule fixed in advance says a sound method with no adverse gate is implemented; it does improve the player projections, which matters for player-level uses (valuing prospects and young players), and its effect on team forecasts is negligible.
The slightly negative standings check (z = −1.39, a difference of 0.03 points on 160 team-seasons) is flagged: if the model owner prefers, the priors can be switched off for the standings forecast by setting `implement` to false in the decision file, with no other change.

**Limitations.** Points-based equivalencies; league names as free text; players who never reached the NHL are absent, so the factors are conditional on making it; the player-level target is the model's own value estimate.
