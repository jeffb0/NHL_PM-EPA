# PM-EPA: Puck-Movement Expected Goals Added

An NHL player-valuation and forecasting model built end-to-end from public play-by-play (2017-18 to 2025-26 regular seasons, 11,052 games), plus the tools that run it on the current season.

At its core, **PM-EPA** measures how many expected goals a skater adds by changing the state of the puck while his team has it (moving it, winning it or losing it), against the value of the state his team had before. It is the hockey analogue of NFL EPA:

```
PM-EPA(action) = V(state after) − V(state before)        V = expected goals the possession will produce
```

Around it sit further components (defense, finishing, turnovers, faceoffs, penalties, special teams, goaltending) on one goals-above-replacement scale, a projection and aging layer, team and game forecasts, and a daily pipeline for the live season.

> **Read this first.** Public NHL play-by-play records faceoffs, hits, takeaways, giveaways, shots and stoppages, but **no passes, carries or zone entries.** PM-EPA therefore observes puck movement only *between* recorded events. It credits visible endpoints (the player who takes the puck away, carries it himself, shoots or gives it away) and splits unobserved movement between those players and the teammates on the ice. Treat it as a **transition and possession-value metric with partially pooled credit**, not a pass-by-pass ledger. Details: [limitations](docs/pm_epa/limitations.md).

## How the work is done

Every model change follows the same discipline: the hypothesis, test and adoption rule are written down *before* the result is seen (each `docs/<topic>/methodology.md` is a pre-registration), gates are reported rather than tuned, and any change after the fact is logged as a deviation. Many of the ideas tested here did not survive; those results are kept and summarised below alongside the ones that did.

## Headline results

<!-- BEGIN HEADLINE -->
| Check | Result |
|---|---|
| Value model calibration (untouched 2023-26 seasons) | slope **1.004**, intercept −0.003 |
| Decision-point value V_dec, out-of-sample R² | **0.42** (test) |
| Shot xG (cross-fitted) | AUC **0.773**, Σxg/Σgoals **1.007** |
| Year-over-year r, 5v5 PM-EPA/60, ≥500 min (season-centred) | forwards **0.53**, defense **0.52** (targets ≥ 0.40 / 0.35) |
| …same, players who changed teams | forwards **0.37**, defense **0.30** |
| Split-half reliability (≥800 min, Spearman-Brown) | forwards **0.72**, defense **0.78** |
| r with 3-season 5v5 offensive xG RAPM | forwards **0.59–0.71**, defense **0.42–0.62**; adds R² beyond ixG/60 + G+A1/60 in every window |
| r with G+A1/60 (convergent) / individual SH% (discriminant) | 0.40 / 0.15 (F); 0.35 / 0.25 (D) |
| Accounting identity (every possession's components = its xG) | max residual **1.1e-16** over 1.88M possessions |

**Face validity (5v5, ≥500 min).**
* **Defense:** Cale Makar leads in each of the last three seasons; Quinn Hughes, Adam Fox, Evan Bouchard, Devon Toews, Zach Werenski and Miro Heiskanen fill out the top.
* **Forwards:** elite play-drivers (Matthews, McDavid, Jack Hughes, Crosby, Point, Hischier) sit alongside high-motor retrieval and net-front forwards (Hyman, Hagel, Brady Tkachuk). This is a *transition/possession* metric, so pass-first playmakers rank lower than in points-based stats.
<!-- END HEADLINE -->

Full evidence: [validation report](reports/validation_report.md) · [model card](reports/model_card.md).

## How PM-EPA works

1. **Possessions.** Each game's event stream becomes possessions (1.93M in all). A possession starts with a faceoff win, a takeaway or an inferred loose-puck recovery, and ends with a shot, giveaway, loss or stoppage ([definitions §3](docs/pm_epa/definitions.md)).
2. **Shot xG.** A gradient-boosted shot model (distance, angle, type, rebound, rush, prior event, strength, off-wing and more), cross-fitted by season.
3. **Two value functions**, both in expected goals, never in goals:
   * **V_start(s):** what a possession is worth once the team *gains* the puck in state s (location × how it was gained × strength × score).
   * **V_dec(s):** the average outcome of possessions whose *terminal decision* (shoot, or lose the puck) happens at s; the analogue of NFL expected points at the snap.
4. **Credit.**
   * Movement Δ = V_dec(where the possession got to) − V_start(where it began). Same player at both ends: 100% to him (an observed carry). Otherwise 35% to the start player, 35% to the end player, 30% to the on-ice skaters.
   * Takeaways earn the value of the state they create; giveaways cost the value of the decision point they wasted.
   * A shot earns only its *attempt* value, V_att − V_dec. The quality of the release (xG − V_att) is excluded, and goals never count.
5. **Validation as a gate:** calibration, year-over-year stability (including players who changed teams), split-half reliability, convergent and discriminant correlations, RAPM alignment, team-level sums, ablations, and an exact accounting identity.

The first build failed face validity (stars at the bottom); the diagnosis and fix are in the [Phase 6 summary](docs/pm_epa/phase_summaries/phase_6_summary.md).

**Worked example (production settings, λ = 0).** A forward steals the puck in the neutral zone (V_start = 0.020). Six seconds later he shoots from the high slot himself: V_dec there = 0.060, a typical attempt from there V_att = 0.075, and this shot's xG = 0.080.

| component | value (xG) |
|---|---|
| takeaway: value of the state he created | +0.020 |
| carry: 0.060 − 0.020 (same player at both ends) | +0.040 |
| shot attempt: 0.075 − 0.060 | +0.015 |
| **PM-EPA** | **+0.075** |
| *shot quality 0.080 − 0.075 (excluded)* | *(+0.005)* |

More examples, including real 2025-26 possessions: [attribution examples](docs/pm_epa/attribution_examples.md).

## Quickstart

Requires Python 3.11 (installed by [`uv`](https://github.com/astral-sh/uv)) and about 1 GB of disk. No API keys are needed.

```bash
make install    # create .venv and install the pinned requirements
make smoke      # full pipeline on 50 games in about a minute, in isolated output directories
make all        # ingest 9 seasons (about 15 min, cached afterwards), then the pipeline, tests and notebooks
make test       # unit tests
```

Individual steps: `python -m src.pipeline --only xg value epa metrics`, or `--from value`. All tunables live in [`config.yaml`](config.yaml) (validated by `src/config.py`).

## The player valuation

Each component is a separate, separately validated model, and all of them sit on one goals-above-replacement scale (5.90 goals per win, estimated from the data).

| Component | What it measures | Key validation | Docs |
|---|---|---|---|
| **PM-EPA** (possession) | value added by moving, winning and losing the puck | see headline results | [methodology](docs/pm_epa/methodology.md) |
| **DPM** (defense) | 5v5 xG against prevented per 60: prior-chained stint RAPM, with an exact breakdown of *how* from the possession ledger | next-season xGA ΔMSE −0.048 (≈ 6.8 SE); YOY F 0.41, D 0.38 | [methodology](docs/defense/methodology.md) · [report](reports/defense/validation_report.md) |
| **FIN** (finishing) | goals above a finishing-only xG, from a hierarchical shooter/goalie model checked against full-Bayes NUTS | next-season log loss z = +7.2 (raw G/xG is *worse* than no player information) | [methodology](docs/finishing/methodology.md) · [report](reports/finishing/validation_report.md) |
| **Turnovers** | the opponent's possession value after a player's turnovers, role-adjusted (TO-RA) | adds to team goal differential (PM-EPA slope 1.68 → 1.05) | [methodology](docs/components/methodology.md) · [report](reports/components/validation_report.md) |
| **Faceoffs** | taker skill, valued by the ledger's start values | z = 24.5 vs no taker; a win is worth 0.0245 xG | same |
| **Penalties** | drawing and taking, by the net xG of the man advantage | 0.175 xG per 5v5 minor; both repeatable (YOY 0.48, 0.52) | same |
| **Special teams** | PP, PK, 4v4 and 3v3: strength-specific RAPM plus finishing in that state | beats the baseline out of sample (PP/PK z = 6.6); PP fails YOY for defensemen (0.18) | [methodology](docs/specialteams/methodology.md) · [report](reports/specialteams/validation_report.md) |
| **Goaltending** | goals above replacement from a logistic mixed model on unblocked shots | beats the goalie-free baseline (z = 2.75); single-season YOY 0.13 (gate 0.15) | [methodology](docs/goaltending/methodology.md) · [report](reports/goaltending/validation_report.md) |

**Combining them.** A joint calibration asked whether every component is worth one goal per goal. Team-level slopes on shrunk components mostly reflect shrinkage compression, not scale error, and with goaltending included the pre-registered adoption test fails (z = 1.58), so **all weights stay at 1** ([report](reports/calibration/validation_report_all_gk.md)). Position-specific weights are published as a variant. The preferred skater total, `total_gar_all_tr`, adds an on-ice transition term by an explicit judgment call that overrides its failed gates ([decision log](docs/transition/methodology.md)). `data/outputs/player_value_all_in.csv` stacks skaters and goalies in one table.

**Does it explain teams?** The total predicts next-season team goal differential well against a goalie-only baseline (out-of-sample R² 0.33, z = 4.6, better in all six test seasons) but is **no better than a team's own prior goal differential** (RMSE 39.9 vs 39.4) and is leakage-sensitive; read it as a decomposition of team results into player credit, not a forecast that beats recent results ([report](reports/wholemodel/validation_report.md)).

## Forecasting

| Model | What it does | Result | Docs |
|---|---|---|---|
| **Projection and aging** | each skater's rate one to three seasons ahead with 80% intervals; position-specific aging curves | beats Marcel (z = 5.2) and age terms help (z = 4.7); **fails at team level** against last season's rates (z = −2.75) | [methodology](docs/projection/methodology.md) · [report](reports/projection/validation_report.md) |
| **Standings** | bottom-up points per 82 from opening-night rosters: projected rates × projected minutes, plus a portable team residual and deployment-aware minutes | RMSE 13.96 points over 2021–2025, against 13.70 for last season's goal differential: parity, not an improvement | [methodology](docs/standings/methodology.md) · [results](reports/standings/candidates_narrative.md) · [2026-27 forecast](reports/standings/forecast_2026.md) |
| **Game model** | every skater's rating starts at his projection and is updated after each game from the expected-goal surprise, by ice time; goalies update by shots | log loss 0.6585 on 6,502 games vs 0.6613 for a tuned team rating (z = 1.99); in-season updating z = 5.7; player attribution z = 2.4 | [methodology](docs/gamemodel/methodology.md) · [report](reports/gamemodel/validation_report.md) |

## Experiments

Every study below was pre-registered with its adoption rule. "Adopted" means it is used by the published model.

| Study | Question | Result | Status |
|---|---|---|---|
| [Pre-NHL priors](reports/prospects/narrative.md) | Do league-translated junior/college/European numbers and draft slot help young players' projections? | players z = 2.62; no team-level gain | adopted |
| [Elite-prospect priors](reports/prospects/elite_narrative.md) | Do top picks debuting at 18 deserve a large premium? | no: they averaged about a generic rookie; no forecast gain | adopted (no harm) |
| [Early-sample updating](reports/prospects/hybrid_narrative.md) | How fast should a young player's own games move his rating? | reliable from 5 games (z = 2.25), 40 games z = 4.57 | adopted |
| [Score effects](reports/scoreeffects/narrative.md) | Should chances be re-weighted by score state? | small at 5v5 except teams trailing in the third; team-rating gain z = 2.84 | adopted |
| [Chemistry](reports/chemistry/narrative.md) | Do players have pairwise synergy beyond their individual value? | no (z = −2.66; synergies uncorrelated across windows) | not adopted |
| [Possession scale](reports/possession/narrative.md) | Is the possession component under-scaled? | tuning chose no rescaling | not adopted |
| [Team residual](reports/residual/narrative.md) | What is the part of team results the components miss? | 45% is 5v5 play that is ranked right but under-scaled | descriptive |
| [5v5 de-shrinkage](reports/deshrink/narrative.md) | Does scaling the 5v5 block fix that? | no: every comparison slightly worse | not adopted |
| [Team context](reports/teamcontext/validation_report.md) | How much of a player's value travels to a new team? | 74% with chain-free components; destination context helps movers' projections (z = 5.1) | adopted in the projection |
| [Chain-free components](reports/chainfree/validation_report.md) | Is persistence an artefact of chained priors? | partly (defense YOY 0.78 → 0.30) but it does not fix team-level forecasts | variant published |
| [NHL EDGE tracking](reports/edge/validation_report.md) | Do skating-speed and distance data improve projections? | no (z = −0.63); physical decline is already in the value history | not adopted |
| [Rest and travel](reports/schedule/narrative.md) | Do rest days, travel, time zones or road trips matter beyond back-to-backs? | no (z −1.6 to +0.2) | not adopted |
| [Situational "angles"](reports/angles/narrative.md) | Do 19 broadcaster and betting-trend situations (revenge, letdown, streaks…) predict anything? | no (joint p = 0.94) | not adopted |
| [Model structure](reports/structure/narrative.md) | Team layer, shot volume, tired or rusty goalies, home ice by team, star power? | none improves forecasts out of sample | not adopted |
| [Day-of information](reports/infoquality/narrative.md) | How much are confirmed lineups and starters worth to the model? | almost nothing (goalie term worth 0.0005 log loss) | descriptive |

## The live season

The trained models are applied to the current season without refitting anything.

```bash
python -m src.live.update 2026                 # fetch completed games and run them through the trained models into data/live/
python -m src.live.game_report <game_id>       # one-game report: game flow, skater impact by situation, takeaways
python -m src.live.daily                       # forecasts for today and tomorrow, prices, picks, morning summary
python -m src.live.afternoon                   # re-forecast games whose starting goalie was not confirmed in the morning
python -m src.live.weekly                      # season xG standings, top skaters, team percentiles
python -m src.live.predict WPG DET 2026-10-04  # one game
```

* **Forecasts** use the frozen game model with in-season updating. Each morning run snapshots the published rosters and reads projected lineups, starting goalies and injury lists; an afternoon run re-forecasts games whose starter was unconfirmed. On the second night of a back-to-back the rested goalie is assumed unless a starter is confirmed.
* **Prospective market test** ([protocol](docs/market/methodology.md)): every forecast is logged before puck drop and scored against the margin-free consensus of three sportsbooks. Nothing is claimed before 300 scored games. Running totals: [market summary](reports/live/market_summary.md).
* **Picks ledger** ([rules](docs/market/picks.md), [ledger](reports/live/picks.md), [record](data/live/public/picks_record.csv)): moneyline picks where the model is at least 3 points above break-even, sized at 1–3 units, settled daily. It is bookkeeping, not advice; the model has no demonstrated edge.
* **Public and private files.** The forecast, odds and scoring logs contain third-party data (sportsbook prices, lineup feeds), so they live in a separate private repository under `data/private/`. Each run publishes a SHA-256 fingerprint of every private log ([fingerprints](data/live/public/fingerprints.csv)), so the private record can be checked against what existed at the time.

Game reports are single-game descriptions, not ratings.

## Using PM-EPA responsibly

* **EV = 5v5.** Qualify at ≥ 500 5v5 minutes and **read the 95% CI before the rank**.
* **`pm_direct_per60_ev` / `pm_pooled_per60_ev`** show how much of a number is individually observed and how much is shared with linemates.
* **Own-zone giveaways are nearly free in the headline number (λ = 0).** `pm_epa_per60_ev_lam1` also charges the opponent's gain from a turnover; read both when turnover risk matters.
* **2024-25 onward is a different scorekeeping regime:** recorded giveaways doubled. Compare players within regime.
* **PM-EPA is offense and transition value.** It does not measure defensive positioning, and on its own it is not WAR; the combined table is the all-in view.

## Outputs

| File | Contents |
|---|---|
| `data/outputs/pm_epa_player_leaderboards.csv` | player-season PM-EPA: totals and per-60 by strength, per-100 possessions, position percentile, 95% bootstrap CI, direct vs pooled split |
| `data/outputs/player_value_all_in.csv` | every component on one goals-above-replacement scale, skaters and goalies |
| `data/outputs/player_projection.csv` | one- to three-season projections with intervals |
| `data/outputs/pm_epa_team_season.csv`, `pm_epa_team_game.csv`, `pm_epa_player_game.csv` | team and game aggregates |
| `data/value_surfaces/` | value tables by strength and the cross-fitted V_start / V_dec / V_att models |
| `reports/` | validation reports, model cards, figures and metrics for every component and study |

## Repository map

```
config.yaml              all tunables (validated by src/config.py)
src/
  ingestion/             NHL API scraper and parser; rosters, career pages, EDGE summaries
  possession/ xg/ states/ value/ epa/ metrics/
                         possession chains, shot xG, state spaces, value functions, the PM-EPA engine, aggregation
  defense/ finishing/ components/ specialteams/ goaltending/
                         the other valuation components
  valuation/ calibration/ transition/ wholemodel/
                         replacement level, goals per win, joint calibration, the combined table, whole-model tests
  projection/ edge/ teamcontext/ chainfree/
                         projection and aging, tracking features, team context, chain-free variants
  standings/ gamemodel/ prospects/
                         team and game forecasting, young-player priors
  chemistry/ residual/ scoreeffects/ schedule/ angles/ structure/ deshrink/ infoquality/
                         the experiments in the table above
  live/                  current-season pipeline: update, forecasts, odds, picks, reports
  validation/ viz/       validation suite and figures
tests/                   unit tests (pytest)
notebooks/               analysis notebooks (generated by notebooks/_build_notebooks.py)
docs/<topic>/            methodology and pre-registration for each component and study
reports/<topic>/         results, validation reports and model cards
schemas/                 SQL DDL and the transitions Parquet schema
```

## Documentation

PM-EPA: [methodology](docs/pm_epa/methodology.md) · [formulas](docs/pm_epa/formulas.md) · [definitions](docs/pm_epa/definitions.md) · [attribution examples](docs/pm_epa/attribution_examples.md) · [edge cases](docs/pm_epa/edge_cases.md) · [limitations and roadmap](docs/pm_epa/limitations.md) · [literature review](docs/pm_epa/literature_review.md) · [data model](docs/pm_epa/data_model.md) · [post-mortem](docs/pm_epa/post_mortem.md). Every other component and study has its own `docs/<topic>/methodology.md` and `reports/<topic>/` results, linked from the tables above.

## License and data

Code is released under the [MIT License](LICENSE). NHL data are used via the league's public web APIs for research; this project is not affiliated with the NHL.
