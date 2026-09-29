# PM-EPA — Puck-Movement Expected Goals Added

**PM-EPA** measures how many expected goals a skater adds by changing the state of the puck while his team has it (moving it, winning it, or losing it), against the value of the state his team had before. It is the hockey analogue of NFL EPA (Expected Points Added), built end-to-end from public NHL play-by-play for the 2017-18 → 2025-26 regular seasons (11,052 games).

```
PM-EPA(action) = V(state after) − V(state before)        V = expected goals the possession will produce
```

> **Read this first.** Public NHL play-by-play records faceoffs, hits, takeaways, giveaways, shots and stoppages, but **no passes, carries or zone entries.** PM-EPA therefore observes puck movement only *between* recorded events. It credits visible endpoints (the player who takes the puck away, carries it himself, shoots, or gives it away) and splits unobserved movement between those players and the teammates on the ice. The engine is ready for tracking-grade event data. Until then, treat PM-EPA as a **transition and possession-value metric with partially pooled credit**, not a pass-by-pass ledger. Details: [limitations](docs/pm_epa/limitations.md).

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

Full evidence: [`reports/validation_report.md`](reports/validation_report.md) · model card: [`reports/model_card.md`](reports/model_card.md).

## How it works (one screen)

1. **Possessions.** Each game's event stream becomes 1.93M possessions. A possession starts with a faceoff win, a takeaway, or an inferred loose-puck recovery, and ends with a shot, giveaway, loss or stoppage. ([definitions §3](docs/pm_epa/definitions.md))
2. **Shot xG.** A gradient-boosted shot model (distance, angle, type, rebound, rush, prior event, strength, off-wing…) is cross-fitted by season; AUC 0.77, calibration 1.01.
3. **Two value functions**, both in expected goals and never in goals:
   * **V_start(s):** what a possession is worth once the team *gains* the puck in state s. State = location × how it was gained × strength × score.
   * **V_dec(s):** the average outcome of possessions whose *terminal decision* (shoot, or lose the puck) happens at s. This is the analogue of NFL EP at the snap.
4. **Credit.**
   * Movement Δ = V_dec(where the possession got to) − V_start(where it began).
     * Same player at both ends → 100% to him (observed carry).
     * Otherwise 35% to the start player, 35% to the end player, 30% to the on-ice skaters.
   * Takeaways earn the value of the state they create.
   * Giveaways cost the value of the decision point they wasted.
   * A shot earns only its *attempt* value, V_att − V_dec: getting a shot off rather than losing the puck. The *quality* of the release (xG − V_att) is excluded, and goals never count.
5. **Validation as a gate.**
   * Calibration and year-over-year stability, including a team-changer subset.
   * Split-half reliability; convergent and discriminant correlations.
   * RAPM alignment; team-level sums; ablations.
   * An exact accounting identity: each possession's components sum to its xG.

Every design choice (grid, shrinkage, λ, credit shares) was set by a protocol written *before* seeing its result. The first build failed face validity, with stars at the bottom. It was diagnosed and fixed in the open: see the [Phase 6 summary](docs/pm_epa/phase_summaries/phase_6_summary.md).

## Worked example (production settings, λ = 0)

A forward steals the puck in the neutral zone. V_start = 0.020. Six seconds later he shoots from the high slot himself: V_dec there = 0.060, a typical attempt from there V_att = 0.075, and this shot's xG = 0.080.

| component | value (xG) |
|---|---|
| takeaway: value of the state he created | +0.020 |
| carry: 0.060 − 0.020 (same player at both ends) | +0.040 |
| shot attempt: 0.075 − 0.060 | +0.015 |
| **PM-EPA** | **+0.075** |
| *shot quality 0.080 − 0.075 (excluded)* | *(+0.005)* |

More examples, including real 2025-26 possessions: [`docs/pm_epa/attribution_examples.md`](docs/pm_epa/attribution_examples.md).

## Quickstart

Requires Python 3.11 (installed automatically by [`uv`](https://github.com/astral-sh/uv)) and ~1 GB of disk. No API keys are needed.

```bash
make install
```

```bash
make smoke
```

`make smoke` runs the full pipeline on 50 games in about a minute, using isolated output directories.

```bash
make all
```

`make all` reproduces everything: ingest 9 seasons (~15 min, cached thereafter), then run the pipeline, tests and notebooks.

Individual steps: `python -m src.pipeline --only xg value epa metrics`, or `--from value`. All tunables live in [`config.yaml`](config.yaml).

## Outputs

| File | Contents |
|---|---|
| `data/outputs/pm_epa_player_leaderboards.csv` | player-season: TOI splits, PM-EPA totals and per-60 (EV/PP/PK/total), per-100 possessions, play counts, mean/SD per play, position percentile, 95% bootstrap CI, direct vs pooled split |
| `data/outputs/pm_epa_team_season.csv`, `pm_epa_team_game.csv`, `pm_epa_player_game.csv` | team and game aggregates |
| `data/value_surfaces/v_state_{5v5,pp,pk,4v4_3v3}.parquet` | tabular value tables (+ `v_state_metadata.json`: N, metrics, sha256) |
| `data/value_surfaces/v_models.joblib` | cross-fitted V_start / V_dec / V_att models |
| `data/processed/transitions.parquet`, `credits.parquet` | every movement segment and every credit row (conserves value) |
| `reports/figures/` | rink value surfaces, calibration, YOY scatter, distributions, components, example plays (PNG + SVG) |

## Defensive component: DPM (v1.0)

The second component of the planned all-in valuation. **Defensive PM-EPA** estimates the 5v5 xG against a skater prevents per 60 minutes. It uses a ridge regression on stints (RAPM) that adjusts for teammates, opponents, zone starts, score and venue, and chains each season's prior to the one before. The PM-EPA ledger splits every opponent possession's xG exactly into components, so each player's number comes with an additive breakdown of *how*: recoveries, movement allowed, forced losses, shot quality, and turnover- or faceoff-born chances.

| Check | Result |
|---|---|
| Held-out next-season xGA prediction vs league average | ΔMSE −0.048 (≈ 6.8 SE) |
| Season-centred YOY r (≥ 500 min) | F 0.41 · D 0.38 (team changers 0.32 · 0.31) |
| r with PM-EPA/60 | 0.06 (F) · 0.14 (D), little overlap, so it adds a new axis |

The pre-registered hypothesis, that the denser ledger components would beat total xGA as a signal, was tested and **rejected**. The headline is therefore prior-informed xGA RAPM, with the ledger as its exact breakdown. `python -m src.defense.run` (or `make defense`) writes `data/outputs/dpm_player_season.csv` and `dpm_player_window.csv`. Docs: [methodology & pre-registration](docs/defense/methodology.md) · [validation report](reports/defense/validation_report.md) · [model card](reports/defense/model_card.md).

## Using PM-EPA responsibly

* **EV = 5v5.** Qualify at ≥ 500 5v5 minutes and **read the 95% CI before the rank**.
* **The `pm_direct_per60_ev` / `pm_pooled_per60_ev` columns** show how much of a number is individually observed and how much is shared with linemates.
* **Own-zone giveaways are nearly free in the headline number (λ = 0).** `pm_epa_per60_ev_lam1` is the risk-adjusted companion that also charges the opponent's gain from a turnover. Read both when turnover risk matters.
* **2024-25 onward is a different scorekeeping regime:** recorded giveaways doubled. Compare players within regime.
* **PM-EPA is offense and transition value.** It does not measure defensive positioning, and it is not WAR.

## Name

**PM-EPA (Puck-Movement Expected Goals Added)** is recommended. "EPA" signals the NFL lineage and delta logic, while "Puck-Movement" states the scope: value from what a player does to the puck's state, not finishing. The alternative considered, *Transition xG Added (TxGA)*, reads well, but it collides with "xGA" (expected goals against) and undersells the offensive-zone movement component.

## Repository map

```
config.yaml                 all tunables (pydantic-validated in src/config.py)
src/ingestion/              NHL API scraper + parser (events, shifts, players)
src/possession/             possession chains, on-ice + TOI
src/xg/                     shot xG features + model
src/states/                 state spaces (26 zones, 10/20-ft grids)
src/value/                  V_start / V_dec / V_att estimators, bake-offs
src/epa/                    PM-EPA attribution engine
src/metrics/                player/team/game aggregation, bootstrap CIs
src/validation/             validation suite, RAPM, ablations, report
src/defense/                Defensive PM-EPA: stints, ridge RAPM, pre-registered selection, report
src/viz/                    rink drawing + figures
tests/                      81 unit tests (pytest)
notebooks/                  01–06 analysis notebooks (generated by notebooks/_build_notebooks.py)
docs/pm_epa/                methodology, formulas, definitions, examples, edge cases, limitations, literature, plan, phase summaries, post-mortem
reports/                    validation report, model card, figures, metrics JSON
schemas/                    SQL DDL + transitions Parquet schema
```

## Documentation

[Methodology](docs/pm_epa/methodology.md) · [Formulas](docs/pm_epa/formulas.md) · [Definitions](docs/pm_epa/definitions.md) · [Attribution examples](docs/pm_epa/attribution_examples.md) · [Edge cases](docs/pm_epa/edge_cases.md) · [Limitations & roadmap](docs/pm_epa/limitations.md) · [Literature review](docs/pm_epa/literature_review.md) · [Data model](docs/pm_epa/data_model.md) · [Implementation plan](docs/pm_epa/implementation_plan.md) · [Post-mortem](docs/pm_epa/post_mortem.md)

Data © NHL, used via its public web APIs for research. This project is not affiliated with the NHL.
