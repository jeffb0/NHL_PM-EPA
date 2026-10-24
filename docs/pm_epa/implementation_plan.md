# PM-EPA Implementation Plan (Phase 0)

**Project:** Puck-Movement Expected Goals Added (PM-EPA) for NHL skaters
**Repo root:** the root of this repository is the `pm-epa-nhl/` project root from the brief; the directory layout below follows the brief's §10.1 without the extra nesting level.
**Date:** 2026-09-27

---

## 0. The constraint that shapes everything

The brief describes PM-EPA in terms of receptions, carries and completed passes. **Public NHL play-by-play (PBP) records none of those.** A probe of the `api-web.nhle.com` gamecenter feed (2017-18 and 2024-25 samples) shows the event vocabulary is:

| Event | Coordinates | Player(s) | Possession meaning |
|---|---|---|---|
| `faceoff` | yes | winner, loser | winner's team gains possession (state reset) |
| `takeaway` | yes | taker | owner team gains possession by force |
| `giveaway` | yes | giver | owner team loses possession unforced |
| `shot-on-goal`, `missed-shot`, `goal` | yes | shooter (+ goalie, + up to 2 assisters on goals) | owner team had possession; possession terminal |
| `blocked-shot` | yes (block location) | shooter, blocker | shooting team had possession; terminal |
| `hit` | yes | hitter, hittee | *ambiguous* — not a possession event |
| `penalty`, `delayed-penalty`, `stoppage`, `period-end` | penalties only | — | stoppage / terminal |

No passes, no carries, no zone entries, no dump-ins, no receptions. Typical sampling density is **~1 possession-bearing event every 8–10 s**, so most real puck movement happens *between* observed events.

**Consequence (design decision D0).** We implement PM-EPA as a *data-granularity-agnostic engine* and run it on PBP at the finest granularity PBP supports:

* A **possession** is inferred from the event stream (Phase 3).
* Inside a possession, puck movement is observed as the **state change between consecutive observed events**. Each such step is a *movement segment*.
* A segment whose start and end events share the same player within a short window is an **observed carry** and is credited 100% to that player.
* A segment whose endpoints are different (or unknown) players contains at least one **unobserved pass/carry**. Its value is split between the known endpoint players and a residual **on-ice pool** (all skaters of the possessing team on the ice, from shift charts). Shares are configurable and *chosen by evidence* (Phase 6/8 ablations), not intuition.
* The engine's action vocabulary (`recovery`, `carry`, `pass_complete`, `turnover`, `shot`) is the one in the brief; if a tracking-grade feed (Stathletes/Sportlogiq/NHL EDGE) is ever available, passes and carries plug straight into the same attribution code with zero unobserved residual. This upgrade path is documented in `limitations.md`.

This is stated up front in every deliverable. It is the single most important caveat on the metric, and we will not paper over it.

## 1. Milestones

| Phase | Deliverable(s) | Gate / checkpoint |
|---|---|---|
| 0 | this plan | — |
| 1 | `literature_review.md` | adopted/rejected ideas explicit |
| 2 | raw snapshot (9 RS seasons, 2017-18 → 2025-26), `events.parquet`, data dictionary | row counts per season within ±3% of expected; coordinate-normalization sanity (≥95% of unblocked shots in attacking half) |
| 3 | `possessions.parquet`, `segments.parquet`, `tests/test_possession.py` | ≥20 random chains hand-audited; all tests pass |
| 4 | zone vs grid10 vs grid20 bake-off table | choose by CV deviance + YOY stability of V, not taste |
| 5 | shot-xG model + calibration report; V(s) via Markov (A), MC baseline, hazard (B), GBM (C); rink heatmaps | xG: test-season log-loss/Brier/ECE; V: calibration slope ∈ [0.9, 1.1], monotonicity report |
| 6 | PM-EPA engine + attribution + `attribution_examples.md` (unit-tested numbers) | hand math == code |
| 7 | full run; `pm_epa_player_leaderboards.csv`, team + game aggregates | team sums reconcile to possession totals exactly (telescoping identity) |
| 8 | `validation_report.md` + figures | YOY/split-half/convergent/discriminant/RAPM/ablations with numbers |
| 9 | README, model card, formulas, edge cases, limitations, notebooks, `make all` | clean `make all` from raw snapshot |
| 10 | `post_mortem.md` | top-3 weaknesses/risks + tracking wish-list |

Each phase ends with `docs/pm_epa/phase_summaries/phase_X_summary.md`.

## 2. Architecture

```
data/raw (gz JSON, immutable)
   └─ src/ingestion  ──► data/processed/events.parquet        (one row per PBP event, normalized)
                          data/processed/shifts.parquet        (player shifts)
                          data/processed/players.parquet       (id, name, position, handedness)
   └─ src/xg         ──► data/processed/shots_xg.parquet       (xG per unblocked attempt; cross-fitted)
   └─ src/possession ──► data/processed/possessions.parquet    (possession chains)
                          data/processed/observations.parquet  (in-possession observed states)
   └─ src/states     ──► state_id per observation (zones/grid; strength; score)
   └─ src/value      ──► data/value_surfaces/v_state_{5v5,pp,pk,4v4_3v3}.parquet (+ metadata json)
   └─ src/epa        ──► data/processed/transitions.parquet    (segment/turnover rows with credit)
                          data/processed/credits.parquet       (player-level credit rows, conserves value)
   └─ src/metrics    ──► data/outputs/*.csv                    (player/team/game aggregates, bootstrap CIs)
   └─ src/validation ──► reports/validation_report.md
   └─ src/viz        ──► reports/figures/**                    (PNG+SVG)
```

`python -m src.pipeline` runs everything after ingestion; `make all` runs ingestion (cached) + pipeline + tests.

## 3. Key technical choices (to be confirmed/overturned by evidence)

| # | Choice | Why | Test that can overturn it |
|---|---|---|---|
| D0 | Segment-based movement on PBP + on-ice residual pool | only defensible option with public data | — (data constraint) |
| D1 | V(s) = expected xG of the *current possession's* terminal shot (brief §2.1) | direct EPA analogue; xG not goals as reward (guardrail §14) | calibration vs future xG and vs goals |
| D2 | Markov-chain V via linear solve `(I−P)V = r` on observed-event states | pools information across possessions (TD-style), lower variance than Monte Carlo | CV deviance vs MC / GBM / hazard |
| D3 | Separate value tables per strength (5v5, PP, PK, 4v4/3v3); EN excluded | surfaces differ materially; EN distorts | ablation: pooled + strength fixed effect |
| D4 | Hierarchical empirical-Bayes shrinkage cell → parent zone | sparse cells (3v3, PK) | CV deviance with/without shrinkage |
| D5 | Turnover swing weight λ on the opponent's value: **started at λ=1, changed to λ=0** by the pre-registered rule (Phase 8) | λ=1 is EP symmetry, but its cost cannot be balanced at the player level when the positive side of puck handling is unobserved | λ ablation: team-changer YOY, convergent, discriminant |
| D6 | Shot split into *attempt* (V_att − V_dec, included, symmetric with giveaways) and *quality* (xG − V_att, excluded) | brief §4.2: the release is the outcome state; but charging decision failures (giveaways) without crediting successes biases against puck handlers (found in Phase 6) | discriminant validity vs finishing; face validity |
| D7 | Cross-fitting (leave-one-season-out) for xG and V used in player metrics | removes in-sample leakage without sacrificing sample | compare to single temporal split |
| D8 | TOI and on-ice rosters from official shift charts | needed for per-60, on-ice pool, RAPM | reconcile shift TOI vs official where available |

## 4. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Event sparsity → most value lands in the unattributed/on-ice pool, weakening "individual" claims | high | high | report direct vs pooled components separately; test individual-ness via YOY for team-changers |
| Scorekeeper (rink) bias in giveaway/takeaway/hit counts | high (well documented) | medium | rink effect diagnostic; home/away split; document; optional rink adjustment |
| Coordinate direction unavailable pre-2019 (`homeTeamDefendingSide` absent) | certain | high if wrong | infer per game-period from shot locations; validate ≥95% of shots in attacking half |
| Blocked-shot owner/zone semantics changed across seasons | observed | medium | derive shooter team from roster, not `eventOwnerTeamId` |
| Shift-chart gaps for some games | medium | low | drop affected games from per-60 denominators *and* numerators consistently; log |
| Overfitting V to noise in small strength states | medium | medium | shrinkage + min-N + parent blending mandatory for 4v4/3v3 |
| Face-invalid leaderboards | medium | high | investigate before narrative; never tune to "look right" |
| Scrape throttling | low | low | backoff; cached snapshot |

## 5. Assumptions (stated so none are hidden)

1. Regular season only (game type 2). Playoffs out of scope for v1.
2. Shootouts excluded; regular-season OT (3v3 since 2015-16) included in the 4v4/3v3 surface.
3. Empty-net states (either goalie pulled) are excluded from V estimation and from player credit by default (config `exclude_empty_net`).
4. "EV" in leaderboards means **5v5** with both goalies in net. 4v4/3v3 are reported in their own column family via `total`.
5. PBP coordinates are in feet, x ∈ [−100, 100], y ∈ [−42.5, 42.5]; all states are normalized so the possessing team attacks +x.
6. Scorer-recorded giveaways/takeaways/hits are noisy and rink-biased; we use them but quantify the bias.
7. xG is the reward currency; goals are only used (a) as the xG model's label and (b) as a secondary calibration check.
