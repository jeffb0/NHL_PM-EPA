# Phase 9 — Polish, docs, exports

**Deliverables (brief §12).**

| # | Deliverable | Location |
|---|---|---|
| 1–6 | Methodology, formulas, attribution examples (unit-tested Part A + auto-generated real Part B), limitations, edge cases, README | `docs/pm_epa/`, `README.md` |
| 7 | Working pipeline + smoke test | `python -m src.pipeline`, `make smoke`, `make all` |
| 8 | Trained artifacts + metadata | `data/value_surfaces/v_state_{5v5,pp,pk,4v4_3v3}.parquet`, `v_state_metadata.json` (N, per-level k, test metrics, sha256), `v_models.joblib` |
| 9 | Shot xG model + calibration report | `data/outputs/models/xg_temporal.joblib`, validation report §1.5, `reports/figures/calibration/` |
| 10 | Leaderboards (all brief columns + direct/pooled split, λ = 1 companion, CIs); team-season, team-game, player-game | `data/outputs/*.csv` |
| 11 | Validation report | `reports/validation_report.md` |
| 12 | Figures, PNG + SVG | `reports/figures/{value_surfaces,calibration,validation,examples}/` |
| 13 | Model card | `reports/model_card.md` |
| 14 | Naming recommendation | README "Name": PM-EPA |
| 15 | Leaderboard sample (top 20 × F/D × last 3 seasons) + commentary | validation report §11, interpretation |
| — | Schema + ERD + data dictionary | `schemas/pm_epa_events.sql`, `schemas/pm_epa_transitions.parquet`, `docs/pm_epa/data_model.md` |
| — | Notebooks 01–06 | `notebooks/` (built by `_build_notebooks.py`, executed by `make notebooks`) |

**Engineering.**
* 72 unit tests (≥ 20 required); ruff clean; pinned dependencies (`requirements.txt`); Python 3.11 via uv.
* Every tunable is in `config.yaml`, pydantic-validated; each run's effective config is logged to `reports/metrics/run_log.json`.
* `make all` runs ingest (cached, idempotent) → pipeline → tests → notebooks.

**Last-mile changes.**
* **λ = 1 companion column (`pm_epa_per60_ev_lam1`).** At λ = 0, defensive-zone giveaways cost ≈ 0.0003 xG, so a risk-adjusted rate ships alongside.
* **Radial monotonicity check** replaced the fixed-y lane check, which wrongly flagged real angle effects on wide lanes.

**Next:** self-critique (post-mortem).
