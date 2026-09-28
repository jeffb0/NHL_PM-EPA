# Phase 5 — Shot xG + value surfaces

**Shot xG** (`src/xg/`). Logistic spline vs histogram GBM, trained on 2017–21 and compared on 2022.

| | log-loss | AUC | Σxg/Σgoals |
|---|---|---|---|
| logit (valid 2022) | 0.2318 | 0.753 | 1.094 |
| **GBM (valid 2022)** | **0.2269** | **0.763** | 1.091 |
| GBM (temporal test 2023-25) | 0.2289 | 0.751 | 1.129 |
| GBM + isotonic (test) | 0.2273 | 0.751 | 1.028 |
| GBM cross-fitted (all seasons) | 0.2189 | 0.773 | 1.008 |

A pure temporal holdout reveals league-level drift: 2023-25 goals are over-predicted by 13%. The final per-shot xG is cross-fitted by season (per-season calibration 0.97–1.08; 2025-26 is the worst at 1.08). Top features: distance, empty net, |y|, shot type, angle, time since the previous event.

**V(s)** (`src/value/`). Selected spec: **GBM on continuous location + kind + strength + score** (Phase 4). Test seasons (2023-26, scored once): R² 0.226, calibration slope 1.004, intercept −0.003. V predictions are stable season to season (r 0.987–0.993 for every pair, including the 2023→24 scoring-regime break at 0.987). Cross-fitted within scoring regime for player metrics.

**Approach comparison** (test seasons): GBM+score 0.2257, zones EB 0.2231, grid10 EB 0.2232, GBM 0.2234, hazard 0.2226. All are well calibrated (slopes 0.996–1.004). Markov = MC by construction on PBP (Phase 3).

**Artifacts.**
* `data/value_surfaces/v_models.joblib`: cross-fitted models by season, plus an all-season model.
* `v_state_{5v5,pp,pk,4v4_3v3}.parquet`: tabular grid10 EB tables.
* `v_state_metadata.json`: N, per-level k, test metrics, sha256.
* Figures in `reports/figures/value_surfaces/`.

**Bug found and fixed:** per-level shrinkage (see Phase 4).

**Next:** PM-EPA engine.
