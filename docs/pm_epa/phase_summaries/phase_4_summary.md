# Phase 4 — State-space design & selection

**Candidates.** Three spatial schemes (26 named sub-zones; 10 × 8.5 ft cells = 200; 20 × 17 ft cells = 50), each × state *kind* (control / faceoff / 12 loose-puck origins) × strength group (5v5 / PP / PK / 4v4_3v3), ± score bucket (−2…+2), ± hit mid-observations. Plus a continuous-location GBM.

**Evidence** (validation season 2022, out-of-sample MSE vs realized possession xG; full table in validation report §1.1):

| spec | R² vs const | ΔMSE×1e6 vs best (± SE) |
|---|---|---|
| GBM continuous + score | **0.2044** | 0 |
| zones + score (EB auto) | 0.2038 | +2.44 ± 0.87 |
| grid10 + score | 0.2035 | +3.46 ± 0.87 |
| GBM continuous | 0.2018 | +9.75 ± 0.98 |
| zones / grid10 / grid20 (EB, k≥10) | 0.2012–0.2016 | ≈ +10.5 ± 1.1 |
| grid10, no shrinkage (k=0) | 0.1981 | — |

**Conclusions.**
1. **Spatial resolution barely matters for predicting possession outcomes from *starts*.** Zones, grid10 and grid20 tie (R² 0.2014 ± 0.0002). Nearly all the signal is in *kind* (faceoff vs loose-puck origin vs control) and coarse zone.
2. **Shrinkage matters only for fine grids.** With k = 0, grid10 loses 0.003 R² to over-fitting. A bug in the first implementation used one method-of-moments k for all hierarchy levels, which over-shrank grid10 to R² 0.141. It was fixed to per-level k, with a regression test.
3. **Score state adds real signal:** +0.0024 R², a ΔMSE of ~11 SE. It enters V (the value of *states*, not credit), so trailing-team aggression is modelled rather than credited to players.
4. **Hits as mid-observations do not help** (Markov with hits R² 0.2006 vs MC 0.2014). Hits stay out of possession inference.

**Decision (by the pre-registered rule).** GBM + score was the only candidate within 2 SE of the best, so it was chosen. Tabular zone/grid tables are still exported for transparency (`v_state_*.parquet`, grid10 EB). Because the headline metric is dominated by starts, a supplementary *control-surface* evaluation (validation report §1.3) checks the fine-location surface that movement deltas use.

**Next:** xG + value surfaces.
