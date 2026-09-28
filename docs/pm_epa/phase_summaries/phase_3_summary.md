# Phase 3 — Possession chains + transitions

**Done.** `src/possession/chains.py` (rules in `definitions.md` §3), `src/possession/build.py` → 1,925,919 possessions (174/game), 3.13M observations; hit-augmented variant for ablation. 21 unit tests in `tests/test_possession.py`.

**Structural finding.** Without hits, every possession has ≤2 observations (start, terminal). Hits consistent with the current possessor add a third observation to only 2.3% of possessions. Consequently the Markov value estimator (approach A) coincides with Monte Carlo except under the hits option. This is reported, not hidden.

**20-chain hand audit** (sample in notebook 02). The audit exposed four issues, all fixed and each covered by a test:
1. Goals seeded loose-puck starts, producing negative-duration possessions from PBP ordering glitches. Goals no longer seed starts.
2. The inferred-start window of 45 s admitted starts where possession had plausibly changed hands unobserved. It is now 25 s (≈ p60 of loose-start durations); the long-gap movement threshold is 30 s.
3. Losses revealed later (icing, unobserved loss) were charged to on-ice players at the *last observation*, sometimes 30+ s before. They are now charged at the ending event (`end_event_idx`).
4. Out-of-order event times are guarded.

**Median possession duration** is 13–20 s depending on start type: most movement is unobserved, which motivates the split/pool attribution.

**Next:** state-space bake-off.
