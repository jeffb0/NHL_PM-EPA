# Phase 10 — Self-critique & handoff

**Done.** `docs/pm_epa/post_mortem.md`. It covers:
* 6 failures caught during the project, with how they were caught and the lesson from each;
* top weaknesses (unobserved passes; scorer-bias dependence; λ = 0 own-zone blind spot; profile-specific forward face validity);
* top misuse risks (WAR substitution; ranking inside CIs; cross-regime or cross-rink comparisons);
* the strongest evidence;
* the tracking-data wish-list.

**Handoff notes.**
* Reproduce: `make install && make all`. The raw snapshot is cached in `data/raw/`, so a re-run skips the network.
* Re-run from any step: `python -m src.pipeline --from value`.
* Safe knobs: `config.yaml` → `epa.*` (shares, pool), `value.lambda_opponent`, `metrics.min_toi_ev`. After any change, re-run `validate ablations report` and re-check the §0 gate table and the §11 leaderboards.
* Highest-value next steps: hierarchical credit-share estimation (R6), rink-adjusted turnovers (R3), and the monotone V_dec ≤ V_att fix.
