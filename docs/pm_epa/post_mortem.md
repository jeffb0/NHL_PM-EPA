# PM-EPA Post-Mortem (Phase 10)

A critical self-review written after the final run. Numbers cite `reports/validation_report.md`.

## What went wrong along the way (and what that teaches)

| # | Failure | How it was caught | Fix | Lesson |
|---|---|---|---|---|
| 1 | One method-of-moments shrinkage k reused at every hierarchy level over-shrank finer grids (grid10 R² 0.141 vs 0.201) | bake-off anomaly: a finer grid lost to its own parent zones | per-level k + regression test | Hierarchical EB needs level-specific priors; "auto" defaults deserve tests |
| 2 | 80 possessions broke the accounting identity (max residual 0.064) | the identity check on the full dataset | propagate exclusions across takeaway links | An exact identity is a cheap, powerful bug detector; keep it in CI |
| 3 | **First leaderboard inverted.** Stars at the bottom, r(PM-EPA, giveaways/60) = −0.89 | brief's face-validity gate | V_dec for decision points; attempt/quality split; λ = 0 | Observation asymmetry (failures recorded, successes invisible) biases any PBP possession metric against puck carriers |
| 4 | The most *stable* variants were the *least valid*: the inverted pre-fix design had the highest forward YOY (0.66) | ablation table | stability used only jointly with convergent validity and face validity | Giveaway rates are highly repeatable (style, role, rink scorer). Stability alone would have selected the wrong metric |
| 5 | A config edit silently failed, so a "final" run used λ = 1 | leaderboard values contradicted a λ = 0 spot check | assert on every programmatic config edit; the run log records the config | Verify effective config, not intended config |
| 6 | Pooled YOY r rewarded a variant (regime-naive cross-fitting) whose "stability" was whole seasons shifting in level | a +0.14 YOY jump with no plausible mechanism; model values proved nearly identical on old seasons | season-centred YOY added to the report and gates | Pooled correlations across seasons need season-centring whenever levels can drift |

## Top weaknesses

1. **Most credit for puck movement is not individually observed.** Only carries (same player at both ends), takeaways, giveaways and shot attempts are individual. Everything between events is split 35/35/30, with the 30% pooled across on-ice skaters. Those shares are a symmetric-ignorance prior, and the ablations show YOY is insensitive to them, which also means the data cannot tell us the right split. Passers are systematically under-credited.
2. **Dependence on scorer-recorded turnovers.** Giveaways and takeaways vary by rink, and the recording regime changed in 2024-25 (giveaways ×2). Turnover components carry scorer bias into player values. Players on rinks that record many giveaways, and puck-dominant players in general, are penalized relative to a tracking-based truth.
3. **Own-zone giveaways are nearly free at λ = 0**, the production setting chosen by the pre-registered rule: ≈ 0.0003 xG per defensive-zone giveaway. The risk-adjusted λ = 1 companion column mitigates this but does not fix it. A principled fix needs the opponent's gain to be balanced by the unobserved positives of puck handling, which requires pass/carry data.
4. **Forward face validity is profile-specific.** PM-EPA rewards possession *gains* and getting pucks to good decision points: forechecking, retrieval, net-front play. Elite playmakers whose value lives in passes (unobserved) rank lower than their reputations. The defense rankings are strongly face-valid; the forward rankings are defensible but narrower.

## Top 3 risks of misuse

1. **Using PM-EPA as WAR or an all-in-one rating.** It excludes finishing, defense without the puck, penalties and goaltending by design. Ranking players on it alone, or summing it with other components, will mislead.
2. **Ranking inside the confidence interval.** Typical 95% CIs at 1,000 5v5 minutes are wide relative to the spread of players. Adjacent ranks are rarely distinguishable; use the bootstrap CI columns.
3. **Comparing across the 2024-25 scoring break or across rinks.** A drop in a player's PM-EPA from 2023-24 to 2024-25 may be a recording artifact. Home-rink scorer effects persist season to season.

## Strongest empirical evidence

<!-- BEGIN EVIDENCE -->
1. **Calibration on untouched data.** V_start's test-season calibration slope is 1.004. V_dec explains 42% of out-of-sample variance in possession outcomes, and it rises monotonically toward the net at 5v5 (1 radial violation in 899 steps).
2. **Stability that survives the hardest controls.** Season-centred YOY is 0.53 (F) and 0.52 (D). Among players who changed teams, PM-EPA/60 still carries over: 0.37 (F) and 0.30 (D).
3. **Independent on-ice evidence.** PM-EPA/60 correlates 0.59–0.71 (F) with 5v5 offensive xG RAPM, a regression that never sees PM-EPA's inputs beyond shots. It adds explanatory power over ixG/60 + G+A1/60 in all 6 window × position fits.
4. **Not finishing.** r with (G − ixG) is −0.004 for forwards.
5. **Face validity after, and only after, fixing real design flaws.** The final leaderboards put Makar first on defense in three straight seasons, with recognizable elite puck movers behind him. Those flaws were found by the face-validity gate and were never tuned toward a desired list.
<!-- END EVIDENCE -->

## What tracking data would change

| Need | Why | Expected effect |
|---|---|---|
| Pass events with passer, receiver, start/end coordinates, completed or failed | Replaces the 35/35/30 prior with observed actors | Pooled share → ~0; passers credited; convergent validity with assists should rise |
| Carry segments (entries, exits, rushes) with start/end | Observed carries today are only the takeaway→own-shot subset | Direct credit for transition play (Tulsky et al. 2013 showed this is where repeatable skill lives) |
| Receptions and pressure (defender distances; NHL EDGE) | V(s) conditional on pressure; separates "safe" from "forced" turnovers | Fairer turnover debits; less scorer bias |
| Puck location at every stoppage/change of possession | Removes inferred starts and unobserved losses | Cleaner possession boundaries; loss attribution |
| Consistent, audited turnover definitions | Removes the 2024-25 regime break and rink bias | Cross-season comparability |

## If I had another week

1. Estimate credit shares with a hierarchical model: player effects on segment Δ, ridge-regularized with on-ice teammates (roadmap R6). This replaces the fixed 35/35/30 with data-driven, uncertainty-aware splits.
2. Rink-adjust turnover components out of sample (R3), and re-check YOY and face validity.
3. Build the defensive mirror metric (R5).
4. Add a CI job that runs `make smoke`, the identity check and the face-validity diff on every change.
