## Interpretation (hand-written; every number below is from the tables in this report)

**Bottom line.** PM-EPA clears every stability and reliability gate in the brief. It is well calibrated, it is not a finishing proxy, and it carries information about on-ice impact beyond boxscore creation. It falls short of the brief's aspirational convergent-validity target, for a reason that is structural rather than a bug: passes are invisible in public PBP.

1. **Value model.** V(s) is essentially perfectly calibrated on untouched test seasons (slope 1.004, intercept −0.003; §1.2).
   * The decision-point surface V_dec, which drives movement deltas, explains ~40% of the variance in possession outcomes out of sample (§1.3b), with a monotone rise toward the net (§1.4).
   * Spatial resolution barely matters for start states. What matters is *how* the puck was won (faceoff vs loose-puck origin), plus score state (§1.1).
2. **Stability (the hard gate).** Within-regime YOY r is 0.56 for forwards and 0.61 for defense, against targets of ≥0.40 and ≥0.35 (§2). Odd/even-game split-half r at ≥800 min is 0.56 (F) and 0.63 (D), which is 0.72 and 0.78 after Spearman-Brown correction to full-season reliability (§3).
   * The best test of individual signal is players who **changed teams**, getting new linemates and a new system. Their YOY is 0.44 (F) and 0.44 (D).
   * Forwards: PM-EPA's YOY is slightly below on-ice xGF/60 (0.60) and ixG/60 (0.64), and above G+A1/60 (0.51).
   * Defensemen: it is well above on-ice xGF/60 (0.43) and G+A1/60 (0.37).
2b. **Pooled YOY is mildly inflated by season-level drift.** Season-centred YOY is F 0.53 and D 0.52, still well above target; team-changers are 0.37 (F) and 0.30 (D). These stricter numbers are the ones to quote. The ablation that looked best on stability, regime-naive cross-fitting (pooled 0.70), is *identical* to the baseline once centred (0.54 / 0.52). Its "gain" was whole seasons shifting in level (§6).
3. **Stability is not validity.** The *most* stable variants in the ablations (§6) include the pre-fix design whose leaderboard was inverted, and giveaways/60 alone (YOY 0.75). Scorer-recorded giveaway rates are extremely repeatable: style, role, and the home rink's scorer (rink-index persistence r = 0.76, §9.3). Choices were therefore made on team-changer stability *jointly* with convergent and discriminant validity, under a rule written before the ablations ran (methodology §5).
4. **Convergent validity is moderate, and it is the right kind.** r with G+A1/60 is 0.40 (F) and 0.35 (D), below the brief's 0.6–0.8 aspiration (PARTIAL). r with ixG/60 is 0.66 (F), and with on-ice xGF/60 0.63 (F) and 0.56 (D).
   * The gap to assists is expected: assists are passes, and passes are exactly what PBP does not record, so a primary-assist-heavy playmaker's contribution lands mostly in the pooled share.
   * Against **5v5 xG RAPM** (offense), PM-EPA/60 correlates 0.59–0.71 for forwards and 0.42–0.62 for defense across three windows (§7). It **adds explanatory power beyond ixG/60 + G+A1/60** in every window: incremental R² +0.014 to +0.034 (F) and +0.025 to +0.047 (D).
5. **Not a finishing proxy.**
   * r with (G − ixG) is −0.004 for forwards.
   * r with individual SH% is 0.15 (F) and 0.25 (D, at the bound).
   * The residual SH% association runs through shot *location* (V_dec), not the release: shot quality xG − V_att is excluded.
6. **Predictive validity.** For defensemen, this season's PM-EPA/60 predicts next season's on-ice xGF/60 better than ixG/60 or G+A1/60 (r 0.37 vs 0.35 and 0.33; §5). For forwards it trails the boxscore rates.
7. **Robust to the scoring-regime break.** Across 2023→2024, giveaways/60 YOY collapses from 0.75 to 0.49 (F), because the definition changed. PM-EPA/60 holds at 0.57 (F) and 0.56 (D). Decision-relative turnover charges and within-regime cross-fitting absorb most of the change.
8. **Team level.** Summed team PM-EPA correlates 0.69 with team xGF (target 0.70; PARTIAL) and 0.37 with goal differential. Its +0.29 correlation with xGA is expected for an offense-only metric: run-and-gun teams move the puck into dangerous areas at both ends.
9. **Known biases, quantified.**
   * Home PM-EPA/60 exceeds road by ~28% (§9.3). Some of that is genuine home advantage, and some is home-scorer recording.
   * Rink giveaway indices range from 0.19 to 1.84 of league average.
   * Neither is adjusted in v1.0 (roadmap R3).

### Decisions taken from the ablations (§6), by the pre-registered rule

* **λ = 0** (λ = 0.5 and 1 are worse on team-changer YOY and much worse on convergent validity).
* **Shares 0.35/0.35 kept.** The best alternative, 0.15/0.55, improved team-changer YOY by only +0.010, under the 0.02 margin.
* **Regime-aware cross-fitting kept.** The naive variant's advantage is a season-level artifact, and it loses convergent validity.
* **The V-model choice does not matter.** All tabular and GBM variants land within ±0.004 YOY.
* **Finding: `no_onice_pool` raises team-changer YOY to 0.46/0.48 at a small convergent cost.** Individually observed credit transfers across teams better than linemate-shared credit, which argues for the direct/pooled split we publish. It is not adopted: it breaks credit conservation (team sums), and it was not a pre-registered candidate.

### Deviations from the brief's targets (and why they are accepted)

| Target | Result | Why it is accepted, not tuned away |
|---|---|---|
| Convergent r 0.6–0.8 with creation stats | 0.40 (G+A1/60, F); 0.66 (ixG/60, F) | Assists are passes, and passes are unobserved. Chasing assist correlation would require crediting assisters from *goal* events only, which is exactly the results-contamination the brief forbids. |
| Team PM-EPA vs xGF ≥ 0.70 | 0.69 | Within sampling error of the target. PM-EPA excludes shot quality and unattributed start values by design, so it is not a decomposition of xGF. |
| Discriminant ≤ 0.25 | 0.25 (D) | At the bound. Driven by shot location, reported rather than adjusted. |
