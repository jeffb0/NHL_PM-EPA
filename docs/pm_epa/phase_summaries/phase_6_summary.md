# Phase 6 — PM-EPA engine + attribution

**Done.**
* `src/epa/engine.py`: components, attribution weights, on-ice pooling, accounting identity.
* `src/epa/run.py`: cross-fitted valuation of every observation, opponent counterfactuals, on-ice lists, exclusion propagation.
* `tests/test_epa_attribution.py`: 15 tests reproducing every Part A number in `attribution_examples.md`.
* **Identity:** at λ = 0 the components of each of 1,876,659 possessions sum to its terminal xG to within 1.1e-16.

## Bugs found and fixed

1. **Unbalanced takeaway pairs (80 possessions).** When the taker's possession was excluded because an empty net occurred later in it, the loser's debit disappeared, and 80 possessions failed the identity (max residual 0.064). Exclusion is now propagated across takeaway links (`propagate_exclusion`, tested).

## Face-validity failure, diagnosis, and design changes

The first full run failed the brief's face-validity gate (§12.4.15):

* **Every** qualified player was negative.
* The *bottom* of the 2025-26 forward list was MacKinnon, Pastrnak, J. Hughes, Eichel, Malkin and Celebrini; the top was checking-line forwards.
* PM-EPA/60 correlated **−0.89 with giveaways/60** and −0.07 with G+A1/60.

We did not tune the leaderboard. We decomposed it by component (top-40 G+A1/60 forwards vs bottom-80) and traced two structural causes:

1. **Decision points were valued with the wrong function.** Shot and giveaway spots were valued with V_start's `control` surface. That surface is fit only on takeaways (~19k/season), so it was nearly flat: V ≈ 0.036 at every shot location, out-of-sample R² 0.03 (control-surface evaluation). Moving the puck into the slot earned nothing.
   **Fix:** a second value function, **V_dec(s) = E[G_p | terminal decision at s]**, fit on terminal observations; out-of-sample R² 0.39 (validation) and 0.42 (test). This is NFL EP at the snap.
2. **Decisions were counted one-sidedly.** With V_dec, a giveaway costs `0 − V_dec(s)` (always < 0), but the matching success term `xG − V_dec(s)` was excluded as "shot decision". That is the NFL equivalent of charging interceptions but not crediting completions.
   **Fix:** split the shot into **attempt** = V_att(s) − V_dec(s), included and mean-zero with giveaways at each spot, and **quality** = xG − V_att(s), the release, excluded per brief §4.2. V_att(s) = E[xG | attempt at s].

After fix 1 alone, the defense list became face-valid (top 5: Makar, Toews, Hutson, Slavin, Carlile), but forwards still netted equal for stars and grinders (−0.110 vs −0.111 per 60).

After fix 2, forwards were still flat: r with G+A1/60 was 0.06, and the giveaway component was dominated by the λ = 1 opponent-value term.

3. **λ selected by the pre-registered rule: λ = 0.**

   | | λ = 1 | λ = 0 |
   |---|---|---|
   | team-changer YOY (mean of F, D) | 0.372 | 0.438 |
   | r with G+A1/60, F / D | 0.06 / 0.14 | 0.40 / 0.35 |
   | \|r\| with individual SH% | ≤ 0.12 | ≤ 0.25 |

   With λ = 0 the leaderboards are face-valid:
   * Makar is #1 on defense in all three recent seasons; Q. Hughes, Fox, Bouchard, Toews and Werenski follow.
   * Forwards: Matthews, McDavid, J. Hughes, Crosby, Point and Hischier appear alongside Hyman, Hagel and B. Tkachuk.

   **Cost:** defensive-zone giveaways become nearly free (≈ 0.0003 xG). A risk-adjusted λ = 1 companion column is shipped (`pm_epa_per60_ev_lam1`).

Shares: 0.15/0.55 beat 0.35/0.35 on team-changer YOY by < 0.02, so the default was kept (rule). Both pre-fix designs are kept as ablations: `no_decision_value(pre-v1.0)` and `no_shot_attempt_credit`.

**Next:** full dataset + validation.
