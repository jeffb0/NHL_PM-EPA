# Chemistry and roster-fit model — results

Pre-registration: [docs/chemistry/methodology.md](../../docs/chemistry/methodology.md). Metrics: `reports/metrics/chemistry.json`. Tuned on two forward folds and frozen: additive penalty 1,500, rank 8, interaction penalty 1,000.

## Gates

| gate | rule | statistic | result |
|---|---|---|---|
| CH-1 | interaction model beats additive on next-season stints (z > 2) | pooled z = −2.66 (new combinations −2.67) | FAIL |
| CH-2 | roster-fit covariate improves the full game model (z > 2) | z = −0.97 | FAIL |
| CH-3 | no harm (both game comparisons z > −2) | −0.97 and −0.48 | PASS |

**Decision by the pre-registered rule: not implemented.**

## Detail

| test season | additive error | with interactions | z | z, new combinations |
|---|---|---|---|---|
| 2021 | 258.277 | 258.274 | 0.98 | 1.32 |
| 2022 | 283.584 | 283.610 | −3.06 | −3.12 |
| 2023 | 271.477 | 271.463 | 1.40 | 1.58 |
| 2024 | 263.413 | 263.437 | −2.48 | −2.80 |
| 2025 | 286.029 | 286.050 | −2.01 | −2.07 |

(Minute-weighted squared error of home-minus-away expected goals per 60; each model fitted on the two previous seasons.)

| game model (6,502 games, log loss) | without | with roster-fit covariate | z |
|---|---|---|---|
| full model P | 0.65854 | 0.65922 | −0.97 |
| preseason-only | 0.66920 | 0.66973 | −0.48 |

* Stability of pair synergies between non-overlapping fits (2021–22 against 2023–24, 574 players with 500+ minutes in both): correlation **0.01**.
* Same data, different random start: correlation of pair synergies 0.99 (the fit is reproducible; it is not an optimisation problem).
* Spread of the fitted synergy term across stints: 0.09–0.25 expected goals per 60, against 0.37–0.44 for the additive term.

## Interpretation

**No portable chemistry is detectable in these data.** The interaction model fits a synergy term of meaningful size, reproducibly, but that term does not carry forward: it makes next-season stint predictions worse in three of five seasons
(pooled z = −2.66), and a player's fitted pair synergies in one two-season window are uncorrelated (0.01) with the same pairs' synergies two seasons later. Reproducible within a window and unrelated across windows is the signature of fitting noise.
The methodology named this possibility in advance (§7): the trait-stability figure says noise, not real but unusable structure.

**It does not help predict games either.** The roster-fit covariate has a positive fitted coefficient because it is correlated with team strength (0.51 with the model's expected margin), but adding it worsens out-of-sample log loss (z = −0.97).

**The tuning result was a warning.** On the two tuning folds the interaction model's gain over the additive one was 0.006 on 235.5 (0.003%), obtained only at the weakest penalty in the grid; any stronger penalty returns the additive model. A gain that small on two folds was noise, and the test seasons showed it.

**What this means.** At the resolution of 5v5 expected goals per stint, a hockey unit is, as far as these data can tell, the sum of its players: either real chemistry is small, or it lives in things expected goals do not see
(finishing on specific passes, which needs tracking or pass data). This closes the hypothesis that non-additive lineup effects are the information missing from the preseason prior. The portable team residual (docs/standings/improvements.md) is therefore
more likely additive player value that the components under-measure (the under-scaling of possession value found earlier) or team-level context, not chemistry.

**Limitations.** One model family (symmetric pairwise, within team), 5v5 expected goals only, two-season windows, grid chosen in advance and not widened; a forward–defense or line-specific structure, a goals target, or tracking-based pass data could differ.
