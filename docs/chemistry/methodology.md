# Chemistry and roster-fit model (low-rank lineup interactions) — methodology and pre-registration

*Written 2026-10-03 before any interaction model was fitted. The design, tuning protocol, tests and rule in §5–§6 are fixed here; changes after results are seen are logged in §9.
Motivation: the preseason player prior loses to a team rating (docs/gamemodel), young-player information does not close the gap (docs/prospects), and last season's unexplained team result is portable with players
(docs/standings/improvements.md). One candidate explanation is that a lineup is worth more or less than the sum of its players.*

## 1. Idea (ported from recommender systems and basketball lineup-synergy models)

Every player-value component in this repository is additive: a five-man unit is the sum of five numbers. A **factorization machine** adds a pairwise term through a small vector of latent traits per player:
the synergy of a unit L is Σ over pairs i < j in L of v_i · v_j. Because the traits belong to players, not to pairs, the model can score **pairs that have never played together**, which a pair-by-pair
(with-or-without-you) table cannot. That is what would make it useful for trades, signings and a new season's roster.

## 2. Data and target

The 5v5 stints of the defensive component (`def_stints`: five skaters a side, minutes, the expected-goal ledger per side). Target per stint: home expected goals minus away expected goals per 60 minutes (the sum of the ledger components),
weighted by stint minutes. Covariates as in the on-ice models: an intercept (home ice), the home zone-start and score-state codes.

## 3. Models

* **A (additive):** ŷ = c + Σ_home β_i − Σ_away β_i, ridge penalty α_b on β (minutes units), solved exactly.
* **F (additive + interactions):** ŷ = A + syn(home) − syn(away), syn(L) = ½(‖Σ_{i∈L} v_i‖² − Σ_{i∈L} ‖v_i‖²), rank k, ridge penalty α_v on V; fitted jointly by L-BFGS from the additive solution with small random traits (fixed seed).
* Each model for target season T is fitted on seasons T − 2 and T − 1 pooled (one parameter set per player), so nothing from T is used.
* **Tuning (once, then frozen):** α_b ∈ {500, 1,500, 5,000, 15,000} for A, then k ∈ {2, 4, 8} and α_v ∈ {1e3, 1e4, 1e5} for F, by forward error on two folds (fit 2017–18 → predict 2019; fit 2018–19 → predict 2020).

## 4. Roster-fit term for a game

For a team's dressed skaters with projected ice-time shares π_i (the game model's projected minutes divided by 60, summing to five), the expected synergy is ½(‖Σ π_i v_i‖² − Σ π_i² ‖v_i‖²)
(pairs weighted by the product of their shares; line structure is ignored). The game covariate is home minus away, with the traits fitted on the two seasons before the game's season.

## 5. Tests

1. **CH-1, stint level, forward:** on the stints of each test season 2021–2025, minute-weighted squared error of F against A (both fitted on the two previous seasons); paired z from per-game sums of the difference.
   Reported separately for stints that include at least one skater who was not on that team the season before (**new combinations**, the portability claim).
2. **CH-2, game level:** the game model of docs/gamemodel (hyperparameters frozen, original rate model) with the roster-fit covariate added to the logistic step (coefficient fitted on earlier seasons, expanding); paired z on per-game log loss
   over the 6,502 test games, for the full model P and for the preseason-only model.
3. **Reported:** the tuned rank and penalties; the spread of the synergy term against the additive term; the fitted game coefficient; stability of a player's traits across non-overlapping fits (correlation of pair synergies v_i · v_j between the 2021–22 and 2023–24 fits for pairs present in both).

## 6. Gates and the implementation rule

| gate | rule |
|---|---|
| CH-1 | F beats A on next-season stints (paired z > 2) |
| CH-2 | the roster-fit covariate improves the full game model (paired z > 2) |
| CH-3 | no harm: neither game-model comparison has z < −2 |

A priori the method is rated **less certain** (many parameters, a non-convex fit, expected-goal targets that are noisy per stint). By the model owner's standing rule (docs/standings/improvements.md §4) it is implemented **only if CH-2 passes** (z > 2),
which also requires CH-3. A CH-1 pass without CH-2 is reported as a finding about lineups, not adopted into forecasts.

## 7. What the result can say

CH-1 with new combinations passing means there are portable, non-additive lineup effects in expected goals. CH-2 passing means they matter for predicting games. Failure of both means either that hockey lineups are close to additive
at this resolution or that stint-level expected goals are too noisy to identify the traits; the trait-stability figure helps tell these apart (unstable traits point to noise).

## 8. Limitations stated in advance

5v5 only; expected goals, not goals; the interaction is symmetric and within a team (no matchup term against opponents, no forward–defense distinction); the roster-fit term ignores line combinations; two-season fits
assume constant traits; a non-convex fit depends on its starting point (one fixed seed; a second seed is reported as a check); the tuning folds are the two unusual seasons.

## 9. Deviations log

*(any change to §3–§6 after results are seen is recorded here with its reason)*

1. **Covariate coding (before any result).** §2 called the zone-start field a code; it is text (offensive, defensive, neutral, on the fly). It enters as three indicators with on-the-fly starts as the reference; the score state enters linearly (−2 to 2).
2. **Outcome, applied mechanically (2026-10-03):** CH-1 fails (pooled z = −2.66: the interaction model is *worse* than the additive one on next-season stints), CH-2 fails (z = −0.97), CH-3 passes. **Not implemented.**
   The tuned penalty was at the weak end of its grid (α_v = 1,000, rank 8); stronger penalties reproduce the additive model. No wider grid was tried after seeing the result.
