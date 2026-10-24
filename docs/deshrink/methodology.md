# De-shrinking the 5v5 play block of the player ratings — methodology and pre-registration

*Written 2026-10-12 before the fix was computed or any test run. Motivated by the residual decomposition (docs/residual), which found the 5v5 play block under-scaled. The definition, tests and rule below are fixed in advance; changes after results are seen are logged in §7.*

## 1. The problem (from docs/residual)

The 5v5 play components of a player's value (possession, defense, turnovers, faceoffs) rank teams correctly but have about half the spread of teams' actual 5v5 expected-goal differential: over 2017–2025 the slope of observed on explained is 1.62.
The regularised player estimates are each shrunk toward zero, so a strong team's players sum to too little. What is left over is stable for a team (0.65 year over year) and travels with players at nearly full value (0.94 ± 0.15).

Today the forecast patches this with the **carried team residual**: every skater gets the same per-minute share of last season's whole team residual. That patch has two faults: it gives a team's weakest play-driver the same top-up as its best, and it carries shooting luck, special-teams and empty-net residuals at the same rate as the persistent 5v5 part.

## 2. The fix

For a forecast season T:

1. **κ(T)** = the slope through the origin of observed on explained 5v5 play, both centred within season, over team-seasons **before T** (the quantity reported as 1.62 for all seasons; here it never uses season T or later).
2. Every player-season's four 5v5 play rates (`r_pm`, `r_def`, `r_to`, `r_fo`) are multiplied by κ(T) and the total rate is recomputed.
3. Everything downstream is rebuilt from the scaled panel: the rate model (trained on seasons before T), and the carried team residual, which is now what the scaled components still fail to explain.

So the top-up for under-measured 5v5 play goes to each player in proportion to his own measured 5v5 play instead of equally to his teammates, and the carried lump shrinks to what remains. Nothing is tuned on game outcomes.

## 3. Honest expectation, and what was already known

* docs/possession found that rescaling possession alone moves forecasts little, because the carried residual absorbs most of a rescaling. The same will apply here: the team total changes little; the allocation among players changes.
* docs/residual found that a better preseason prior improved the preseason-only model (z = 1.80) but that **with in-season updating the gain disappeared (z = 0.14)**, because the updates learn the same thing within weeks.
  A fix to the prior should therefore be expected to matter early in a season and for the standings projection, and little for mid-season game forecasts.
* This hypothesis was suggested by looking at the 2017–2025 seasons, so the backtest below is not a clean test of it (docs/residual §8.3). The clean test is prospective.

## 4. Tests

Game model of docs/gamemodel as configured for its backtest, 6,502 games of 2021–2025; priors rebuilt for each season with κ(T).

* **DS-1:** preseason-only model with the fix against the published prior (paired z on log loss).
* **DS-2:** full model (in-season updating) with the fix against the published one.
* **Reported:** κ(T) by season; after scaling, the slope of observed on explained 5v5 play and the SD of the team residual (all seasons, κ from all seasons, descriptive);
  the same two comparisons restricted to each team's first 20 games of a season; the standings backtest of the published configuration with the fix (κ from all seasons before 2021 for every forecast season, since that pipeline takes one factor).

## 5. Rule

The mechanism is measured and the factor uses no future information, so the method is rated sound in advance. By the author's standing rule:

* **implement** in the forward forecasts (2026-27 preseason ratings, with κ from 2017–2025) if DS-1 and DS-2 both have z above −1;
* a reliable improvement is claimed only for a comparison with z above 2;
* if either z is below −1, not implemented, and reported.

Implementation changes the forward forecasts, so it is logged in docs/market/methodology.md with its date.

## 6. Results

See reports/deshrink/narrative.md. Run 2026-10-12: DS-1 z = −0.25, DS-2 z = −0.59 (both above −1; neither an improvement); first-20-games z = −1.12 and −0.87; standings RMSE 13.99 → 14.10 (z = −1.73).

## 7. Deviations

1. **Implementation held (2026-10-12).** §5 says implement when both primary z are above −1, and they are. The fix is nevertheless **not switched on** in the forward forecasts, pending the author's decision, because every reading is negative
   and the two reported checks where §3 expected the fix to matter most (early-season games, standings) are the most negative. The descriptive result also weakens the "sound in advance" rating: scaling removes only 2 of the 14 goals of 5v5 play residual SD.
   `reports/metrics/deshrink.json` records `implement: true` (the rule's output) and `implemented: false`.
2. **Owner's decision (2026-10-12): leave the fix off.** The forward forecasts and the standings projection are unchanged.
