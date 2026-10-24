# Elite-prospect priors and faster learning for inexperienced players — methodology and pre-registration

*Written 2026-10-03 before any of this was fitted or scored. Motivation: the game model rates Gavin McKenna (first overall, 2026, playing at 18) as a generic rookie (0.13 goals above replacement per 60), and the earlier pre-NHL priors (docs/prospects)
do not change that (0.12) because a single log-of-pick term learned nothing. The author's specification (2026-10-03): a rookie's expected production should depend on his draft position; top-three picks in their debut season should sit
between a top-six forward and a star; players who make the NHL at 18, straight after being drafted, deserve extra weight because teams rarely keep such players unless they have clearly earned it; and the prior should give way to actual NHL performance
as experience accumulates. The design, tests and rules below are fixed here; changes after results are seen are logged in §8.*

## 1. Two parts

**A. Rookie prior from draft slot and debut timing.** Replace the single log-of-pick term by attributes that can separate elite prospects:

* draft slot bins: picks 1–3, 4–10, 11–32 (later picks and undrafted players are the reference);
* debut timing: **D0** = first NHL season is the season of his draft year (the 18-year-old case), **D1** = the season after;
* the existing NHL-equivalent production from his last pre-NHL season (league factors estimated from earlier transitions only).

All are player attributes known before the season being forecast (for a player with no NHL game yet, the debut season is the season being forecast, since he is on the roster). As in docs/prospects each is multiplied by the fade
w = M0 / (m_eff + M0) (M0 = 1,000 discounted NHL minutes), so a debut-season player carries the full prior and a three-year regular about a fifth of it: the prior gives way to the player's own record as it accumulates, season by season.
They enter the projection ridge (same model, same training floor, λ by leave-one-season-out) in place of the two draft columns of docs/prospects.

**B. Faster in-season learning for inexperienced players.** Within a season the game model moves every dressed skater's rating by the team's expected-goal surprise in proportion to ice time. A player with little NHL history has a more uncertain rating and should move faster.
Each skater's share of the update is multiplied by g_i = 1 + κ w_i, with w_i = M0 / (career modeled minutes before the season + minutes played so far this season + M0); the total team step is unchanged (the shares are renormalised).
κ ∈ {0, 0.5, 1, 2, 4} is tuned on the 2019 and 2020 seasons (preseason priors from part A if adopted) and frozen. This is how the model "converges to actual performance" in season; it uses team results while the player dresses, weighted by ice time, because no per-game individual value estimate exists in the game model.

## 2. Why estimate rather than set the numbers

The author's prior (top-three picks between a top-six forward, about 0.35 per 60, and a star, about 0.60) is a reasonable belief, but the panel has real debut seasons to learn from. The estimate is reported next to that range.
Draft slot bins have few players (about three top-three picks a year, not all of whom debut immediately), so the ridge will shrink them; the implied rating for a top-three D0 forward and its bootstrap interval are reported so the shrinkage is visible.

## 3. Tests

1. **Player level:** expanding-window forecasts for target seasons 2020–2025 (docs/projection §4). Minute-weighted squared error of the next-season rate against the current model M, paired z by player-cluster bootstrap, on:
   **true rookies** (no NHL game before the season, ≥ 200 minutes), **first-round rookies**, thin-history players (m_eff < 1,000), and all players (≥ 300 minutes).
2. **Game level:** the game model (hyperparameters frozen) with preseason ratings from the new rate model, preseason-only and full, paired z on per-game log loss over the 6,502 test games.
3. **Part B:** with the adopted preseason ratings, the full game model with the tuned κ against κ = 0 on the 6,502 test games.
4. **Reported:** fitted coefficients; implied debut ratings by slot and timing for forwards and defensemen, with intervals; the 2026-27 rookies most affected; the author's range for comparison.

## 4. Gates and rules

| gate | rule |
|---|---|
| EP-1 | part A is better on true rookies (paired z > 2) |
| EP-2 | part A does no harm: all players z > −2, and both game-model comparisons z > −2 |
| EP-3 | part B improves the full game model (paired z > 2) |

**Part A** is rated sound in advance (draft position and debut age are standard, well-motivated priors). By the author's standing rule it is implemented if EP-2 holds and the true-rookie z is above −1; a reliable improvement is claimed only if EP-1 passes.
**Part B** changes the validated in-season dynamics for every player, so it is implemented only if EP-3 passes; if tuning selects κ = 0 there is nothing to implement.

If the estimated top-three D0 rating falls outside the author's 0.35–0.60 range, the estimate is used (not the range) and the difference is reported; the author can override with an explicit decision, logged as such.

## 5. What "implemented" means

The new rate model becomes the source of preseason ratings in the forward game forecasts (`src/live/predict.py`) and the rate model of the standings forecast. Because the forward forecasts are part of the prospective market test (docs/market), the change is logged there with its date,
and the `model` column of the forecast log changes; results before and after are reported separately. The published game-model validation report (docs/gamemodel) is not re-run.

## 6. Limitations stated in advance

Few elite picks (wide intervals); only players who reached the NHL are in the data, so the prior is conditional on making a roster; draft position is known for drafted players only; NHL-equivalent production is points-based;
part B learns from team results while the player dresses, not from his own shifts; a player's debut can be delayed by injury or development path rather than quality.

## 7. Deliverables

`src/prospects/elite.py`, `reports/metrics/elite_prospects.json`, `reports/prospects/elite_narrative.md`, tests, and the logged model change if adopted.

## 8. Deviations log

*(any change to §1–§4 after results are seen is recorded here with its reason)*

1. **Outcome, applied mechanically (2026-10-03).** Part A: EP-1 fails (true rookies z = 0.29), EP-2 holds (all players z = 0.65; game model preseason-only z = −1.08, full z = −1.52). By §4 it is **implemented**, with the estimated values, not the author's range.
   Part B: tuning chose κ = 4 (the edge of the grid) but the test seasons show no gain (z = −0.25); EP-3 fails and it is **not implemented**. The grid was not widened.
2. **The estimate is far below the author's range.** Top-three forwards debuting at 18 are projected at 0.16 goals above replacement per 60 (90% interval 0.12–0.19), against 0.35–0.60. The 14 such debuts in the data averaged 0.11. Reported to the author; no override has been made.
3. **Effective date.** The forward forecasts use the new rate model from the 2026-10-04 run; the 2026-10-03 forecasts were made with the original one.
