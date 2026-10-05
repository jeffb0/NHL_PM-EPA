# Score-effects adjustment of expected goals — methodology and pre-registration

*Written 2026-10-04 before any factor was estimated or any test run. Motivation (model owner, after WPG 3 at DET 2, where Winnipeg led 3–0 and was out-chanced 4.88 to 3.59): chances conceded while protecting a lead are not the same evidence
about a team as chances conceded in a tied game, though they are not nothing either. The design, test and rule below are fixed here; changes after results are seen are logged in §7.*

## 1. What score effects are

Teams that trail take more of the shots and teams that lead take fewer, more so late in games. Raw expected-goal totals therefore overstate the trailing team and understate the leading team relative to how they would play at even score.
The game model's in-season update moves player ratings by each game's expected-goal surprise, so it marks down teams that win by building leads; its methodology lists this as a limitation.

## 2. The adjustment

* **Factors.** At 5v5, for each score state d (the shooting team's goal differential, capped at −2 … +2) and period p (1, 2, 3): the league rate of expected goals for per minute in that state, r(d, p), from the unblocked 5v5 shots (cross-fitted expected goals) and the
  5v5 ice time in that state (the stint table). The weight is **w(d, p) = r(0, p) / r(d, p)**: a shot taken in state (d, p) counts as it would have at even score in the same period. Tied-game shots are unchanged; a trailing team's shots are discounted and a leading team's shots are marked up, by the league-average amounts.
* **Scope.** 5v5 shots in periods 1–3 only. Special teams, 4v4, 3v3, pulled-goalie play and overtime are left raw (pulled-goalie expected goals are an extreme score effect but a small share and already separate in the reports).
* **No future information.** For season T the factors are estimated on seasons before T. The live forecasts use all completed seasons.
* **Score-adjusted expected goals** for a team-game = adjusted 5v5 + raw everything else.

## 3. Uses

1. **Game reports (display):** score-adjusted expected goals are shown beside the raw figure for each team. This is description, not a model change, and is added regardless of the test.
2. **Game model (tested below):** the in-season rating update uses the score-adjusted expected-goal differential in place of the raw one.

## 4. Test

The game model of docs/gamemodel with rate model and inputs as currently used for the backtest (original rate model; hyperparameters frozen at λ = 0.04, α = 0, n0 = 3,000), 6,502 test games, 2021–2025.

* **Primary:** paired z on per-game log loss, model updated with score-adjusted differentials against the model updated with raw differentials.
* **Reported:** the factor table; the same comparison for the team-rating benchmark G2 (so the benchmark gets the same improvement if there is one); the comparison with λ and α re-tuned on 2019–2020 using adjusted differentials (in case the adjustment changes how fast ratings should move or how much weight goals deserve);
  how much the adjustment changes game-level differentials (correlation and typical size); log loss by season.

## 5. Gates and rule

| gate | rule |
|---|---|
| SE-1 | the score-adjusted update improves the game model (paired z > 2) |
| SE-2 | no harm (paired z > −2) |

The adjustment is rated sound in advance (a standard correction in hockey analytics with a clear mechanism and no fitted model parameters beyond league-average rates). By the owner's standing rule it is implemented in the forward forecasts if the primary z is above −1; a reliable improvement is claimed only if SE-1 passes.
Because the forward forecasts are part of the prospective market test, adoption is logged in docs/market/methodology.md as a model change with its date.

## 6. Limitations stated in advance

League-average factors (teams differ in how they play with a lead); the differential is capped at two goals; period is a coarse proxy for time remaining; 5v5 only; the factors also absorb whatever makes better teams lead more often (a small selection effect);
the expected-goal model itself is unchanged.

## 7. Deviations log

*(any change to §2–§5 after results are seen is recorded here with its reason)*

1. **Outcome, applied mechanically (2026-10-04).** SE-1 fails (z = 0.39), SE-2 holds. By §5 the adjustment is **implemented** in the forward forecasts (from the 2026-10-05 run) and shown in the game reports. No design change.
