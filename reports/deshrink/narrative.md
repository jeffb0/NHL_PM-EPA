# De-shrinking the 5v5 play block — results

*Run 2026-10-12 under docs/deshrink/methodology.md (pre-registered the same day, before the run). Numbers are in reports/metrics/deshrink.json. One deviation: the rule says implement; implementation is held for the author's decision (methodology §7).*

## Verdict

**The fix is built and it does not improve the forecasts.** Every comparison is slightly on the wrong side of zero. The pre-registered rule (implement unless z is below −1) is met on both primary tests, but the two checks where the fix was expected to matter most,
early-season games and the standings projection, are the most negative. It is **not switched on** in the live forecast; that is a deviation from the rule and is the author's call.

## Tests (6,502 games, 2021–2025)

| comparison | published | de-shrunk | paired z |
|---|---|---|---|
| DS-1, preseason-only model | 0.66920 | 0.66927 | −0.25 |
| DS-2, full model with in-season updating | 0.65854 | 0.65866 | −0.59 |
| preseason-only, each team's first 20 games (1,652 games) | 0.67690 | 0.67756 | −1.12 |
| full model, first 20 games | 0.67418 | 0.67465 | −0.87 |
| standings backtest, points per 82 games (RMSE; κ = 1.38) | 13.99 | 14.10 | −1.73 |

By season the full model is worse in 2021, 2023 and 2025 and better in 2022 and 2024, never by more than 0.0007.

## What the fix did

* **The factor.** κ estimated only from earlier seasons runs from 1.38 (for 2019) to 1.60 (for 2025); from all nine seasons it is 1.62, the figure in docs/residual.
* **Ratings.** In 2025 the de-shrunk ratings correlate 0.97 with the published ones and are 17% more spread out (SD 0.202 against 0.173 goals per 60); the average player moves 0.04.
* **The residual it was meant to remove is mostly still there.** With the block scaled so that the slope is exactly 1, the 5v5 play residual's SD falls only from 14.2 to 12.1 goals per team-season, and the whole team residual's from 22.1 to 20.1.

## Why it does not help

**Under-scaling was the smaller part of the 5v5 play residual.** The components correlate 0.85 with teams' actual 5v5 expected-goal differential. A scale factor fixes the slope but cannot touch the 28% of variance that the components do not track at all,
and that part is 12 of the original 14 goals of residual SD. docs/residual read the slope of 1.62 as the main story; this test shows it is not. Most of what the carried residual carries is team-level 5v5 play that is uncorrelated with the sum of the players' measured values.

**The carried residual was already doing the job.** Because the lump is recomputed after scaling, a team's total barely changes; the fix only moves credit within a team, toward players with large measured 5v5 values and away from the rest.
The backtest says the equal per-minute split was no worse than that, and slightly better early in the season. That is consistent with docs/residual's other finding, that most of a departing player's share of the residual stays with the team.

## What this means for the earlier claim

In the previous write-up this was called the model's largest measured weakness, with the fix "tested at z = 1.80". That figure was for a different change (splitting the carried residual by bucket), on the preseason-only model, and the same study found the gain vanished with in-season updating (z = 0.14).
The claim should not have been made without that qualification. The direct test here settles it: scaling the 5v5 block is not an improvement.

## Limitations

The hypothesis came from the same seasons it is tested on; a single factor for four components and both positions; κ for the standings pipeline is one value rather than one per season; classical paired z.
