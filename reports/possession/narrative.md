# Possession under-scaling — results

Pre-registration: [docs/possession/methodology.md](../../docs/possession/methodology.md). Metrics: `reports/metrics/possession_scale.json`.

## Outcome

**Tuning selected no rescaling, so nothing is implemented.** On the two tuning seasons (2019, 2020) the preseason-only game model's log loss rises with every increase of either factor (0.67444 at 1 | 1, 0.67457 at 1.5 | 1.5, 0.67536 at 3 | 4).
By the pre-registered rule the hypothesis is "not supported by the tuning seasons"; PS-1 and PS-2 are recorded as not passed, PS-3 (no harm) trivially holds.

## The declared context variants on the test seasons (reported; not a clean test)

| variant | preseason-only game model | full game model | standings (published model) | standings (roster model alone) |
|---|---|---|---|---|
| κ = 1.5 for both positions | −0.00023 log loss, z = 3.48 | −0.00011, z = 2.68 | 13.99 → 13.96, z = 1.28 | 14.60 → 14.55, z = 2.60 |
| κ = 3 for defensemen only | −0.00030, z = 2.75 | −0.00007, z = 1.01 | | |

(6,502 games, 2021–2025. Better in four of five seasons for both variants; worse in 2025.)

## Interpretation

**The tuning seasons and the test seasons disagree.** The rule fixed in advance used 2019–2020 to choose the factors, and those seasons say "do not rescale". The two variants declared for context do better on 2021–2025, consistently
(four of five seasons) and with z of 2.7–3.5. That evidence is encouraging but not clean: the 1.5 figure came from team-level fits that included the test seasons, the comparison was not the gated one, and it is one of two variants shown.
The honest status is "probably real, small, and not established by the protocol".

**Either way it is small.** In the best case the rescaling improves the preseason-only model by 0.0002–0.0003 log loss. The gap between that model and a tuned team rating is 0.008, so even taken at face value under-scaled possession value explains about 3% of what the preseason player prior is missing. As §2 expected, most of a rescaling's effect on a team total is absorbed by a smaller carried residual:
the roster model alone gains 0.05 points of standings RMSE (z = 2.6) and the published model 0.03 (z = 1.3).

**What this closes.** Three candidate explanations for the preseason prior's weakness have now been tested on the game-level test bed: young players (no team-level effect), chemistry (none), possession scale (at most 3% of the gap).
What remains is the portable team residual itself, which the forecast already uses as a lump. It behaves like additive value that travels with players and that none of the components measures; identifying it is a measurement problem
(what do those players do that the event data do not credit?), not a re-weighting problem.

**Option for the model owner.** Adopting κ = 1.5 would be a judgment call outside the pre-registered rule (as the TR total was). Its effect on forecasts is a fraction of a point per team. It is not implemented.

**Limitations.** Two unusual tuning seasons; a coarse grid; context variants informed by all-season estimates; game tests on the original rate model.
