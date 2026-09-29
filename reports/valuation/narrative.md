## Interpretation, sanity checks and open decisions

**The three components are not yet on equal footing.** They are all in goals, but the team reconciliation shows they do not carry equal weight:

* Regressing team 5v5 goal differential on the components jointly gives slopes of **1.68 (possession), 1.18 (defense) and 1.46 (finishing)**, R² 0.46. The SD of summed player value (15.9 goals) is about half the SD of team goal differential (30.9). Team goal differential also contains goaltending, coaching and depth effects that are not player components, so a slope of exactly 1 is not expected. But slopes well above 1 for possession and finishing say those two are being under-weighted relative to defense.
* Possession is the clearest case. PM-EPA credits only individually observed puck actions plus a 30% pooled share, excludes shot quality and start values, and is then shrunk, so its SD is only 0.37 goals per qualified player-season against 2.10 for defense and 1.48 for finishing. This is a scale property of the metric, not a bug in the integration. **No rescaling was applied.** The recommended next step is the joint calibration (regress team goal differential on the components, with shrinkage) as part of the combining framework.
* The reconciliation intercepts are large (about −54 goals) because the sum is measured above replacement, not above average.

**Replacement level.**

* **The DPM replacement level is pooled over all seasons (the default, chosen 2026-09-29).** The pre-registered per-season estimate is noisy: its SD across seasons is 0.036 (F) and 0.029 (D) xGA/60, i.e. 0.5–0.6 goals per 1,000 minutes shifted for every skater of a position, and it turns positive for forwards in 2021 and 2025 (fringe forwards rated *better* than regulars, which is not credible). Under the per-season version defensemen were 72% of the 2025-26 top 25. With the pooled level they are 44%, and 35% of the top 100 across all seasons (they are 36% of players). The per-season version remains available as `def_gar_perseason` / `total_gar_perseason`.
* Pooling assumes the true replacement gap is stable across 2017-2025. If it drifted, pooled values carry a small bias in the early and late seasons.
* Sensitivity to the regular/fringe cutoff is small for rankings (rank correlation 0.995 for 11F/5D and 0.998 for 13F/7D) but not for levels: −0.5 and +0.2 goals per 1,000 minutes.
* The possession replacement rate is *higher* than the regulars' mean for forwards (about +0.05 xG/60 pooled). Depth forwards accrue PM-EPA quickly, because PM-EPA rewards forechecking, retrievals and takeaways. Regulars' possession value above replacement is therefore small (mean +0.29 goals; fringe pool mean 0.0).

**Other cautions.**

* Defense is the largest and least certain component. The median 95% interval on total value is ±2.5 goals (D) to ±2.8 (F). The top of the defense scale (Seider +12.2, Pelech +11.8 goals in 2025-26) reflects a chained prior plus a large DPM spread, and these are not comparable with typical WAR magnitudes.
* Total value intervals combine the component SDs assuming independence. The components are correlated (finishing ~ possession +0.39, finishing ~ defense −0.23), so the total intervals are approximate.
* DPM uncertainty uses the published bootstrap SE (of the un-refit headline). The refit changes point estimates slightly, and the pooled replacement level's own uncertainty is not included.
* Finishing is 5v5 only. Power-play and other-strength finishing is in `fin_gar_other_strengths_memo`, reserved for the special-teams component (its intervals are not computed).
* Skaters who changed teams are assigned to their primary team in the reconciliation.
