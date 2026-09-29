## Interpretation, sanity checks and open decisions

**Scale: turnover and faceoffs explain much of PM-EPA's under-scaling, and shrinkage explains more.** The team reconciliation regresses team 5v5 goal differential on the summed player components:

* **Three components only** (with the updated defense column, which now also excludes unobserved starts): PM-EPA 2.04 ± 0.43, DPM 1.30 ± 0.16, FIN 1.61 ± 0.25 (R² 0.45).
* **Adding turnover cost and faceoffs** (all shrunk): PM-EPA 1.53, DPM 1.14, FIN 1.80, TO 1.51 ± 0.66, FO 1.99 ± 0.80 (R² 0.46).
* **With possession and turnover unshrunk:** PM-EPA 1.16 ± 0.31, DPM 1.11, TO 1.18 ± 0.44, FO 1.88, FIN 1.79 (R² 0.48). Possession, defense and turnover are then close to unit scale.
* **Two separate effects are at work.** The missing turnover and faceoff value was real (the possession slope falls from 2.04 to 1.53). The rest is shrinkage compression: an independent-player prior shrinks effects shared across a team, so team sums of shrunk values understate team totals. Slopes above 1 at team level are therefore expected for every shrunk component and do not mean player values are wrong.
* **Finishing (1.8) and faceoffs (1.9) remain the least well scaled at team level.** Whether finishing would fall to unit scale unshrunk was not tested. **No rescaling was applied.** The joint calibration is the next step and should be built with this shrinkage effect in mind.
* The SD of summed player value (15.3 goals) is still about half the SD of team goal differential (30.9), which also contains goaltending, coaching and depth effects that are not player components.
* Penalties are excluded from the 5v5 regression because they are realized on special teams. Against special-teams goal differential their slope is 1.35 ± 0.51.

**Replacement level.**

* **The DPM replacement level is pooled over all seasons (the default).** The per-season estimate was noisy (SD 0.036 for forwards, 0.029 for defensemen, xGA/60, about 0.5–0.6 goals per 1,000 minutes, and positive for forwards in 2021 and 2025). The per-season version remains available as `def_gar_perseason` / `total_gar_perseason`. Pooling assumes a stable gap between regulars and replacement over 2017-2025.
* **Sensitivity to the regular/fringe cutoff** is small for rankings (rank correlation 0.996 for 11F/5D and 0.999 for 13F/7D) but not for levels (−0.5 and +0.2 goals per 1,000 minutes). In the sensitivity runs the turnover, faceoff and penalty replacement stays at the default cutoff.
* **The possession replacement rate is higher than the regulars' mean for forwards** (about +0.05 xG/60 pooled), because depth forwards accrue PM-EPA quickly. Regulars' possession value above replacement is small (mean +0.29 goals), and their turnover value is negative (−0.20) for the reason given in the components report (the replacement pool handles the puck less).

**Face validity of the combined table (2025-26, top of the list).** Seider 11.1, Robertson 11.1, Pelech 10.7, O'Reilly 9.7, Kopitar 9.3, Zub 9.2, Kyrou 8.9, Brodin 8.8 goals above replacement. O'Reilly and Kopitar are there through faceoffs (+1.8 and +2.0) on top of solid defense and finishing. Defensemen are 36% of the top 25 (36% of players). The bottom is DeAngelo (−8.1), Klingberg (−6.1), Levshunov and Domi.

**Other cautions.**

* **The turnover component moves value away from puck-dominant players** (see the components report): it charges observed giveaways but cannot credit unobserved successful plays. It is right at team level and biased in allocation. The role-adjusted variant (`to_ra_gar`, `total_gar_to_ra`) compares each player with others of similar observed puck involvement; it changes totals by up to about ±0.5 goals a season (mean +0.24 for the most-involved quintile, −0.06 for the least), and the team-level regression fit is slightly better with it (R² 0.473 vs 0.464).
* **Defense is the largest and least certain component.** The median 95% interval on total value is ±2.7 goals (D) to ±3.0 (F). Its top values (Seider +9.9, Pelech +10.1 goals from defense alone in 2025-26) are not comparable with typical WAR magnitudes.
* **Total-value intervals combine component SDs assuming independence.** The components are correlated (finishing ~ turnover −0.47, finishing ~ possession +0.39, defense ~ turnover +0.27), so the total intervals are approximate.
* **DPM uncertainty** uses the published bootstrap SE, and the pooled replacement level's own uncertainty is not included.
* **Finishing is 5v5 only**; power-play finishing is in a memo column reserved for special teams.
* **Skaters who changed teams** are assigned to their primary team in the reconciliation.
