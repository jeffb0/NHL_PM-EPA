## Interpretation

This report is computed with the **role-adjusted turnover component (TO-RA) as the default turnover component** (user decision, 2026-09-29, made after the first version of this calibration compared the two). The original turnover charge is kept as `to_gar_orig` / `total_gar_to_orig`.

**Team level.** The unshrunk possession component runs at about 1.8 times its nominal scale (1.79 ± 0.28) and finishing at about 0.8 (0.81 ± 0.09). The ridge-weight intervals for both exclude 1 (possession 1.53, 1.21–1.82; finishing 0.82, 0.69–0.95). Defense (1.05 unshrunk), turnover (1.18 ± 0.40) and faceoffs (1.32 ± 0.63) are consistent with 1. The shrunk slopes above 1 for finishing (1.87), turnover (2.19) and faceoffs (2.07) sit against unshrunk slopes near 1 or below, which confirms the shrinkage-compression point in §1 of the methodology: independent-player priors shrink team-shared effects, so a slope above 1 on shrunk values does not by itself mean the scale is wrong.

**Decision.** The leave-one-season-out gain from the ridge weights (MSE 468.5 vs 485.2) gives paired z = 1.50, below the pre-registered threshold of 2, so all weights stay at 1. With only 284 team-seasons the honest reading is "suggestive, not established": possession may be under-scaled and finishing over-scaled, but the data did not clear the bar set in advance.

**Player level.** The ridge-weighted total predicts next-season within-team on-ice goal differential worse than the current total (paired z = -7.66; own-rate slope 0.81 vs 0.93). On xG the two are indistinguishable (z = -1.61). CAL-3 therefore fails for the ridge variant, which supports not adopting it; the gate is moot while weights stay at 1.

**Forward component slopes** (next-season goal differential on each season-t component, teammates controlled) are not calibration weights, because they mix scale error with year-to-year persistence and with how each component is observed. Possession (1.99) and finishing (1.58) are above 1, defense (0.50) is below, faceoffs are near zero for goals (0.02 ± 0.24), and **turnover cost is still negative (-0.66 ± 0.35 goals, -1.04 ± 0.24 xG)**, though less so than with the original charge (-0.99 and -1.19 in the first run). Players charged more turnover cost in one season tend to do better the next, once teammates are controlled. Likely contributors, not tested here: turnovers are recorded only for players who touch the puck, and the charge is partly a by-product of playing style that also drives shot volume. The role adjustment removes the involvement-driven part of the level but evidently not all of this relationship.

**Post-hoc variant comparison** (exploratory, `src/calibration/variants.py`; not pre-registered, and run on the same data that raised the question; z is against the current total with TO-RA, positive favors the variant):

| total | goals z | xG z |
|---|---|---|
| with the original (not role-adjusted) turnover | -8.90 | -7.39 |
| without turnover cost | +5.71 | +3.90 |
| without faceoffs | +2.15 | +0.68 |
| without penalties | -5.69 | -3.94 |
| without turnover and penalties | +0.61 | +0.37 |
| without turnover, faceoffs and penalties | +2.83 | +1.35 |

TO-RA predicts next-season results clearly better than the original charge, which is the basis for making it the default. Dropping the turnover component altogether still predicts better than including even TO-RA, so turnover cost remains the weakest component of the total for this purpose. Penalties help. Faceoffs are marginal. No component has been removed: the pre-registered structure stays, and the turnover question is deferred to the whole-model validation.

**Limitations.** 284 team-seasons; primary-team assignment misplaces the share of value of players traded mid-season; DPM is calibrated only on its shrunk values; team residual variance includes goaltending and coaching, which no component measures; and the forward check uses a single next-season horizon.
