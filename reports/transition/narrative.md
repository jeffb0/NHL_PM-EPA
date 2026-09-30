## Interpretation

**Result: not adopted, and the reason is informative.** All three substantive gates fail (TR-1 reliability, TR-2 player-level, TR-3 team-level; TR-4 no-harm passes). The on-ice offense measure does not add validated information to the total beyond what PM-EPA already carries.

**Defensemen's possession value is small but not noise.** PM-EPA for defensemen has a standard deviation of only 0.016 goals per 60, yet it is **more stable than the on-ice measure**: year-over-year correlation 0.51 for PM-EPA against 0.24 for OPM (forwards 0.53 against 0.43). So the credited-action signal is real; it is *scaled down*. The within-position regression of OPM on PM-EPA has slopes of 3.3 (defensemen) and 4.4 (forwards): on-ice offense moves three to four times as much as PM-EPA for the same underlying signal, in line with the earlier team-level finding that PM-EPA is under-scaled (unshrunk slopes 2.84 ± 1.21 for defensemen and 1.37 ± 0.61 for forwards, docs/calibration §5d). The position-specific weights (about 1.6 on possession for both) only partly undo that compression, because the ridge holds weights toward 1.

**The transition residual is mostly noise for defensemen.** After removing the PM-EPA part, TR keeps a standard deviation of 0.068 per 60 but a year-over-year correlation of only 0.13 (forwards 0.28), below the 0.20 gate. Adding it does not improve next-season prediction (all skaters z = 1.14, defensemen z = 0.65, TR coefficient for defensemen 0.32 ± 0.24), and at the team level it makes held-out error worse (562.4 against 549.3, z = −1.27), with a forwards' weight of 0.03 (−0.27 to 0.31) and a defensemen's weight of 1.04 (0.59 to 1.45). The defensemen's unadjusted team slope is 2.62 ± 0.53, so a signal exists in the team sums, but it does not survive out-of-sample use.

**It does not capture the elite offensive defensemen either.** With TR, Quinn Hughes drops from 12.2 to 8.6, Cale Makar from 15.0 to 12.7 and Jaccob Slavin from 18.2 to 16.4, because their on-ice offense is already explained by PM-EPA (negative residuals), while Morrissey rises from 2.87 to 4.21 (TR +1.29) and Roman Josi from 4.6 to 3.7. TR is a residual: it rewards players whose team scores more than their credited actions predict, which is as likely to reflect linemates as a defenseman's own play. Morrissey's rank among 210 defensemen would move only slightly.

**What this says about the review.** The Morrissey/Appleton gap is not a missing offense-transition component. The model's view of Morrissey is a good 5v5 defenseman (65th percentile at 5v5), a poor power-play estimate driven by shorthanded chances against, and a wide interval (2.73 ± 2.7 before the position weights). Two things remain open and data-limited: the scale of defensemen's possession value (a stronger, less-regularized rescaling of PM-EPA for defensemen is supported by the unshrunk slope but not decisively by cross-validation), and the treatment of shorthanded chances against on the power play. A true breakout/entry measure needs puck-tracking event data (passes, carries, entries), which the play-by-play does not contain and the EDGE summaries do not provide.

**Limitations.** Teammate confounding within defensive pairs and lines (the ridge regularizes but does not solve it); OPM has no unshrunk analogue; independent-season fits are noisier for thin samples.

## What drives TR (a follow-up diagnostic)

TR is `OPM − β × PM-EPA`, with β = 3.3 for defensemen. Everything turns on the gap between a defenseman's on-ice offense and 3.3 times his credited-action value.

| 2022 defenseman | PM-EPA per 60 | expected OPM (3.3 × PM) | actual OPM | TR per 60 | TR (goals) |
|---|---|---|---|---|---|
| Josh Morrissey (WPG) | 0.026 | 0.086 | 0.140 | +0.053 | +1.29 |
| Adam Fox (NYR) | 0.053 | 0.175 | 0.169 | −0.005 | −0.13 |
| Devon Toews (COL) | 0.055 | 0.182 | 0.175 | −0.006 | −0.15 |
| Roman Josi (NSH) | 0.055 | 0.182 | 0.139 | −0.041 | −0.87 |
| Jaccob Slavin (CAR) | 0.049 | 0.162 | 0.083 | −0.079 | −1.77 |
| Cale Makar (COL) | 0.059 | 0.195 | 0.073 | −0.122 | −2.19 |
| Quinn Hughes (VAN) | 0.039 | 0.129 | −0.017 | −0.146 | −3.55 |

* **Morrissey (positive):** his credited actions are modest, but the team's offense with him on the ice is good relative to his teammates: 3.03 expected goals for per 60 against the Jets' 2.86 as a team, and an on-ice offense coefficient (0.14) above what his actions predict. That is the signal the component is meant to find: offense that does not show up in credited events (passing, breakouts, positioning).
* **Hughes and Makar (negative):** the reverse. Their credited actions are among the most valuable of any defenseman, but the team's offense while they are on the ice, after adjusting for teammates, does not rise by the amount those actions imply. Hughes is the extreme case: Vancouver had the weakest offense of this group (2.43 expected goals for per 60) and his on-ice figure (2.47) is only at the team level, even though his teammates were weak (the ridge adjusts for that).
* **It is not a linemate or team artifact, as I first suggested.** TR's correlation with the mean on-ice offense of a defenseman's teammates is 0.00, with his defense partner's −0.08, with his team's offense 0.04 and with his ice time 0.04; a regression on those four explains 10% of its variance. It tracks the player's own on-ice offense (correlation 0.71 with OPM per 60, 0.35 with his raw on-ice expected goals for per 60).
* **Part of it is a statistical artifact at the top end.** Across all defensemen TR is negatively related to credited-action value per 60 (−0.29) even though the fit is orthogonal on the totals, so the players with the highest PM-EPA systematically fall short of the linear line. OPM is a noisier, more heavily shrunk measure, so it lags at the extremes; that will push the very best puck-movers down regardless of their true offense. This is one reason the residual is not reliable enough to pass the gates.

**Status.** `total_gar_all_tr` is published as a variant because the model owner likes how it orders defensemen. It has not passed the pre-registered gates (reliability 0.13, next-season z 0.65, team-level z −1.27), so it is a judgment-based alternative, not an evidence-based improvement; the default total is unchanged. If it is ever made the default, that should be recorded as an owner decision.
