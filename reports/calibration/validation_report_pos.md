# Position-specific weights — validation report

Spec and pre-registered rules: [methodology §5d](../../docs/calibration/methodology.md). The all-strength identity with the seven main components entering separately for forwards and defensemen (18 weights with faceoffs, 4v4, 3v3 and the goalie pooled). **Exploratory:** the specification was chosen after four were tried on the same data (§5d).

**The position-specific weights were adopted**: held-out team-level error 549.3, against 588.9 for the pooled ridge and 609.8 for all ones (paired z = 2.24 against pooled, 2.27 against ones; both must exceed 2).

## Gates

| Gate | Result | Pass |
|---|---|---|
| D-cal-pos adoption: z > 2 against all ones and against the pooled ridge | 2.27 and 2.24 | ✅ |
| CAL-3 no harm at player level, deployment-adjusted (position-weighted vs uncalibrated total) | goals z = -0.47, xG z = 1.15 (needs > −2) | ✅ |
| CAL-5 identical to the uncalibrated total when weights are 1 | weights adopted, non-zero by design | ✅ |

## Weights (ridge toward 1, λ = 1.58e+03; applied weights are clipped to [0, 3])

| component | forwards weight (95% CI) | defensemen weight (95% CI) | F unshrunk slope | D unshrunk slope |
|---|---|---|---|---|
| possession | 1.60 (1.18 to 2.00) | 1.66 (1.28 to 2.00) | 1.37 ± 0.61 | 2.84 ± 1.21 |
| defense | 1.84 (1.42 to 2.29) | 0.97 (0.60 to 1.33) | 2.07 ± 0.27 | 0.80 ± 0.26 |
| finishing | 0.74 (0.61 to 0.88) | 1.26 (0.89 to 1.61) | 0.76 ± 0.08 | 1.23 ± 0.23 |
| turnover | 1.24 (0.74 to 1.71) | 1.32 (0.87 to 1.77) | 1.23 ± 0.61 | 1.09 ± 0.74 |
| penalties | 1.34 (0.87 to 1.79) | 0.65 (0.10 to 1.14) | 1.45 ± 0.41 | 0.27 ± 0.56 |
| power play | 1.33 (1.11 to 1.54) | 1.41 (0.95 to 1.88) | 1.31 ± 0.13 | 1.76 ± 0.46 |
| penalty kill | -0.19 (-0.63 to 0.21) | 1.98 (1.47 to 2.41) | -0.98 ± 0.35 | 3.16 ± 0.51 |

Pooled: fo 0.57, 4v4 0.81, 3v3 0.95, gk 0.98. DPM and the special-teams RAPM parts have no unshrunk analogue, so their slopes also carry shrinkage compression.

## Named players, 2022

| player | position | GAR before | GAR position-weighted | change | ± (GAR sd) |
|---|---|---|---|---|---|
| Josh Morrissey | D | 2.73 | 2.87 | 0.14 | 2.68 |
| Mason Appleton | C | 2.73 | 1.35 | -1.38 | 1.05 |
| Adam Fox | D | 16.35 | 21.08 | 4.73 | 2.45 |
| Jaccob Slavin | D | 13.79 | 18.22 | 4.43 | 1.85 |
| Mark Scheifele | C | 13.22 | 10.83 | -2.39 | 2.75 |
| Kyle Connor | L | 10.96 | 7.30 | -3.67 | 2.66 |

## Defensemen against forwards (qualified skaters, 500+ 5v5 minutes)

| pos_group | player-seasons | mean GAR before | mean GAR position-weighted | SD before | SD position-weighted |
|---|---|---|---|---|---|
| D | 1829 | 3.25 | 4.40 | 3.30 | 4.30 |
| F | 3205 | 5.94 | 4.62 | 4.71 | 5.07 |

External-validity look (not a criterion): correlation of total GAR per 60 with ice time per game and with points per 60.

| position | TOI/game before | TOI/game after | points/60 before | points/60 after |
|---|---|---|---|---|
| D | 0.197 | 0.218 | 0.155 | 0.187 |
| F | 0.372 | 0.353 | 0.595 | 0.553 |

Defensemen, 2022: largest gains and losses in rank among the 210 qualified.

| player_name | team | total_gar_all | total_gar_all_ridge | rank_before | rank_after |
|---|---|---|---|---|---|
| Jared Spurgeon | MIN | 13.5 | 17.4 | 4 | 5 |
| Marc-Edouard Vlasic | SJS | 7.0 | 9.1 | 39 | 40 |
| Nicolas Hague | VGK | 3.3 | 4.1 | 112 | 113 |
| Jacob Bryson | BUF | 1.4 | 1.8 | 152 | 153 |
| Joshua Brown | ARI | -2.5 | -4.3 | 207 | 208 |
| Jordan Oesterle | DET | -0.8 | -0.7 | 187 | 188 |
| Tony DeAngelo | PHI | -7.5 | -9.2 | 209 | 210 |
| Brayden McNabb | VGK | 10.7 | 12.7 | 12 | 14 |

| player_name | team | total_gar_all | total_gar_all_ridge | rank_before | rank_after |
|---|---|---|---|---|---|
| Jaccob Slavin | CAR | 13.8 | 18.2 | 2 | 2 |
| Devon Toews | COL | 13.5 | 17.9 | 3 | 3 |
| Adam Fox | NYR | 16.3 | 21.1 | 1 | 1 |
| Chris Tanev | CGY | 11.4 | 14.7 | 10 | 10 |
| Mikey Anderson | LAK | 8.1 | 10.8 | 28 | 28 |
| Jonas Brodin | MIN | 8.7 | 11.8 | 19 | 19 |

## Interpretation

**What the review found about Morrissey and Appleton (2022).** The two come out at the same total, 2.73 goals above replacement, for different reasons. Morrissey's 5v5 value is +3.36 (65th percentile among 210 qualified defensemen: possession +0.64, 88th percentile; finishing +1.37, 96th; defense +0.74, 41st), and his special teams total −0.63, driven by a **−1.61 power-play value (2nd percentile)**. Appleton's total is mostly special-teams value on a small base (529 5v5 minutes): +1.43 on the penalty kill, +0.65 finishing, +0.48 penalties, negative possession. The model rates Morrissey a good 5v5 defenseman with a poor power-play estimate, not an elite one, and his estimate is uncertain (2.73 ± 2.7 goals, one standard deviation), so the model cannot actually separate him from a depth forward, or from an elite defenseman.

**Why his power-play value is negative, and why that is not clearly wrong or clearly right.** With him on ice the Jets' power play produced 8.62 expected goals per 60, against 6.23 without him (+2.4), but also allowed 3.60 shorthanded expected goals against per 60, against 1.68 without him (+1.9). The model, after adjusting for teammates, credits him +0.47 per 60 on offense over a fringe defenseman but charges him +0.9 on shorthanded chances against, a net −0.43 per 60 (−2.3 goals over 314 minutes). Shorthanded xG against is a weak signal that mostly reflects the forwards' pinching and turnovers as much as the point man. We tested whether heavier shrinkage on the defense block (which would soften his charge, to −0.63 goals with ten times the penalty) is supported by held-out prediction: it is not, the held-out error is flat (357.19 against 357.22), so the data cannot say either way and no change was made.

**A structural problem the review did expose: defensemen were systematically compressed relative to forwards, and forwards' penalty-kill value was noise.** Possession value for defensemen has a standard deviation of 0.016 goals per 60, against 0.116 for defense, so a defenseman's total is driven by defense and penalty kill; the correlation of a defenseman's GAR per 60 with ice time is 0.20 (0.37 for forwards) and with points per 60 is 0.16 (0.60 for forwards). In the position-split team-level regression, forwards' penalty-kill value has a slope of −0.98 ± 0.35 (it carries no information about team success; it is mostly the constant per-minute replacement credit for playing the penalty kill), while defensemen's is 3.16 ± 0.51 and defensemen's possession slope is 2.84 ± 1.21 (forwards 1.37). Splitting the seven main components by position improves held-out team-level error (549 against 589 for pooled weights and 610 for all ones; z = 2.24 and 2.27), and the position-weighted total does no harm at the player level (deployment-adjusted goals z = −0.47, xG z = +1.15). It was chosen after four specifications were tried on the same data, so treat the evidence as moderate.

**What changes.** Appleton drops from 2.73 to 1.35 (his penalty-kill credit is switched off and forwards' defense scaled up). Defensemen rise as a group (mean 3.25 to 4.40; forwards 5.94 to 4.62), so defensemen and forwards become more comparable; the correlation of defensemen's GAR per 60 with ice time rises from 0.20 to 0.22 and with points from 0.16 to 0.19. **Morrissey barely moves (2.73 to 2.87) and keeps his place among defensemen (about 124th of 210 before and after)**: the position weights lift every defenseman roughly equally and his gap is in his own power-play and defense estimates. Adam Fox (16.3 to 21.1) and Jaccob Slavin (13.8 to 18.2) rise most in absolute terms.

**So the honest answer to "adjust something so it captures this".** The position-specific weights fix a real structural bias (defensemen undervalued, forwards' penalty-kill value inflated) and are published as a variant (`total_gar_all_pos`; the default stays uncalibrated until you decide). They do not lift Morrissey; that needs a different kind of change. Candidates, none validated yet: (1) a better treatment of shorthanded chances against on the power play (attribute them to the forwards and the play, not the point man, or shrink that block harder), which the held-out data can neither support nor reject; (2) a defensemen-specific transition/possession measure (zone entries and exits, breakouts), which the current possession value hardly captures; (3) showing intervals with every rank, since 2.73 ± 2.7 against 1.35 ± 1.05 is not a distinction the model can make.

