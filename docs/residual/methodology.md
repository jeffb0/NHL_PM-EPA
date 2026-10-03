# Decomposing the portable team residual — methodology and pre-registration

*Written 2026-10-03 before any decomposition was computed. §3 is descriptive (no gate); the forecast test and its rule in §4–§5 are fixed here; changes after results are seen are logged in §8.*

## 1. Question

The published forecast adds to each player his minute share of last season's **team residual**: the team's actual goal differential minus the sum of its players' values and its realised goaltending. That lump has a coefficient near 1
(it re-appears next season with the players), and three candidate explanations have been ruled out or shown to be small (young players, chemistry, possession scale). What is the residual made of, which parts persist, and who carries them?

## 2. Accounting identity (exact)

A team's goal differential (shootouts counted as zero) splits into five observed parts, each compared with the player-value components that are meant to explain it:

| bucket | observed | explained by |
|---|---|---|
| **play5** | 5v5 expected goals for minus against | possession + defense + turnovers + faceoffs |
| **fin5** | 5v5 goals for minus expected goals for (shooting above expected) | finishing |
| **st** | power-play and penalty-kill goals for minus expected goals against in those states | power play + penalty kill + penalties |
| **gk** | expected goals against minus goals against, all states with a goalie in net | realised goaltending (by construction close to zero residual) |
| **other** | 4v4, 3v3, goalie-pulled states, and empty-net goals (the remainder) | 4v4 + 3v3 |

Observed and explained parts are centred within season; bucket residual = observed − explained; the five residuals sum to the team residual (the identity is checked numerically and the largest discrepancy reported).
Expected goals are the cross-fitted shot model; component sums are the projection panel's (rolling-refit finishing and faceoffs).

## 3. Descriptive analysis (reported, no gate)

1. **What it is made of:** each bucket's share of the residual's variance (covariance with the total over the total's variance); for each bucket the correlation and slope of observed on explained (a slope above 1 means the components are under-scaled for that bucket).
2. **What persists:** year-over-year correlation of each bucket's residual for the same team; and the **carry coefficients**: regression of a team's residual in season s on the bucket residuals of season s − 1 carried by its season-s players in proportion to their minutes (one coefficient per bucket; the lump coefficient for comparison).
3. **Who carries it:** the same regression with the lump split into the part carried by forwards, the part carried by defensemen, and the previous season's residual of the team itself (what stays with the organisation), identified by player movement.

## 4. Forecast test

**Split prior:** in the game model's preseason ratings the lump (coefficient 1) is replaced by Σ_b c_b(T) × bucket residual per skater-minute, with c_b(T) the carry coefficients fitted on seasons before T (expanding; with fewer than two seasons, c_b = 1).
For a fair read of what the split adds, a **fitted lump** (one coefficient fitted the same way) is also run.

* **RD-1:** split prior against the published prior, preseason-only game model, 6,502 games (paired z on log loss).
* **RD-2:** split prior against the fitted lump, same model.
* **RD-3 (no harm):** the full game model with the split prior against the published one, z > −2.
* Reported: the standings backtest of the published configuration with the split prior.

## 5. Implementation rule

The split replaces a working one-parameter term with five fitted coefficients on roughly 100–250 team-seasons, so parsimony applies: **implement only if RD-1 passes (z > 2) and RD-3 holds.** If only the fitted lump helps, that one-parameter change is reported as a candidate but not adopted in this step.

## 6. What the result can say

If the residual is mostly **play5** and that bucket carries, the components under-measure 5v5 play-driving (a measurement problem in possession and defense). If it is mostly **fin5** or **gk**, it is shooting and save percentage beyond the model, and a carry coefficient near zero would say it is luck that the lump is wrongly carrying.
If the part that stays with the team is large, it is coaching or system. A split prior that beats the lump means the lump mixes persistent and non-persistent parts.

## 7. Limitations stated in advance

Team-season counts are small (about 30 a season), so bucket coefficients are noisy and collinear for stable rosters; primary-team assignment misplaces traded players; the goaltending bucket is near zero by construction and says nothing about goalie persistence;
5v5 expected goals carry score effects; the descriptive regressions use all seasons and classical standard errors.

## 8. Deviations log

*(any change to §2–§5 after results are seen is recorded here with its reason)*

1. **Outcome, applied mechanically (2026-10-03):** RD-1 fails (split against published prior, preseason-only game model: z = 1.80), RD-2 fails (split against fitted lump: z = 1.34), RD-3 holds (full model z = 0.14). **Not implemented.**
2. **Clarification of §3.3 made while coding, before results:** "what stays with the organisation" is measured as the previous season's team residual (scaled to this season's minutes) minus the parts carried in by this season's forwards and defensemen, i.e. the residual share of players who left.
3. No other change. The descriptive findings suggest follow-ups (carry the 5v5 play bucket alone; rescale the whole 5v5 play block); those would be new, post-hoc hypotheses on seasons already examined and are **not** tested here.
