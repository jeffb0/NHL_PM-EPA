# Improving the player-to-team forecast — candidates, tests and decision rules (pre-registration)

*Written 2026-09-30 before any candidate was built or scored. The ranking, tests and rules below are fixed here; changes after results are seen are logged in §7.
Baseline: the standings backtest (docs/standings/methodology.md), M1 RMSE 14.58 points per 82 over 2021–2025, against 13.70 for last season's goal differential (B2).*

## 1. What the evidence says the gap is

* The roster model loses to last season's goal differential by about 0.9 points, and a team-level blend of the two (M3) does not help (z = −0.82).
* Even with perfect roster knowledge (the actual players and minutes) it is 13.9, still not better than B2, so most of the gap is **not** about who plays; it is information that last
  season's *results* carry and the player values do not.
* The team-context study found that about 74% of a team's persistence travels with its players and about 26% stays with the team; the whole-model validation found a team-level slope
  of 1.16 and the backtest a calibration slope of 1.22: player sums are compressed.
* The projection layer is trained to minimise player-level error (minute-weighted); its summed forecast did worse at the team level than last season's raw rates (docs/projection, PR-6). Player-level
  optimality does not imply team-level optimality, because errors that are shared by teammates (system, coaching, the ridge's shrinkage toward zero) add up in a team sum while independent errors cancel.
* Opening-roster uncertainty costs about 0.7 points (oracle roster 13.9 vs 14.6); availability (games missed) is part of it. **An availability/injury model is deliberately not a candidate here**:
  the author deferred it (2026-09-30).

## 2. Candidates considered and the ranking

Considered and not taken forward: fitting a free slope on the team sum (M2 already did worse, 15.1); a separate goalie-share model (goalie term SD is about 7 goals, small relative to the gap);
richer aging (already in the projection layer); EDGE covariates (null result, docs/edge); availability (deferred, above).

**1. Portable team residual (C1), novel, highest expected gain.** Last season's result contains a part the player values did not explain (the team residual: actual goal differential minus
the season's own player-value sum and realised goaltending). B2 and M3 use the whole team result at the team level, so it cannot follow players who move, and it keeps credit for players who
leave. C1 attributes last season's team residual to that season's skaters in proportion to their minutes (a residual per skater-minute), and each player **carries** his share to his new roster in
proportion to his projected minutes. One coefficient `c` (how much of the carried residual is real) is fitted on earlier seasons. It targets directly the information that B2 has and M1 lacks, and
it respects roster turnover, which a team-level blend cannot.

**2. Aggregation-consistent fine-tuning (C2), novel, higher variance.** Keep the player-level projection, then learn a **team-level correction** on features aggregated the way the forecast aggregates:
for forwards and defensemen separately, the minute-weighted sums of the player features (ten shrunk component rates, the no-history indicator, the four age terms), 30 features. A ridge (toward zero,
i.e. toward the player model; λ by leave-one-season-out) on the team residual (actual goal differential per 82 minus M1's). This lets team outcomes re-weight which components and which ages persist
**at the team level**, where teammates' shared errors add up. Risk: 30 features on 62–190 training team-seasons.

**3. Deployment-aware minutes (C3), sound, modest expected gain.** The minutes model uses only a player's own history and age; minutes are then scaled within team. Coaches give ice time by
quality **relative to the rest of the roster**. C3 adds two features to the minutes model: the player's shrunk history total rate and its deviation from the mean of his team-position group on the roster.
A player who joins a weaker roster gets more minutes, one who joins a deeper roster fewer, before the within-team rescaling.

## 3. Tests (same backtest, same information rules as docs/standings §3)

Test seasons 2021–2025 (160 team-seasons); every fitted quantity (c, the ridge, the minutes model) uses seasons < T only; the goal-differential-to-points map is M1's.

* **Primary:** RMSE of points per 82; paired z of the candidate against M1 (z > 0 favours the candidate).
* **Secondary (reported):** paired z against B2 and M3; calibration slope; number of test seasons in which the candidate beats M1; for C3, the within-team minutes-share error against actual shares.
* **Combination (reported, not a gate):** all implemented candidates together.

## 4. Decision rules (the author's instruction, made mechanical)

A priori soundness (stated before the results): **C1 sound** (one fitted parameter, a transparent accounting identity, motivated by the team-context finding); **C3 sound** (standard coaching behaviour, two features);
**C2 less certain** (many fitted parameters on few team-seasons; an outcome-trained correction can overfit).

| outcome | sound (C1, C3) | less certain (C2) |
|---|---|---|
| z > 2 against M1 (reliable improvement) | implement | implement |
| −1 < z ≤ 2 (no reliable change) | implement | do not implement |
| z ≤ −1 (worse) | do not implement | do not implement |

"Implement" means the candidate becomes part of the published standings forecast (`src/standings/next.py`), the 2026-27 forecast is regenerated, and the change is logged in docs/standings §9.
If more than one is implemented, the combination is used only if it is not worse than the best single candidate by z < −1; otherwise the best single one is used.

## 5. What a result can say

A C1 gain would mean last season's unexplained team result is partly portable with players (for example, the player values miss something the players take with them). A C2 gain would mean the
player-level projection weights persistence differently from what team outcomes support. A C3 gain would mean roster context shapes deployment enough to matter at the team level. Parity means the
extra information is too noisy to use with five test seasons.

## 6. Limitations stated in advance

Five test seasons and 160 team-seasons; the first test season has only two training seasons (2019, 2020, both unusual); C1 and M3 both use last season's results, so C1 is not independent of the history baselines;
C2 has high variance; the player values share full-data hyperparameters (as in the backtest).

## 7. Deviations log

*(any change to §2–§4 after results are seen is recorded here with its reason)*

No change to §2–§4. Outcomes, applied mechanically (2026-09-30): C1 z = 1.33 (sound → implemented), C2 z = 0.01 (less certain → not implemented), C3 z = 1.81 (sound → implemented); the combination
C1 + C3 is not worse than either (z = 0.80 against C1, 1.21 against C3) and is used. Implementation details for the 2026-27 forecast (not part of the backtest): C1's coefficient is refitted on all seasons 2019–2025 (0.84);
C3's roster-relative feature is computed over a dressed-size pre-selection (14 forwards and 8 defensemen by the baseline minutes model) because a published camp roster is larger than the backtest's opening lineups;
the 80% interval uses the combination's out-of-sample RMSE (13.96). Results: [report](../../reports/standings/candidates_narrative.md).
