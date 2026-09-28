# PM-EPA Definitions

Authoritative definitions of every term used in code and reports. Code references are given as `module:function`.

## 1. Data units

| Term | Definition |
|---|---|
| **Season** | Identified by its start year: `2023` = 2023-24. Regular season only (game type 2). |
| **Event** | One row of NHL play-by-play (`api-web.nhle.com/v1/gamecenter/{id}/play-by-play`), kept types: faceoff, hit, giveaway, takeaway, shot-on-goal, missed-shot, blocked-shot, goal, penalty, delayed-penalty, stoppage, period-start/end, game-end. Shootout events are dropped. |
| **Acting team** | The team credited with an event: faceoff winner, hitter, giver/taker, shooter (for blocked shots the shooter's team is looked up from the roster, because `eventOwnerTeamId` semantics for blocks are not stable across seasons), penalized team. |
| **Normalized coordinates** | Feet, rotated so the acting team attacks toward +x (opponent goal line x = +89, own goal line x = −89, blue lines ±25). y is rotated with x, so +y is always the attacker's left. Attack direction per game-period is inferred by a vote over zone-coded events and cross-checked against `homeTeamDefendingSide` (agreement 99.97% over 27,433 game-periods). |
| **Strength** | From one team's perspective, using the event's `situationCode`: `5v5`, `PP` (more skaters), `PK` (fewer), `4v4`, `3v3`, `EN` (either goalie pulled), `other`. **EV = 5v5 with both goalies in net.** Strength groups for value surfaces: 5v5, PP, PK, 4v4_3v3. |
| **Score diff** | Possessing-team goals minus opponent goals *before* the event. |
| **Scoring regime** | 2017-18 → 2023-24 and 2024-25 → 2025-26. From 2024-25 the NHL's recorded giveaways roughly doubled and takeaways fell ~35% (`data/processed/data_quality.md`); turnover-derived states are not comparable across the break. |
| **On-ice** | From official shift charts: skater on ice for an event at time t if `start < t ≤ end` (faceoffs: `start ≤ t < end`); goalies excluded (`src/possession/onice.py`). |
| **Usable game** | Shift chart present and ≥90% of 5v5 events matched with exactly 5+5 skaters (10,901 of 11,052 games). Only usable games enter per-60 rates, for numerator and denominator alike. |

## 2. Value units

| Term | Definition |
|---|---|
| **xG** | Probability that an unblocked shot attempt is a goal, from the in-house shot model (`src/xg/model.py`), cross-fitted by season. **Blocked attempts have xG = 0** (a blocked attempt cannot score). Goals are valued at their xG, never 1.0 (guardrail: no results leakage; see methodology §4.1 for the brief's conflicting instruction and our resolution). |
| **G_p (possession return)** | xG of possession p's terminal shot if it ended with an unblocked attempt; 0 otherwise. |
| **V(s)** | Expected possession return given the puck is *alive under the possessing team's control* in state s: `V(s) = E[G_p | s]`. |
| **Alive observation** | An observation where the puck is controlled and play continues: a possession *start* (faceoff win, takeaway, inferred recovery) or a *mid* observation (hit on the possessing team, only when `hits_as_observations`). V is trained only on alive observations. |
| **Decision point / V_dec(s)** | The location of a possession's terminal decision (shot, giveaway, lost puck). `V_dec(s) = E[G_p | terminal decision at s]` is estimated on terminal observations and values the "after" state of the last movement segment. |
| **V_att(s)** | Expected xG of a shot attempt from s (blocked = 0). It splits a shot's terminal value into *attempt* (V_att − V_dec, in PM-EPA) and *quality* (xG − V_att, excluded). |

## 3. Possession rules (`src/possession/chains.py`)

A **possession** is a maximal run of play during which one team is inferred to control the puck.

**Opens** when
1. a team wins a faceoff (start kind `faceoff`);
2. a team records a takeaway (`takeaway`); an open opponent possession is closed as `takeaway_against`, with a pseudo-observation `lost_takeaway` at the mirrored takeaway location;
3. a team records a terminal event with no open possession of its own. Its start is then **inferred** from the most recent loose-puck origin, if that origin is at most `max_inferred_start_gap_s` = 25 s earlier and not later than the event:

| origin (preceding event) | same team recovers | other team recovers | location used |
|---|---|---|---|
| missed shot | `loose_own_miss` | `loose_opp_miss` | coarse zone only (the miss location is not where the puck was recovered) |
| shot on goal (not frozen) | `loose_own_sog` | `loose_opp_sog` | coarse zone only |
| blocked shot | `loose_own_block` | `loose_opp_block` | block location |
| giveaway | `loose_own_giveaway` | `loose_opp_giveaway` | giveaway location |
| possession lost without a recorded event | — | `loose_unobserved` | none |
| same-team shot within `rebound_window_s` = 3 s *and* the new event is a shot | `rebound` | — | coarse zone only |

   With no qualifying origin, the start is `unobserved` and there is no start observation. The possession then earns no movement credit.

**Closes** when
* the possessing team shoots (`shot_sog`, `shot_miss`, `shot_block`, `goal`) or gives the puck away (`giveaway`);
* the opponent records a takeaway (`takeaway_against`);
* the opponent records any other possession event (`unobserved_loss`, e.g. an uncredited dump-in or failed pass);
* a stoppage, penalty or period end occurs (`stoppage:<reason>`, `penalty`, `period_end`);
* a faceoff occurs while it is open (`unrecorded_stoppage`, a data defect; value unattributed).

**Never possession events.** Hits (unless `hits_as_observations`, which adds a *mid* observation only when the hittee's team already has an open possession), delayed-penalty markers, and period starts. A goal never seeds a loose start, because a faceoff always follows.

**Observations.** Every possession stores its observations in time order: at most one start, then optional mid observations, then at most one terminal observation. Possessions have ≤2 observations without hits; with hits, 2.3% gain a third.

## 4. State space (`src/states/mapping.py`)

`state_id = strength_group | kind | spatial [| score bucket]`

* **kind:** `control` (takeaway, hit_mid, and arrival states), `faceoff`, or a loose kind from §3.
* **spatial**, by kind precision: a fine cell (`control`, `faceoff`, loose kinds from giveaways/blocks), a coarse zone OZ/NZ/DZ (loose kinds from misses/SOG/rebounds), or none (`loose_unobserved`).
* **Spatial schemes:** `zones` = 26 named sub-zones; `grid10` = 10 × 8.5 ft cells (200); `grid20` = 20 × 17 ft cells (50). Each maps to a parent (grid → zone → coarse) used for shrinkage.

| Sub-zone | Rule (attacking frame) |
|---|---|
| OZ_behind / DZ_behind | \|x\| > 89 |
| OZ_low_slot | 69 ≤ x ≤ 89, \|y\| ≤ 11 |
| OZ_high_slot | 54 ≤ x < 69, \|y\| ≤ 11 |
| OZ_circle_L/R | 54 ≤ x ≤ 89, 11 < \|y\| ≤ 30 |
| OZ_boards_L/R | 54 ≤ x ≤ 89, \|y\| > 30 |
| OZ_point_C/L/R | 25 ≤ x < 54 (C if \|y\| ≤ 11) |
| NZ_att_C/L/R, NZ_own_C/L/R | 0 ≤ x < 25 / −25 < x < 0; C if \|y\| ≤ 14 |
| DZ_* | mirror of OZ_* |

## 5. PM-EPA terms (`src/epa/engine.py`)

| Term | Definition |
|---|---|
| **Segment** | Consecutive observations (o_i, o_{i+1}) of one possession. Δ = V(o_{i+1}) − V(o_i). |
| **Observed carry** | Segment whose endpoints have the same known player within `carry_max_gap_s` = 12 s (never a faceoff start). Credit: 100% to that player. |
| **Split segment** | Any other segment with gap ≤ `max_segment_gap_s` = 30 s. It contains ≥1 unobserved pass/carry. Credit: `start_share` = 0.35 to the start player (0 if unknown or a faceoff winner), `end_share` = 0.35 to the end player (0 if unknown), and the remainder split equally across the possessing team's on-ice skaters at the end event. |
| **Long-gap segment** | Gap > 30 s. 100% to the on-ice pool. |
| **Rebound segment** | Starts at a `rebound` state. Unattributed (brief §11.1). |
| **Takeaway swing** | Taker: `V_T(s) + λ·V_O(s̄)`. Loser's on-ice skaters: `−(V_O(s̄) + λ·V_T(s))`, where s̄ = mirrored state. Zero-sum at λ = 1. |
| **Giveaway** | Giver: `−(V_T(s) + λ·V_O(loose_opp_giveaway at s̄))`. |
| **Unobserved loss / icing / offside** | On-ice pool at the end event: `−(V_T(s_last) + λ·V_O(loose_unobserved))`, or `−V_T(s_last)` for icing and offside. Other stoppages are unattributed. |
| **Shot attempt** | Shooter: `V_att(s) − V_dec(s)`: getting an attempt off at s rather than the average decision outcome there. In PM-EPA (direct). Mean-zero with giveaways at each spot. |
| **Shot quality** | Shooter: `xG − V_att(s)`: the release (type, block, rebound luck). Reported, **excluded** from PM-EPA (brief §4.2). |
| **λ (lambda_opponent)** | Weight on the opponent's possession value after a turnover. **Production: λ = 0**, selected by the pre-registered rule; λ = 0.5 and 1 (EP symmetry) are ablated. |
| **Direct PM-EPA** | carry + move_start + move_end + takeaway + giveaway + shot_attempt. |
| **Pooled PM-EPA** | move_pool + takeaway_against + unobserved_loss + stoppage_loss. |
| **PM-EPA** | Direct + pooled. |
| **PM-EPA/60 (EV)** | 60 × (5v5 PM-EPA) / (5v5 TOI minutes) over usable games. |
