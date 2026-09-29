# Team context ("system / chemistry") — methodology and pre-registration

*Written 2026-09-29 before any team-context analysis was run. The gates in §4 are fixed here; changes after results are seen are
logged in §7. Scope is deliberately bounded (model owner's request): investigate, quantify, and add what helps; do not try to separate
chemistry from everything else the model already captures.*

## 1. Question

The projection layer and the whole-model validation suggest that a player's value estimate carries the context of the team he plays for
(system, coaching, linemates, arena), so last season's roster sum forecasts a team better than talent-smoothed history. This work asks:

1. **How much of a player's value travels?** Compare persistence of value estimates for players who change primary team between
   consecutive seasons (movers) with players who stay (stayers).
2. **Does the destination team's context predict a mover's next-season value beyond his own history?** A positive destination-context
   effect (and a negative origin-context effect) for movers is the signature of a system/chemistry component that the individual
   estimates do not remove.
3. **If so, does adding it improve projections**, at the player level and in the team-level forecast?

Data limits stated up front: the repository has **no coach data**, so "Coach X versus Coach A" cannot be measured directly. The team-season
is the unit of context; coaching, system, linemate and arena effects are bundled together and cannot be separated here. Coach data
(the NHL API game pages list head coaches) would be a new ingestion.

## 2. Definitions

* **Mover:** a skater whose primary team in season T differs from his primary team in T − 1 (both seasons with ≥ 300 modeled minutes for the
  persistence comparison). Traded-mid-season players are assigned to their primary team only (a limitation).
* **Value rate:** the projection layer's component and total rates per 60 modeled minutes (finishing and faceoffs are rolling-refit values).
* **Context of a team in season s, for player i** (leave-one-out): the minute-weighted mean total value rate, at season s, of the
  team's other skaters. `ctx_origin` = the origin team's context in T − 1; `ctx_dest` = the destination team's context in T − 1 (both known when
  a player signs); for stayers they coincide. A second, results-based context `ctx_gd` = the destination team's goal differential per game in T − 1.

## 3. Analyses

* **A. Travel of value:** for each of total, possession, defense, finishing, turnover, penalties, power play, penalty kill: correlation and
  minute-weighted slope of next-season rate on this season's rate, for stayers and movers separately (season-centred by position; player-cluster
  bootstrap intervals). The ratio of the mover to the stayer slope is the share of persistence that travels.
* **B. Context regression:** next-season total rate on the shrunk multi-season history feature, age terms, `ctx_dest` and `ctx_origin`
  (WLS by next-season minutes, player-cluster SEs), for movers, and for all players with a single `ctx` for stayers.
* **C. Predictive test:** the projection ridge (docs/projection) with `ctx_dest`, `ctx_origin`, a mover indicator, and (variant) `ctx_gd` as extra
  features, evaluated exactly as in the projection layer (expanding window, test seasons 2020–2025, players with ≥ 300 target minutes,
  player-cluster bootstrap z), overall and for movers; and the team-level M protocol with the roster-weighted sum of these projections.

## 4. Gates (failures are reported, not tuned away)

| gate | rule |
|---|---|
| TC-1 | value travels less than it stays: the mover slope on the total rate is below the stayer slope (difference with a player-cluster bootstrap interval excluding 0) |
| TC-2 | for movers, the `ctx_dest` coefficient in analysis B is positive with a cluster-robust t > 2 |
| TC-3 | adding context features improves the player-level projection for movers (paired z > 2) and does not hurt overall (z > −2) |
| TC-4 | team level: projections with context features beat those without (paired z > 2) |

Reported, not gated: the share of persistence that travels by component; the `ctx_origin` coefficient; the results-based context variant; mover
counts by season.

## 5. Interpretation rules

A positive, significant destination effect with gains in TC-3 supports a team-context component worth adding to forecasts. A null result means
either that context is already removed by the individual estimates (the ridge and stint models adjust for teammates) or that team-season is too
crude a unit (coaching and system change within a team across seasons). Neither would show that chemistry does not exist. This work cannot
separate system from teammates from arena; those remain bundled.

## 6. Limitations stated in advance

* Team-season context is a coarse proxy: coaches change, systems change within a team, lines are re-formed.
* Movers are a selected group (contracts, trades, age); their persistence is not the persistence of a random player.
* Primary-team assignment misplaces traded players; many movers are near replacement level and thinly observed.
* Small samples: roughly a fifth of players move in a season, and only movers with ≥ 300 minutes in both seasons enter A.

## 7. Deviations log

*(any change to §2–§4 after results are seen is recorded here with its reason)*

1. **Two additions after the main results (not pre-registered, no gate).** Analysis A was repeated on the chain-free defense and special-teams components
   (docs/chainfree) because the chained prior carries a player's own coefficient across teams and inflates how much of the published defensive
   value travels; analysis B was split by position for movers. Both are descriptive.
2. **A usage tool, `src/teamcontext/move.py`,** projects a player under a chosen destination team from the context-feature ridge. It is a convenience,
   not a validated forecast of any individual move.
