# Phase 2 — Data ingestion

**Done.**
* `src/ingestion/nhl_api.py` scraped 9 regular seasons (2017-18 → 2025-26): 11,052 games, PBP + shift charts + rosters; 298 MB gzipped raw snapshot with a manifest. 0 missing PBP files; 0 missing shift files (though 57 shift charts are empty).
* `src/ingestion/parse.py` → `events.parquet` (3.43M events), `shifts.parquet` (8.39M shifts), `players.parquet`, `games.parquet`, `data/processed/data_quality.md`.
* `src/possession/onice.py` → on-ice skaters per event, TOI by strength per player-game, and a shift-quality table. 10,901 games usable (shift chart present and ≥90% of 5v5 events matched 5+5).

**Checks.**
* Attack-direction inference agrees with `homeTeamDefendingSide` in 99.97% of 27,433 game-periods.
* 97.5–97.9% of unblocked shots per season land in the attacking half after normalization.
* Blocked shots: the acting team is always the shooter's roster team; 99.8% fall in the shooter's attacking half.
* TOI sanity: team skater-minutes ≈ 595 per game; McDavid averaged 22.6 min/game in 2024-26.

**Findings that changed the design.**
1. `eventOwnerTeamId`/`zoneCode` semantics for blocked shots are unstable, so the shooter's team is taken from the roster.
2. **Scorekeeping regime break in 2024-25:** giveaways ~20–26k → 39–40k per season; takeaways ~18–19k → 12k; the NZ share of giveaways rose from 12% to 18%. `data.scoring_regimes` was added, V is cross-fitted within regime, and YOY is reported within and across the break.

**Next:** possession chains.
