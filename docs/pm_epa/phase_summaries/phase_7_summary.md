# Phase 7 — Full dataset computation

**Done.** Full 9-season pipeline with the production configuration: V_start = GBM + score, V_dec / V_att = GBM + score, λ = 0, shares 0.35/0.35/0.30, cross-fitted within scoring regime.

| Output | Size |
|---|---|
| possessions (non-excluded) | 1,876,659 |
| movement segments (`transitions.parquet`) | 1,166,827 |
| credit rows (`credits.parquet`, pool expanded) | 13,159,783 |
| player-seasons | 8,321 (5,034 qualified at ≥ 500 5v5 min) |
| team-seasons / team-games | `pm_epa_team_season.csv` / `pm_epa_team_game.csv` |

**Sanity checks.**
* **Accounting identity (λ = 0):** max |Σ components − G_p| = 1.1e-16 over all possessions.
* **League level:** mean qualified PM-EPA/60 (5v5) is +0.06 to +0.10 for forwards and −0.02 to +0.02 for defense per season. It is near zero by construction: movement and decision terms are mean-zero, and forwards are positive because they take most shot attempts and forecheck takeaways.
* **Face validity (brief §12.4.15).**
  * Defense leaders are the league's elite puck-movers: Makar is #1 in 2023-24, 2024-25 and 2025-26; also Q. Hughes, Fox, Bouchard, Toews, Werenski, Heiskanen, Theodore, Karlsson, Josi.
  * Forward leaders combine elite play-drivers (Matthews, McDavid, J. Hughes, Crosby, Point, Hischier, Guentzel) with high-motor forechecking and net-front forwards (Hyman, Hagel, Carrier, B. Tkachuk).
  * Career bottoms are recognizably poor transition players (Yandle, Benn, Gudas at D; Hoffman, Laine at F).
* **Face-validity caveat.** The 2025-26 forward list tilts toward forecheckers (Carrier, Acciari, Bowman), plausibly from the new turnover-recording regime (more recorded takeaway-like events for pressure). Pass-first playmakers (e.g., Draisaitl, Kucherov) rank lower than their reputations because their value lives in unobserved passes.

**Process note.** A programmatic config edit (λ 1 → 0) silently failed, and the first "final" run used λ = 1. It was caught because its leaderboard contradicted the λ = 0 spot check; the run was discarded and repeated. Config edits are now asserted, and `run_log.json` records the effective config.

**Next:** validation (hard gate).
