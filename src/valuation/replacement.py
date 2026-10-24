"""Regular / fringe status per player-season (methodology §9), applied identically to every component.

Per season and team (``primary_team``), forwards and defensemen are ranked separately by all-strength
TOI. The top ``n_f`` forwards and ``n_d`` defensemen are regulars; everyone else, including players
with very few games, is fringe. The fringe pool is the replacement population.
"""

from __future__ import annotations

import polars as pl


def status(player_season: pl.DataFrame, n_f: int = 12, n_d: int = 6) -> pl.DataFrame:
    """player_id, season, pos_group, fringe (bool), team_rank."""
    limit = pl.when(pl.col("pos_group") == "D").then(pl.lit(n_d)).otherwise(pl.lit(n_f))
    return (
        player_season.select("player_id", "season", "pos_group", "primary_team", "toi_total")
        .with_columns(
            pl.col("toi_total")
            .rank("ordinal", descending=True)
            .over("season", "primary_team", "pos_group")
            .alias("team_rank")
        )
        .with_columns((pl.col("team_rank") > limit).alias("fringe"))
        .select("player_id", "season", "pos_group", "fringe", "team_rank")
    )
