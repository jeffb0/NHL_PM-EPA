"""Team-context features (docs/teamcontext/methodology.md §2)."""

from __future__ import annotations

import polars as pl


def add_context(p: pl.DataFrame, gd_pg: pl.DataFrame | None = None) -> pl.DataFrame:
    """Add to the projection panel, for each skater-season (player i, season T, team k):

    * ``ctx_dest``   minute-weighted mean total value rate of the other skaters on team k in season T - 1 (leave-one-out);
    * ``ctx_origin`` the same for the team the player was on in T - 1 (equals ``ctx_dest`` for stayers and for players with no T - 1 season);
    * ``mover``      1 if the player's primary team changed between T - 1 and T, else 0 (0 when there is no T - 1 season);
    * ``ctx_gd``     goal differential per game of team k in T - 1 (from ``gd_pg``: season, team, gd_pg), 0 when unknown.

    Only season T - 1 information enters, so the features are known when a player signs.
    """
    tot = p.group_by("season", "team").agg((pl.col("m") * pl.col("r_total")).sum().alias("S"), pl.col("m").sum().alias("M"))
    dest = tot.select((pl.col("season") + 1).alias("season"), "team", pl.col("S").alias("S_d"), pl.col("M").alias("M_d"))
    origin = tot.select((pl.col("season") + 1).alias("season"), pl.col("team").alias("team_prev"), pl.col("S").alias("S_o"), pl.col("M").alias("M_o"))
    prev = p.select("player_id", (pl.col("season") + 1).alias("season"), pl.col("team").alias("team_prev"), pl.col("m").alias("m_prev"), pl.col("r_total").alias("r_prev"))
    x = p.join(dest, on=["season", "team"], how="left").join(prev, on=["player_id", "season"], how="left").join(origin, on=["season", "team_prev"], how="left")
    own_s, own_m = pl.col("m_prev") * pl.col("r_prev"), pl.col("m_prev")
    same = pl.col("team_prev") == pl.col("team")
    loo_dest_s = pl.col("S_d") - pl.when(same).then(own_s).otherwise(0.0)
    loo_dest_m = pl.col("M_d") - pl.when(same).then(own_m).otherwise(0.0)
    ctx_dest = pl.when(loo_dest_m > 0).then(loo_dest_s / loo_dest_m).otherwise(None)
    ctx_origin = pl.when(pl.col("team_prev").is_not_null() & ((pl.col("M_o") - own_m) > 0)).then((pl.col("S_o") - own_s) / (pl.col("M_o") - own_m)).otherwise(None)
    x = x.with_columns(ctx_dest.alias("ctx_dest"), ctx_origin.alias("ctx_origin"), (pl.col("team_prev").is_not_null() & ~same).cast(pl.Float64).alias("mover"))
    mean_ctx = float(x["ctx_dest"].drop_nulls().mean())
    x = x.with_columns(pl.col("ctx_dest").fill_null(mean_ctx), pl.col("ctx_origin").fill_null(pl.col("ctx_dest")).fill_null(mean_ctx))
    if gd_pg is not None:
        g = gd_pg.select((pl.col("season") + 1).alias("season"), "team", pl.col("gd_pg").alias("ctx_gd"))
        x = x.join(g.with_columns(pl.col("season").cast(x["season"].dtype)), on=["season", "team"], how="left").with_columns(pl.col("ctx_gd").fill_null(0.0))
    else:
        x = x.with_columns(pl.lit(0.0).alias("ctx_gd"))
    return x.drop("S_d", "M_d", "S_o", "M_o", "m_prev", "r_prev", "team_prev")
