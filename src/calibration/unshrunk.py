"""Unshrunk component values above replacement (methodology §2).

Team-level scale checks must use quantities that no prior has shrunk, or shrinkage compression (independent-player priors
shrink effects shared across a team) is mistaken for scale error. Possession and turnover already carry unshrunk memo
columns in ``player_value_season``; this module adds finishing and faceoffs. DPM has no unshrunk analogue (a RAPM
coefficient needs regularization) and is used as published.
"""

from __future__ import annotations

import polars as pl

from src.components import faceoff
from src.config import load_config
from src.valuation.replacement import status


def fin_raw(proc, cfg) -> pl.DataFrame:
    """Goals minus xG_f at 5v5 above the fringe pool's rate per attempt (by season and position)."""
    ps = pl.read_parquet(proc / "player_season.parquet")
    st = status(ps, cfg.valuation.regular_forwards, cfg.valuation.regular_defense).select("player_id", "season", "fringe")
    f = (pl.read_parquet(proc / "fin_player_season.parquet").select("player_id", "season", "pos_group", "attempts_5v5", "goals_5v5", "xg_f_5v5")
         .join(st, on=["player_id", "season"], how="left").with_columns(pl.col("fringe").fill_null(True), (pl.col("goals_5v5") - pl.col("xg_f_5v5")).alias("exc")))
    rep = f.filter(pl.col("fringe")).group_by("season", "pos_group").agg((pl.col("exc").sum() / pl.col("attempts_5v5").sum()).alias("rep_per_attempt"))
    return f.join(rep, on=["season", "pos_group"], how="left").with_columns(
        (pl.col("exc") - pl.col("rep_per_attempt").fill_null(0.0) * pl.col("attempts_5v5")).alias("fin_raw_gar")).select("player_id", "season", "fin_raw_gar")


def fo_raw(proc) -> pl.DataFrame:
    """Sum of swing x (win - league win probability at that dot) above the fringe pool's rate per draw."""
    d = faceoff.draws(proc)
    p0 = d.group_by("dot").agg(pl.col("y").mean().alias("p0"))
    d = d.join(p0, on="dot")
    rows = pl.concat([
        d.select(pl.col("home_taker").alias("player_id"), "season", (pl.col("S") * (pl.col("y") - pl.col("p0"))).alias("c")),
        d.select(pl.col("away_taker").alias("player_id"), "season", (-pl.col("S") * (pl.col("y") - pl.col("p0"))).alias("c")),
    ])
    a = rows.group_by("player_id", "season").agg(pl.col("c").sum().alias("csum"), pl.len().alias("draws"))
    meta = faceoff.taker_meta(proc).select("player_id", "season", "fringe")
    a = a.join(meta, on=["player_id", "season"], how="left").with_columns(pl.col("fringe").fill_null(True))
    rep = a.filter(pl.col("fringe")).group_by("season").agg((pl.col("csum").sum() / pl.col("draws").sum()).alias("rep_per_draw"))
    return a.join(rep, on="season", how="left").with_columns((pl.col("csum") - pl.col("rep_per_draw").fill_null(0.0) * pl.col("draws")).alias("fo_raw_gar")).select("player_id", "season", "fo_raw_gar")


def build(proc=None) -> pl.DataFrame:
    """player_value_season plus fin_raw_gar and fo_raw_gar (0 where a player has no attempts / draws)."""
    cfg = load_config()
    proc = proc or cfg.path(cfg.data.processed_dir)
    t = pl.read_parquet(proc / "player_value_season.parquet")
    return (t.join(fin_raw(proc, cfg), on=["player_id", "season"], how="left").join(fo_raw(proc), on=["player_id", "season"], how="left")
            .with_columns(pl.col("fin_raw_gar").fill_null(0.0), pl.col("fo_raw_gar").fill_null(0.0)))
