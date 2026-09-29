"""Unshrunk component values above replacement (methodology §2).

Team-level scale checks must use quantities that no prior has shrunk, or shrinkage compression (independent-player priors
shrink effects shared across a team) is mistaken for scale error. Possession and turnover already carry unshrunk memo
columns in ``player_value_season``; this module adds finishing and faceoffs. DPM has no unshrunk analogue (a RAPM
coefficient needs regularization) and is used as published.
"""

from __future__ import annotations

import json

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


def pen_raw(proc) -> pl.DataFrame:
    """Penalty value without empirical-Bayes shrinkage: cost of a minor x (drawn - taken minor-equivalents above the fringe rates)."""
    cost = json.loads((load_config().path(load_config().data.metrics_dir) / "pen_component.json").read_text(encoding="utf-8"))["cost_minor"]
    p = pl.read_parquet(proc / "pen_player_season.parquet")
    return p.select("player_id", "season", (cost * ((pl.col("drawn_w") - pl.col("taken_w")) - (pl.col("rep_drawn") - pl.col("rep_taken")) * pl.col("toi_ev"))).alias("pen_raw_gar"))


def fin_raw_st(proc, cfg) -> pl.DataFrame:
    """Goals minus xG_f above the fringe pool's rate per attempt, by special-teams state (replacement pooled over seasons by position)."""
    from src.specialteams.finish import shot_class
    shots = pl.read_parquet(proc / "fin_shots.parquet", columns=["game_id", "event_idx", "season", "shooter", "pos_group", "goal", "strength_g", "own_skaters", "opp_skaters"])
    xg = pl.read_parquet(proc / "fin_xg.parquet").select("game_id", "event_idx", "xg_f")
    st = status(pl.read_parquet(proc / "player_season.parquet"), cfg.valuation.regular_forwards, cfg.valuation.regular_defense).select(pl.col("player_id").alias("shooter"), "season", "fringe")
    d = (shots.filter(pl.col("strength_g") != "5v5").join(xg, on=["game_id", "event_idx"], how="left").join(st, on=["shooter", "season"], how="left")
         .with_columns(pl.col("fringe").fill_null(True), shot_class(pl.col("own_skaters"), pl.col("opp_skaters")).alias("cls"), (pl.col("goal").cast(pl.Float64) - pl.col("xg_f")).alias("exc"))
         .filter(pl.col("cls") != "other"))
    rep = d.filter(pl.col("fringe")).group_by("pos_group", "cls").agg((pl.col("exc").sum() / pl.len()).alias("rep"))
    a = d.group_by("shooter", "season", "pos_group", "cls").agg(pl.col("exc").sum().alias("exc"), pl.len().alias("n")).join(rep, on=["pos_group", "cls"], how="left")
    a = a.with_columns((pl.col("exc") - pl.col("rep").fill_null(0.0) * pl.col("n")).alias("fin_raw_st")).rename({"shooter": "player_id"})
    return a.select(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64), "cls", "fin_raw_st")


def st_raw(proc, cfg) -> pl.DataFrame:
    """st_<cls>_raw_gar = published special-teams RAPM part + unshrunk finishing part (the RAPM part has no unshrunk analogue)."""
    from src.specialteams.run import build_table
    tab = build_table(cfg)
    fin = fin_raw_st(proc, cfg)
    out = tab.select("player_id", "season", *[f"st_{c}_rapm_gar" for c in ("pp", "pk", "4v4", "3v3")])
    for c in ("pp", "pk", "4v4", "3v3"):
        f = fin.filter(pl.col("cls") == c).select("player_id", "season", pl.col("fin_raw_st").alias(f"_f_{c}"))
        out = out.join(f, on=["player_id", "season"], how="full", coalesce=True)
    fills = [pl.col(f"st_{c}_rapm_gar").fill_null(0.0) for c in ("pp", "pk", "4v4", "3v3")] + [pl.col(f"_f_{c}").fill_null(0.0) for c in ("pp", "pk", "4v4", "3v3")]
    out = out.with_columns(*fills)
    return out.select("player_id", "season", *[(pl.col(f"st_{c}_rapm_gar") + pl.col(f"_f_{c}")).alias(f"st_{c}_raw_gar") for c in ("pp", "pk", "4v4", "3v3")])


def build(proc=None, all_strength: bool = False) -> pl.DataFrame:
    """player_value_season plus fin_raw_gar and fo_raw_gar (0 where a player has no attempts / draws); with ``all_strength``
    also pen_raw_gar and the four st_<cls>_raw_gar."""
    cfg = load_config()
    proc = proc or cfg.path(cfg.data.processed_dir)
    t = pl.read_parquet(proc / "player_value_season.parquet")
    t = (t.join(fin_raw(proc, cfg), on=["player_id", "season"], how="left").join(fo_raw(proc), on=["player_id", "season"], how="left")
         .with_columns(pl.col("fin_raw_gar").fill_null(0.0), pl.col("fo_raw_gar").fill_null(0.0)))
    if all_strength:
        extra = pen_raw(proc).join(st_raw(proc, cfg), on=["player_id", "season"], how="full", coalesce=True)
        t = t.join(extra.with_columns(pl.col("player_id").cast(t["player_id"].dtype), pl.col("season").cast(t["season"].dtype)), on=["player_id", "season"], how="left")
        t = t.with_columns([pl.col(c).fill_null(0.0) for c in extra.columns if c.endswith("_raw_gar")])
    return t
