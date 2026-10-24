"""Finishing above the fringe pool by special-teams state (methodology §4).

Uses the saved shot-level shifts of the finishing model (``fin_shot_shifts.parquet``, rows in the order of
``fin_shots.parquet``): goals above replacement = sum over a player's unblocked attempts of
expit(l + T) - expit(l + T_ref), with T_ref the fringe pool's mean shift over 5v5 shots of the same position and
season, exactly as in ``src.finishing.model.player_effects``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import expit

from src.valuation.replacement import status


def shot_class(own: pl.Expr, opp: pl.Expr) -> pl.Expr:
    return (
        pl.when(own > opp)
        .then(pl.lit("pp"))
        .when(own < opp)
        .then(pl.lit("pk"))
        .when((own == 4) & (opp == 4))
        .then(pl.lit("4v4"))
        .when((own == 3) & (opp == 3))
        .then(pl.lit("3v3"))
        .otherwise(pl.lit("other"))
    )


def class_gar(d: pl.DataFrame) -> pl.DataFrame:
    """Finishing goals above the fringe pool by special-teams state from shot-level shifts.

    ``d``: one row per shot with shooter, season, pos_group, strength_g, own_skaters, opp_skaters, l, T, fringe. The
    reference shift is
    the fringe pool's mean over the same season's 5v5 shots by position (as in ``src.finishing.model.player_effects``);
    the function
    uses only the rows it is given, so a rolling fold passes only its own season.
    """
    d5 = d.filter(pl.col("strength_g") == "5v5")
    avg = {s: v for s, v in d5.group_by("season").agg(pl.col("T").mean()).iter_rows()}
    rep = {
        (g, s): v
        for g, s, v in d5.filter(pl.col("fringe")).group_by("pos_group", "season").agg(pl.col("T").mean()).iter_rows()
    }
    ref = np.array([rep.get((g, s), avg.get(s, 0.0)) for g, s in zip(d["pos_group"].to_list(), d["season"].to_list())])
    g_rep = expit(d["l"].to_numpy() + d["T"].to_numpy()) - expit(d["l"].to_numpy() + ref)
    d = d.with_columns(
        pl.Series("g", g_rep), shot_class(pl.col("own_skaters"), pl.col("opp_skaters")).alias("cls")
    ).filter(pl.col("strength_g") != "5v5")
    return (
        d.group_by("shooter", "season", "cls")
        .agg(pl.col("g").sum().alias("fin_gar_st"), pl.len().alias("attempts_st"))
        .rename({"shooter": "player_id"})
        .with_columns(pl.col("season").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    )


def by_class(proc, n_f: int, n_d: int) -> tuple[pl.DataFrame, dict]:
    """(player_id, season, cls, fin_gar_st, attempts_st) over non-5v5 attempts, plus consistency checks (gate ST-6)."""
    shots = pl.read_parquet(
        proc / "fin_shots.parquet",
        columns=["shooter", "season", "pos_group", "strength_g", "own_skaters", "opp_skaters"],
    )
    sh = pl.read_parquet(proc / "fin_shot_shifts.parquet")
    assert shots.height == sh.height and all(
        (shots[c] == sh[c]).all() for c in ("shooter", "season", "pos_group", "strength_g")
    ), "fin_shots and fin_shot_shifts are misaligned"
    st = status(pl.read_parquet(proc / "player_season.parquet"), n_f, n_d).select(
        pl.col("player_id").alias("shooter"), "season", "fringe"
    )
    d = (
        shots.select("own_skaters", "opp_skaters")
        .hstack(sh)
        .join(st, on=["shooter", "season"], how="left")
        .with_columns(pl.col("fringe").fill_null(True))
    )
    assert d.height == sh.height
    out = class_gar(d)
    fin = pl.read_parquet(proc / "fin_player_season.parquet").select("player_id", "season", "fin_gar_other")
    tot = (
        out.group_by("player_id", "season")
        .agg(pl.col("fin_gar_st").sum())
        .join(
            fin.with_columns(pl.col("season").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)),
            on=["player_id", "season"],
            how="full",
            coalesce=True,
        )
        .fill_null(0.0)
    )
    chk = {
        "max_abs_diff_vs_fin_gar_other": float((tot["fin_gar_st"] - tot["fin_gar_other"]).abs().max()),
        "share_of_abs_gar_in_other_class": float(
            out.filter(pl.col("cls") == "other")["fin_gar_st"].abs().sum() / out["fin_gar_st"].abs().sum()
        ),
        "attempts_by_class": {k: int(v) for k, v in out.group_by("cls").agg(pl.col("attempts_st").sum()).iter_rows()},
    }
    return out.filter(pl.col("cls") != "other"), chk
