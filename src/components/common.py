"""Shared pieces for the turnover, faceoff and penalty components."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import load_config
from src.valuation.replacement import status


def base(proc) -> pl.DataFrame:
    """Skater-seasons with the regular/fringe status, position group and 5v5 TOI (minutes)."""
    cfg = load_config()
    ps = pl.read_parquet(proc / "player_season.parquet")
    st = status(ps, cfg.valuation.regular_forwards, cfg.valuation.regular_defense).select(
        "player_id", "season", "fringe"
    )
    return (
        ps.select("player_id", "season", "pos_group", "toi_ev", "primary_team")
        .join(st, on=["player_id", "season"], how="left")
        .with_columns(pl.col("fringe").fill_null(True))
    )


def good_game_ids(proc) -> pl.DataFrame:
    return pl.read_parquet(proc / "shift_quality.parquet").filter(pl.col("onice_match_rate") >= 0.9).select("game_id")


def eb_rate_component(d: pl.DataFrame, min_toi: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Empirical-Bayes goals above replacement from a per-60 rate with a bootstrap SE.

    ``d``: player_id, season, pos_group, fringe, toi_ev (minutes), total (raw goals), se60 (bootstrap SE of the per-60
    rate).
    Prior mean = pooled raw rate of the player's status group (season x position); tau^2 from qualified regulars
    (method of moments). Replacement = pooled raw fringe rate. Returns (per player-season, replacement rates).
    """
    d = d.with_columns(pl.col("toi_ev").fill_null(0.0), pl.col("total").fill_null(0.0)).with_columns(
        pl.when(pl.col("toi_ev") > 0).then(pl.col("total") / pl.col("toi_ev") * 60).alias("rate")
    )
    grp = d.group_by("season", "pos_group", "fringe").agg(
        (pl.col("total").sum() / pl.col("toi_ev").sum() * 60).alias("m_status"),
        pl.col("toi_ev").sum().alias("pool_toi"),
    )
    q = d.filter(
        (~pl.col("fringe"))
        & (pl.col("toi_ev") >= min_toi)
        & pl.col("rate").is_not_null()
        & pl.col("se60").is_not_null()
    )
    tau = (
        q.group_by("season", "pos_group")
        .agg(pl.col("rate").var().alias("v_x"), (pl.col("se60") ** 2).mean().alias("v_se"))
        .with_columns((pl.col("v_x") - pl.col("v_se")).clip(1e-8, None).alias("tau2"))
    )
    rep = grp.filter(pl.col("fringe")).select("season", "pos_group", pl.col("m_status").alias("rep_per60"))
    d = (
        d.join(
            grp.select("season", "pos_group", "fringe", "m_status"), on=["season", "pos_group", "fringe"], how="left"
        )
        .join(tau.select("season", "pos_group", "tau2"), on=["season", "pos_group"], how="left")
        .join(rep, on=["season", "pos_group"], how="left")
    )
    se = pl.col("se60").fill_null(10.0).clip(1e-4, None)
    d = (
        d.with_columns(
            (pl.col("tau2") / (pl.col("tau2") + se**2)).alias("B"),
            pl.col("rate").fill_null(pl.col("m_status")).alias("x"),
        )
        .with_columns(
            (pl.col("m_status") + pl.col("B") * (pl.col("x") - pl.col("m_status"))).alias("post_per60"),
            (pl.col("B") * se**2).sqrt().alias("post_sd60"),
        )
        .with_columns(
            ((pl.col("post_per60") - pl.col("rep_per60")) * pl.col("toi_ev") / 60).alias("gar"),
            (pl.col("post_sd60") * pl.col("toi_ev") / 60).alias("gar_sd"),
        )
    )
    return d.select(
        "player_id", "season", "rate", "post_per60", "rep_per60", "gar", "gar_sd", pl.col("B").alias("shrink_B")
    ), rep


def season_centred_yoy(tab: pl.DataFrame, col: str, min_toi: float) -> dict:
    """Season-centred year-over-year r among qualified players, by position group."""
    t = tab.filter(pl.col("toi_ev") >= min_toi).with_columns(
        (pl.col(col) - pl.col(col).mean().over("season", "pos_group")).alias("_c")
    )
    nxt = t.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("_c").alias("_n"))
    j = t.join(nxt, on=["player_id", "season"])
    out = {"all": {"r": float(np.corrcoef(j["_c"], j["_n"])[0, 1]), "n": j.height}}
    for (g,), x in j.group_by(["pos_group"]):
        out[g] = {"r": float(np.corrcoef(x["_c"], x["_n"])[0, 1]), "n": x.height}
    return out


def paired_z(d: np.ndarray) -> dict:
    """Mean and z of a vector of paired differences (positive = first model better)."""
    m, s = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean": m, "se": s, "z": m / s if s > 0 else float("nan"), "n": int(len(d))}
