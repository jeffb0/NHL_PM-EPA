"""Bucket decomposition of the team residual and its carry with players (docs/residual/methodology.md §2-§3)."""

from __future__ import annotations

import numpy as np
import polars as pl

BUCKETS = ("play5", "fin5", "st", "gk", "other")
MODEL = {"play5": ("pm", "def", "to", "fo"), "fin5": ("fin",), "st": ("pp", "pk", "pen"), "other": ("4v4", "3v3")}


def observed(proc, res: pl.DataFrame) -> pl.DataFrame:
    """Per (season, team_id): the five observed parts of the goal differential (they sum to it exactly)."""
    sh = pl.read_parquet(
        proc / "fin_shots.parquet", columns=["season", "team_id", "opp_team_id", "strength", "goal", "xg_v1"]
    )
    f = sh.group_by("season", "team_id", "strength").agg(
        pl.col("goal").sum().alias("gf"), pl.col("xg_v1").sum().alias("xf")
    )
    a = sh.group_by("season", pl.col("opp_team_id").alias("team_id"), "strength").agg(
        pl.col("goal").sum().alias("ga"), pl.col("xg_v1").sum().alias("xa")
    )
    j = f.join(a, on=["season", "team_id", "strength"], how="full", coalesce=True).with_columns(
        pl.col("gf", "xf", "ga", "xa").fill_null(0.0)
    )
    s5, st = pl.col("strength") == "5v5", pl.col("strength").is_in(["PP", "PK"])
    o = j.group_by("season", "team_id").agg(
        (pl.col("xf") - pl.col("xa")).filter(s5).sum().alias("play5"),
        (pl.col("gf") - pl.col("xf")).filter(s5).sum().alias("fin5"),
        (pl.col("gf") - pl.col("xa")).filter(st).sum().alias("st"),
        (pl.col("xa") - pl.col("ga")).sum().alias("gk"),
        (pl.col("gf") - pl.col("xa")).filter(~s5 & ~st).sum().alias("other"),
        (pl.col("gf") - pl.col("ga")).sum().alias("_shots_gd"),
    )
    o = res.select("season", "team_id", "team", "gd").join(
        o.with_columns(pl.col("team_id").cast(res["team_id"].dtype)), on=["season", "team_id"], how="left"
    )
    # empty-net goals: the remainder
    return o.with_columns((pl.col("other") + pl.col("gd") - pl.col("_shots_gd")).alias("other")).drop("_shots_gd")


def explained(proc, p: pl.DataFrame, res: pl.DataFrame) -> pl.DataFrame:
    """Per (season, team): the component sums meant to explain each bucket, and the skater minutes."""
    sk = p.group_by("season", "team").agg(
        *[
            sum((pl.col(f"r_{c}") * pl.col("m") / 60).sum() for c in comps).alias(f"m_{b}")
            for b, comps in MODEL.items()
        ],
        pl.col("m").sum().alias("M"),
    )
    gs = (
        pl.read_parquet(proc / "gk_shots.parquet", columns=["season", "team_id", "goal", "xg_f"])
        .group_by("season", "team_id")
        .agg((pl.col("xg_f") - pl.col("goal")).sum().alias("m_gk"))
    )
    return (
        res.select("season", "team_id", "team")
        .join(sk, on=["season", "team"], how="left")
        .join(gs.with_columns(pl.col("team_id").cast(res["team_id"].dtype)), on=["season", "team_id"], how="left")
    )


def table(proc, p: pl.DataFrame, res: pl.DataFrame) -> pl.DataFrame:
    """Per team-season: observed, explained (both centred within season) and residual for each bucket, their sum R, and
    the residual per skater-minute."""
    t = observed(proc, res).join(explained(proc, p, res).drop("team"), on=["season", "team_id"])
    for b in BUCKETS:
        t = t.with_columns(
            (pl.col(b) - pl.col(b).mean().over("season")).alias(f"o_{b}"),
            (pl.col(f"m_{b}") - pl.col(f"m_{b}").mean().over("season")).alias(f"e_{b}"),
        )
        t = t.with_columns((pl.col(f"o_{b}") - pl.col(f"e_{b}")).alias(f"R_{b}")).with_columns(
            (pl.col(f"R_{b}") / pl.col("M")).alias(f"rho_{b}")
        )
    return t.with_columns(pl.sum_horizontal([f"R_{b}" for b in BUCKETS]).alias("R")).with_columns(
        (pl.col("R") / pl.col("M")).alias("rho")
    )


def carried(p: pl.DataFrame, t: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Per (season s, team): for each rho column, the sum over the team's season-s skaters of (that rate for the
    player's season s-1 team) x his season-s minutes;
    also split by forwards and defensemen for the lump (c_F, c_D)."""
    prev = p.select("player_id", (pl.col("season") + 1).alias("season"), pl.col("team").alias("team_prev"))
    rho = t.select((pl.col("season") + 1).alias("season"), pl.col("team").alias("team_prev"), *cols)
    j = (
        p.select("player_id", "season", "team", "pos_group", "m")
        .join(prev, on=["player_id", "season"], how="left")
        .join(rho, on=["season", "team_prev"], how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in cols])
    )
    return j.group_by("season", "team").agg(
        *[(pl.col(c) * pl.col("m")).sum().alias(f"c_{c}") for c in cols],
        (pl.col("rho") * pl.col("m")).filter(pl.col("pos_group") == "F").sum().alias("c_F"),
        (pl.col("rho") * pl.col("m")).filter(pl.col("pos_group") == "D").sum().alias("c_D"),
    )


def ols0(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares through the origin with classical standard errors."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    cov = np.linalg.pinv(X.T @ X) * float(r @ r) / max(len(y) - X.shape[1], 1)
    return b, np.sqrt(np.diag(cov))
