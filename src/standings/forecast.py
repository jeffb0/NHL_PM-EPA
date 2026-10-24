"""Bottom-up team forecast from an opening-night roster (docs/standings/methodology.md §3-§4)."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.projection.model import Spec
from src.standings import data as sd

log = logging.getLogger("standings")
KEEP = {"F": 13, "D": 7}
NEW_GOALIE_SHOTS = 600.0


def roster_frame(
    p: pl.DataFrame, bio: pl.DataFrame, roster: pl.DataFrame, T: int, extra: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Design rows for rostered skaters (player_id, team_id, pos_group) with history features from seasons <= T - 1 and
    age at T."""
    h = pd_.history(p).filter(pl.col("t") == T - 1).drop("t")
    d = roster.join(h, on="player_id", how="left").join(bio, on="player_id", how="left")
    d = d.with_columns(
        pl.col("m_eff").is_null().cast(pl.Float64).alias("no_history"),
        pd_.age_at(pl.lit(T), pl.col("birth_date")).alias("age"),
        pl.lit(T, pl.Int64).alias("season"),
    )
    d = d.with_columns(
        [
            pl.col(c).fill_null(0.0)
            for c in d.columns
            if c.startswith(("rt_", "rbar_")) or c in ("m_eff", "m_last", "m_mean3")
        ]
    )
    if extra is not None:
        d = d.join(extra, on="player_id", how="left")
    for c in ["height_in", "weight_lb", "age"] + (list(extra.columns[1:]) if extra is not None else []):
        med = d.group_by("pos_group").agg(pl.col(c).median().alias("_med"))
        d = (
            d.join(med, on="pos_group", how="left")
            .with_columns(pl.col(c).fill_null(pl.col("_med")).fill_null(pl.col(c).median()))
            .drop("_med")
        )
    return d


def allocate_minutes(d: pl.DataFrame, m_hat: np.ndarray, target: dict[str, float]) -> pl.DataFrame:
    """Keep each team's top 13 forwards and 7 defensemen by projected minutes; rescale to the league-mean team totals by
    position."""
    d = d.with_columns(pl.Series("m_hat", np.maximum(m_hat, 1.0)))
    d = (
        d.with_columns(pl.col("m_hat").rank("ordinal", descending=True).over("team_id", "pos_group").alias("_rk"))
        .filter(pl.col("_rk") <= pl.col("pos_group").replace_strict(KEEP, return_dtype=pl.Int64))
        .drop("_rk")
    )
    tot = d.group_by("team_id", "pos_group").agg(pl.col("m_hat").sum().alias("_tot"))
    return (
        d.join(tot, on=["team_id", "pos_group"])
        .with_columns(
            (
                pl.col("m_hat") / pl.col("_tot") * pl.col("pos_group").replace_strict(target, return_dtype=pl.Float64)
            ).alias("mins")
        )
        .drop("_tot")
    )


def team_minutes(p: pl.DataFrame, res: pl.DataFrame, season: int) -> dict[str, float]:
    """League-mean team forward and defense modeled minutes per 82 games in ``season``."""
    tg = float(res.filter(pl.col("season") == season)["gp"].sum())
    return {
        g: float(p.filter((pl.col("season") == season) & (pl.col("pos_group") == g))["m"].sum()) / tg * 82.0
        for g in ("F", "D")
    }


def goalie_value(
    gk: pl.DataFrame, T: int, rates: pl.DataFrame, per: pl.DataFrame, lg: pl.DataFrame, max_goalies: int | None = None
) -> pl.DataFrame:
    """Team goalie value (goals per 82 against an average goalie): expected shot share x shots x rolling persistent save
    rate, goalies on the opening roster."""
    g = gk.filter(pl.col("season") == T).select("team_id", "goalie")
    s1 = per.filter(pl.col("season") == T - 1).select("goalie", pl.col("shots").alias("s1"))
    s2 = per.filter(pl.col("season") == T - 2).select("goalie", pl.col("shots").alias("s2"))
    r = rates.filter(pl.col("season") == T)
    rep = float(r["rep_v"][0]) if r.height else 0.0
    g = (
        g.join(s1, on="goalie", how="left")
        .join(s2, on="goalie", how="left")
        .join(r.select(pl.col("goalie").cast(pl.Int64), "rate_v"), on="goalie", how="left")
        .with_columns(
            (pl.col("s1").fill_null(0.0) + 0.5 * pl.col("s2").fill_null(0.0)).alias("w"),
            pl.col("rate_v").fill_null(rep).alias("rate"),
        )
    )
    g = g.with_columns(pl.when(pl.col("w") <= 0).then(NEW_GOALIE_SHOTS).otherwise(pl.col("w")).alias("w"))
    # a published camp roster lists prospects and AHL goalies; the backtest's opening lineups had the two who dressed
    if max_goalies:
        g = g.filter(pl.col("w").rank("ordinal", descending=True).over("team_id") <= max_goalies)
    g = g.with_columns((pl.col("w") / pl.col("w").sum().over("team_id")).alias("share"))
    shots = float(lg.filter(pl.col("season") == T - 1)["shots82"].item())
    return g.group_by("team_id").agg((pl.col("share") * shots * pl.col("rate")).sum().alias("G"))


def _skater_sum(d: pl.DataFrame, rate: np.ndarray, name: str) -> pl.DataFrame:
    return (
        d.with_columns(pl.Series("_r", rate))
        .group_by("team_id")
        .agg((pl.col("_r") * pl.col("mins") / 60.0).sum().alias(name))
    )


def build(proc, cfg, seasons: list[int]) -> pl.DataFrame:
    """One row per (season T, team): S (projected skaters), S_R (last-season rates), S_res (results-target projection),
    G, and the oracle-roster S_oracle and G_oracle."""
    from src.wholemodel import build as wb

    roll = pl.read_parquet(proc / "wm_rolling_values.parquet")
    p = pd_.panel(proc, roll)
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    res = sd.results(proc)
    sk, gk = sd.openers(proc)
    per, lg = sd.goalie_usage(proc)
    rates = pl.read_parquet(proc / "gk_rolling_rates.parquet")
    M = Spec()
    # minutes are modeled conditional on playing >= 200 minutes
    fr_min = pd_.design_frame(p, bio, 1)
    fr_val = pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR)
    fr_all = pd_.design_frame(p, bio, 1, targets_only=False)
    p2 = p.join(pr.results_outcome(proc), on=["player_id", "season"], how="left").with_columns(
        pl.col("y_res").fill_null(0.0),
        pl.col("sh_pp").fill_null(0.0),
        pl.col("sh_pk").fill_null(0.0),
        pl.col("sh_o").fill_null(0.0),
    )
    Sres = Spec(extra=("sh_pp", "sh_pk", "sh_o"))
    fr_res = pd_.design_frame(p2, bio, 1, floor=pr.REVISED_FLOOR)
    gr = (
        wb.goalie_roll(proc)
        .join(res.select("season", "team", "team_id"), on=["season", "team"])
        .select("season", "team_id", pl.col("G_roll").alias("G_oracle"))
    )
    out = []
    for T in seasons:
        ros = sk.filter(pl.col("season") == T).select("team_id", "player_id", "pos_group")
        d = roster_frame(p, bio, ros, T)
        models = pm.fit_models(pr.train_rows(fr_val, 1, T), M)
        mm = pr._minutes_fit(pr.train_rows(fr_min, 1, T))
        m_hat = pr._minutes_pred(mm, d)
        alloc = allocate_minutes(d, m_hat, team_minutes(p, res, T - 1))
        keep = alloc.select("team_id", "player_id")
        rate_k = pm.predict_models(models, alloc, M)
        S = _skater_sum(alloc, rate_k, "S")
        # R: last season's rolling rate (players with no history get the position mean of the season)
        last = p.filter(pl.col("season") == T - 1).select("player_id", pl.col("r_total").alias("r_last"))
        posmean = (
            p.filter(pl.col("season") == T - 1)
            .group_by("pos_group")
            .agg((pl.col("r_total") * pl.col("m")).sum().alias("a"), pl.col("m").sum().alias("b"))
            .select("pos_group", (pl.col("a") / pl.col("b")).alias("pm_"))
        )
        r_last = (
            alloc.join(last, on="player_id", how="left")
            .join(posmean, on="pos_group", how="left")
            .with_columns(pl.coalesce("r_last", "pm_").alias("rr"))["rr"]
            .to_numpy()
        )
        S_R = _skater_sum(alloc, r_last, "S_R")
        # P-res: projection of next-season results with last season's state shares as the usage controls
        sh = p2.filter(pl.col("season") == T - 1).select("player_id", "sh_pp", "sh_pk", "sh_o")
        dres = roster_frame(p2, bio, ros, T, extra=sh)
        mres = pm.fit_models(pr.train_rows(fr_res, 1, T), Sres, target="y_res")
        ares = alloc.join(dres.select("player_id", "sh_pp", "sh_pk", "sh_o"), on="player_id", how="left", suffix="_x")
        S_res = _skater_sum(
            ares,
            pm.predict_models(
                mres, ares.with_columns(pl.col(c).fill_null(0.0) for c in ("sh_pp", "sh_pk", "sh_o")), Sres
            ),
            "S_res",
        )
        G = goalie_value(gk, T, rates, per, lg)
        # oracle roster: the players who actually played in T, with actual minutes (projection-layer forecast)
        f = pr.forecast(fr_all, fr_val, 1, T, M, p)
        ab = res.filter(pl.col("season") == T).select("team", "team_id")
        S_or = (
            f.with_columns((pl.col("proj") / 60.0 * pl.col("m")).alias("s"))
            .group_by("team")
            .agg(pl.col("s").sum().alias("S_oracle"))
            .join(ab, on="team")
            .drop("team")
        )
        t = (
            res.filter(pl.col("season") == T)
            .select("season", "team_id", "team")
            .join(S, on="team_id", how="left")
            .join(S_R, on="team_id", how="left")
            .join(S_res, on="team_id", how="left")
            .join(G, on="team_id", how="left")
            .join(S_or, on="team_id", how="left")
            .join(gr.filter(pl.col("season") == T).drop("season"), on="team_id", how="left")
        )
        out.append(t.with_columns(pl.col("team_id").cast(pl.Int64), pl.lit(len(keep)).alias("n_kept")))
        log.info("season %d: %d teams forecast, %d skaters kept", T, t.height, keep.height)
    return pl.concat(out)
