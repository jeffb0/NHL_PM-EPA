"""Game-level inputs: results, lineups, goalies, preseason priors (docs/gamemodel/methodology.md §2-§3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.prospects import features as pf
from src.standings import candidates as cd
from src.standings import data as sd
from src.standings import forecast as fc

POS_DEFAULT_MIN = {"F": 14.0, "D": 19.0}


@dataclass
class Games:
    """Regular-season games in date order with per-game lineups and goalie lines."""
    table: pl.DataFrame                                   # game_id, season, game_date, home_id, away_id, gd, xgd, home_win, b2b_home, b2b_away
    lineups: dict                                         # (game_id, team_id) -> (player_ids int64 array, minutes array)
    goalies: dict                                         # (game_id, team_id) -> list of (goalie, shots, saved), starter first


def load_games(proc) -> Games:
    g = pl.read_parquet(proc / "games.parquet")
    so = pl.col("last_period_type") == "SO"
    g = g.with_columns(pl.when(so).then(0).otherwise(pl.col("home_final") - pl.col("away_final")).cast(pl.Float64).alias("gd"), (pl.col("home_final") > pl.col("away_final")).cast(pl.Float64).alias("home_win"),
                       pl.col("game_date").str.to_date("%Y-%m-%d").alias("date"))
    x = pl.read_parquet(proc / "fin_shots.parquet", columns=["game_id", "team_id", "xg_v1"]).group_by("game_id", "team_id").agg(pl.col("xg_v1").sum().alias("xg"))
    g = (g.join(x.rename({"team_id": "home_id", "xg": "xg_h"}), on=["game_id", "home_id"], how="left").join(x.rename({"team_id": "away_id", "xg": "xg_a"}), on=["game_id", "away_id"], how="left")
         .with_columns((pl.col("xg_h") - pl.col("xg_a")).alias("xgd")))
    # back-to-back: the team played on the previous calendar day
    tg = pl.concat([g.select("game_id", "date", pl.col("home_id").alias("team_id")), g.select("game_id", "date", pl.col("away_id").alias("team_id"))]).sort("team_id", "date")
    tg = tg.with_columns(((pl.col("date") - pl.col("date").shift(1).over("team_id")).dt.total_days() == 1).fill_null(False).cast(pl.Float64).alias("b2b"))
    g = (g.join(tg.select("game_id", pl.col("team_id").alias("home_id"), pl.col("b2b").alias("b2b_home")), on=["game_id", "home_id"])
         .join(tg.select("game_id", pl.col("team_id").alias("away_id"), pl.col("b2b").alias("b2b_away")), on=["game_id", "away_id"]))
    pg = pl.read_parquet(proc / "player_game.parquet", columns=["game_id", "player_id", "team_id", "toi_all"]).filter(pl.col("toi_all") > 0)
    lu = pg.group_by("game_id", "team_id").agg(pl.col("player_id"), (pl.col("toi_all") / 60.0).alias("mins"))
    lineups = {(r[0], r[1]): (np.asarray(r[2], dtype=np.int64), np.asarray(r[3], dtype=float)) for r in lu.iter_rows()}
    gs = pl.read_parquet(proc / "gk_shots.parquet", columns=["game_id", "goalie", "team_id", "goal", "xg_f"]).drop_nulls("goalie").group_by("game_id", "team_id", "goalie").agg(
        pl.len().alias("shots"), (pl.col("xg_f") - pl.col("goal")).sum().alias("saved")).sort("shots", descending=True)
    goalies: dict = {}
    for gid, tid, gk, n, s in gs.iter_rows():
        goalies.setdefault((gid, tid), []).append((int(gk), float(n), float(s)))
    ok = [(gid, h, a) for gid, h, a in g.select("game_id", "home_id", "away_id").iter_rows() if (gid, h) in lineups and (gid, a) in lineups and (gid, h) in goalies and (gid, a) in goalies]
    keep = pl.DataFrame({"game_id": [o[0] for o in ok]})
    g = g.join(keep, on="game_id").drop_nulls("xgd").sort("date", "game_id")
    return Games(g.select("game_id", "season", "date", "home_id", "away_id", "home_abbrev", "away_abbrev", "gd", "xgd", "home_win", "b2b_home", "b2b_away"), lineups, goalies)


@dataclass
class Priors:
    theta0: dict            # season -> {player_id: goals above replacement per 60 at the start of the season}
    pos: dict               # player_id -> "F" | "D"
    toi_last: dict          # season -> {player_id: mean minutes per game in season - 1}
    goalie0: dict           # season -> ({goalie: prior saved per shot}, replacement rate)
    shots_pg: dict          # season -> league-mean unblocked shots faced per team-game in season - 1
    home_adv: float         # mean home margin (blend-free: goals) in 2017-2018


def build_priors(proc, cfg, seasons: list[int], games: Games, yp: pl.DataFrame | None = None) -> Priors:
    """With ``yp`` (docs/prospects) the rate model includes the pre-NHL columns."""
    roll = pl.read_parquet(proc / "wm_rolling_values.parquet")
    p = pd_.panel(proc, roll)
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    res = sd.results(proc)
    resid = cd.team_residual(proc, p, res)
    posdf = pl.read_parquet(proc / "players.parquet").select(pl.col("player_id").cast(pl.Int64), pl.when(pl.col("position") == "D").then(pl.lit("D")).otherwise(pl.lit("F")).alias("pos_group"))
    pos = dict(posdf.iter_rows())
    pg = pl.read_parquet(proc / "player_game.parquet", columns=["player_id", "season", "toi_all"]).filter(pl.col("toi_all") > 0)
    tl = pg.group_by("season", "player_id").agg((pl.col("toi_all").mean() / 60.0).alias("m"))
    fr_val = pf.with_yp(pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR), yp)
    rates = pl.read_parquet(proc / "gk_rolling_rates.parquet")
    _, lg = sd.goalie_usage(proc)
    M = pf.spec(yp)
    theta0, toi_last, goalie0, shots_pg = {}, {}, {}, {}
    for T in seasons:
        who = pg.filter(pl.col("season") == T).select(pl.col("player_id").cast(pl.Int64)).unique().join(posdf, on="player_id", how="left").with_columns(pl.col("pos_group").fill_null("F"), pl.lit(0, pl.Int64).alias("team_id"))
        d = pf.with_yp(fc.roster_frame(p, bio, who, T), yp)
        rate =pm.predict_models(pm.fit_models(pr.train_rows(fr_val, 1, T), M), d, M)
        prev = p.filter(pl.col("season") == T - 1).select("player_id", "team").join(resid.filter(pl.col("season") == T - 1).select("team", "rho"), on="team", how="left").select("player_id", "rho")
        rho = d.select("player_id").join(prev, on="player_id", how="left")["rho"].fill_null(0.0).to_numpy()
        theta0[T] = dict(zip(d["player_id"].to_list(), (rate + 60.0 * rho).tolist()))
        toi_last[T] = dict(tl.filter(pl.col("season") == T - 1).select("player_id", "m").iter_rows())
        r = rates.filter(pl.col("season") == T)
        goalie0[T] = (dict(r.select("goalie", "rate_v").iter_rows()), float(r["rep_v"][0]))
        shots_pg[T] = float(lg.filter(pl.col("season") == T - 1)["shots82"].item()) / 82.0
    early = games.table.filter(pl.col("season") <= 2018)
    return Priors(theta0, pos, toi_last, goalie0, shots_pg, float(early["gd"].mean()))
