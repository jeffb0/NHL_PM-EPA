"""Standings data: actual results, opening-night rosters, history baselines (docs/standings/methodology.md §3, §5)."""

from __future__ import annotations

import polars as pl


def team_table(proc) -> pl.DataFrame:
    """One row per (season, team_id): abbreviation and games played."""
    g = pl.read_parquet(proc / "games.parquet")
    both = pl.concat([g.select("season", pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")), g.select("season", pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team"))])
    return both.group_by("season", "team_id").agg(pl.col("team").first(), pl.len().alias("gp"))


def results(proc) -> pl.DataFrame:
    """Actual points, goal differential and expected goal differential per team-season, and their per-82-game scalings.

    Shootout decisions count as ties in goals (the shootout goal is not a game goal); a win is 2 points, an overtime or shootout loss 1."""
    g = pl.read_parquet(proc / "games.parquet")
    so = pl.col("last_period_type") == "SO"
    hg = pl.when(so).then(0).otherwise(pl.col("home_final") - pl.col("away_final"))
    home_win = pl.col("home_final") > pl.col("away_final")
    extra = pl.col("last_period_type") != "REG"
    home = g.select("season", pl.col("home_id").alias("team_id"), hg.alias("gd"), pl.when(home_win).then(2).when(extra).then(1).otherwise(0).alias("pts"))
    away = g.select("season", pl.col("away_id").alias("team_id"), (-hg).alias("gd"), pl.when(~home_win).then(2).when(extra).then(1).otherwise(0).alias("pts"))
    r = pl.concat([home, away]).group_by("season", "team_id").agg(pl.col("gd").sum().alias("gd"), pl.col("pts").sum().alias("pts"), pl.len().alias("gp"))
    x = _xgd(proc)
    r = r.join(x, on=["season", "team_id"], how="left")
    return r.with_columns((pl.col("pts") / pl.col("gp") * 82).alias("pts82"), (pl.col("gd") / pl.col("gp") * 82).alias("gd82"), (pl.col("xgd") / pl.col("gp") * 82).alias("xgd82")).join(
        team_table(proc).select("season", "team_id", "team"), on=["season", "team_id"])


def _xgd(proc) -> pl.DataFrame:
    sh = pl.read_parquet(proc / "fin_shots.parquet", columns=["season", "team_id", "opp_team_id", "xg_v1"])
    f = sh.group_by("season", "team_id").agg(pl.col("xg_v1").sum().alias("xgf"))
    a = sh.group_by("season", pl.col("opp_team_id").alias("team_id")).agg(pl.col("xg_v1").sum().alias("xga"))
    return f.join(a, on=["season", "team_id"]).select("season", "team_id", (pl.col("xgf") - pl.col("xga")).alias("xgd"))


def prior(res: pl.DataFrame) -> pl.DataFrame:
    """Previous-season points, goal and expected goal differential per 82 keyed by the season they predict (team_id persists through relocations)."""
    return res.select((pl.col("season") + 1).alias("season"), "team_id", pl.col("pts82").alias("pts82_prev"), pl.col("gd82").alias("gd82_prev"), pl.col("xgd82").alias("xgd82_prev"))


def openers(proc, n_games: int = 2) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Skaters and goalies who dressed for each team in its first ``n_games`` of the season.

    Returns (skaters: season, team_id, player_id, pos_group; goalies: season, team_id, goalie)."""
    g = pl.read_parquet(proc / "games.parquet").select("game_id", "season", "game_date", "home_id", "away_id")
    tg = pl.concat([g.select("game_id", "season", "game_date", pl.col("home_id").alias("team_id")), g.select("game_id", "season", "game_date", pl.col("away_id").alias("team_id"))])
    first = tg.sort("game_date", "game_id").with_columns(pl.int_range(pl.len()).over("season", "team_id").alias("k")).filter(pl.col("k") < n_games).select("game_id", "team_id")
    pos = pl.read_parquet(proc / "players.parquet").select(pl.col("player_id").cast(pl.Int64), pl.when(pl.col("position") == "D").then(pl.lit("D")).when(pl.col("position") == "G").then(pl.lit("G")).otherwise(pl.lit("F")).alias("pos_group"))
    pg = pl.read_parquet(proc / "player_game.parquet", columns=["game_id", "player_id", "team_id", "season", "toi_all"]).filter(pl.col("toi_all") > 0)
    sk = pg.join(first, on=["game_id", "team_id"]).select("season", "team_id", "player_id").unique().join(pos, on="player_id", how="left").filter(pl.col("pos_group").is_in(["F", "D"]))
    gk = pl.read_parquet(proc / "gk_shots.parquet", columns=["game_id", "season", "goalie", "team_id"]).unique().join(first, on=["game_id", "team_id"]).select("season", "team_id", "goalie").unique()
    return sk.with_columns(pl.col("player_id").cast(pl.Int64), pl.col("team_id").cast(pl.Int64)), gk.with_columns(pl.col("goalie").cast(pl.Int64), pl.col("team_id").cast(pl.Int64))


def goalie_usage(proc) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(per goalie-season shots faced, per season league-mean shots against per team per 82 games)."""
    s = pl.read_parquet(proc / "gk_shots.parquet", columns=["season", "goalie", "team_id"])
    per = s.group_by("season", "goalie").agg(pl.len().alias("shots")).with_columns(pl.col("goalie").cast(pl.Int64))
    gp = team_table(proc).group_by("season").agg(pl.col("gp").sum().alias("tg"))
    lg = s.group_by("season").agg(pl.len().alias("n")).join(gp, on="season").select("season", (pl.col("n") / pl.col("tg") * 82).alias("shots82"))
    return per, lg
