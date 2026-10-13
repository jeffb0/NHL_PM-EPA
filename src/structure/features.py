"""Inputs for the structural candidates (docs/structure/methodology.md §2): goalie fatigue, team home ice, scoring environment, games played, shot volume."""

from __future__ import annotations

import numpy as np
import polars as pl

RUSTY_DAYS, RUSTY_TEAM_GAMES, HOME_SHRINK, TOTAL_PRIOR_GAMES, EARLY_GAMES = 10, 5, 400.0, 20.0, 15


def goalie_flags(g: pl.DataFrame, starters: dict) -> pl.DataFrame:
    """Per game: ``gk_tired`` and ``gk_rusty`` as home minus away. ``g``: game_id, season, date, home_id, away_id in any order; ``starters``: (game_id, team_id) -> goalie id."""
    last_team: dict = {}          # team -> (season, date, starter) of its previous game
    last_start: dict = {}         # (season, goalie) -> date of his previous start
    played: dict = {}             # (season, team) -> games so far
    rows = []
    for gid, season, d, hid, aid in g.sort("date", "game_id").select("game_id", "season", "date", "home_id", "away_id").iter_rows():
        v = {}
        for side, tid in (("h", hid), ("a", aid)):
            gk = starters.get((gid, tid))
            p = last_team.get(tid)
            ls = last_start.get((season, gk))
            n = played.get((season, tid), 0)
            v[f"tired_{side}"] = float(gk is not None and p is not None and p[0] == season and (d - p[1]).days == 1 and p[2] == gk)
            v[f"rusty_{side}"] = float(gk is not None and ((ls is not None and (d - ls).days >= RUSTY_DAYS) or (ls is None and n >= RUSTY_TEAM_GAMES)))
        rows.append({"game_id": gid, "gk_tired": v["tired_h"] - v["tired_a"], "gk_rusty": v["rusty_h"] - v["rusty_a"], **v})
        for tid in (hid, aid):
            gk = starters.get((gid, tid))
            last_team[tid] = (season, d, gk)
            played[(season, tid)] = played.get((season, tid), 0) + 1
            if gk is not None:
                last_start[(season, gk)] = d
    return pl.DataFrame(rows)


def home_ice_team(g: pl.DataFrame) -> pl.DataFrame:
    """Per game: the home team's own home advantage in goals from earlier seasons, relative to the league's and shrunk toward zero. ``g``: game_id, season, home_id, away_id, gd."""
    out = []
    for T in sorted(g["season"].unique().to_list()):
        past = g.filter(pl.col("season") < T).drop_nulls("gd")
        cur = g.filter(pl.col("season") == T).select("game_id", "home_id")
        if past.height == 0:
            out.append(cur.select("game_id", pl.lit(0.0).alias("home_ice_team")))
            continue
        h = past.group_by(pl.col("home_id").alias("team")).agg(pl.col("gd").mean().alias("gh"), pl.len().alias("nh"))
        a = past.group_by(pl.col("away_id").alias("team")).agg((-pl.col("gd")).mean().alias("ga"), pl.len().alias("na"))
        league = float(past["gd"].mean())
        t = h.join(a, on="team").with_columns((((pl.col("gh") - pl.col("ga")) / 2 - league) * (pl.col("nh") + pl.col("na")) / (pl.col("nh") + pl.col("na") + HOME_SHRINK)).alias("home_ice_team"))
        out.append(cur.join(t.select(pl.col("team").alias("home_id"), "home_ice_team"), on="home_id", how="left").select("game_id", pl.col("home_ice_team").fill_null(0.0)))
    return pl.concat(out)


def environment(g: pl.DataFrame) -> pl.DataFrame:
    """Per game: the expected total of expected goals (the mean of the two teams' running totals per game, with last season's average as a prior of 20 games) and the fewer of the two teams' games played.
    ``g``: game_id, season, date, home_id, away_id, xg_total (may be null)."""
    league = {s: float(v) for s, v in g.drop_nulls("xg_total").group_by("season").agg(pl.col("xg_total").mean()).iter_rows()}
    prev_avg: dict = {}           # (season, team) -> its average total that season
    acc: dict = {}                # (season, team) -> [sum, games with a total, games played]
    rows = []
    for gid, season, hid, aid, tot in g.sort("date", "game_id").select("game_id", "season", "home_id", "away_id", "xg_total").iter_rows():
        base = league.get(season - 1, float(np.mean(list(league.values()))))
        ex, n = [], []
        for tid in (hid, aid):
            s = acc.get((season, tid), [0.0, 0, 0])
            ex.append((TOTAL_PRIOR_GAMES * prev_avg.get((season - 1, tid), base) + s[0]) / (TOTAL_PRIOR_GAMES + s[1]))
            n.append(s[2])
        rows.append({"game_id": gid, "exp_total": float(np.mean(ex)), "min_played": int(min(n))})
        for tid in (hid, aid):
            s = acc.setdefault((season, tid), [0.0, 0, 0])
            s[2] += 1
            if tot is not None:
                s[0], s[1] = s[0] + tot, s[1] + 1
                prev_avg[(season, tid)] = s[0] / s[1]
    return pl.DataFrame(rows)


def volume_target(shots: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Per game: the home-minus-away unblocked-attempt differential priced at the league's expected goals per attempt in earlier seasons (the first season uses its own).
    ``shots``: game_id, season, team_id, xg_v1; ``games``: game_id, home_id, away_id."""
    per = shots.group_by("season").agg(pl.col("xg_v1").mean().alias("v")).sort("season")
    price, run = {}, []
    for s, v in per.iter_rows():
        price[s] = float(np.mean(run)) if run else float(v)
        run.append(v)
    c = shots.group_by("game_id", "season", "team_id").agg(pl.len().alias("n"))
    j = (games.join(c.rename({"team_id": "home_id", "n": "nh"}), on=["game_id", "home_id"]).join(c.select("game_id", pl.col("team_id").alias("away_id"), pl.col("n").alias("na")), on=["game_id", "away_id"]))
    return j.select("game_id", ((pl.col("nh").cast(pl.Float64) - pl.col("na")) * pl.col("season").replace_strict(price, return_dtype=pl.Float64)).alias("vol_gd"))
