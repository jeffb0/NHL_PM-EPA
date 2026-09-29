"""Goals per win, derived from the data (methodology §8).

OLS through the origin of (wins - GP/2) on regular-season goal differential across team-seasons.
Every game has one winner (shootout winners count), so a .500 team has GD ~ 0 and the intercept is fixed;
the slope is wins per goal and goals-per-win is its inverse. Shootouts are excluded from goal
differential (events carry no shootout goals), so a shootout win is a win with zero goal margin.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def team_seasons(proc) -> pl.DataFrame:
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "season", "home_id", "away_id", "home_final", "away_final")
    goals = pl.read_parquet(proc / "events.parquet", columns=["game_id", "type", "team_id", "strength"]).filter(pl.col("type") == "goal")
    gf = goals.group_by("game_id", "team_id").agg(pl.len().cast(pl.Int64).alias("gf"))  # signed: gf - ga must not wrap
    long = pl.concat([
        games.select("game_id", "season", pl.col("home_id").alias("team_id"), pl.col("away_id").alias("opp_id"), (pl.col("home_final") > pl.col("away_final")).alias("win")),
        games.select("game_id", "season", pl.col("away_id").alias("team_id"), pl.col("home_id").alias("opp_id"), (pl.col("away_final") > pl.col("home_final")).alias("win")),
    ])
    long = (long.join(gf, on=["game_id", "team_id"], how="left").join(gf.rename({"team_id": "opp_id", "gf": "ga"}), on=["game_id", "opp_id"], how="left")
            .with_columns(pl.col("gf").fill_null(0), pl.col("ga").fill_null(0)))
    return long.group_by("season", "team_id").agg(pl.len().cast(pl.Int64).alias("gp"), pl.col("win").sum().cast(pl.Int64).alias("wins"), pl.col("gf").sum(), pl.col("ga").sum()).with_columns(
        (pl.col("gf") - pl.col("ga")).alias("gd"), (pl.col("wins") - pl.col("gp") / 2).alias("excess_wins"))


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    return float((x * y).sum() / (x * x).sum())


def goals_per_win(ts: pl.DataFrame, reps: int = 1000, seed: int = 0) -> dict:
    x, y = ts["gd"].to_numpy().astype(float), ts["excess_wins"].to_numpy().astype(float)
    b = slope_through_origin(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = np.array([slope_through_origin(x[k], y[k]) for k in (rng.integers(0, n, n) for _ in range(reps))])
    per_season = {int(s): 1.0 / slope_through_origin(g["gd"].to_numpy().astype(float), g["excess_wins"].to_numpy().astype(float)) for (s,), g in ts.group_by(["season"])}
    A = np.column_stack([np.ones(n), x])
    b_int = float(np.linalg.lstsq(A, y, rcond=None)[0][1])
    return {"goals_per_win": 1.0 / b, "wins_per_goal": b, "ci95": [float(1 / np.quantile(bs, 0.975)), float(1 / np.quantile(bs, 0.025))],
            "r2": float(1 - ((y - b * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()), "n_team_seasons": int(n),
            "with_intercept_goals_per_win": 1.0 / b_int, "per_season_goals_per_win": dict(sorted(per_season.items()))}
