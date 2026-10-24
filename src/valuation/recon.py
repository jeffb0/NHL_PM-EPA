"""Team-level reconciliation helpers: 5v5 team goal differential vs summed player components."""

from __future__ import annotations

import numpy as np
import polars as pl


def team_gd5(proc) -> pl.DataFrame:
    """season, team (abbrev), gd5 = 5v5 goals for - against."""
    games = pl.read_parquet(proc / "games.parquet")
    abbr = pl.concat(
        [
            games.select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")),
            games.select(pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team")),
        ]
    ).unique("team_id")
    goals = pl.read_parquet(
        proc / "events.parquet", columns=["season", "type", "team_id", "opp_team_id", "strength"]
    ).filter((pl.col("type") == "goal") & (pl.col("strength") == "5v5"))
    # signed: gf - ga must not wrap
    gf = goals.group_by("season", "team_id").agg(pl.len().cast(pl.Int64).alias("gf"))
    ga = goals.group_by("season", pl.col("opp_team_id").alias("team_id")).agg(pl.len().cast(pl.Int64).alias("ga"))
    return (
        gf.join(ga, on=["season", "team_id"], how="full", coalesce=True)
        .fill_null(0)
        .with_columns((pl.col("gf") - pl.col("ga")).alias("gd5"))
        .join(abbr, on="team_id")
        .select("season", "team", "gd5")
    )


def team_gd_st(proc) -> pl.DataFrame:
    """season, team (abbrev), st_gd = goals for - against at strengths other than 5v5 (PP, PK, 4v4, 3v3; empty net
    excluded)."""
    games = pl.read_parquet(proc / "games.parquet")
    abbr = pl.concat(
        [
            games.select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")),
            games.select(pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team")),
        ]
    ).unique("team_id")
    goals = pl.read_parquet(
        proc / "events.parquet", columns=["season", "type", "team_id", "opp_team_id", "strength"]
    ).filter((pl.col("type") == "goal") & pl.col("strength").is_in(["PP", "PK", "4v4", "3v3"]))
    gf = goals.group_by("season", "team_id").agg(pl.len().cast(pl.Int64).alias("gf"))
    ga = goals.group_by("season", pl.col("opp_team_id").alias("team_id")).agg(pl.len().cast(pl.Int64).alias("ga"))
    return (
        gf.join(ga, on=["season", "team_id"], how="full", coalesce=True)
        .fill_null(0)
        .with_columns((pl.col("gf") - pl.col("ga")).alias("st_gd"))
        .join(abbr, on="team_id")
        .select("season", "team", "st_gd")
    )


def team_gd_all(proc) -> pl.DataFrame:
    """season, team (abbrev), gd_all = goals for - against at every event strength (shootouts are not in the event
    strengths)."""
    games = pl.read_parquet(proc / "games.parquet")
    abbr = pl.concat(
        [
            games.select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")),
            games.select(pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team")),
        ]
    ).unique("team_id")
    goals = pl.read_parquet(
        proc / "events.parquet", columns=["season", "type", "team_id", "opp_team_id", "strength"]
    ).filter((pl.col("type") == "goal") & pl.col("strength").is_in(["5v5", "PP", "PK", "4v4", "3v3", "EN"]))
    gf = goals.group_by("season", "team_id").agg(pl.len().cast(pl.Int64).alias("gf"))
    ga = goals.group_by("season", pl.col("opp_team_id").alias("team_id")).agg(pl.len().cast(pl.Int64).alias("ga"))
    return (
        gf.join(ga, on=["season", "team_id"], how="full", coalesce=True)
        .fill_null(0)
        .with_columns((pl.col("gf") - pl.col("ga")).alias("gd_all"))
        .join(abbr, on="team_id")
        .select("season", "team", "gd_all")
    )


def team_sums(t: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Sum player-season columns to team-season by primary team (``team`` column)."""
    return t.group_by("season", "team").agg(*[pl.col(c).sum() for c in cols])


def regress(j: pl.DataFrame, regressors: list[str], y: str = "gd5") -> dict:
    """OLS of the team goal differential on the regressors (with intercept). Slopes are goals of GD per goal of
    component."""
    yv = j[y].to_numpy().astype(float)
    X = np.column_stack([np.ones(j.height)] + [j[c].to_numpy().astype(float) for c in regressors])
    coef = np.linalg.lstsq(X, yv, rcond=None)[0]
    res = yv - X @ coef
    n, k = X.shape
    s2 = float(res @ res / (n - k))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {
        "n": int(n),
        "r2": float(1 - (res @ res) / ((yv - yv.mean()) ** 2).sum()),
        "slopes": {c: {"slope": float(coef[i + 1]), "se": float(se[i + 1])} for i, c in enumerate(regressors)},
    }
