"""Rest, travel and schedule-density features from game dates and venues alone (docs/schedule/methodology.md §2)."""

from __future__ import annotations

import numpy as np
import polars as pl

# arena latitude, longitude and standard-time UTC offset
ARENAS = {
    "ANA": (33.81, -117.88, -8),
    "ARI": (33.53, -112.26, -7),
    "BOS": (42.37, -71.06, -5),
    "BUF": (42.88, -78.88, -5),
    "CAR": (35.80, -78.72, -5),
    "CBJ": (39.97, -83.01, -5),
    "CGY": (51.04, -114.05, -7),
    "CHI": (41.88, -87.67, -6),
    "COL": (39.75, -105.01, -7),
    "DAL": (32.79, -96.81, -6),
    "DET": (42.34, -83.06, -5),
    "EDM": (53.55, -113.50, -7),
    "FLA": (26.16, -80.33, -5),
    "LAK": (34.04, -118.27, -8),
    "MIN": (44.94, -93.10, -6),
    "MTL": (45.50, -73.57, -5),
    "NJD": (40.73, -74.17, -5),
    "NSH": (36.16, -86.78, -6),
    "NYI": (40.72, -73.65, -5),
    "NYR": (40.75, -73.99, -5),
    "OTT": (45.30, -75.93, -5),
    "PHI": (39.90, -75.17, -5),
    "PIT": (40.44, -79.99, -5),
    "SEA": (47.62, -122.35, -8),
    "SJS": (37.33, -121.90, -8),
    "STL": (38.63, -90.20, -6),
    "TBL": (27.94, -82.45, -5),
    "TOR": (43.64, -79.38, -5),
    "UTA": (40.77, -111.90, -7),
    "VAN": (49.28, -123.11, -8),
    "VGK": (36.10, -115.18, -8),
    "WPG": (49.89, -97.14, -6),
    "WSH": (38.90, -77.02, -5),
}
BLOCKS = {
    "R": ["rest_adv", "layoff_h", "layoff_a"],
    "T": ["km_h", "km_a", "east_a", "west_a"],
    "D": ["trip_a", "stand_h", "dense_h", "dense_a"],
}
REST_CAP, LAYOFF, RUN_CAP = 3, 5, 6


def km(a: str, b: str) -> float:
    """Great-circle distance between two teams' arenas in kilometres."""
    (la1, lo1, _), (la2, lo2, _) = ARENAS[a], ARENAS[b]
    p1, p2, dl = np.radians(la1), np.radians(la2), np.radians(lo2 - lo1)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(h)))


def team_games(g: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game with that team's schedule state before the game. ``g``: game_id, season, date, home_abbrev,
    away_abbrev."""
    rows = []
    long = pl.concat(
        [
            g.select(
                "game_id",
                "season",
                "date",
                pl.col("home_abbrev").alias("team"),
                pl.col("home_abbrev").alias("venue"),
                pl.lit(1).alias("home"),
            ),
            g.select(
                "game_id",
                "season",
                "date",
                pl.col("away_abbrev").alias("team"),
                pl.col("home_abbrev").alias("venue"),
                pl.lit(0).alias("home"),
            ),
        ]
    ).sort("season", "team", "date", "game_id")
    for (_, team), t in long.group_by(["season", "team"], maintain_order=True):
        prev_date, prev_venue, prev_home, run, dates = None, team, None, 0, []
        for gid, d, venue, home in t.select("game_id", "date", "venue", "home").iter_rows():
            days = (d - prev_date).days if prev_date is not None else None
            run = run + 1 if home == prev_home else 1  # consecutive games at home / on the road
            # hours the venue is east (+) or west (-) of the team's home zone
            tz = ARENAS[venue][2] - ARENAS[team][2]
            rows.append(
                {
                    "game_id": gid,
                    "team": team,
                    "rest": float(min(days, REST_CAP)) if days is not None else float(REST_CAP),
                    "layoff": float(days is not None and days >= LAYOFF),
                    "km": km(prev_venue, venue) / 1000.0,
                    "east": float(max(tz, 0)),
                    "west": float(max(-tz, 0)),
                    "run": float(min(run, RUN_CAP)),
                    "dense": float(sum(1 for x in dates if 1 <= (d - x).days <= 3) >= 2),
                }
            )
            prev_date, prev_venue, prev_home = d, venue, home
            dates.append(d)
    return pl.DataFrame(rows)


def game_features(g: pl.DataFrame) -> pl.DataFrame:
    """One row per game with the eleven features of the three blocks."""
    tg = team_games(g)
    h = tg.rename({c: f"{c}_H" for c in tg.columns if c not in ("game_id", "team")}).rename({"team": "home_abbrev"})
    a = tg.rename({c: f"{c}_A" for c in tg.columns if c not in ("game_id", "team")}).rename({"team": "away_abbrev"})
    j = (
        g.select("game_id", "home_abbrev", "away_abbrev")
        .join(h, on=["game_id", "home_abbrev"])
        .join(a, on=["game_id", "away_abbrev"])
    )
    return j.select(
        "game_id",
        (pl.col("rest_H") - pl.col("rest_A")).alias("rest_adv"),
        pl.col("layoff_H").alias("layoff_h"),
        pl.col("layoff_A").alias("layoff_a"),
        pl.col("km_H").alias("km_h"),
        pl.col("km_A").alias("km_a"),
        pl.col("east_A").alias("east_a"),
        pl.col("west_A").alias("west_a"),
        pl.col("run_A").alias("trip_a"),
        pl.col("run_H").alias("stand_h"),
        pl.col("dense_H").alias("dense_h"),
        pl.col("dense_A").alias("dense_a"),
    )
