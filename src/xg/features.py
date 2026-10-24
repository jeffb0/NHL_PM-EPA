"""Shot-level features for the xG model (brief §6.3).

Computed for every shot attempt (SOG, miss, block, goal); the model is *trained* on
unblocked attempts (blocked shots cannot become goals, so their terminal xG is 0).
Prior-event context uses the immediately preceding event in the same game-period.
"""

from __future__ import annotations

import numpy as np
import polars as pl

SHOT_EVENTS = ["shot-on-goal", "missed-shot", "blocked-shot", "goal"]
UNBLOCKED = ["shot-on-goal", "missed-shot", "goal"]
SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around"]
PREV_TYPES = ["faceoff", "hit", "giveaway", "takeaway", "shot-on-goal", "missed-shot", "blocked-shot"]


def build_shot_features(
    events: pl.DataFrame, players: pl.DataFrame, rush_window_s: float = 4, rebound_window_s: float = 3
) -> pl.DataFrame:
    ev = events.filter(~pl.col("type").is_in(["period-start", "delayed-penalty", "game-end"])).sort(
        "game_id", "event_idx"
    )
    prev = [
        pl.col(c).shift(1).over("game_id", "period").alias(f"prev_{c}")
        for c in ["type", "team_id", "x", "y", "t_period"]
    ]
    ev = ev.with_columns(prev)
    shots = ev.filter(pl.col("type").is_in(SHOT_EVENTS) & pl.col("x").is_not_null() & pl.col("y").is_not_null())
    # previous event coordinates in the shooter's frame
    same = pl.col("prev_team_id") == pl.col("team_id")
    px = pl.when(same).then(pl.col("prev_x")).otherwise(-pl.col("prev_x"))
    py = pl.when(same).then(pl.col("prev_y")).otherwise(-pl.col("prev_y"))
    dt = (pl.col("t_period") - pl.col("prev_t_period")).clip(0, 120)
    hand = players.select("player_id", "shoots")
    shots = (
        shots.with_columns(
            px.alias("prev_x_s"),
            py.alias("prev_y_s"),
            dt.alias("dt_prev"),
            ((89 - pl.col("x")) ** 2 + pl.col("y") ** 2).sqrt().alias("dist"),
            pl.arctan2(pl.col("y").abs(), 89 - pl.col("x")).degrees().alias("angle"),
        )
        .with_columns(
            ((pl.col("x") - pl.col("prev_x_s")) ** 2 + (pl.col("y") - pl.col("prev_y_s")) ** 2)
            .sqrt()
            .fill_null(0)
            .alias("dist_prev"),
        )
        .with_columns(
            (pl.col("dist_prev") / pl.col("dt_prev").clip(1, None)).alias("speed_prev"),
            (
                same
                & pl.col("prev_type").is_in(["shot-on-goal", "missed-shot", "blocked-shot"])
                & (pl.col("dt_prev") <= rebound_window_s)
            )
            .fill_null(False)
            .alias("rebound"),
            ((pl.col("prev_x_s") < 25) & (pl.col("dt_prev") <= rush_window_s) & pl.col("prev_type").is_in(PREV_TYPES))
            .fill_null(False)
            .alias("rush"),
            (pl.col("opp_goalie") == 0).alias("empty_net"),
            (pl.col("own_goalie") == 0).alias("own_goalie_pulled"),
            pl.col("score_diff").clip(-3, 3).alias("score_c"),
            (pl.col("period") >= 4).alias("ot"),
            pl.when(pl.col("shot_type").is_in(SHOT_TYPES))
            .then(pl.col("shot_type"))
            .otherwise(pl.lit("other"))
            .alias("shot_type_c"),
            pl.when(pl.col("prev_type").is_in(PREV_TYPES))
            .then(pl.col("prev_type"))
            .otherwise(pl.lit("other"))
            .alias("prev_type_c"),
            (pl.col("type") == "goal").cast(pl.Int8).alias("goal"),
        )
        .join(hand, left_on="player1", right_on="player_id", how="left")
        .with_columns(
            (((pl.col("shoots") == "L") & (pl.col("y") < 0)) | ((pl.col("shoots") == "R") & (pl.col("y") > 0)))
            .fill_null(False)
            .alias("off_wing"),
        )
    )
    return shots


NUMERIC = ["dist", "angle", "x", "dt_prev", "dist_prev", "speed_prev", "score_c", "t_period"]
BOOL = ["rebound", "rush", "empty_net", "own_goalie_pulled", "ot", "off_wing", "is_home"]
CATEG = {
    "shot_type_c": SHOT_TYPES + ["other"],
    "prev_type_c": PREV_TYPES + ["other"],
    "strength": ["5v5", "PP", "PK", "4v4", "3v3", "EN"],
}


def design_matrix(shots: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Numeric design matrix shared by the logit and GBM models."""
    cols, names = [], []
    for c in NUMERIC:
        cols.append(shots[c].cast(pl.Float64).fill_null(0).to_numpy())
        names.append(c)
    y_abs = shots["y"].abs().cast(pl.Float64).to_numpy()
    cols.append(y_abs)
    names.append("y_abs")
    for c in BOOL:
        cols.append(shots[c].cast(pl.Float64).fill_null(0).to_numpy())
        names.append(c)
    for c, levels in CATEG.items():
        v = shots[c].fill_null("other").to_numpy()
        for lv in levels:
            cols.append((v == lv).astype(float))
            names.append(f"{c}={lv}")
    # interaction terms that matter for a linear model (distance x type, rebound x angle)
    d = shots["dist"].to_numpy()
    for lv in ["tip-in", "deflected", "slap", "backhand"]:
        cols.append(d * (shots["shot_type_c"].to_numpy() == lv))
        names.append(f"dist*{lv}")
    cols.append(shots["angle"].to_numpy() * shots["rebound"].cast(pl.Float64).to_numpy())
    names.append("angle*rebound")
    return np.column_stack(cols), names
