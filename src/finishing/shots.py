"""Finishing shot table: unblocked, non-empty-net skater attempts with shooter, goalie, xG features.

Built from the PM-EPA event tables (no new ingestion). Adds what the shared xG does not carry: the
goalie in net (``player2`` on shot events), shooter position, and the xG_f context features
(previous-event ownership, lateral movement from the shooter's own team's previous event, time since
the last faceoff, own-team attempts in the previous 10 s). Every dropped row is accounted for in
``fin_shots_checks.json`` (methodology gate F5).
"""

from __future__ import annotations

import json
import logging

import polars as pl

from src.config import load_config
from src.xg.features import SHOT_EVENTS, UNBLOCKED, build_shot_features

log = logging.getLogger(__name__)

STRENGTH_GROUPS = {"5v5": "5v5", "PP": "PP", "PK": "PK", "4v4": "4v4_3v3", "3v3": "4v4_3v3", "EN": "pulled"}
# shooter x shot-type deviations are relative to "base" (wrist, snap, wrap-around and the small residual types)
TYPE_GROUPS = {"slap": "slap", "backhand": "back", "tip-in": "tip", "deflected": "tip"}
V2_EXTRA = [
    "prev_same",
    "lat_same",
    "dx_same",
    "prev_y_abs_same",
    "cross_slot",
    "secs_since_fo",
    "own_skaters",
    "opp_skaters",
    "team_att_10s",
]


def _context(events: pl.DataFrame) -> pl.DataFrame:
    """Per-event context that needs the event stream: seconds since the last faceoff, own attempts in the last 10 s."""
    ev = events.filter(~pl.col("type").is_in(["period-start", "delayed-penalty", "game-end"])).sort(
        "game_id", "event_idx"
    )
    fo = (
        ev.with_columns(pl.when(pl.col("type") == "faceoff").then(pl.col("t_period")).otherwise(None).alias("_t"))
        .with_columns(pl.col("_t").forward_fill().over("game_id", "period").alias("t_fo"))
        .select(
            "game_id",
            "event_idx",
            (pl.col("t_period") - pl.col("t_fo").fill_null(0)).clip(0, 1200).alias("secs_since_fo"),
        )
    )
    sh = ev.filter(pl.col("type").is_in(SHOT_EVENTS)).sort("game_id", "period", "team_id", "event_idx")
    prev = [
        ((pl.col("t_period") - pl.col("t_period").shift(k).over("game_id", "period", "team_id")) <= 10)
        .fill_null(False)
        .cast(pl.Int8)
        for k in (1, 2, 3)
    ]
    sh = sh.select("game_id", "event_idx", pl.sum_horizontal(prev).alias("team_att_10s"))
    return fo.join(sh, on=["game_id", "event_idx"], how="left")


def add_v2_features(shots: pl.DataFrame) -> pl.DataFrame:
    """xG_f (feature set F1) extras that depend only on columns already in ``shots``."""
    same = (pl.col("prev_team_id") == pl.col("team_id")).fill_null(False)
    return shots.with_columns(
        same.alias("prev_same"),
        pl.when(same).then((pl.col("y") - pl.col("prev_y_s")).abs()).otherwise(0.0).fill_null(0.0).alias("lat_same"),
        pl.when(same).then(pl.col("x") - pl.col("prev_x_s")).otherwise(0.0).fill_null(0.0).alias("dx_same"),
        pl.when(same).then(pl.col("prev_y_s").abs()).otherwise(0.0).fill_null(0.0).alias("prev_y_abs_same"),
        (same & (pl.col("prev_y_s") * pl.col("y") < 0) & (pl.col("prev_y_s").abs() >= 8) & (pl.col("dt_prev") <= 3))
        .fill_null(False)
        .alias("cross_slot"),
    )


def build(proc) -> tuple[pl.DataFrame, dict]:
    cfg = load_config()
    events = pl.read_parquet(proc / "events.parquet")
    players = pl.read_parquet(proc / "players.parquet")
    goalies = set(players.filter(pl.col("position") == "G")["player_id"].to_list())
    pos = players.select(
        pl.col("player_id").alias("player1"),
        pl.when(pl.col("position") == "D").then(pl.lit("D")).otherwise(pl.lit("F")).alias("pos_group"),
        (pl.col("position") == "G").alias("shooter_is_goalie"),
    )
    allshots = build_shot_features(events, players, cfg.xg.rush_window_s, cfg.possession.rebound_window_s)
    total_goals = events.filter(pl.col("type") == "goal").height
    chk = {"goals_in_events": total_goals}
    ctx = _context(events)
    s = (
        allshots.join(ctx, on=["game_id", "event_idx"], how="left")
        .join(pos, on="player1", how="left")
        .with_columns(pl.col("shooter_is_goalie").fill_null(False))
    )
    s = add_v2_features(s)
    chk["goals_in_shot_features_table"] = int(s["goal"].sum())
    chk["goals_without_coordinates"] = total_goals - chk["goals_in_shot_features_table"]
    chk["goals_on_blocked_rows"] = int(s.filter(pl.col("type") == "blocked-shot")["goal"].sum())  # must be 0
    unb = s.filter(pl.col("type").is_in(UNBLOCKED))
    chk["goals_empty_net"] = int(unb.filter(pl.col("empty_net"))["goal"].sum())
    chk["goals_by_goalie_shooters"] = int(unb.filter(~pl.col("empty_net") & pl.col("shooter_is_goalie"))["goal"].sum())
    fin = unb.filter(~pl.col("empty_net") & ~pl.col("shooter_is_goalie") & pl.col("pos_group").is_not_null())
    chk["goals_in_finishing_table"] = int(fin["goal"].sum())
    chk["goals_unaccounted"] = int(
        total_goals
        - chk["goals_without_coordinates"]
        - chk["goals_empty_net"]
        - chk["goals_by_goalie_shooters"]
        - chk["goals_in_finishing_table"]
    )
    v1 = pl.read_parquet(proc / "shots_xg.parquet").select("game_id", "event_idx", pl.col("xg").alias("xg_v1"))
    fin = fin.join(v1, on=["game_id", "event_idx"], how="left").with_columns(
        pl.col("player1").alias("shooter"),
        pl.col("player2").alias("goalie"),
        pl.col("strength").replace_strict(STRENGTH_GROUPS, default="other").alias("strength_g"),
        pl.col("shot_type").replace_strict(TYPE_GROUPS, default="base").fill_null("base").alias("tg"),
        pl.col("goal").cast(pl.Int8),
    )
    chk["rows"] = fin.height
    chk["rows_missing_xg_v1"] = int(fin["xg_v1"].is_null().sum())
    chk["rows_duplicate_keys"] = int(fin.height - fin.select("game_id", "event_idx").unique().height)
    chk["rows_goalie_null"] = int(fin["goalie"].is_null().sum())
    chk["goalie_ids_not_goalies"] = int(
        fin.filter(pl.col("goalie").is_not_null() & ~pl.col("goalie").is_in(list(goalies))).height
    )
    chk["rows_by_strength"] = {r["strength_g"]: r["len"] for r in fin.group_by("strength_g").agg(pl.len()).to_dicts()}
    chk["rows_by_type_group"] = {r["tg"]: r["len"] for r in fin.group_by("tg").agg(pl.len()).to_dicts()}
    keep = [
        "game_id",
        "event_idx",
        "season",
        "period",
        "t_period",
        "team_id",
        "opp_team_id",
        "type",
        "shooter",
        "goalie",
        "pos_group",
        "goal",
        "xg_v1",
        "strength_g",
        "tg",
        "y",
        "shot_type_c",
        "prev_type_c",
        "strength",
        "dist",
        "angle",
        "x",
        "dt_prev",
        "dist_prev",
        "speed_prev",
        "score_c",
        "rebound",
        "rush",
        "empty_net",
        "own_goalie_pulled",
        "ot",
        "off_wing",
        "is_home",
        "score_diff",
    ] + V2_EXTRA
    return fin.select(keep).sort("game_id", "event_idx"), chk


def run() -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    fin, chk = build(proc)
    fin.write_parquet(proc / "fin_shots.parquet")
    met = cfg.path(cfg.data.metrics_dir)
    (met / "fin_shots_checks.json").write_text(json.dumps(chk, indent=2))
    log.info(
        "finishing shots: %d rows, %d goals; unaccounted goals = %d",
        fin.height,
        chk["goals_in_finishing_table"],
        chk["goals_unaccounted"],
    )
    return fin


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
