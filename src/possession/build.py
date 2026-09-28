"""Run possession-chain inference over every game -> possessions + observations Parquet."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from src.config import load_config
from src.possession.chains import ChainParams, build_game

log = logging.getLogger(__name__)

EVENT_COLS = [
    "game_id", "event_idx", "period", "t_game", "type", "team_id", "opp_team_id", "player1", "player2",
    "x", "y", "reason", "home_id", "home_skaters", "away_skaters", "home_goalie", "away_goalie",
    "home_score", "away_score",
]
POSS_ID_STRIDE = 10_000  # poss_id = game_id * stride + local index (max ~400 per game)


def _one(args: tuple[int, list[dict], ChainParams]) -> tuple[list[dict], list[dict]]:
    gid, events, params = args
    return build_game(events, params, poss_id_start=gid * POSS_ID_STRIDE)


def build_all(events: pl.DataFrame, params: ChainParams, workers: int = 4) -> tuple[pl.DataFrame, pl.DataFrame]:
    ev = events.select(EVENT_COLS).sort("game_id", "event_idx")
    tasks = [(gid, g.to_dicts(), params) for (gid,), g in ev.group_by(["game_id"], maintain_order=True)]
    poss, obs = [], []
    with ProcessPoolExecutor(workers) as pool:
        for p, o in pool.map(_one, tasks, chunksize=64):
            poss.extend(p)
            obs.extend(o)
    poss_df = pl.DataFrame(poss, infer_schema_length=None)
    obs_df = pl.DataFrame(obs, infer_schema_length=None)
    return poss_df, obs_df


def run(hits: bool | None = None) -> None:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    events = pl.read_parquet(proc / "events.parquet")
    params = ChainParams(
        rebound_window_s=cfg.possession.rebound_window_s,
        max_inferred_start_gap_s=cfg.possession.max_inferred_start_gap_s,
        hits_as_observations=cfg.possession.hits_as_observations if hits is None else hits,
    )
    sfx = "_hits" if hits else ""
    poss, obs = build_all(events, params)
    season = events.select("game_id", "season").unique()
    poss = poss.join(season, on="game_id")
    obs = obs.join(season, on="game_id")
    poss.write_parquet(proc / f"possessions{sfx}.parquet")
    obs.write_parquet(proc / f"observations{sfx}.parquet")
    log.info("possessions=%d observations=%d", poss.height, obs.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    run(hits="--hits" in sys.argv or None)
