"""Process the current season's completed games with the trained models:  python -m src.live.update [season]

Fetches new games, then runs parse -> on-ice -> possessions -> expected goals (saved all-seasons model) -> possession value (saved value surfaces)
-> player-game table, into data/live/ so the historical tables under data/processed/ are untouched. No model is refitted here."""

from __future__ import annotations

import logging
import os
import sys

import joblib
import numpy as np
import polars as pl
import yaml

from src.config import ROOT, load_config

log = logging.getLogger("live")


def live_config(season: int) -> None:
    """Point every path at data/live/ (raw files and trained models stay where they are) via $PM_EPA_CONFIG."""
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    base["data"].update({"seasons": [season], "processed_dir": "data/live/processed", "output_dir": "data/live/outputs", "figures_dir": "reports/live/figures", "metrics_dir": "reports/live/metrics"})
    base["data"]["scoring_regimes"][-1] = sorted(set(base["data"]["scoring_regimes"][-1]) | {season})
    path = ROOT / "data/live/config.live.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    os.environ["PM_EPA_CONFIG"] = str(path)
    cfg = load_config()
    for d in (cfg.data.processed_dir, cfg.data.output_dir, cfg.data.metrics_dir, cfg.data.figures_dir):
        cfg.path(d).mkdir(parents=True, exist_ok=True)


def xg_all_model() -> dict:
    """The expected-goals model fitted on every historical season (same specification as the cross-fitted models); cached."""
    from src.xg.features import UNBLOCKED, build_shot_features, design_matrix
    from src.xg.model import make_model
    path = ROOT / "data/outputs/models/xg_all.joblib"
    if path.exists():
        return joblib.load(path)
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    proc = ROOT / base["data"]["processed_dir"]
    import json
    chosen = json.loads((ROOT / base["data"]["metrics_dir"] / "xg_metrics.json").read_text())["chosen"]
    shots = build_shot_features(pl.read_parquet(proc / "events.parquet"), pl.read_parquet(proc / "players.parquet"), base["xg"]["rush_window_s"], base["possession"]["rebound_window_s"])
    unb = shots.filter(pl.col("type").is_in(UNBLOCKED))
    X, names = design_matrix(unb)
    model = make_model(chosen, names, base["seed"]).fit(X, unb["goal"].to_numpy())
    out = {"model": model, "names": names, "seasons": sorted(unb["season"].unique().to_list())}
    joblib.dump(out, path)
    log.info("fitted all-seasons xG model on %d shots", unb.height)
    return out


def apply_xg(cfg) -> None:
    from src.xg.features import UNBLOCKED, build_shot_features, design_matrix
    proc = cfg.path(cfg.data.processed_dir)
    shots = build_shot_features(pl.read_parquet(proc / "events.parquet"), pl.read_parquet(proc / "players.parquet"), cfg.xg.rush_window_s, cfg.possession.rebound_window_s)
    X, names = design_matrix(shots)
    m = xg_all_model()
    assert names == m["names"], "xG feature columns changed"
    unb = shots["type"].is_in(UNBLOCKED).to_numpy()
    xg = np.zeros(shots.height)
    xg[unb] = m["model"].predict_proba(X[unb])[:, 1]
    shots.select("game_id", "event_idx", "season", "type", "team_id", "player1", "x", "y", "dist", "angle", "shot_type_c", "rebound", "rush", "strength", "empty_net", "goal").with_columns(pl.Series("xg", xg)).write_parquet(proc / "shots_xg.parquet")


def run(season: int = 2026) -> None:
    live_config(season)
    cfg = load_config()
    raw, proc = cfg.path(cfg.data.raw_dir), cfg.path(cfg.data.processed_dir)
    sched = raw / "schedule" / f"{season}.json.gz"
    if sched.exists():
        sched.unlink()                                   # the schedule is cached once per season; an ongoing season needs it refreshed
    from src.ingestion.nhl_api import ingest
    ingest([season])
    from src.ingestion import parse
    parse.run()
    from src.possession import build, onice
    onice.run()
    build.run()
    apply_xg(cfg)
    from src.value.fit import terminal_values
    terminal_values(pl.read_parquet(proc / "possessions.parquet"), pl.read_parquet(proc / "shots_xg.parquet")).write_parquet(proc / "possessions_valued.parquet")
    from src.epa import run as epa_run
    _, credits, _ = epa_run.run()
    from src.metrics.aggregate import player_game_table
    players = pl.read_parquet(proc / "players.parquet")
    pg = player_game_table(credits, proc, cfg.epa.include_turnovers).join(players.select("player_id", "first_name", "last_name", "position"), on="player_id", how="left")
    pg.write_parquet(proc / "player_game.parquet")
    q = pl.read_parquet(proc / "shift_quality.parquet")
    log.info("live update: %d games, %d player-game rows; %d games below the on-ice match threshold", pl.read_parquet(proc / "games.parquet").height, pg.height, q.filter(pl.col("onice_match_rate") < 0.9).height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
