"""Tiny end-to-end run (a few dozen games, two seasons) in an isolated directory.

Proves every stage runs from raw API data to leaderboards without the full 9-season
snapshot. Uses $PM_EPA_CONFIG to point all paths at data/smoke/ and reports/smoke/.
Numbers from the smoke run are NOT meaningful (tiny sample).
"""

from __future__ import annotations

import logging
import os

import yaml

from src.config import ROOT

log = logging.getLogger("smoke")


def run(n_games: int = 25) -> None:
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    base["data"].update({
        "seasons": [2024, 2025], "raw_dir": "data/raw", "processed_dir": "data/smoke/processed",
        "value_dir": "data/smoke/value_surfaces", "output_dir": "data/smoke/outputs", "figures_dir": "reports/smoke/figures", "metrics_dir": "reports/smoke/metrics",
        "scoring_regimes": [[2024, 2025]],
    })
    base["splits"] = {"train": [2024], "validation": [2025], "test": [2025], "cross_fit": True}
    base["metrics"]["bootstrap_reps"] = 50
    base["metrics"]["min_toi_ev"] = 20
    cfg_path = ROOT / "data/smoke/config.smoke.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(base))
    os.environ["PM_EPA_CONFIG"] = str(cfg_path)

    from src.ingestion.nhl_api import ingest
    ingest([2024, 2025], limit=n_games)
    # restrict parsing to the first n games per season (raw dir may hold the full snapshot)
    from src.ingestion import parse as parse_mod

    orig_glob = parse_mod.Path.glob

    def limited(self, pattern):
        items = sorted(orig_glob(self, pattern))
        return items[:n_games] if "pbp" in str(self) else items

    parse_mod.Path.glob = limited  # type: ignore[method-assign]
    try:
        parse_mod.run()
    finally:
        parse_mod.Path.glob = orig_glob  # type: ignore[method-assign]
    from src.possession import build, onice
    onice.run()
    build.run()
    from src.xg import model
    model.run()
    from src.value import fit
    fit.run()
    from src.epa import run as epa_run
    epa_run.run()
    from src.metrics import aggregate
    aggregate.run()
    from src.validation import validate
    validate.run(include_rapm=False)
    import polars as pl
    lb = pl.read_csv(ROOT / "data/smoke/outputs/pm_epa_player_leaderboards.csv")
    assert lb.height > 0 and lb["pm_epa_per60_ev"].is_not_null().any(), "smoke leaderboard empty"
    log.info("SMOKE OK: %d player-season rows", lb.height)
