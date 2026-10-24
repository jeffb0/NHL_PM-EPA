"""NHL EDGE tracking summaries per skater-season (docs/edge/methodology.md §2).

    python -m src.ingestion.edge [--seasons 2021 2022 ...] [--limit N]

One call per skater-season to ``https://api-web.nhle.com/v1/edge/skater-detail/{playerId}/{seasonId}/2`` (regular
season). Responses are cached
verbatim under ``data/raw/edge/<season>/<playerId>.json.gz`` (a definitive 404 is cached as a marker), so the run is
idempotent and resumable.
EDGE statistics exist from the 2021-22 season. Parsed features are written to
``data/processed/edge_player_season.parquet``.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from src.config import load_config, season_id
from src.ingestion.nhl_api import _get_json, _write_gz, read_gz

log = logging.getLogger(__name__)
URL = "https://api-web.nhle.com/v1/edge/skater-detail/{pid}/{sid}/2"
FIRST_EDGE_SEASON = 2021


def _fetch(pid: int, season: int, raw_dir: Path, timeout: int) -> str:
    path = raw_dir / "edge" / str(season) / f"{pid}.json.gz"
    if path.exists():
        return "cached"
    data = _get_json(URL.format(pid=pid, sid=season_id(season)), timeout)
    _write_gz(path, data if data is not None else {"missing": True})
    time.sleep(0.02)
    return "missing" if data is None else "ok"


def fetch(pairs: list[tuple[int, int]], raw_dir: Path, timeout: int, workers: int = 6) -> dict[str, int]:
    counts = {"ok": 0, "cached": 0, "missing": 0}
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(_fetch, pid, s, raw_dir, timeout) for pid, s in pairs]
        for i, f in enumerate(as_completed(futs), 1):
            counts[f.result()] += 1
            if i % 500 == 0:
                log.info("EDGE %d / %d fetched (%s)", i, len(pairs), counts)
    return counts


def _g(d: dict | None, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parse(raw_dir: Path, pairs: list[tuple[int, int]]) -> pl.DataFrame:
    rows = []
    for pid, s in pairs:
        path = raw_dir / "edge" / str(s) / f"{pid}.json.gz"
        if not path.exists():
            continue
        d = read_gz(path)
        if d.get("missing"):
            continue
        zt = d.get("zoneTimeDetails") or {}
        rows.append(
            {
                "player_id": pid,
                "season": s,
                "edge_gp": _g(d, "player", "gamesPlayed"),
                "max_speed_mph": _g(d, "skatingSpeed", "speedMax", "imperial"),
                "max_speed_pctile": _g(d, "skatingSpeed", "speedMax", "percentile"),
                "bursts_over20": _g(d, "skatingSpeed", "burstsOver20", "value"),
                "bursts_pctile": _g(d, "skatingSpeed", "burstsOver20", "percentile"),
                "shot_speed_mph": _g(d, "topShotSpeed", "imperial"),
                "shot_speed_pctile": _g(d, "topShotSpeed", "percentile"),
                "distance_mi": _g(d, "totalDistanceSkated", "imperial"),
                "distance_pctile": _g(d, "totalDistanceSkated", "percentile"),
                "distance_max_game_mi": _g(d, "distanceMaxGame", "imperial"),
                "oz_share": zt.get("offensiveZonePctg"),
                "dz_share": zt.get("defensiveZonePctg"),
                "oz_share_ev": zt.get("offensiveZoneEvPctg"),
            }
        )
    schema = {
        "player_id": pl.Int64,
        "season": pl.Int64,
        "edge_gp": pl.Int64,
        **{
            k: pl.Float64
            for k in (
                "max_speed_mph",
                "max_speed_pctile",
                "bursts_over20",
                "bursts_pctile",
                "shot_speed_mph",
                "shot_speed_pctile",
                "distance_mi",
                "distance_pctile",
                "distance_max_game_mi",
                "oz_share",
                "dz_share",
                "oz_share_ev",
            )
        },
    }
    return pl.DataFrame(rows, schema=schema)


def run(seasons: list[int] | None = None, limit: int | None = None) -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    raw_dir = cfg.path(cfg.data.raw_dir)
    seasons = seasons or [s for s in cfg.data.seasons if s >= FIRST_EDGE_SEASON]
    ps = pl.read_parquet(proc / "player_season.parquet").filter(
        pl.col("season").is_in(seasons) & (pl.col("gp") > 0) & (pl.col("position") != "G")
    )
    pairs = sorted({(int(p), int(s)) for p, s in ps.select("player_id", "season").iter_rows()})
    if limit:
        pairs = pairs[:limit]
    log.info("EDGE: %d skater-seasons over %s", len(pairs), seasons)
    counts = fetch(pairs, raw_dir, cfg.data.request_timeout_s)
    log.info("EDGE fetched: %s", counts)
    out = parse(raw_dir, pairs)
    out.write_parquet(proc / "edge_player_season.parquet")
    log.info("EDGE parsed: %d rows", out.height)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.seasons, a.limit)


if __name__ == "__main__":
    main()
