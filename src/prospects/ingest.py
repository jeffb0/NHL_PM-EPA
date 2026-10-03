"""Fetch each skater's career page (pre-NHL seasons, draft):  python -m src.prospects.ingest   (docs/prospects/methodology.md §2)."""

from __future__ import annotations

import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from src.config import load_config
from src.ingestion.nhl_api import _get_json, _write_gz

log = logging.getLogger("prospects")
URL = "https://api-web.nhle.com/v1/player/{pid}/landing"


def player_ids(proc, raw) -> list[int]:
    ids = set(pl.read_parquet(proc / "players.parquet").filter(pl.col("position") != "G")["player_id"].to_list())
    nxt = raw / "rosters_next"
    if nxt.exists():
        for f in nxt.glob("*/*/*.json.gz"):
            d = json.load(gzip.open(f))
            ids.update(p["id"] for grp in ("forwards", "defensemen") for p in d.get(grp, []))
    return sorted(int(i) for i in ids)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    proc, raw = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.raw_dir)
    out = raw / "player_landing"
    todo = [i for i in player_ids(proc, raw) if not (out / f"{i}.json.gz").exists()]
    log.info("%d players to fetch", len(todo))

    def one(pid: int) -> bool:
        d = _get_json(URL.format(pid=pid), cfg.data.request_timeout_s)
        if d is None:
            return False
        _write_gz(out / f"{pid}.json.gz", {k: d.get(k) for k in ("playerId", "position", "birthDate", "draftDetails", "seasonTotals")})
        return True

    with ThreadPoolExecutor(4) as pool:
        ok = sum(pool.map(one, todo))
    log.info("fetched %d of %d", ok, len(todo))


if __name__ == "__main__":
    main()
