"""Snapshot the published rosters for an upcoming season (dated, never overwritten):  python -m src.ingestion.rosters_next 2026

Rosters change daily until opening night, so each run writes ``data/raw/rosters_next/<season>/<YYYY-MM-DD>/<TEAM>.json.gz`` and a manifest."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from src.config import load_config, season_id
from src.ingestion.nhl_api import ROSTER_URL, _get_json, _write_gz


def main(season: int) -> None:
    cfg = load_config()
    raw = cfg.path(cfg.data.raw_dir)
    teams = sorted(p.stem.replace(".json", "") for p in (raw / "rosters" / str(season - 1)).glob("*.json.gz"))
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    out = raw / "rosters_next" / str(season) / stamp
    man = {}
    for t in teams:
        if (out / f"{t}.json.gz").exists():                         # one snapshot per team per day: a second run the same day keeps the first
            man[t] = "kept"
            continue
        d = _get_json(ROSTER_URL.format(team=t, sid=season_id(season)), cfg.data.request_timeout_s)
        if d is None:
            man[t] = "missing"
            continue
        _write_gz(out / f"{t}.json.gz", d)
        man[t] = sum(len(d.get(g, [])) for g in ("forwards", "defensemen", "goalies"))
    (out / "manifest.json").write_text(json.dumps({"fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"), "players_by_team": man}, indent=2))
    print(man)


if __name__ == "__main__":
    main(int(sys.argv[1]))
