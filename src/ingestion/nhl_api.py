"""Raw data ingestion from the public NHL APIs.

Sources (all public, no key required):
  * schedule:   https://api.nhle.com/stats/rest/en/game?cayenneExp=season=<id> and gameType=2
  * PBP:        https://api-web.nhle.com/v1/gamecenter/<gameId>/play-by-play
  * shifts:     https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<gameId>
  * rosters:    https://api-web.nhle.com/v1/roster/<TEAM>/<seasonId>   (handedness, position)

Raw responses are stored verbatim as gzipped JSON under ``data/raw/<kind>/<season>/`` so
every downstream number can be regenerated from an immutable snapshot. Fetching is
idempotent: existing files are skipped. A manifest records file hashes + fetch times.

CLI:  python -m src.ingestion.nhl_api --seasons 2017-2025 [--limit N]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests

from src.config import load_config, season_id

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://api.nhle.com/stats/rest/en/game?cayenneExp=season={sid}%20and%20gameType={gt}"
PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
SHIFT_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}"
ROSTER_URL = "https://api-web.nhle.com/v1/roster/{team}/{sid}"

_session = requests.Session()
_session.headers["User-Agent"] = "pm-epa-research/0.1 (reproducible academic analysis)"


def _get_json(url: str, timeout: int, retries: int = 5) -> dict | None:
    """GET with exponential backoff. Returns None on a definitive 404."""
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"status {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            wait = 2**attempt
            log.warning("GET %s failed (%s); retry in %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def _write_gz(path: Path, obj: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, separators=(",", ":")).encode()
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(raw)
    tmp.rename(path)
    return hashlib.sha256(raw).hexdigest()


def read_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read())


def fetch_schedule(season: int, raw_dir: Path, game_type: int, timeout: int) -> list[int]:
    path = raw_dir / "schedule" / f"{season}.json.gz"
    if not path.exists():
        data = _get_json(SCHEDULE_URL.format(sid=season_id(season), gt=game_type), timeout)
        assert data is not None, f"no schedule for {season}"
        _write_gz(path, data)
    data = read_gz(path)
    # gameStateId 7 == final. Anything else (postponed/cancelled) is skipped.
    return sorted(g["id"] for g in data["data"] if g.get("gameStateId") == 7)


def _fetch_game(gid: int, season: int, raw_dir: Path, timeout: int) -> dict:
    out = {"game_id": gid}
    for kind, url in (("pbp", PBP_URL), ("shifts", SHIFT_URL)):
        path = raw_dir / kind / str(season) / f"{gid}.json.gz"
        if path.exists():
            out[kind] = "cached"
            continue
        data = _get_json(url.format(gid=gid), timeout)
        if data is None:
            out[kind] = "missing"
            continue
        out[kind] = _write_gz(path, data)
    return out


def fetch_rosters(season: int, raw_dir: Path, timeout: int) -> None:
    """Rosters give handedness (off-wing feature) and canonical positions."""
    teams: set[str] = set()
    for p in (raw_dir / "pbp" / str(season)).glob("*.json.gz"):
        g = read_gz(p)
        teams.update({g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]})
        if len(teams) >= 33:
            break
    for team in sorted(teams):
        path = raw_dir / "rosters" / str(season) / f"{team}.json.gz"
        if path.exists():
            continue
        data = _get_json(ROSTER_URL.format(team=team, sid=season_id(season)), timeout)
        if data is not None:
            _write_gz(path, data)


def ingest(seasons: list[int], limit: int | None = None) -> None:
    cfg = load_config()
    raw_dir = cfg.path(cfg.data.raw_dir)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for season in seasons:
        gids = fetch_schedule(season, raw_dir, cfg.data.game_type, cfg.data.request_timeout_s)
        if limit:
            gids = gids[:limit]
        log.info("season %s: %d final games", season, len(gids))
        results = []
        with ThreadPoolExecutor(cfg.data.ingest_workers) as pool:
            futs = [
                pool.submit(_fetch_game, g, season, raw_dir, cfg.data.request_timeout_s) for g in gids
            ]
            for i, f in enumerate(as_completed(futs), 1):
                results.append(f.result())
                if i % 200 == 0:
                    log.info("season %s: %d/%d games", season, i, len(gids))
        fetch_rosters(season, raw_dir, cfg.data.request_timeout_s)
        missing = {k: sum(r.get(k) == "missing" for r in results) for k in ("pbp", "shifts")}
        manifest[str(season)] = {
            "n_games": len(gids),
            "missing": missing,
            "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log.info("season %s done; missing=%s", season, missing)


def _parse_seasons(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None, help="e.g. 2017-2025 or 2023,2024")
    ap.add_argument("--limit", type=int, default=None, help="games per season (smoke tests)")
    args = ap.parse_args()
    seasons = _parse_seasons(args.seasons) if args.seasons else load_config().data.seasons
    ingest(seasons, args.limit)
