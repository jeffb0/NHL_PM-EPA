"""Daily forecast and odds log:  python -m src.live.daily [YYYY-MM-DD] [--no-update]   (docs/market/methodology.md §2)

Refreshes the season's completed games, forecasts every game of the given day and the next that has not started, captures prices, and appends to
data/live/logs/forecast_log.csv and odds_log.csv. Rows are only ever appended."""

from __future__ import annotations

import csv
import logging
import sys
from datetime import UTC, date, datetime, timedelta

import requests

from src.config import ROOT
from src.live import lineups as ln
from src.live import odds as od
from src.live import predict as lp

log = logging.getLogger("live")
LOGS = ROOT / "data/live/logs"
F_COLS = ["made_utc", "game_id", "game_date", "start_utc", "away", "home", "p_home", "p_away", "exp_goals_home", "exp_goals_away", "model_mu", "home_goalie", "home_goalie_source", "away_goalie", "away_goalie_source",
          "home_lineup_source", "away_lineup_source", "home_games", "away_games", "completed_games", "model",
          "home_removed", "away_removed", "home_added", "away_added", "home_questionable", "away_questionable", "unmatched_out", "roster_snapshot"]
O_COLS = ["captured_utc", "game_id", "game_date", "away", "home", "book", "market", "side", "line", "american", "source"]
MODEL = "game model P (docs/gamemodel); from 2026-10-04: rate model v3 (elite-prospect priors, docs/prospects/elite.md) and early-sample adjustment for thin-history players (docs/prospects/hybrid.md); inputs v2: daily rosters, injury list and listed starting goalies"


def slate(day: str) -> list[dict]:
    d = requests.get(f"https://api-web.nhle.com/v1/score/{day}", timeout=30).json()
    return [{"game_id": g["id"], "game_date": day, "start_utc": g.get("startTimeUTC"), "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"], "state": g.get("gameState")}
            for g in d.get("games", []) if g.get("gameType") == 2]


def _append(path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():                                              # a log written with fewer columns is rewritten once with the new header (old rows keep their values; new columns are blank)
        with path.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames != cols:
                old = list(rd)
                with path.open("w", newline="", encoding="utf-8") as out:
                    w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(old)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def run(day: str | None = None, update: bool = True, season: int = 2026) -> dict:
    day = day or date.today().isoformat()
    if update:
        from src.live.update import run as live_update
        live_update(season)
    if update:
        from src.ingestion.rosters_next import main as roster_snapshot
        roster_snapshot(season)                                    # today's published rosters: trades, call-ups and assignments
        from src.prospects.ingest import main as career_pages
        career_pages()                                             # draft slot and pre-NHL seasons for players new to the rosters
    ctx = lp.prepare(season)
    snap = ctx["snap"]
    roster: dict = {}
    for team, name, pid in snap.select("team", "name", "player_id").iter_rows():
        roster.setdefault(team, {})[ln.norm(name)] = pid
    feed = {}
    for tomorrow in (False, True):
        for g in ln.fetch(tomorrow):
            feed[(g["away"], g["home"])] = ln.resolve(g, roster)
    now = datetime.now(UTC)
    stamp = now.isoformat(timespec="seconds")
    nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    today = slate(day)
    games = today + slate(nxt)
    plays_today = {g["away"] for g in today} | {g["home"] for g in today}        # these teams are on a back-to-back in tomorrow's games
    f_rows, o_rows, skipped = [], [], []
    for g in games:
        started = g["state"] not in ("FUT", "PRE") or (g["start_utc"] and datetime.fromisoformat(g["start_utc"].replace("Z", "+00:00")) <= now)
        if started:
            skipped.append(g["game_id"])
            continue
        news = feed.get((g["away"], g["home"]))
        f = lp.forecast(ctx, g["away"], g["home"], g["game_date"], news=news, force_b2b=plays_today if g["game_date"] == nxt else None)
        th, ta = f["teams"][g["home"]], f["teams"][g["away"]]
        f_rows.append({"made_utc": stamp, "game_id": g["game_id"], "game_date": g["game_date"], "start_utc": g["start_utc"], "away": g["away"], "home": g["home"], "p_home": round(f["p_home_win"], 5), "p_away": round(f["p_away_win"], 5),
                       "exp_goals_home": round(f["expected_goals_home"], 3), "exp_goals_away": round(f["expected_goals_away"], 3), "model_mu": round(f["model_mu"], 4), "home_goalie": th["goalie"], "home_goalie_source": th["goalie_source"],
                       "away_goalie": ta["goalie"], "away_goalie_source": ta["goalie_source"], "home_lineup_source": th["lineup_source"], "away_lineup_source": ta["lineup_source"], "home_games": th["games_played"], "away_games": ta["games_played"],
                       "completed_games": f["completed_games_this_season"], "model": MODEL, "home_removed": "; ".join(th["removed"]), "away_removed": "; ".join(ta["removed"]), "home_added": "; ".join(th["added"]),
                       "away_added": "; ".join(ta["added"]), "home_questionable": "; ".join(th["questionable"]), "away_questionable": "; ".join(ta["questionable"]),
                       "unmatched_out": "; ".join(th["unmatched_out"] + ta["unmatched_out"]), "roster_snapshot": ctx["snapshot_stamp"]})
        prices, url = od.fetch(g["away"], g["home"])
        o_rows += [{"captured_utc": stamp, "game_id": g["game_id"], "game_date": g["game_date"], "away": g["away"], "home": g["home"], **p, "source": url} for p in prices]
    _append(LOGS / "forecast_log.csv", F_COLS, f_rows)
    _append(LOGS / "odds_log.csv", O_COLS, o_rows)
    with_odds = len({r["game_id"] for r in o_rows})
    log.info("lineup feed matched %d of %d games", sum(1 for g in games if (g["away"], g["home"]) in feed), len(games))
    log.info("logged %d forecasts (%d with prices) for %s and %s; %d games already started", len(f_rows), with_odds, day, nxt, len(skipped))
    # score finished games against the market, then lock today's picks and settle yesterday's (docs/market/picks.md)
    from src.live import picks as pk
    from src.live import score as sc
    sc.run()
    ledger = pk.run(day)
    from src.live import digest
    digest.run(day)                                                # reports/live/daily_digest.md: the summary that is emailed
    return {"forecasts": f_rows, "odds_rows": len(o_rows), "games_with_odds": with_odds, "skipped": skipped, "picks": ledger}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else None, update="--no-update" not in sys.argv)
