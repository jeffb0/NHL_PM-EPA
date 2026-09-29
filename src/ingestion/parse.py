"""Parse raw NHL JSON into normalized, analysis-ready Parquet tables.

Outputs (data/processed/):
  games.parquet    one row per game
  events.parquet   one row per PBP event (normalized coordinates, strength, score)
  shifts.parquet   one row per player shift
  players.parquet  one row per player (name, position, handedness)

Conventions (documented in docs/pm_epa/definitions.md):
  * ``team_id`` is the *acting* team of the event, defined per event type
    (faceoff winner, hitter, giver/taker, shooter). For shots we derive the team
    from the shooter's roster entry, never from ``eventOwnerTeamId``, because the
    owner of ``blocked-shot`` changed meaning across seasons.
  * ``x, y`` are rotated so the acting team attacks toward +x (opponent goal line at
    x=+89). ``y`` is rotated with x so left/right from the attacker's view is preserved.
  * ``strength`` is from the acting team's perspective.
  * ``score_diff`` is acting-team goals minus opponent goals *before* the event.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from src.config import load_config
from src.ingestion.nhl_api import read_gz

log = logging.getLogger(__name__)

KEEP_TYPES = {
    "faceoff", "hit", "giveaway", "takeaway", "shot-on-goal", "missed-shot", "blocked-shot",
    "goal", "penalty", "delayed-penalty", "stoppage", "period-start", "period-end", "game-end",
}
SHOT_TYPES = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
UNBLOCKED = {"shot-on-goal", "missed-shot", "goal"}


def _mmss(s: str | None) -> float:
    if not s:
        return np.nan
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def strength_label(own_sk: int, opp_sk: int, own_g: int, opp_g: int) -> str:
    """Strength from one team's perspective. Empty net (either goalie pulled) is its own class."""
    if own_g == 0 or opp_g == 0:
        return "EN"
    if own_sk == opp_sk:
        return {5: "5v5", 4: "4v4", 3: "3v3"}.get(own_sk, "other")
    return "PP" if own_sk > opp_sk else "PK"


def strength_expr() -> pl.Expr:
    """Vectorized ``strength_label`` over own/opp skater + goalie columns."""
    own, opp = pl.col("own_skaters"), pl.col("opp_skaters")
    return (
        pl.when(own.is_null()).then(None)
        .when((pl.col("own_goalie") == 0) | (pl.col("opp_goalie") == 0)).then(pl.lit("EN"))
        .when((own == opp) & (own == 5)).then(pl.lit("5v5"))
        .when((own == opp) & (own == 4)).then(pl.lit("4v4"))
        .when((own == opp) & (own == 3)).then(pl.lit("3v3"))
        .when(own == opp).then(pl.lit("other"))
        .when(own > opp).then(pl.lit("PP"))
        .otherwise(pl.lit("PK"))
    )


def _parse_situation(code: str | None) -> tuple[int, int, int, int]:
    """situationCode = awayGoalie, awaySkaters, homeSkaters, homeGoalie (e.g. '1551')."""
    if not code or len(code) != 4 or not code.isdigit():
        return (1, 5, 5, 1)
    ag, a_sk, h_sk, hg = (int(c) for c in code)
    return ag, a_sk, h_sk, hg


def _acting(p: dict, d: dict, team_of: dict[int, int]) -> tuple[int | None, int | None, int | None]:
    """Return (acting_team, player1, player2) for an event."""
    t = p["typeDescKey"]
    owner = d.get("eventOwnerTeamId")
    if t == "faceoff":
        p1, p2 = d.get("winningPlayerId"), d.get("losingPlayerId")
    elif t == "hit":
        p1, p2 = d.get("hittingPlayerId"), d.get("hitteePlayerId")
    elif t in ("giveaway", "takeaway"):
        p1, p2 = d.get("playerId"), None
    elif t in ("shot-on-goal", "missed-shot"):
        p1, p2 = d.get("shootingPlayerId"), d.get("goalieInNetId")
    elif t == "blocked-shot":
        p1, p2 = d.get("shootingPlayerId"), d.get("blockingPlayerId")
    elif t == "goal":
        p1, p2 = d.get("scoringPlayerId"), d.get("goalieInNetId")
    elif t == "penalty":
        p1, p2 = d.get("committedByPlayerId"), d.get("drawnByPlayerId")
    else:
        p1, p2 = None, None
    team = team_of.get(p1) if p1 is not None else None
    if team is None and t == "blocked-shot" and p2 is not None and team_of.get(p2) is not None:
        team = -team_of[p2]  # placeholder: opponent of blocker, resolved by caller
    if team is None:
        team = owner
    return team, p1, p2


def parse_game(pbp: dict) -> tuple[dict, list[dict], list[dict]]:
    gid = pbp["id"]
    home, away = pbp["homeTeam"]["id"], pbp["awayTeam"]["id"]
    team_of = {r["playerId"]: r["teamId"] for r in pbp.get("rosterSpots", [])}
    players = [
        {
            "player_id": r["playerId"],
            "team_id": r["teamId"],
            "first_name": r["firstName"]["default"],
            "last_name": r["lastName"]["default"],
            "position": r["positionCode"],
        }
        for r in pbp.get("rosterSpots", [])
    ]
    plays = sorted(
        (p for p in pbp["plays"] if p["typeDescKey"] in KEEP_TYPES),
        key=lambda p: p["sortOrder"],
    )
    rows: list[dict] = []
    h_score = a_score = 0
    for i, p in enumerate(plays):
        d = p.get("details", {}) or {}
        pdsc = p["periodDescriptor"]
        ptype = pdsc.get("periodType", "REG")
        if ptype == "SO":
            continue
        period = pdsc["number"]
        team, p1, p2 = _acting(p, d, team_of)
        if team is not None and team < 0:  # blocked shot fallback: opponent of blocker's team
            team = away if -team == home else home
        ag, a_sk, h_sk, hg = _parse_situation(p.get("situationCode"))
        t = _mmss(p.get("timeInPeriod"))
        rows.append(
            {
                "game_id": gid,
                "event_idx": i,
                "event_id": p["eventId"],
                "period": period,
                "period_type": ptype,
                "t_period": t,
                "t_game": (period - 1) * 1200 + t,
                "type": p["typeDescKey"],
                "team_id": team,
                "is_home": None if team is None else team == home,
                "player1": p1,
                "player2": p2,
                "assist1": d.get("assist1PlayerId"),
                "assist2": d.get("assist2PlayerId"),
                "x_raw": d.get("xCoord"),
                "y_raw": d.get("yCoord"),
                "zone_code_raw": d.get("zoneCode"),
                "owner_team_raw": d.get("eventOwnerTeamId"),
                "home_defending_side": p.get("homeTeamDefendingSide"),
                "shot_type": d.get("shotType"),
                "reason": d.get("reason") or d.get("descKey"),
                "secondary_reason": d.get("secondaryReason"),
                "penalty_type": d.get("typeCode") if p["typeDescKey"] == "penalty" else None,
                "penalty_min": d.get("duration"),
                "home_skaters": h_sk,
                "away_skaters": a_sk,
                "home_goalie": hg,
                "away_goalie": ag,
                "home_score": h_score,
                "away_score": a_score,
            }
        )
        if p["typeDescKey"] == "goal":
            if team == home:
                h_score += 1
            else:
                a_score += 1
    game = {
        "game_id": gid,
        "season": int(str(pbp["season"])[:4]),
        "game_date": pbp["gameDate"],
        "home_id": home,
        "away_id": away,
        "home_abbrev": pbp["homeTeam"]["abbrev"],
        "away_abbrev": pbp["awayTeam"]["abbrev"],
        "home_final": pbp["homeTeam"].get("score"),
        "away_final": pbp["awayTeam"].get("score"),
        "last_period_type": pbp.get("gameOutcome", {}).get("lastPeriodType"),
    }
    return game, rows, players


def parse_shifts(js: dict, gid: int) -> list[dict]:
    out = []
    for r in js.get("data", []):
        if r.get("typeCode") != 517:  # 517 = shift, 505 = goal marker
            continue
        s, e = _mmss(r.get("startTime")), _mmss(r.get("endTime"))
        if np.isnan(s) or np.isnan(e) or e <= s:
            continue
        out.append(
            {
                "game_id": gid,
                "player_id": r["playerId"],
                "team_id": r["teamId"],
                "period": r["period"],
                "start": s,
                "end": e,
            }
        )
    return out


def _parse_one(args: tuple[Path, Path | None]) -> tuple[dict, list[dict], list[dict], list[dict]]:
    pbp_path, shift_path = args
    pbp = read_gz(pbp_path)
    game, rows, players = parse_game(pbp)
    shifts = parse_shifts(read_gz(shift_path), pbp["id"]) if shift_path and shift_path.exists() else []
    return game, rows, players, shifts


def infer_direction(ev: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Attack sign of the home team per game-period (+1 => home attacks +x).

    Primary: vote over events with a zone code, excluding blocked shots (whose zone code
    is relative to the blocker in some seasons) and neutral-zone/center-ice events:
        vote = sign(x_raw) if zone=='O' else -sign(x_raw)   (for the owner team)
    converted to the home team's frame. Fallback: ``homeTeamDefendingSide``.
    Agreement between the two is reported as a data-quality metric.
    """
    g = ev.join(games.select("game_id", "home_id"), on="game_id")
    votes = (
        g.filter(
            pl.col("zone_code_raw").is_in(["O", "D"])
            & (pl.col("type") != "blocked-shot")
            & (pl.col("x_raw").abs() > 25)
            & pl.col("owner_team_raw").is_not_null()
        )
        .with_columns(
            (
                pl.col("x_raw").sign()
                * pl.when(pl.col("zone_code_raw") == "O").then(1).otherwise(-1)
                * pl.when(pl.col("owner_team_raw") == pl.col("home_id")).then(1).otherwise(-1)
            ).alias("vote")
        )
        .group_by("game_id", "period")
        .agg(pl.col("vote").sum().alias("vote_sum"), pl.len().alias("n_votes"))
    )
    side = (
        g.filter(pl.col("home_defending_side").is_not_null())
        .group_by("game_id", "period")
        .agg(pl.col("home_defending_side").mode().first())
        .with_columns(
            pl.when(pl.col("home_defending_side") == "left").then(1).otherwise(-1).alias("side_sign")
        )
    )
    out = (
        g.select("game_id", "period").unique()
        .join(votes, on=["game_id", "period"], how="left")
        .join(side.select("game_id", "period", "side_sign"), on=["game_id", "period"], how="left")
        .with_columns(
            pl.when(pl.col("vote_sum").fill_null(0) != 0)
            .then(pl.col("vote_sum").sign())
            .otherwise(pl.col("side_sign"))
            .alias("home_attack_sign")
        )
    )
    return out


def run(seasons: list[int] | None = None) -> None:
    cfg = load_config()
    seasons = seasons or cfg.data.seasons
    raw = cfg.path(cfg.data.raw_dir)
    out_dir = cfg.path(cfg.data.processed_dir)
    tasks = []
    for season in seasons:
        for p in sorted((raw / "pbp" / str(season)).glob("*.json.gz")):
            tasks.append((p, raw / "shifts" / str(season) / p.name))
    log.info("parsing %d games", len(tasks))
    games, rows, players, shifts = [], [], [], []
    with ProcessPoolExecutor(4) as pool:
        for g, r, pl_, sh in pool.map(_parse_one, tasks, chunksize=32):
            games.append(g)
            rows.extend(r)
            players.extend(pl_)
            shifts.extend(sh)
    games_df = pl.DataFrame(games)
    ev = pl.DataFrame(rows, infer_schema_length=None)
    ev = ev.join(games_df.select("game_id", "season", "home_id", "away_id"), on="game_id")

    direction = infer_direction(ev, games_df)
    ev = ev.join(direction.select("game_id", "period", "home_attack_sign", "side_sign"), on=["game_id", "period"], how="left")
    team_sign = (
        pl.when(pl.col("is_home")).then(pl.col("home_attack_sign")).otherwise(-pl.col("home_attack_sign"))
    )
    own_sk = pl.when(pl.col("is_home")).then(pl.col("home_skaters")).otherwise(pl.col("away_skaters"))
    opp_sk = pl.when(pl.col("is_home")).then(pl.col("away_skaters")).otherwise(pl.col("home_skaters"))
    own_g = pl.when(pl.col("is_home")).then(pl.col("home_goalie")).otherwise(pl.col("away_goalie"))
    opp_g = pl.when(pl.col("is_home")).then(pl.col("away_goalie")).otherwise(pl.col("home_goalie"))
    ev = ev.with_columns(
        (pl.col("x_raw") * team_sign).cast(pl.Float64).alias("x"),
        (pl.col("y_raw") * team_sign).cast(pl.Float64).alias("y"),
        pl.when(pl.col("is_home")).then(pl.col("away_id")).otherwise(pl.col("home_id")).alias("opp_team_id"),
        own_sk.alias("own_skaters"),
        opp_sk.alias("opp_skaters"),
        own_g.alias("own_goalie"),
        opp_g.alias("opp_goalie"),
        (
            pl.when(pl.col("is_home"))
            .then(pl.col("home_score") - pl.col("away_score"))
            .otherwise(pl.col("away_score") - pl.col("home_score"))
        ).alias("score_diff"),
    ).with_columns(strength_expr().alias("strength"))
    ev = ev.sort("game_id", "event_idx")
    ev.write_parquet(out_dir / "events.parquet")
    games_df.write_parquet(out_dir / "games.parquet")
    pl.DataFrame(shifts).write_parquet(out_dir / "shifts.parquet")

    # Players: latest roster-spot identity + handedness from team rosters.
    pdf = pl.DataFrame(players).unique(subset=["player_id"], keep="last")
    hand = []
    for season in seasons:
        for f in (raw / "rosters" / str(season)).glob("*.json.gz"):
            js = read_gz(f)
            for grp in ("forwards", "defensemen", "goalies"):
                for r in js.get(grp, []):
                    hand.append({"player_id": r["id"], "shoots": r.get("shootsCatches"), "roster_position": r.get("positionCode")})
    if hand:
        hdf = pl.DataFrame(hand).unique(subset=["player_id"], keep="last")
        pdf = pdf.join(hdf, on="player_id", how="left")
    pdf.drop("team_id").write_parquet(out_dir / "players.parquet")
    _quality_report(ev, direction, games_df, len(shifts), out_dir)


def _quality_report(ev: pl.DataFrame, direction: pl.DataFrame, games: pl.DataFrame, n_shifts: int, out_dir: Path) -> None:
    unb = ev.filter(pl.col("type").is_in(list(UNBLOCKED)) & pl.col("x").is_not_null())
    by_season = (
        ev.group_by("season")
        .agg(
            pl.col("game_id").n_unique().alias("games"),
            pl.len().alias("events"),
            (pl.col("type") == "giveaway").sum().alias("giveaways"),
            (pl.col("type") == "takeaway").sum().alias("takeaways"),
            pl.col("type").is_in(list(SHOT_TYPES)).sum().alias("shot_attempts"),
            (pl.col("type") == "goal").sum().alias("goals"),
            (pl.col("x_raw").is_null() & pl.col("type").is_in(list(SHOT_TYPES) + ["giveaway", "takeaway", "faceoff", "hit"])).mean().alias("pct_missing_coords"),
        )
        .join(
            unb.group_by("season").agg((pl.col("x") > 0).mean().alias("pct_unblocked_in_attacking_half")),
            on="season",
        )
        .sort("season")
    )
    d = direction.filter(pl.col("side_sign").is_not_null() & pl.col("vote_sum").is_not_null() & (pl.col("vote_sum") != 0))
    agree = (d["vote_sum"].sign() == d["side_sign"]).mean() if d.height else float("nan")
    lines = [
        "# Data quality report (auto-generated by src/ingestion/parse.py)",
        "",
        by_season.to_pandas().to_markdown(index=False, floatfmt=".4f"),
        "",
        f"* Direction inference: zone-vote vs `homeTeamDefendingSide` agreement = {agree:.4f} over {d.height} game-periods where both exist.",
        f"* Game-periods with undetermined direction: {direction.filter(pl.col('home_attack_sign').is_null()).height}",
        f"* Shifts parsed: {n_shifts}",
    ]
    (out_dir / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
