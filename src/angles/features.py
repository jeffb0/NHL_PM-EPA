"""Situational angles from commentary and trend betting, each known before the game (docs/angles/methodology.md §2)."""

from __future__ import annotations

import polars as pl

TEAM = ["streak_w3", "streak_l3", "last10", "luck10", "blowout_loss", "blowout_win", "shutout_loss", "scored5", "prev_ot", "b2b_after_blowout", "b2b_after_ot", "front_b2b", "revenge", "revenge_blowout", "rematch_lost", "h2h",
        "hot_goalie"]
HOME_ONLY, AWAY_ONLY = ["home_after_trip"], ["trip_finale"]
ANGLES = [*TEAM, *HOME_ONLY, *AWAY_ONLY]
CONTINUOUS = ["last10", "luck10", "h2h", "hot_goalie"]
FOLK = {"streak_w3": 1, "streak_l3": -1, "last10": 1, "luck10": 1, "blowout_loss": 1, "blowout_win": -1, "shutout_loss": 1, "scored5": 1, "prev_ot": -1, "b2b_after_blowout": 1, "b2b_after_ot": -1, "front_b2b": -1,
        "revenge": 1, "revenge_blowout": 1, "rematch_lost": 1, "h2h": 1, "hot_goalie": 1, "home_after_trip": -1, "trip_finale": -1}       # claimed effect on the team in the situation
BLOWOUT, STREAK, TRIP = 3, 3, 3


def team_states(g: pl.DataFrame, starters: dict | None = None) -> pl.DataFrame:
    """One row per team-game with every angle for that team, from games before this one.
    ``g``: game_id, season, date, home_abbrev, away_abbrev, home_final, away_final, last_period_type, xgd (home minus away, may be null).
    ``starters``: (game_id, team) -> (goalie id, goals saved above expected in that game)."""
    starters = starters or {}
    rows = []
    for r in g.sort("date", "game_id").iter_rows(named=True):
        ot = r["last_period_type"] != "REG"
        for side, me, opp, gf, ga, sgn in (("H", r["home_abbrev"], r["away_abbrev"], r["home_final"], r["away_final"], 1.0), ("A", r["away_abbrev"], r["home_abbrev"], r["away_final"], r["home_final"], -1.0)):
            rows.append({"game_id": r["game_id"], "season": r["season"], "date": r["date"], "team": me, "opp": opp, "home": side == "H", "gf": gf, "ga": ga, "ot": ot,
                         "luck": None if r["xgd"] is None else sgn * ((0.0 if r["last_period_type"] == "SO" else r["home_final"] - r["away_final"]) - r["xgd"])})
    long = pl.DataFrame(rows).sort("team", "date", "game_id")
    out = []
    for (team,), t in long.group_by(["team"], maintain_order=True):
        hist: list[dict] = []                    # this season's games so far
        meet: dict[str, list[int]] = {}          # opponent -> results (+1 / -1) in every season
        gk: dict[int, list[float]] = {}          # goalie -> goals saved above expected in his starts this season
        rs = t.to_dicts()
        for i, r in enumerate(rs):
            if hist and hist[-1]["season"] != r["season"]:
                hist, gk = [], {}
            p = hist[-1] if hist else None
            nxt = rs[i + 1] if i + 1 < len(rs) and rs[i + 1]["season"] == r["season"] else None
            won = [h["gf"] > h["ga"] for h in hist]
            run_w = next((k for k, w in enumerate(reversed(won)) if not w), len(won))
            run_l = next((k for k, w in enumerate(reversed(won)) if w), len(won))
            b2b = p is not None and (r["date"] - p["date"]).days == 1
            last = next((h for h in reversed(hist) if h["opp"] == r["opp"]), None)
            road = next((k for k, h in enumerate(reversed(hist)) if h["home"]), len(hist))          # straight road games before tonight
            luck = [h["luck"] for h in hist[-10:] if h["luck"] is not None]
            m = meet.get(r["opp"], [])[-6:]
            st = starters.get((r["game_id"], team))
            prior = gk.get(st[0], [])[-5:] if st else []
            out.append({"game_id": r["game_id"], "team": team,
                        "streak_w3": float(run_w >= STREAK), "streak_l3": float(run_l >= STREAK), "last10": sum(won[-10:]) / len(won[-10:]) - 0.5 if len(won) >= 5 else 0.0,
                        "luck10": sum(luck) / len(luck) if len(hist) >= 5 and luck else 0.0,
                        "blowout_loss": float(p is not None and p["ga"] - p["gf"] >= BLOWOUT), "blowout_win": float(p is not None and p["gf"] - p["ga"] >= BLOWOUT),
                        "shutout_loss": float(p is not None and p["gf"] == 0), "scored5": float(p is not None and p["gf"] >= 5), "prev_ot": float(p is not None and p["ot"]),
                        "b2b_after_blowout": float(b2b and p["ga"] - p["gf"] >= BLOWOUT), "b2b_after_ot": float(b2b and p["ot"]), "front_b2b": float(nxt is not None and (nxt["date"] - r["date"]).days == 1),
                        "revenge": float(last is not None and last["ga"] > last["gf"]), "revenge_blowout": float(last is not None and last["ga"] - last["gf"] >= BLOWOUT),
                        "rematch_lost": float(p is not None and p["opp"] == r["opp"] and p["ga"] > p["gf"]), "h2h": sum(m) / len(m) if len(m) >= 3 else 0.0,
                        "hot_goalie": sum(prior) if len(prior) >= 3 else 0.0,
                        "home_after_trip": float(r["home"] and road >= TRIP),
                        "trip_finale": float((not r["home"]) and road >= TRIP - 1 and nxt is not None and nxt["home"])})
            hist.append(r)
            meet.setdefault(r["opp"], []).append(1 if r["gf"] > r["ga"] else -1)
            if st:
                gk.setdefault(st[0], []).append(st[1])
    return pl.DataFrame(out)


def game_angles(g: pl.DataFrame, starters: dict | None = None) -> pl.DataFrame:
    """One row per game: each team angle as home minus away, the one-sided angles as they are, and both sides' raw values (suffix _H / _A) for the records."""
    ts = team_states(g, starters)
    h = ts.rename({c: f"{c}_H" for c in ANGLES}).rename({"team": "home_abbrev"})
    a = ts.rename({c: f"{c}_A" for c in ANGLES}).rename({"team": "away_abbrev"})
    j = g.select("game_id", "home_abbrev", "away_abbrev").join(h, on=["game_id", "home_abbrev"]).join(a, on=["game_id", "away_abbrev"])
    return j.with_columns(*[(pl.col(f"{c}_H") - pl.col(f"{c}_A")).alias(c) for c in TEAM], pl.col("home_after_trip_H").alias("home_after_trip"), (-pl.col("trip_finale_A")).alias("trip_finale")).drop("home_abbrev", "away_abbrev")
