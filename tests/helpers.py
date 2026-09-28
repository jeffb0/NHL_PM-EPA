"""Synthetic event builders for unit tests (no network / no data files needed)."""

from __future__ import annotations

HOME, AWAY = 1, 2


def ev(idx: int, t: float, typ: str, team: int | None = None, p1: int | None = None, x: float | None = None,
       y: float | None = None, p2: int | None = None, period: int = 1, reason: str | None = None,
       home_sk: int = 5, away_sk: int = 5, home_g: int = 1, away_g: int = 1, home_score: int = 0,
       away_score: int = 0) -> dict:
    """One event in the parsed schema. ``x, y`` are in the acting team's attacking frame."""
    opp = None if team is None else (AWAY if team == HOME else HOME)
    return {
        "game_id": 2024020001, "event_idx": idx, "period": period, "t_game": (period - 1) * 1200 + t,
        "type": typ, "team_id": team, "opp_team_id": opp, "player1": p1, "player2": p2, "x": x, "y": y,
        "reason": reason, "home_id": HOME, "home_skaters": home_sk, "away_skaters": away_sk,
        "home_goalie": home_g, "away_goalie": away_g, "home_score": home_score, "away_score": away_score,
    }
