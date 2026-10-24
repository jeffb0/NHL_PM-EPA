"""Possession-chain inference from NHL play-by-play.

A *possession* is a maximal run of play during which one team is inferred to control the
puck. Public PBP has no pass/carry events, so a possession is observed only at the events
listed below. Rules (full spec: docs/pm_epa/definitions.md §3):

Possession-bearing events (acting team = T)
  faceoff        T wins -> closes any open possession, opens T possession   (start: faceoff)
  takeaway       opens T possession (start: takeaway). If the opponent had an open possession
                 it closes as ``takeaway_against`` with a pseudo-observation at the loss point.
  giveaway       terminal for T (end: giveaway)
  shot-on-goal / missed-shot / blocked-shot / goal    terminal for T (end: shot_*)
Context-only events
  hit            not a possession event. Optionally (``hits_as_observations``) a *mid*
                 observation when the hittee's team already has an open possession.
  stoppage / penalty / period-end / game-end   close any open possession (terminal value 0).
Inferred (loose-puck) starts
  When T records a terminal event without an open possession, T's possession start is
  inferred from the preceding loose-puck origin (a shot, a giveaway, or an unobserved loss)
  if it happened within ``max_inferred_start_gap_s`` (and not after the event). The start kind records the origin, e.g.
  ``loose_opp_miss`` (T recovered after an opponent's missed shot) or ``rebound`` (T's own
  shot within ``rebound_window_s``). Otherwise the possession has no start observation
  (``unobserved``) and generates no movement credit.

All coordinates in the outputs are in the possessing team's attacking frame (+x = toward
the opponent's net). Events store coordinates in the *acting* team's frame, so observations
taken from an opponent's event are mirrored: (x, y) -> (-x, -y).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SHOT_END = {
    "shot-on-goal": "shot_sog",
    "missed-shot": "shot_miss",
    "blocked-shot": "shot_block",
    "goal": "goal",
}
SHOT_ORIGIN = {"shot-on-goal": "sog", "missed-shot": "miss", "blocked-shot": "block"}
STOP_TYPES = {"stoppage", "penalty", "period-end", "game-end"}


@dataclass
class ChainParams:
    rebound_window_s: float = 3.0
    max_inferred_start_gap_s: float = 25.0
    hits_as_observations: bool = False


@dataclass
class _Open:
    poss_id: int
    team: int
    period: int
    start_type: str
    start_link: int | None
    obs: list[dict] = field(default_factory=list)


def _team_view(ev: dict, team: int) -> dict[str, Any]:
    """Strength / score / coordinates of event ``ev`` from ``team``'s perspective."""
    home = ev["home_id"]
    is_home = team == home
    own_sk = ev["home_skaters"] if is_home else ev["away_skaters"]
    opp_sk = ev["away_skaters"] if is_home else ev["home_skaters"]
    own_g = ev["home_goalie"] if is_home else ev["away_goalie"]
    opp_g = ev["away_goalie"] if is_home else ev["home_goalie"]
    if own_g == 0 or opp_g == 0:
        strength = "EN"
    elif own_sk == opp_sk:
        strength = {5: "5v5", 4: "4v4", 3: "3v3"}.get(own_sk, "other")
    else:
        strength = "PP" if own_sk > opp_sk else "PK"
    sd = ev["home_score"] - ev["away_score"]
    x, y = ev.get("x"), ev.get("y")
    if x is not None and ev.get("team_id") is not None and ev["team_id"] != team:
        x, y = -x, -y
    return {"strength": strength, "score_diff": sd if is_home else -sd, "x": x, "y": y}


class ChainBuilder:
    """Builds possessions + observations for one game. Pure Python, easily unit-tested."""

    def __init__(self, params: ChainParams, poss_id_start: int = 0):
        self.p = params
        self.next_id = poss_id_start
        self.possessions: list[dict] = []
        self.observations: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _obs(
        self,
        cur: _Open,
        ev: dict,
        kind: str,
        role: str,
        player: int | None,
        x: float | None,
        y: float | None,
        t: float | None = None,
        event_idx: int | None = None,
        view_team_event: dict | None = None,
    ) -> None:
        view = _team_view(view_team_event or ev, cur.team)
        cur.obs.append(
            {
                "poss_id": cur.poss_id,
                "game_id": ev["game_id"],
                "period": ev["period"],
                "team_id": cur.team,
                "seq": len(cur.obs),
                "event_idx": ev["event_idx"] if event_idx is None else event_idx,
                "t": ev["t_game"] if t is None else t,
                "obs_kind": kind,
                "role": role,
                "player_id": player,
                "x": x,
                "y": y,
                "strength": view["strength"],
                "score_diff": view["score_diff"],
            }
        )

    def _open(self, team: int, period: int, start_type: str, link: int | None) -> _Open:
        cur = _Open(self.next_id, team, period, start_type, link)
        self.next_id += 1
        return cur

    def _close(self, cur: _Open, end_type: str, ev: dict | None, terminal_event_idx: int | None) -> None:
        self.possessions.append(
            {
                "poss_id": cur.poss_id,
                "game_id": ev["game_id"] if ev else cur.obs[0]["game_id"],
                "period": cur.period,
                "team_id": cur.team,
                "start_type": cur.start_type,
                "start_link": cur.start_link,
                "end_type": end_type,
                "terminal_event_idx": terminal_event_idx,
                "end_event_idx": ev["event_idx"] if ev else None,
                "start_t": cur.obs[0]["t"] if cur.obs else (ev["t_game"] if ev else None),
                "end_t": ev["t_game"] if ev else (cur.obs[-1]["t"] if cur.obs else None),
                "n_obs": len(cur.obs),
            }
        )
        self.observations.extend(cur.obs)

    # ------------------------------------------------------------------ main loop
    def run(self, events: list[dict]) -> None:
        cur: _Open | None = None
        pending: dict | None = None  # loose-puck origin after a terminal event
        period = None
        for ev in events:
            typ = ev["type"]
            if ev["period"] != period:
                if cur is not None:
                    self._close(cur, "period_end", None, None)
                cur, pending, period = None, None, ev["period"]
            if typ == "period-start" or typ == "delayed-penalty":
                continue
            if typ in STOP_TYPES:
                if cur is not None:
                    reason = ev.get("reason") if typ == "stoppage" else typ
                    end = (
                        f"stoppage:{reason}" if typ == "stoppage" else ("penalty" if typ == "penalty" else "period_end")
                    )
                    self._close(cur, end, ev, ev["event_idx"])
                cur, pending = None, None
                continue
            team = ev.get("team_id")
            if team is None:
                continue
            if typ == "hit":
                hittee_team = ev["opp_team_id"]
                if (
                    self.p.hits_as_observations
                    and cur is not None
                    and cur.team == hittee_team
                    and ev.get("x") is not None
                ):
                    v = _team_view(ev, hittee_team)
                    self._obs(cur, ev, "hit_mid", "mid", ev.get("player2"), v["x"], v["y"])
                continue
            if typ == "faceoff":
                if cur is not None:
                    self._close(cur, "unrecorded_stoppage", ev, None)
                cur = self._open(team, ev["period"], "faceoff", None)
                self._obs(cur, ev, "faceoff", "start", ev.get("player1"), ev.get("x"), ev.get("y"))
                pending = None
                continue
            if typ == "takeaway":
                link = None
                if cur is not None and cur.team != team:
                    # opponent loses the puck at the takeaway location (their frame = mirrored)
                    v = _team_view(ev, cur.team)
                    self._obs(cur, ev, "lost_takeaway", "end", None, v["x"], v["y"])
                    self._close(cur, "takeaway_against", ev, ev["event_idx"])
                    link = cur.poss_id
                elif cur is not None:
                    self._close(cur, "unobserved_loss", ev, None)
                cur = self._open(team, ev["period"], "takeaway", link)
                self._obs(cur, ev, "takeaway", "start", ev.get("player1"), ev.get("x"), ev.get("y"))
                pending = None
                continue
            if typ not in SHOT_END and typ != "giveaway":
                continue
            # ---- terminal event for `team`
            if cur is None or cur.team != team:
                if cur is not None:  # opponent lost it somewhere we did not observe
                    last = cur.obs[-1] if cur.obs else None
                    self._close(cur, "unobserved_loss", ev, None)
                    pending = {
                        "origin": "unobserved",
                        "team": cur.team,
                        "t": last["t"] if last else ev["t_game"],
                        "event_idx": last["event_idx"] if last else ev["event_idx"],
                        "x": None,
                        "y": None,
                        "poss_id": cur.poss_id,
                        "event": None,
                    }
                cur = self._start_inferred(team, ev, pending)
            if typ == "giveaway":
                self._obs(cur, ev, "giveaway", "end", ev.get("player1"), ev.get("x"), ev.get("y"))
                self._close(cur, "giveaway", ev, ev["event_idx"])
                pending = {
                    "origin": "giveaway",
                    "team": team,
                    "t": ev["t_game"],
                    "event_idx": ev["event_idx"],
                    "x": ev.get("x"),
                    "y": ev.get("y"),
                    "poss_id": cur.poss_id,
                    "event": ev,
                }
            else:
                self._obs(cur, ev, "shot", "end", ev.get("player1"), ev.get("x"), ev.get("y"))
                self._close(cur, SHOT_END[typ], ev, ev["event_idx"])
                # a goal is always followed by a faceoff: it never seeds a loose-puck start
                pending = (
                    None
                    if typ == "goal"
                    else {
                        "origin": SHOT_ORIGIN[typ],
                        "team": team,
                        "t": ev["t_game"],
                        "event_idx": ev["event_idx"],
                        "x": ev.get("x"),
                        "y": ev.get("y"),
                        "poss_id": cur.poss_id,
                        "event": ev,
                    }
                )
            cur = None
        if cur is not None:
            self._close(cur, "period_end", None, None)

    def _start_inferred(self, team: int, ev: dict, pending: dict | None) -> _Open:
        """Open a possession for ``team`` whose start was not directly observed."""
        gap = None if pending is None else ev["t_game"] - pending["t"]
        if gap is None or gap < 0 or gap > self.p.max_inferred_start_gap_s:
            return self._open(team, ev["period"], "unobserved", None)
        own = pending["team"] == team
        origin = pending["origin"]
        if origin in ("sog", "miss", "block") and own and gap <= self.p.rebound_window_s and ev["type"] in SHOT_END:
            kind = "rebound"
        elif origin == "unobserved":
            kind = "loose_unobserved"
        else:
            kind = f"loose_{'own' if own else 'opp'}_{origin}"
        cur = self._open(team, ev["period"], kind, pending["poss_id"])
        src = pending.get("event")
        if src is not None and pending["x"] is not None:
            x, y = (pending["x"], pending["y"]) if own else (-pending["x"], -pending["y"])
        else:
            x = y = None
        self._obs(
            cur,
            ev,
            kind,
            "start",
            None,
            x,
            y,
            t=pending["t"],
            event_idx=pending["event_idx"],
            view_team_event=src or ev,
        )
        return cur


def build_game(events: list[dict], params: ChainParams, poss_id_start: int = 0) -> tuple[list[dict], list[dict]]:
    b = ChainBuilder(params, poss_id_start)
    b.run(events)
    return b.possessions, b.observations
