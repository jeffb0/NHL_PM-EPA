"""Game-level rating models: player ratings with in-season updates (P and its ablations) and team-rating benchmarks (docs/gamemodel/methodology.md §3-§4)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

from src.gamemodel.data import POS_DEFAULT_MIN, Games, Priors

TEAM_MINUTES = 300.0
EXPERIENCE_M0 = 1000.0
RECENT = 10


def skater_update(theta: dict, ids: np.ndarray, mins: np.ndarray, step: float, gain: np.ndarray | None = None) -> None:
    """Move each dressed skater's rating so the team's strength with this lineup changes by ``step`` (normalised least-mean-squares step).
    ``gain`` (docs/prospects/elite.md §1B) gives some players a larger share of the move; the team's total step is unchanged."""
    x = mins / 60.0
    g = np.ones(len(x)) if gain is None else gain
    d = step * g * x / float((g * x * x).sum())
    for i, di in zip(ids.tolist(), d.tolist()):
        theta[i] = theta.get(i, 0.0) + di


def run_player_season(games: Games, pri: Priors, season: int, lam: float, alpha: float, n0: float, attribution: str = "player", lineup: str = "actual", update: bool = True, feature=None, kappa: float = 0.0, early=None):
    """Expected home goal differential (mu) for every game of ``season`` in date order, with ratings updated after each game.

    With ``feature`` (a function of the dressed player ids and their projected minutes) the home-minus-away value of that roster feature is returned as a second array."""
    t = games.table.filter(pl.col("season") == season)
    theta = dict(pri.theta0[season])
    toi_last, (g0, rep), shots_pg, h = pri.toi_last[season], pri.goalie0[season], pri.shots_pg[season], pri.home_adv
    recent: dict = defaultdict(list)
    career = pri.career.get(season, {}) if kappa else {}
    played: dict = defaultdict(float)                              # minutes this season, for the experience-weighted update
    cum: dict = {}                                                 # games and summed game lines this season, for the early-sample adjustment
    gsaved, gshots = defaultdict(float), defaultdict(float)
    team_u: dict = defaultdict(float)
    prev_lu, prev_gk = {}, {}
    mu_out, feat_out = np.zeros(t.height), np.zeros(t.height)
    fval: dict = {}

    def strength(gid: int, team: int) -> float:
        ids, _ = games.lineups[(gid, team)]
        gk = games.goalies[(gid, team)][0][0]
        if lineup == "previous" and team in prev_lu:
            ids, gk = prev_lu[team], prev_gk[team]
        m = np.array([float(np.mean(recent[i][-RECENT:])) if recent[i] else toi_last.get(i, POS_DEFAULT_MIN[pri.pos.get(i, "F")]) for i in ids.tolist()])
        m *= TEAM_MINUTES / m.sum()
        if feature is not None:
            fval[team] = feature(ids, m)
        if early is None:
            s = float(sum(theta.get(i, 0.0) * mi for i, mi in zip(ids.tolist(), m.tolist())) / 60.0)
        else:                                                       # thin-history players: prior plus the adjustment from their games so far (docs/prospects/hybrid.md)
            s = float(sum((theta.get(i, 0.0) + early.adj(i, cum.get(i))) * mi for i, mi in zip(ids.tolist(), m.tolist())) / 60.0)
        rate = (n0 * g0.get(gk, rep) + gsaved[gk]) / (n0 + gshots[gk])
        return s + rate * shots_pg + (team_u[team] if attribution == "team" else 0.0)

    for k, (gid, hid, aid, gd, xgd) in enumerate(t.select("game_id", "home_id", "away_id", "gd", "xgd").iter_rows()):
        mu = strength(gid, hid) - strength(gid, aid) + h
        mu_out[k] = mu
        if feature is not None:
            feat_out[k] = fval[hid] - fval[aid]
        e = alpha * gd + (1 - alpha) * xgd - mu
        for team, sign in ((hid, 1.0), (aid, -1.0)):
            ids, mins = games.lineups[(gid, team)]
            if update and lam > 0:
                if attribution == "player":
                    gain = None
                    if kappa:
                        exp_ = np.array([career.get(i, 0.0) + played[i] for i in ids.tolist()])
                        gain = 1.0 + kappa * EXPERIENCE_M0 / (exp_ + EXPERIENCE_M0)
                    skater_update(theta, ids, mins, sign * lam * e, gain)
                else:
                    team_u[team] += sign * lam * e
            if update:
                for gk, n, saved in games.goalies[(gid, team)]:
                    gshots[gk] += n
                    gsaved[gk] += saved
            for i, mi in zip(ids.tolist(), mins.tolist()):
                recent[i].append(mi)
                played[i] += mi
            if early is not None:
                for i, line in zip(ids.tolist(), games.stats.get((gid, team), ())):       # a game being forecast has no lines yet
                    if i in early.thin:
                        c = cum.setdefault(i, np.zeros(8))
                        c[0] += 1.0
                        c[1:] += line
            prev_lu[team], prev_gk[team] = ids, games.goalies[(gid, team)][0][0]
    return (mu_out, feat_out) if feature is not None else mu_out


def run_team_rating(games: Games, k: float, rho: float, alpha: float, h: float) -> np.ndarray:
    """Team rating in goals per game over every season in order (benchmarks G1 with alpha = 1 and G2): mu for every game of the table."""
    R: dict = defaultdict(float)
    mu_out = np.zeros(games.table.height)
    cur = None
    for i, (season, hid, aid, gd, xgd) in enumerate(games.table.select("season", "home_id", "away_id", "gd", "xgd").iter_rows()):
        if season != cur:
            for key in list(R):
                R[key] *= rho
            cur = season
        mu = R[hid] - R[aid] + h
        mu_out[i] = mu
        e = alpha * gd + (1 - alpha) * xgd - mu
        R[hid] += k * e / 2
        R[aid] -= k * e / 2
    return mu_out


def logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 30) -> np.ndarray:
    """Logistic regression with an intercept by Newton's method (a tiny ridge keeps it stable)."""
    A = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    b = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ b)))
        H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-8 * np.eye(A.shape[1])
        step = np.linalg.solve(H, A.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def logit_pred(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(X.shape[0]), X]) if X.size else np.ones((X.shape[0], 1))
    return 1 / (1 + np.exp(-(A @ b)))


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
