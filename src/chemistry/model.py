"""Additive and factorization-machine lineup models on 5v5 stints (docs/chemistry/methodology.md §3-§4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse
from scipy.optimize import minimize

LEDGER = ("start_turnover", "start_faceoff", "start_recovery", "start_unobserved", "move", "term", "quality")
COV = ("zone_OZ", "zone_DZ", "zone_NZ", "score")


@dataclass
class Design:
    Ah: sparse.csr_matrix  # stints x players, 1 where the player is on the ice for the home side
    Aa: sparse.csr_matrix
    C: np.ndarray  # covariates: intercept, zone, score (centred on the training means)
    y: np.ndarray  # home minus away expected goals per 60
    w: np.ndarray  # stint minutes
    game: np.ndarray
    home: np.ndarray  # stints x 5 player ids (for subsetting)
    away: np.ndarray


@dataclass
class Params:
    c: np.ndarray
    beta: np.ndarray
    V: np.ndarray  # players x k (k = 0 for the additive model)
    players: np.ndarray  # sorted player ids
    centre: np.ndarray  # means of the zone and score covariates in training


def stint_frame(proc) -> pl.DataFrame:
    cols = [
        "game_id",
        "season",
        "minutes",
        "home_on",
        "away_on",
        "zone_home",
        "score_home",
        *[f"h_{c}" for c in LEDGER],
        *[f"a_{c}" for c in LEDGER],
    ]
    d = pl.read_parquet(proc / "def_stints.parquet", columns=cols).filter(pl.col("minutes") > 0)
    xg = pl.sum_horizontal([f"h_{c}" for c in LEDGER]) - pl.sum_horizontal([f"a_{c}" for c in LEDGER])
    # on-the-fly starts are the reference
    zone = [(pl.col("zone_home") == z).cast(pl.Float64).alias(f"zone_{z}") for z in ("OZ", "DZ", "NZ")]
    return d.select(
        "game_id",
        "season",
        "minutes",
        "home_on",
        "away_on",
        *zone,
        pl.col("score_home").cast(pl.Float64).alias("score"),
        (xg / pl.col("minutes") * 60).alias("y"),
    )


def _incidence(ids: np.ndarray, players: np.ndarray) -> sparse.csr_matrix:
    idx = np.searchsorted(players, ids)
    idx = np.clip(idx, 0, len(players) - 1)
    known = players[idx] == ids  # players unseen in training contribute nothing
    rows = np.repeat(np.arange(ids.shape[0]), ids.shape[1]).reshape(ids.shape)
    return sparse.csr_matrix((np.ones(int(known.sum())), (rows[known], idx[known])), shape=(ids.shape[0], len(players)))


def design(
    d: pl.DataFrame, players: np.ndarray | None = None, centre: np.ndarray | None = None
) -> tuple[Design, np.ndarray, np.ndarray]:
    home = d["home_on"].list.to_array(5).to_numpy().astype(np.int64)
    away = d["away_on"].list.to_array(5).to_numpy().astype(np.int64)
    if players is None:
        players = np.unique(np.concatenate([home.ravel(), away.ravel()]))
    zs = np.column_stack([d[c].fill_null(0.0).to_numpy() for c in COV])
    w = d["minutes"].to_numpy()
    if centre is None:
        centre = (w[:, None] * zs).sum(0) / w.sum()
    C = np.column_stack([np.ones(d.height), zs - centre])
    return (
        Design(
            _incidence(home, players),
            _incidence(away, players),
            C,
            d["y"].to_numpy(),
            w,
            d["game_id"].to_numpy(),
            home,
            away,
        ),
        players,
        centre,
    )


def fit_additive(D: Design, players: np.ndarray, centre: np.ndarray, alpha_b: float) -> Params:
    """Ridge on the player effects (covariates unpenalised), solved from the normal equations."""
    X = sparse.hstack([sparse.csr_matrix(D.C), D.Ah - D.Aa]).tocsr()
    XtW = X.T.multiply(D.w).tocsr()
    q = D.C.shape[1]
    pen = np.concatenate([np.zeros(q), np.full(len(players), alpha_b)])
    theta = np.linalg.solve((XtW @ X).toarray() + np.diag(pen + 1e-8), XtW @ D.y)
    return Params(theta[:q], theta[q:], np.zeros((len(players), 0)), players, centre)


def _syn(A: sparse.csr_matrix, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = A @ V
    return 0.5 * ((S * S).sum(1) - A @ (V * V).sum(1)), S


def predict(D: Design, p: Params) -> np.ndarray:
    out = D.C @ p.c + (D.Ah - D.Aa) @ p.beta
    if p.V.shape[1]:
        out = out + _syn(D.Ah, p.V)[0] - _syn(D.Aa, p.V)[0]
    return out


def objective(theta: np.ndarray, D: Design, k: int, alpha_b: float, alpha_v: float) -> tuple[float, np.ndarray]:
    """Minute-weighted squared error plus ridge penalties, scaled by total minutes; value and gradient."""
    q, P = D.C.shape[1], D.Ah.shape[1]
    c, beta, V = theta[:q], theta[q : q + P], theta[q + P :].reshape(P, k)
    sh, Sh = _syn(D.Ah, V)
    sa, Sa = _syn(D.Aa, V)
    X = D.Ah - D.Aa
    e = D.y - (D.C @ c + X @ beta + sh - sa)
    W = D.w.sum()
    val = ((D.w * e * e).sum() + alpha_b * (beta * beta).sum() + alpha_v * (V * V).sum()) / W
    r = -2.0 * D.w * e
    gV = (
        (D.Ah.T @ (r[:, None] * Sh) - (D.Ah.T @ r)[:, None] * V)
        - (D.Aa.T @ (r[:, None] * Sa) - (D.Aa.T @ r)[:, None] * V)
        + 2 * alpha_v * V
    )
    grad = np.concatenate([D.C.T @ r, X.T @ r + 2 * alpha_b * beta, gV.ravel()]) / W
    return float(val), grad


def fit_fm(D: Design, add: Params, k: int, alpha_b: float, alpha_v: float, seed: int = 0, maxiter: int = 200) -> Params:
    rng = np.random.default_rng(seed)
    V0 = rng.normal(0.0, 0.02, (len(add.players), k))
    theta0 = np.concatenate([add.c, add.beta, V0.ravel()])
    res = minimize(
        objective,
        theta0,
        args=(D, k, alpha_b, alpha_v),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxcor": 20},
    )
    q, P = D.C.shape[1], len(add.players)
    return Params(res.x[:q], res.x[q : q + P], res.x[q + P :].reshape(P, k), add.players, add.centre)


def roster_synergy(V: np.ndarray, pi: np.ndarray) -> float:
    """Expected synergy of a roster whose players are on the ice with probabilities ``pi`` (pairs weighted by the
    product of shares)."""
    s = (pi[:, None] * V).sum(0)
    return float(0.5 * ((s * s).sum() - ((pi * pi)[:, None] * V * V).sum()))
