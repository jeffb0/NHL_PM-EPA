"""Weighted ridge RAPM on stints, solved in closed form (methodology §4).

Each stint gives two rows, one per attacking team:
    y_c = 60 * (component c of the attacking team's ledger in the stint) / minutes,  weight = minutes
    X   = [intercept | offense block (attacking skaters) | defense block (defending skaters) | covariates]

Only the player blocks are penalized, so the intercept and covariates (home attack, zone start,
score state) are estimated as fixed effects. With W = diag(minutes):

    (X'WX + alpha * D) beta = X'W y + alpha * D * beta_prior        D = diag(1 on player columns)

W y = 60 * sums, so the right-hand side needs no division. All components share X'WX, so every
target is solved with one Cholesky factorization per alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import linalg, sparse

from src.defense.stints import COMPONENTS

COVARIATES = ["home_attack", "zone_OZ", "zone_NZ", "zone_DZ", "score_m2", "score_m1", "score_p1", "score_p2"]
FLIP_ZONE = {"OZ": "DZ", "DZ": "OZ", "NZ": "NZ", "fly": "fly"}


@dataclass
class Design:
    X: sparse.csr_matrix  # rows = 2 * stints
    w: np.ndarray  # minutes
    S: np.ndarray  # 60 * component sums, shape (rows, len(targets))
    targets: list[str]
    goals: np.ndarray  # attacking-team goals (secondary target)
    groups: np.ndarray  # game_id per row
    players: list[int]
    n_cov: int

    @property
    def P(self) -> int:
        return len(self.players)

    @property
    def pen(self) -> np.ndarray:
        d = np.zeros(self.X.shape[1])
        d[1 : 1 + 2 * self.P] = 1.0
        return d

    def y(self) -> np.ndarray:
        return self.S / self.w[:, None]


def targets_from(stints: pl.DataFrame, side: str) -> dict[str, np.ndarray]:
    """Component sums for one attacking side (h/a), plus the total xG."""
    out = {c: stints[f"{side}_{c}"].to_numpy() for c in COMPONENTS}
    out["xg"] = np.sum([out[c] for c in COMPONENTS], axis=0)
    return out


def build_design(stints: pl.DataFrame, targets: list[str], players: list[int] | None = None) -> Design:
    """Design over the given stints. ``players`` fixes the column order (players not listed are dropped
    from X, i.e. treated as average), which is how a fitted model is applied to a later season."""
    home_on, away_on = stints["home_on"].to_list(), stints["away_on"].to_list()
    if players is None:
        players = sorted({p for lst in home_on + away_on for p in lst})
    idx = {p: i for i, p in enumerate(players)}
    P, n = len(players), stints.height
    rows, cols = [], []
    for r in range(n):
        for att, dfn, rr in ((home_on[r], away_on[r], r), (away_on[r], home_on[r], n + r)):
            for p in att:
                if p in idx:
                    rows.append(rr)
                    cols.append(idx[p])
            for p in dfn:
                if p in idx:
                    rows.append(rr)
                    cols.append(P + idx[p])
    Xp = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n, 2 * P))
    zone_h = stints["zone_home"].to_numpy()
    zone_a = np.array([FLIP_ZONE[z] for z in zone_h])
    sc = stints["score_home"].to_numpy()
    zone = np.r_[zone_h, zone_a]
    score = np.r_[sc, -sc]
    cov = np.column_stack(
        [
            np.r_[np.ones(n), np.zeros(n)],
            zone == "OZ",
            zone == "NZ",
            zone == "DZ",
            score == -2,
            score == -1,
            score == 1,
            score == 2,
        ]
    ).astype(float)
    X = sparse.hstack([sparse.csr_matrix(np.ones((2 * n, 1))), Xp, sparse.csr_matrix(cov)]).tocsr()
    th, ta = targets_from(stints, "h"), targets_from(stints, "a")
    S = 60.0 * np.column_stack([np.r_[th[t], ta[t]] for t in targets])
    mins = stints["minutes"].to_numpy()
    gid = stints["game_id"].to_numpy()
    goals = np.r_[stints["h_goals"].to_numpy(), stints["a_goals"].to_numpy()]
    return Design(X.tocsr(), np.r_[mins, mins], S, targets, goals, np.r_[gid, gid], players, cov.shape[1])


def gram(d: Design, row_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """X'WX and X'WY (Y = S / w), optionally with extra row weights (bootstrap)."""
    rw = d.w if row_weight is None else d.w * row_weight
    Xw = d.X.multiply(rw[:, None]).tocsr()
    A = (d.X.T @ Xw).toarray()
    S = d.S if row_weight is None else d.S * row_weight[:, None]
    return A, np.asarray(d.X.T @ S)


# added to the diagonal; 0 for the 5v5 models (unchanged), set by the special-teams fits whose thin states can leave
# empty covariate columns
JITTER = 0.0


def solve(A: np.ndarray, B: np.ndarray, pen: np.ndarray, alpha: float, prior: np.ndarray | None = None) -> np.ndarray:
    rhs = B if prior is None else B + alpha * pen[:, None] * prior
    c = linalg.cho_factor(
        A + alpha * np.diag(pen) + (JITTER * np.eye(len(pen)) if JITTER else 0.0), lower=False, check_finite=False
    )
    return linalg.cho_solve(c, rhs, check_finite=False)


def cv_alpha(d: Design, alphas: list[float], folds: int, seed: int) -> dict[str, dict[float, float]]:
    """Game-grouped K-fold CV: minute-weighted MSE of y for every (target, alpha)."""
    rng = np.random.default_rng(seed)
    games = np.unique(d.groups)
    fold_of = dict(zip(rng.permutation(games), np.arange(len(games)) % folds))
    f = np.array([fold_of[g] for g in d.groups])
    A, B = gram(d)
    Y = d.y()
    err = {t: {a: 0.0 for a in alphas} for t in d.targets}
    wsum = 0.0
    for k in range(folds):
        te = f == k
        dk = Design(d.X[te], d.w[te], d.S[te], d.targets, d.goals[te], d.groups[te], d.players, d.n_cov)
        Ak, Bk = gram(dk)
        for a in alphas:
            beta = solve(A - Ak, B - Bk, d.pen, a)
            pred = d.X[te] @ beta
            for j, t in enumerate(d.targets):
                err[t][a] += float(np.sum(d.w[te] * (pred[:, j] - Y[te, j]) ** 2))
        wsum += d.w[te].sum()
    return {t: {a: v / wsum for a, v in e.items()} for t, e in err.items()}


def fit(
    d: Design,
    alphas: dict[str, float],
    prior: dict[str, np.ndarray] | None = None,
    gram_ab: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Coefficients, shape (n_cols, len(targets)); each target uses its own alpha."""
    A, B = gram_ab if gram_ab is not None else gram(d)
    out = np.zeros((A.shape[0], len(d.targets)))
    for a in sorted(set(alphas[t] for t in d.targets)):
        js = [j for j, t in enumerate(d.targets) if alphas[t] == a]
        pr = None if prior is None else np.column_stack([prior[d.targets[j]] for j in js])
        out[:, js] = solve(A, B[:, js], d.pen, a, pr)
    return out


def bootstrap(
    d: Design, alphas: dict[str, float], reps: int, seed: int, prior: dict[str, np.ndarray] | None = None
) -> np.ndarray:
    """Game-block Poisson bootstrap: coefficient draws, shape (reps, n_cols, len(targets))."""
    rng = np.random.default_rng(seed)
    games, inv = np.unique(d.groups, return_inverse=True)
    draws = []
    for _ in range(reps):
        m = rng.poisson(1.0, len(games)).astype(float)[inv]
        draws.append(fit(d, alphas, prior, gram(d, m)))
    return np.stack(draws)


def coef_frame(d: Design, beta: np.ndarray) -> pl.DataFrame:
    """Per-player offense and defense coefficients for every target (raw xG/60 units, not sign-flipped)."""
    P = d.P
    data = {"player_id": d.players}
    for j, t in enumerate(d.targets):
        data[f"off_{t}"] = beta[1 : 1 + P, j]
        data[f"def_{t}"] = beta[1 + P : 1 + 2 * P, j]
    return pl.DataFrame(data)


def player_minutes(d: Design) -> pl.DataFrame:
    """5v5 minutes on ice per player (each stint counted once, via the defense block)."""
    P = d.P
    mins = np.asarray(d.X[:, 1 + P : 1 + 2 * P].multiply(d.w[:, None]).sum(0)).ravel()
    return pl.DataFrame({"player_id": d.players, "toi_5v5": mins})


def prior_vector(d: Design, prev: pl.DataFrame | None, rho: float) -> dict[str, np.ndarray] | None:
    """Prior mean for the player blocks: rho x last season's coefficients (0 for new players)."""
    if prev is None or rho == 0:
        return None
    m = pl.DataFrame({"player_id": d.players}).join(prev, on="player_id", how="left").fill_null(0.0)
    out = {}
    for t in d.targets:
        v = np.zeros(d.X.shape[1])
        v[1 : 1 + d.P] = rho * m[f"off_{t}"].to_numpy()
        v[1 + d.P : 1 + 2 * d.P] = rho * m[f"def_{t}"].to_numpy()
        out[t] = v
    return out
