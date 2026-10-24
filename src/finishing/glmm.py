"""Penalized (Laplace) logistic mixed model on sparse designs, with EM variance components.

Model:  y_i ~ Bernoulli(expit(offset_i + x_i' beta)),   beta_j ~ N(0, s2[g(j)]) for penalized columns,
unpenalized for fixed columns (tiny ridge for numerical stability).

* ``newton``       penalized Newton / IRLS to the mode (concave, so it converges in a handful of steps)
* ``em_variance``  empirical-Bayes variance components: s2_g <- mean(b_g^2 + diag(H^-1)_g), where H is the
                   penalized Hessian at the mode (Laplace posterior precision)
* ``posterior_cov`` H^-1 from the stored Cholesky factor (dense; feasible for ~2e4 columns)

Solved densely (Hessian is sparse-product-built, then dense Cholesky). Methodology §5 tier 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg
from scipy.linalg import lapack
from scipy.sparse import csr_matrix
from scipy.special import expit

log = logging.getLogger(__name__)

FIXED_RIDGE = 1e-6
S2_FLOOR = 1e-6


@dataclass
class Mode:
    beta: np.ndarray
    chol: tuple  # cho_factor output at the mode
    prec: np.ndarray
    iters: int
    converged: bool
    loglik: float = 0.0
    extra: dict = field(default_factory=dict)

    def posterior_cov(self) -> np.ndarray:
        inv, info = lapack.dpotri(self.chol[0], lower=0)
        if info != 0:
            raise np.linalg.LinAlgError(f"dpotri failed: {info}")
        return np.triu(inv) + np.triu(inv, 1).T


def precision_vector(n_cols: int, groups: dict[str, np.ndarray], s2: dict[str, float]) -> np.ndarray:
    prec = np.full(n_cols, FIXED_RIDGE)
    for g, idx in groups.items():
        prec[idx] = 1.0 / max(s2[g], S2_FLOOR)
    return prec


def _hessian(X: csr_matrix, w: np.ndarray, prec: np.ndarray) -> np.ndarray:
    H = (X.T @ X.multiply(w[:, None]).tocsr()).toarray()
    H[np.diag_indices_from(H)] += prec
    return H


def newton(
    X: csr_matrix,
    y: np.ndarray,
    offset: np.ndarray,
    prec: np.ndarray,
    beta0: np.ndarray | None = None,
    max_iter: int = 30,
    tol: float = 1e-5,
    max_step: float = 5.0,
) -> Mode:
    beta = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    converged = False
    n_it = 0
    for _ in range(max_iter):
        n_it += 1
        p = expit(X @ beta + offset)
        g = X.T @ (y - p) - prec * beta
        H = _hessian(X, p * (1 - p), prec)
        c = linalg.cho_factor(H, lower=False, check_finite=False)
        step = linalg.cho_solve(c, g, check_finite=False)
        m = float(np.abs(step).max())
        if m > max_step:
            step *= max_step / m
        beta = beta + step
        if m < tol:
            converged = True
            break
    p = np.clip(expit(X @ beta + offset), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)) - 0.5 * np.sum(prec * beta**2))
    return Mode(beta, c, prec, n_it, converged, ll)


def em_variance(
    X: csr_matrix,
    y: np.ndarray,
    offset: np.ndarray,
    groups: dict[str, np.ndarray],
    s2_init: dict[str, float],
    iters: int = 60,
    tol_sd: float = 1e-4,
    beta0: np.ndarray | None = None,
) -> tuple[dict[str, float], Mode, list[dict]]:
    """Empirical-Bayes variance components by SQUAREM-accelerated EM.

    Plain EM here converges linearly at a rate of ~0.98-0.99 per step (the persistent and season-deviation
    variances are weakly separated), so a "small change per step" stopping rule stops far from the fixed
    point. SQUAREM (Varadhan & Roland 2008) extrapolates over pairs of EM steps; ``iters`` counts cycles
    (three EM evaluations each). Convergence is an *absolute* bound on the standard-deviation scale,
    max_g |sqrt(F(s)_g) - sqrt(s_g)| < ``tol_sd`` (logit units): a relative bound never triggers for a variance
    component whose true value is ~0, which approaches the boundary sub-linearly.
    Returns (s2, mode at the final s2, per-cycle trace).
    """
    names = list(groups)
    state = {"beta": beta0}

    def F(v: np.ndarray) -> np.ndarray:
        s2 = {g: max(float(x), S2_FLOOR) for g, x in zip(names, v)}
        mode = newton(X, y, offset, precision_vector(X.shape[1], groups, s2), state["beta"])
        state["beta"] = mode.beta
        diag = np.diag(mode.posterior_cov())
        return np.array([max(float(np.mean(mode.beta[groups[g]] ** 2 + diag[groups[g]])), S2_FLOOR) for g in names])

    s = np.array([s2_init[g] for g in names], dtype=float)
    trace: list[dict] = []
    converged = False
    for cyc in range(iters):
        s1 = F(s)
        r = s1 - s
        res_sd = float(np.max(np.abs(np.sqrt(s1) - np.sqrt(s))))
        trace.append({"cycle": cyc + 1, "em_residual_sd": res_sd, **{f"s2_{g}": float(x) for g, x in zip(names, s)}})
        log.info(
            "SQUAREM %d: EM residual (sd scale) %.6f  %s",
            cyc + 1,
            res_sd,
            {g: round(float(np.sqrt(x)), 4) for g, x in zip(names, s)},
        )
        if res_sd < tol_sd:
            converged = True
            break
        s2v = F(s1)
        v = (s2v - s1) - r
        nv = float(np.linalg.norm(v))
        alpha = -float(np.linalg.norm(r)) / nv if nv > 0 else -1.0
        alpha = min(alpha, -1.0)
        cand = np.maximum(s - 2 * alpha * r + alpha**2 * v, S2_FLOOR)
        s = F(cand)  # stabilizing EM step (keeps the iteration monotone in practice)
    s2 = {g: float(x) for g, x in zip(names, s)}
    mode = newton(X, y, offset, precision_vector(X.shape[1], groups, s2), state["beta"])
    trace[-1]["converged"] = converged
    return s2, mode, trace
