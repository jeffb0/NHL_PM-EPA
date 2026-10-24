"""Weighted ridge projection models (docs/projection/methodology.md §3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.projection.data import COMPS

AGE_KNOTS = (24.0, 28.0, 32.0)
LAMBDAS = np.geomspace(1e-3, 3e1, 15)


@dataclass(frozen=True)
class Spec:
    """Which feature blocks enter: the ten shrunk component rates (or only their total), age terms, height and
    weight."""

    comps: str = "ten"  # "ten" | "total"
    age: bool = True
    hw: bool = True
    extra: tuple[str, ...] = ()  # optional prior covariates (e.g. future EDGE features): columns already in the frame


def age_basis(age: np.ndarray) -> np.ndarray:
    a = age - 27.0
    return np.column_stack([a] + [np.maximum(age - k, 0.0) for k in AGE_KNOTS])


def feature_names(spec: Spec) -> list[str]:
    n = [f"rt_{c}" for c in COMPS] if spec.comps == "ten" else ["rt_total"]
    n += ["no_history"]
    if spec.age:
        n += ["age_lin", "age_h24", "age_h28", "age_h32"]
    if spec.hw:
        n += ["height_in", "weight_lb"]
    return n + list(spec.extra)


def xmat(d: pl.DataFrame, spec: Spec) -> np.ndarray:
    base = [d[f"rt_{c}"].to_numpy() for c in (COMPS if spec.comps == "ten" else ["total"])] + [
        d["no_history"].to_numpy()
    ]
    cols = [np.column_stack(base)]
    if spec.age:
        cols.append(age_basis(d["age"].to_numpy()))
    if spec.hw:
        cols.append(np.column_stack([d["height_in"].to_numpy(), d["weight_lb"].to_numpy()]))
    if spec.extra:
        cols.append(np.column_stack([d[c].to_numpy() for c in spec.extra]))
    return np.hstack(cols)


@dataclass
class Ridge:
    mu: np.ndarray
    sd: np.ndarray
    ybar: float
    beta: np.ndarray  # on standardized features
    lam: float
    names: list[str] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ybar + ((X - self.mu) / self.sd) @ self.beta

    def raw_coef(self) -> np.ndarray:
        return self.beta / self.sd


def wridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> Ridge:
    w = w / w.sum()
    mu = (w[:, None] * X).sum(0)
    sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0))
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    ybar = float((w * y).sum())
    A = Z.T @ (w[:, None] * Z) + lam * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (w * (y - ybar)))
    return Ridge(mu, sd, ybar, beta, lam)


def choose_lambda(X: np.ndarray, y: np.ndarray, w: np.ndarray, season: np.ndarray) -> float:
    """Leave-one-target-season-out weighted squared error over the grid."""
    if len(np.unique(season)) < 2:
        return 0.1  # a single training season cannot be cross-validated by season
    err = np.zeros(len(LAMBDAS))
    for s in np.unique(season):
        te = season == s
        for i, lam in enumerate(LAMBDAS):
            m = wridge(X[~te], y[~te], w[~te], lam)
            err[i] += float((w[te] * (y[te] - m.predict(X[te])) ** 2).sum())
    return float(LAMBDAS[int(np.argmin(err))])


def fit_models(
    train: pl.DataFrame,
    spec: Spec,
    target: str = "r_total",
    weight: str = "m",
    lam: float | dict[str, float] | None = None,
) -> dict[str, Ridge]:
    """One ridge per position group."""
    out = {}
    for g in ("F", "D"):
        d = train.filter(pl.col("pos_group") == g)
        X, y, w, s = xmat(d, spec), d[target].to_numpy(), d[weight].to_numpy(), d["season"].to_numpy()
        lam_g = (lam[g] if isinstance(lam, dict) else lam) if lam is not None else choose_lambda(X, y, w, s)
        m = wridge(X, y, w, lam_g)
        m.names = feature_names(spec)
        out[g] = m
    return out


def predict_models(models: dict[str, Ridge], d: pl.DataFrame, spec: Spec) -> np.ndarray:
    pred = np.zeros(d.height)
    for g in ("F", "D"):
        m = (d["pos_group"] == g).to_numpy()
        if m.any():
            pred[m] = models[g].predict(xmat(d.filter(pl.col("pos_group") == g), spec))
    return pred


def variance_model(m: np.ndarray, resid: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """sigma^2(m) = a + b / m by weighted least squares on squared residuals (both parts kept non-negative)."""
    A = np.column_stack([np.ones(len(m)), 1.0 / m])
    W = w / w.sum()
    coef = np.linalg.solve(A.T @ (W[:, None] * A), A.T @ (W * resid**2))
    return max(float(coef[0]), 0.0), max(float(coef[1]), 1e-9)


def age_curve(models: dict[str, Ridge], spec: Spec, ages: np.ndarray) -> dict[str, np.ndarray]:
    """Partial effect of age on the projected rate (per 60) relative to age 27, other features fixed."""
    out = {}
    for g, m in models.items():
        k = feature_names(spec).index("age_lin")
        coef = m.raw_coef()[k : k + 4]
        out[g] = age_basis(ages) @ coef
    return out
