"""Out-of-sample and stability evaluation for DPM (methodology §6)."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import nnls

from src.defense import ridge
from src.defense.stints import COMPONENTS

MODELS = ["B0", "M0", "M1", "M2", "M3"]
SIMPLICITY = ["M1", "M2", "M0", "M3"]  # D1 tie-break order (simplest first)


def split_blocks(d: ridge.Design, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column index sets of the non-defense part (intercept, offense, covariates) and the defense block."""
    P = d.P
    base = np.r_[0, np.arange(1, 1 + P), np.arange(1 + 2 * P, d.X.shape[1])]
    dfn = np.arange(1 + P, 1 + 2 * P)
    return base, dfn


def features(fit_d: ridge.Design, beta: np.ndarray, next_stints: pl.DataFrame) -> tuple[ridge.Design, np.ndarray, dict[str, np.ndarray]]:
    """Apply season-Y coefficients to season Y+1 stints.

    Returns the Y+1 design (Y's player columns), the base prediction (intercept + covariates + attacker
    offense from the total-xG fit) and one defense feature per target: sum of the defenders' coefficients.
    """
    nd = ridge.build_design(next_stints, ["xg"], players=fit_d.players)
    base_idx, def_idx = split_blocks(nd, beta)
    j_xg = fit_d.targets.index("xg")
    base = nd.X[:, base_idx] @ beta[base_idx, j_xg]
    Xd = nd.X[:, def_idx]
    feats = {t: Xd @ beta[def_idx, j] for j, t in enumerate(fit_d.targets)}
    return nd, base, feats


def defense_term(model: str, feats: dict[str, np.ndarray], core: list[str], w_stack: dict[str, float] | None) -> np.ndarray:
    zero = np.zeros_like(feats["xg"])
    if model == "B0":
        return zero
    if model == "M0":
        return feats["xg"]
    if model == "M1":
        return sum((feats[c] for c in core), zero)
    if model == "M2":
        return sum((feats[c] for c in core + ["quality"]), zero)
    if model == "M3":
        return sum((w_stack[c] * feats[c] for c in COMPONENTS), zero)
    raise ValueError(model)


def stack_weights(pairs: list[tuple[ridge.Design, np.ndarray, dict[str, np.ndarray]]]) -> dict[str, float]:
    """Non-negative weights on the component defense features, fit on the train pairs (M3)."""
    F = np.vstack([np.column_stack([f[c] for c in COMPONENTS]) for _, _, f in pairs])
    r = np.concatenate([d.y()[:, 0] - b for d, b, _ in pairs])
    w = np.concatenate([d.w for d, _, _ in pairs])
    sw = np.sqrt(w)
    coef, _ = nnls(F * sw[:, None], r * sw)
    return dict(zip(COMPONENTS, coef.tolist()))


def pair_losses(nd: ridge.Design, base: np.ndarray, feats: dict[str, np.ndarray], core: list[str],
                w_stack: dict[str, float] | None, target: str = "xg") -> dict[str, pl.DataFrame]:
    """Per-game sums of minute-weighted squared error, one column per model (for the game bootstrap)."""
    y = nd.y()[:, 0] if target == "xg" else 60.0 * nd.goals / nd.w
    cols = {}
    for m in MODELS:
        if m == "M3" and w_stack is None:
            continue
        pred = base + defense_term(m, feats, core, w_stack)
        cols[m] = nd.w * (y - pred) ** 2
    df = pl.DataFrame({"game_id": nd.groups, "w": nd.w, **cols})
    return df.group_by("game_id").agg(pl.all().sum())


def summarize_losses(per_game: pl.DataFrame, reps: int, seed: int) -> dict:
    """Weighted MSE per model, and game-bootstrap SE of each model's difference from every other."""
    models = [m for m in MODELS if m in per_game.columns]
    W = per_game["w"].to_numpy()
    E = np.column_stack([per_game[m].to_numpy() for m in models])
    mse = E.sum(0) / W.sum()
    rng = np.random.default_rng(seed)
    n = len(W)
    boots = []
    for _ in range(reps):
        k = rng.integers(0, n, n)
        boots.append(E[k].sum(0) / W[k].sum())
    boots = np.array(boots)
    out = {"mse": dict(zip(models, mse.tolist())), "n_games": n, "minutes": float(W.sum() / 2)}
    out["diff_se"] = {f"{a}-{b}": float(np.std(boots[:, i] - boots[:, j])) for i, a in enumerate(models) for j, b in enumerate(models) if i != j}
    out["diff"] = {f"{a}-{b}": float(mse[i] - mse[j]) for i, a in enumerate(models) for j, b in enumerate(models) if i != j}
    return out


def pick_model(summary: dict, candidates: list[str]) -> str:
    """D1: lowest MSE; any model within 1 SE of it that is simpler wins (order M1 < M2 < M0 < M3)."""
    mse = summary["mse"]
    best = min(candidates, key=lambda m: mse[m])
    for m in SIMPLICITY:
        if m not in candidates:
            continue
        if m == best or mse[m] - mse[best] <= summary["diff_se"][f"{m}-{best}"]:
            return m
    return best


def headline(coefs: pl.DataFrame, model: str, core: list[str], w_stack: dict[str, float] | None) -> pl.Series:
    """Headline DPM/60 from per-player defense coefficients: sign-flipped so positive = fewer xG against."""
    if model == "M0":
        raw = coefs["def_xg"]
    elif model == "M1":
        raw = sum(coefs[f"def_{c}"] for c in core)
    elif model == "M2":
        raw = sum(coefs[f"def_{c}"] for c in core + ["quality"])
    elif model == "M3":
        raw = sum(w_stack[c] * coefs[f"def_{c}"] for c in COMPONENTS)
    else:
        raise ValueError(model)
    return (-raw).alias("dpm_per60")
