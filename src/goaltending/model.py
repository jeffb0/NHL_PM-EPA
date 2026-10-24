"""Goalie mixed model (docs/goaltending/methodology.md §3).

    logit P(goal) = offset + cell[season, strength] + u_g + d_{g,season} (+ t_{team,season})

``offset`` = logit(xG_f) + the shooter's fitted shift, so shooter quality is held fixed. Random effects: persistent
goalie
``u``, goalie-season deviation ``d``, defending team-season ``t``. The machinery is ``src.finishing.glmm``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse
from scipy.special import expit

from src.finishing import glmm
from src.finishing.model import SCODE

S2_INIT = 0.02


@dataclass
class Layout:
    cells: dict[int, int]
    goalie: dict[int, int]
    goalie_season: dict[int, int]
    team: dict[int, int]
    n_cols: int
    blocks: dict[str, tuple[int, int]]
    groups: dict[str, np.ndarray]
    persistent: bool
    season_dev: bool
    team_effect: bool
    fringe: bool = False


def _keys(df: pl.DataFrame, cell_season: pl.Expr | None = None) -> pl.DataFrame:
    cs = cell_season if cell_season is not None else pl.col("season")
    return df.with_columns(
        (cs * 10 + pl.col("strength_g").replace_strict(SCODE, default=9)).alias("k_cell"),
        (pl.col("goalie") * 10000 + pl.col("season")).alias("k_gs"),
        (pl.col("team_id") * 10000 + pl.col("season")).alias("k_ts"),
    )


def build_layout(
    df: pl.DataFrame, team_effect: bool, persistent: bool = True, season_dev: bool = True, fringe: bool = False
) -> Layout:
    df = _keys(df)
    col = 0
    blocks: dict[str, tuple[int, int]] = {}

    def take(name: str, n: int) -> range:
        nonlocal col
        r = range(col, col + n)
        blocks[name] = (col, col + n)
        col += n
        return r

    cells = dict(zip(sorted(df["k_cell"].unique().to_list()), take("cell", df["k_cell"].n_unique())))
    if fringe:
        take("f", 1)  # unpenalized replacement (fringe goalie-season) effect
    goalies = sorted(df["goalie"].unique().to_list())
    goalie = dict(zip(goalies, take("u", len(goalies)))) if persistent else {}
    gsk = sorted(df["k_gs"].unique().to_list())
    goalie_season = dict(zip(gsk, take("d", len(gsk)))) if season_dev else {}
    tsk = sorted(df["k_ts"].unique().to_list())
    team = dict(zip(tsk, take("t", len(tsk)))) if team_effect else {}
    groups = {
        g: np.array(list(m.values()), dtype=np.int64)
        for g, m in (("u", goalie), ("d", goalie_season), ("t", team))
        if m
    }
    return Layout(cells, goalie, goalie_season, team, col, blocks, groups, persistent, season_dev, team_effect, fringe)


def _map(s: pl.Series, mapping: dict[int, int]) -> np.ndarray:
    if not mapping:
        return np.full(len(s), -1, dtype=np.int64)
    return s.replace_strict(mapping, default=-1, return_dtype=pl.Int64).to_numpy()


def design(df: pl.DataFrame, lay: Layout, cell_season: pl.Expr | None = None) -> sparse.csr_matrix:
    """Sparse design; ids unseen when the layout was built get no column (effect 0). ``cell_season`` overrides the
    season used for the cell."""
    d = _keys(df, cell_season)
    n = d.height
    rows, cols = [], []
    fcol = lay.blocks["f"][0] if lay.fringe else -1
    for idx in (
        _map(d["k_cell"], lay.cells),
        np.where(d["gfringe"].to_numpy(), fcol, -1) if lay.fringe else np.full(n, -1),
        _map(d["goalie"], lay.goalie) if lay.persistent else np.full(n, -1),
        _map(d["k_gs"], lay.goalie_season) if lay.season_dev else np.full(n, -1),
        _map(d["k_ts"], lay.team) if lay.team_effect else np.full(n, -1),
    ):
        m = idx >= 0
        rows.append(np.flatnonzero(m))
        cols.append(idx[m])
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, lay.n_cols))


def fit(
    df: pl.DataFrame,
    lay: Layout,
    s2: dict[str, float] | None = None,
    em_iters: int = 0,
    em_tol_sd: float = 1e-4,
    beta0: np.ndarray | None = None,
):
    """Fit at fixed variance components ``s2`` or estimate them by EM. Returns (mode, s2, trace)."""
    X = design(df, lay)
    y = df["goal"].to_numpy().astype(float)
    off = df["offset"].to_numpy()
    s2_0 = {g: S2_INIT for g in lay.groups} | (s2 or {})
    if em_iters > 0:
        s2_hat, mode, trace = glmm.em_variance(X, y, off, lay.groups, s2_0, em_iters, em_tol_sd, beta0)
        return mode, s2_hat, trace
    mode = glmm.newton(X, y, off, glmm.precision_vector(lay.n_cols, lay.groups, s2_0), beta0)
    return mode, s2_0, []


def shot_probs(
    df: pl.DataFrame, lay: Layout, beta: np.ndarray, cell_season: pl.Expr | None = None
) -> dict[str, np.ndarray]:
    """Per-shot goal probabilities: ``base`` (offset + cell + team, an average goalie), ``rep`` (base + the replacement
    effect), ``v`` (base + persistent, the forecast used everywhere), ``with_status`` (v + own fringe status; tried and
    reported),
    ``full`` (base + own fringe effect + persistent + season deviation), ``cell_only`` (offset + cell)."""
    X = design(df, lay, cell_season)
    off = df["offset"].to_numpy()

    def part(name: str) -> np.ndarray:
        lo, hi = lay.blocks.get(name, (0, 0))
        return np.asarray(X[:, lo:hi] @ beta[lo:hi]).ravel() if hi > lo else np.zeros(df.height)

    cell, own_f, u, d, t = part("cell"), part("f"), part("u"), part("d"), part("t")
    f_rep = float(beta[lay.blocks["f"][0]]) if lay.fringe else 0.0
    return {
        "cell_only": expit(off + cell),
        "base": expit(off + cell + t),
        "rep": expit(off + cell + t + f_rep),
        "v": expit(off + cell + t + u),
        "with_status": expit(off + cell + t + own_f + u),
        "full": expit(off + cell + t + own_f + u + d),
    }


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))
