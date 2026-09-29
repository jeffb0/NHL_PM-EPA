"""Finishing model layout, fitting and per-player-season effects (methodology §5).

    logit P(goal_i) = offset_i + cell[season, strength] + f_pos * fringe + gamma . Z_p
                      + u_p + delta_{p,season} (+ w_{p,type}) + v_g + eps_{g,season}

``Spec`` switches blocks on and off so one implementation serves the production model, the forward-
evaluation folds (no fringe block: replacement status is a same-season label) and the single-season
independent fits used for split-half and year-over-year diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import sparse
from scipy.special import expit

from src.finishing import glmm
from src.finishing.prior import PriorSpec

log = logging.getLogger(__name__)

SCODE = {"5v5": 0, "PP": 1, "PK": 2, "4v4_3v3": 3, "pulled": 4}
TCODE = {"slap": 1, "back": 2, "tip": 3}
S2_INIT = 0.02


@dataclass(frozen=True)
class Spec:
    fringe: bool = True
    season_dev: bool = True
    goalie_season: bool = True
    types: bool = False

    @staticmethod
    def A(fringe: bool = True) -> Spec:
        return Spec(fringe=fringe)

    @staticmethod
    def B(fringe: bool = True) -> Spec:
        return Spec(fringe=fringe, types=True)

    @staticmethod
    def single() -> Spec:
        return Spec(fringe=False, season_dev=False, goalie_season=False, types=False)


@dataclass
class Layout:
    spec: Spec
    cells: dict[int, int]
    fringe: dict[str, int]
    gamma_names: list[str]
    gamma_cols: list[int]
    gamma_table: pl.DataFrame | None
    shooter: dict[int, int]
    shooter_season: dict[int, int]
    goalie: dict[int, int]
    goalie_season: dict[int, int]
    wtype: dict[int, int]
    n_cols: int
    blocks: dict[str, tuple[int, int]]
    groups: dict[str, np.ndarray]
    shooter_pos: dict[int, str] = field(default_factory=dict)


def _map(s: pl.Series, mapping: dict[int, int]) -> np.ndarray:
    if not mapping:
        return np.full(len(s), -1, dtype=np.int64)
    return s.fill_null(-1).replace_strict(mapping, default=-1, return_dtype=pl.Int64).to_numpy()


def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("season") * 10 + pl.col("strength_g").replace_strict(SCODE, default=9)).alias("k_cell"),
        (pl.col("shooter") * 10000 + pl.col("season")).alias("k_ss"),
        (pl.col("goalie").fill_null(-1) * 10000 + pl.col("season")).alias("k_gs"),
        (pl.col("shooter") * 10 + pl.col("tg").replace_strict(TCODE, default=0)).alias("k_w"),
        pl.col("tg").replace_strict(TCODE, default=0).alias("tcode"),
    )


def build_layout(df: pl.DataFrame, spec: Spec, prior: PriorSpec) -> Layout:
    df = add_keys(df)
    col = 0
    blocks: dict[str, tuple[int, int]] = {}

    def take(name: str, n: int) -> range:
        nonlocal col
        r = range(col, col + n)
        blocks[name] = (col, col + n)
        col += n
        return r

    ck = sorted(df["k_cell"].unique().to_list())
    cells = dict(zip(ck, take("cell", len(ck))))
    fringe = dict(zip(["F", "D"], take("fringe", 2))) if spec.fringe else {}
    players = sorted(df["shooter"].unique().to_list())
    Z, zn = prior.covariates(players)
    gcols = list(take("gamma", len(zn)))
    gtab = pl.DataFrame({"shooter": pl.Series(players, dtype=pl.Int64), **{n: Z[:, j] for j, n in enumerate(zn)}}) if zn else None
    shooter = dict(zip(players, take("u", len(players))))
    ssk = sorted(df["k_ss"].unique().to_list()) if spec.season_dev else []
    shooter_season = dict(zip(ssk, take("d", len(ssk)))) if spec.season_dev else {}
    goalies = sorted(g for g in df["goalie"].drop_nulls().unique().to_list())
    goalie = dict(zip(goalies, take("v", len(goalies))))
    gsk = sorted(df.filter(pl.col("goalie").is_not_null())["k_gs"].unique().to_list()) if spec.goalie_season else []
    goalie_season = dict(zip(gsk, take("e", len(gsk)))) if spec.goalie_season else {}
    wk = sorted(df.filter(pl.col("tcode") > 0)["k_w"].unique().to_list()) if spec.types else []
    wtype = dict(zip(wk, take("w", len(wk)))) if spec.types else {}
    pos = dict(df.select("shooter", "pos_group").unique(subset=["shooter"]).iter_rows())
    groups: dict[str, np.ndarray] = {}
    for g in ("F", "D"):
        groups[f"u_{g}"] = np.array([c for p, c in shooter.items() if pos[p] == g], dtype=np.int64)
        if spec.season_dev:
            groups[f"d_{g}"] = np.array([c for k, c in shooter_season.items() if pos[k // 10000] == g], dtype=np.int64)
    groups["v"] = np.array(list(goalie.values()), dtype=np.int64)
    if spec.goalie_season:
        groups["e"] = np.array(list(goalie_season.values()), dtype=np.int64)
    if spec.types:
        groups["w"] = np.array(list(wtype.values()), dtype=np.int64)
    groups = {g: idx for g, idx in groups.items() if len(idx)}
    return Layout(spec, cells, fringe, zn, gcols, gtab, shooter, shooter_season, goalie, goalie_season, wtype, col, blocks, groups, pos)


def design(df: pl.DataFrame, lay: Layout) -> sparse.csr_matrix:
    """Sparse design for ``df`` under a layout; ids unseen when the layout was built get no column (effect 0)."""
    df = add_keys(df)
    n = df.height
    rows, cols, vals = [], [], []

    def add(idx: np.ndarray, v: np.ndarray | None = None) -> None:
        m = idx >= 0
        rows.append(np.flatnonzero(m))
        cols.append(idx[m])
        vals.append(np.ones(m.sum()) if v is None else v[m])

    add(_map(df["k_cell"], lay.cells))
    if lay.fringe:
        f = df["fringe"].to_numpy().astype(bool)
        add(np.where(f, np.where(df["pos_group"].to_numpy() == "D", lay.fringe["D"], lay.fringe["F"]), -1))
    if lay.gamma_names:
        z = df.select(pl.col("shooter").cast(pl.Int64)).join(lay.gamma_table, on="shooter", how="left").fill_null(0.0)
        for j, c in enumerate(lay.gamma_cols):
            v = z[lay.gamma_names[j]].to_numpy()
            add(np.full(n, c, dtype=np.int64), v)
    add(_map(df["shooter"], lay.shooter))
    if lay.spec.season_dev:
        add(_map(df["k_ss"], lay.shooter_season))
    add(_map(df["goalie"], lay.goalie))
    if lay.spec.goalie_season:
        add(_map(df["k_gs"], lay.goalie_season))
    if lay.spec.types:
        add(np.where(df["tcode"].to_numpy() > 0, _map(df["k_w"], lay.wtype), -1))
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    return sparse.csr_matrix((v, (r, c)), shape=(n, lay.n_cols))


def with_offset(df: pl.DataFrame, xg_col: str = "xg_f") -> pl.DataFrame:
    p = np.clip(df[xg_col].to_numpy(), 1e-5, 1 - 1e-5)
    return df.with_columns(pl.Series("offset", np.log(p / (1 - p))))


@dataclass
class FinFit:
    layout: Layout
    mode: glmm.Mode
    s2: dict[str, float]
    trace: list[dict] = field(default_factory=list)

    @property
    def beta(self) -> np.ndarray:
        return self.mode.beta

    def parts(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Linear-predictor contribution of each block, per row of ``df``."""
        X = design(df, self.layout)
        return {b: np.asarray(X[:, lo:hi] @ self.beta[lo:hi]).ravel() for b, (lo, hi) in self.layout.blocks.items() if hi > lo}


def fit_finishing(df: pl.DataFrame, spec: Spec, prior: PriorSpec | None = None, s2: dict[str, float] | None = None,
                  em_iters: int = 0, em_tol_sd: float = 1e-4, beta0: np.ndarray | None = None, layout: Layout | None = None) -> FinFit:
    """Fit at fixed variance components (``s2``) or estimate them by EM (``em_iters`` > 0)."""
    prior = prior or PriorSpec()
    lay = layout or build_layout(df, spec, prior)
    X = design(df, lay)
    y = df["goal"].to_numpy().astype(float)
    off = df["offset"].to_numpy()
    s2_0 = {g: S2_INIT for g in lay.groups} | (s2 or {})
    if em_iters > 0:
        s2_hat, mode, trace = glmm.em_variance(X, y, off, lay.groups, s2_0, em_iters, em_tol_sd, beta0)
        return FinFit(lay, mode, s2_hat, trace)
    mode = glmm.newton(X, y, off, glmm.precision_vector(lay.n_cols, lay.groups, s2_0), beta0)
    return FinFit(lay, mode, s2_0)


def reference_shifts(T: np.ndarray, df: pl.DataFrame, fringe: np.ndarray) -> tuple[dict[int, float], dict[tuple[str, int], float]]:
    """Season league-average shift and per (position, season) fringe-pool shift, over 5v5 shots."""
    d = df.select("season", "pos_group", "strength_g").with_columns(pl.Series("T", T), pl.Series("fr", fringe))
    d5 = d.filter(pl.col("strength_g") == "5v5")
    avg = {s: float(v) for s, v in d5.group_by("season").agg(pl.col("T").mean()).iter_rows()}
    rep = {(g, s): float(v) for g, s, v in d5.filter(pl.col("fr")).group_by("pos_group", "season").agg(pl.col("T").mean()).iter_rows()}
    return avg, rep


def player_effects(fit: FinFit, df: pl.DataFrame, draws: int, seed: int, cov: np.ndarray | None = None) -> pl.DataFrame:
    """Per player-season finishing value (goals) with posterior draws.

    Above average: reference = league-average shooter of that season (5v5 shot-weighted mean shift).
    Above replacement: reference = the fringe pool of the same position and season.
    Goals are computed shot by shot: sum of expit(l_i + T_i) - expit(l_i + T_ref), with l_i the
    offset + season/strength cell + goalie terms, and T_i the shooter's shift (+ fringe effect).
    """
    parts = fit.parts(df)
    p = lambda k: parts.get(k, np.zeros(df.height))  # noqa: E731
    l_i = df["offset"].to_numpy() + p("cell") + p("v") + p("e")
    fr_eff = p("fringe")
    shift = p("gamma") + p("u") + p("d") + p("w")
    T = shift + fr_eff
    fringe = df["fringe"].to_numpy().astype(bool) if "fringe" in df.columns else np.zeros(df.height, bool)
    avg, rep = reference_shifts(T, df, fringe)
    season = df["season"].to_numpy()
    posg = df["pos_group"].to_numpy()
    ref_avg = np.array([avg.get(s, 0.0) for s in season])
    ref_rep = np.array([rep.get((g, s), avg.get(s, 0.0)) for g, s in zip(posg, season)])
    base = expit(l_i)
    g_avg = expit(l_i + T) - expit(l_i + ref_avg)
    g_rep = expit(l_i + T) - expit(l_i + ref_rep)
    is5 = (df["strength_g"].to_numpy() == "5v5")
    tab = df.select("shooter", "season", "pos_group", "goal", "xg_f").with_columns(
        pl.Series("g_avg", g_avg), pl.Series("g_rep", g_rep), pl.Series("T", T), pl.Series("is5", is5), pl.Series("p_l", base))
    agg = []
    for tag, m in (("5v5", tab["is5"]), ("other", ~tab["is5"])):
        a = tab.filter(m).group_by("shooter", "season", "pos_group").agg(
            pl.len().alias(f"attempts_{tag}"), pl.col("goal").sum().alias(f"goals_{tag}"), pl.col("xg_f").sum().alias(f"xg_f_{tag}"),
            pl.col("g_avg").sum().alias(f"fin_gaa_{tag}"), pl.col("g_rep").sum().alias(f"fin_gar_{tag}"), pl.col("T").mean().alias(f"shift_{tag}"))
        agg.append(a)
    out = agg[0].join(agg[1], on=["shooter", "season", "pos_group"], how="full", coalesce=True)
    out = out.with_columns(*[pl.col(c).fill_null(0) for c in out.columns if c.startswith(("attempts", "goals", "xg_f", "fin_"))])
    if draws > 0 and cov is not None:
        sd = _draw_sd(fit, df, l_i, fr_eff, p("gamma"), ref_avg, ref_rep, is5, draws, seed, cov)
        out = out.join(sd, on=["shooter", "season"], how="left")
    return out.rename({"shooter": "player_id"})


def _draw_sd(fit: FinFit, df: pl.DataFrame, l_i, fr_eff, gam, ref_avg, ref_rep, is5, draws, seed, cov) -> pl.DataFrame:
    """Posterior sd and 95% bounds of 5v5 goals above average / replacement, per player-season (5v5 shots)."""
    lay = fit.layout
    rng = np.random.default_rng(seed)
    dk = add_keys(df)
    shooter = dk["shooter"].to_numpy()
    season = dk["season"].to_numpy()
    u_col = _map(dk["shooter"], lay.shooter)
    d_col = _map(dk["k_ss"], lay.shooter_season) if lay.spec.season_dev else np.full(df.height, -1)
    w_col = np.where(dk["tcode"].to_numpy() > 0, _map(dk["k_w"], lay.wtype), -1) if lay.spec.types else np.full(df.height, -1)
    order = np.argsort(shooter, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(shooter[order]) != 0, True])
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ix = order[a:b]
        ix = ix[is5[ix]]
        if len(ix) == 0:
            continue
        J = np.unique(np.concatenate([u_col[ix], d_col[ix], w_col[ix]]))
        J = J[J >= 0]
        pos = {j: k for k, j in enumerate(J)}
        mu, S = fit.beta[J], cov[np.ix_(J, J)]
        L = np.linalg.cholesky(S + 1e-12 * np.eye(len(J)))
        D = mu + rng.standard_normal((draws, len(J))) @ L.T
        T = np.zeros((draws, len(ix))) + (gam[ix] + fr_eff[ix])[None, :]
        for cols_ in (u_col, d_col, w_col):
            c = cols_[ix]
            has = c >= 0
            kk = np.array([pos[j] for j in c[has]], dtype=np.int64)
            T[:, has] += D[:, kk]
        e_base = expit(l_i[ix] + T)
        g_avg = e_base - expit(l_i[ix] + ref_avg[ix])[None, :]
        g_rep = e_base - expit(l_i[ix] + ref_rep[ix])[None, :]
        for s in np.unique(season[ix]):
            m = season[ix] == s
            ga, gr = g_avg[:, m].sum(1), g_rep[:, m].sum(1)
            rows.append((int(shooter[ix[0]]), int(s), float(ga.std()), *np.quantile(ga, [0.025, 0.975]), float(gr.std()), *np.quantile(gr, [0.025, 0.975])))
    return pl.DataFrame(rows, schema=["shooter", "season", "fin_gaa_sd", "fin_gaa_lo", "fin_gaa_hi", "fin_gar_sd", "fin_gar_lo", "fin_gar_hi"], orient="row")


def calibrate_cell_shift(eta: np.ndarray, y: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Per-cell additive logit shift making each cell's predicted goals equal its observed goals."""
    out = np.zeros(len(eta))
    for c in np.unique(cell):
        m = cell == c
        lo, hi = -3.0, 3.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if expit(eta[m] + mid).sum() < y[m].sum():
                lo = mid
            else:
                hi = mid
        out[m] = 0.5 * (lo + hi)
    return out
