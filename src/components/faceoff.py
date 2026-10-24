"""Faceoff skill (FO), methodology §4.

Value of a draw: the ledger books the start value of a faceoff-started possession for the winner. With V(dot) the mean
cross-fitted start value of 5v5 faceoff-started possessions at a dot (winner's frame), winning versus losing a draw at a
location is worth S = V(dot, own frame) + V(dot, opponent's frame). Taker skill is a hierarchical logistic model fitted
with
the finishing model's penalized Laplace machinery; player value = sum over draws of S x (P(win) - P(win at the reference
skill)).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
from scipy import sparse
from scipy.special import expit

from src.components import common
from src.config import load_config
from src.finishing import glmm

log = logging.getLogger(__name__)
DOTS = (
    [(0.0, 0.0)]
    + [(sx * 20.0, sy * 22.0) for sx in (-1, 1) for sy in (-1, 1)]
    + [(sx * 69.0, sy * 22.0) for sx in (-1, 1) for sy in (-1, 1)]
)
DOT_ARR = np.array(DOTS)
S2_INIT = 0.02


def snap(x: np.ndarray, y: np.ndarray, tol: float = 8.0) -> np.ndarray:
    """Index of the nearest faceoff dot, or -1 if farther than ``tol`` feet."""
    d = np.sqrt((x[:, None] - DOT_ARR[None, :, 0]) ** 2 + (y[:, None] - DOT_ARR[None, :, 1]) ** 2)
    k = d.argmin(1)
    return np.where(d[np.arange(len(x)), k] <= tol, k, -1)


def dot_values(proc) -> np.ndarray:
    """V(dot): mean start value of 5v5 faceoff-started possessions by dot, in the winner's frame."""
    o = pl.read_parquet(proc / "obs_valued.parquet", columns=["obs_kind", "role", "strength", "x", "y", "v"]).filter(
        (pl.col("obs_kind") == "faceoff")
        & (pl.col("role") == "start")
        & (pl.col("strength") == "5v5")
        & pl.col("x").is_not_null()
    )
    k = snap(o["x"].to_numpy(), o["y"].to_numpy())
    v = o["v"].to_numpy()
    return np.array([v[k == i].mean() if (k == i).any() else np.nan for i in range(len(DOTS))])


def draws(proc) -> pl.DataFrame:
    """One row per 5v5 draw with both takers known, oriented to the home team."""
    ev = pl.read_parquet(
        proc / "events.parquet",
        columns=[
            "game_id",
            "event_idx",
            "season",
            "type",
            "team_id",
            "home_id",
            "player1",
            "player2",
            "x",
            "y",
            "strength",
        ],
    ).filter(
        (pl.col("type") == "faceoff")
        & (pl.col("strength") == "5v5")
        & pl.col("player1").is_not_null()
        & pl.col("player2").is_not_null()
        & pl.col("x").is_not_null()
    )
    pl_ = pl.read_parquet(proc / "players.parquet").select("player_id", "shoots", "position")
    win_home = (ev["team_id"] == ev["home_id"]).to_numpy()
    sgn = np.where(win_home, 1.0, -1.0)
    xh, yh = ev["x"].to_numpy() * sgn, ev["y"].to_numpy() * sgn
    dot_h = snap(xh, yh)
    dot_a = snap(-xh, -yh)
    v = dot_values(proc)
    swing = np.where((dot_h >= 0) & (dot_a >= 0), v[np.maximum(dot_h, 0)] + v[np.maximum(dot_a, 0)], np.nan)
    d = (
        ev.select("game_id", "event_idx", "season")
        .with_columns(
            pl.Series("home_taker", np.where(win_home, ev["player1"].to_numpy(), ev["player2"].to_numpy())),
            pl.Series("away_taker", np.where(win_home, ev["player2"].to_numpy(), ev["player1"].to_numpy())),
            pl.Series("y", win_home.astype(np.int8)),
            pl.Series("dot", dot_h),
            pl.Series("yh", yh),
            pl.Series("S", swing),
        )
        .filter(pl.col("dot") >= 0)
    )
    hand = {"L": 1.0, "R": -1.0}
    hmap = dict(zip(pl_["player_id"].to_list(), [hand.get(s, 0.0) for s in pl_["shoots"].to_list()]))
    sh = np.array([hmap.get(t, 0.0) for t in d["home_taker"].to_list()])
    sa = np.array([hmap.get(t, 0.0) for t in d["away_taker"].to_list()])
    ys = np.sign(d["yh"].to_numpy())
    known = set(pl_["player_id"].to_list())
    return d.with_columns(pl.Series("side", sh * ys - sa * (-ys))).filter(
        pl.col("home_taker").is_in(list(known)) & pl.col("away_taker").is_in(list(known)) & pl.col("S").is_not_null()
    )


def taker_meta(proc) -> pl.DataFrame:
    """player_id, season, fringe (same-season status) and the variance group (C vs other) of every taker."""
    b = common.base(proc).select("player_id", "season", "fringe")
    pos = pl.read_parquet(proc / "players.parquet").select(
        "player_id", pl.when(pl.col("position") == "C").then(pl.lit("C")).otherwise(pl.lit("O")).alias("grp")
    )
    return b.join(pos, on="player_id", how="left").with_columns(pl.col("grp").fill_null("O"))


def _fr(d: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    f = meta.select(pl.col("player_id"), "season", pl.col("fringe").cast(pl.Float64).alias("fr"))
    for side in ("home", "away"):
        d = d.join(
            f.rename({"player_id": f"{side}_taker", "fr": f"fr_{side}"}), on=[f"{side}_taker", "season"], how="left"
        ).with_columns(pl.col(f"fr_{side}").fill_null(1.0))
    return d.with_columns((pl.col("fr_home") - pl.col("fr_away")).alias("frd"))


class Layout:
    def __init__(self, d: pl.DataFrame, grp: dict[int, str], fringe: bool, season_dev: bool = True):
        self.fringe, self.season_dev = fringe, season_dev
        col = 0
        self.dots = {i: col + i for i in range(len(DOTS))}
        col += len(DOTS)
        self.side_col = col
        col += 1
        self.fr_col = col if fringe else None
        col += 1 if fringe else 0
        takers = sorted(set(d["home_taker"].to_list()) | set(d["away_taker"].to_list()))
        self.u = {t: col + i for i, t in enumerate(takers)}
        col += len(takers)
        self.d = {}
        if season_dev:
            keys = sorted(
                set((d["home_taker"] * 10000 + d["season"]).to_list())
                | set((d["away_taker"] * 10000 + d["season"]).to_list())
            )
            self.d = {k: col + i for i, k in enumerate(keys)}
            col += len(keys)
        self.n = col
        self.groups = {}
        for g in ("C", "O"):
            self.groups[f"u_{g}"] = np.array([c for t, c in self.u.items() if grp.get(t, "O") == g], dtype=np.int64)
            if season_dev:
                self.groups[f"d_{g}"] = np.array(
                    [c for k, c in self.d.items() if grp.get(k // 10000, "O") == g], dtype=np.int64
                )
        self.groups = {k: v for k, v in self.groups.items() if len(v)}


def design(d: pl.DataFrame, lay: Layout) -> sparse.csr_matrix:
    n = d.height
    ent: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (np.arange(n), np.array([lay.dots[i] for i in d["dot"].to_list()]), np.ones(n)),
        (np.arange(n), np.full(n, lay.side_col), d["side"].to_numpy()),
    ]
    if lay.fringe:
        ent.append((np.arange(n), np.full(n, lay.fr_col), d["frd"].to_numpy()))
    for side, sgn in (("home", 1.0), ("away", -1.0)):
        tk = d[f"{side}_taker"].to_numpy()
        keys = [(lay.u, tk)] + ([(lay.d, tk * 10000 + d["season"].to_numpy())] if lay.season_dev else [])
        for mapping, key in keys:
            cols = np.array([mapping.get(x, -1) for x in key])
            m = cols >= 0
            ent.append((np.flatnonzero(m), cols[m], np.full(m.sum(), sgn)))
    r, c, v = (np.concatenate([e[i] for e in ent]) for i in range(3))
    return sparse.csr_matrix((v, (r, c)), shape=(n, lay.n))


def fit(d: pl.DataFrame, grp, fringe: bool, s2=None, em_iters: int = 0, em_tol: float = 1e-4):
    lay = Layout(d, grp, fringe)
    X, y, off = design(d, lay), d["y"].to_numpy().astype(float), np.zeros(d.height)
    s2_0 = {g: S2_INIT for g in lay.groups} | (s2 or {})
    if em_iters > 0:
        s2_hat, mode, trace = glmm.em_variance(X, y, off, lay.groups, s2_0, em_iters, em_tol)
        return lay, mode, s2_hat, trace
    return lay, glmm.newton(X, y, off, glmm.precision_vector(lay.n, lay.groups, s2_0)), s2_0, []


def eta_parts(lay: Layout, beta: np.ndarray, d: pl.DataFrame) -> dict[str, np.ndarray]:
    """Linear predictor pieces for the home taker: fixed part (dot + side + fringe) and each side's talent (u +
    delta)."""
    X = design(d, lay)
    fixed_cols = np.r_[0 : lay.side_col + 1, [lay.fr_col] if lay.fringe else []].astype(int)
    fixed = np.asarray(X[:, fixed_cols] @ beta[fixed_cols]).ravel()
    out = {"fixed": fixed}
    for side in ("home", "away"):
        tk = d[f"{side}_taker"].to_numpy()
        t = np.array([beta[lay.u[x]] if x in lay.u else 0.0 for x in tk])
        if lay.season_dev:
            k = tk * 10000 + d["season"].to_numpy()
            t = t + np.array([beta[lay.d[x]] if x in lay.d else 0.0 for x in k])
        out[f"t_{side}"] = t
    return out


def _ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def forward_eval(d: pl.DataFrame, grp, s2, folds: list[int], min_draws: int, seed: int) -> dict:
    """Fit on seasons < fold (no fringe block), predict the fold; compare M0 / Mraw / Mshr (methodology §7)."""
    L = {m: [] for m in ("M0", "Mraw", "Mshr")}
    slopes = {"Mraw": ([], []), "Mshr": ([], [])}
    per_fold = {}
    for target in folds:
        tr, te = d.filter(pl.col("season") < target), d.filter(pl.col("season") == target)
        lay, mode, _, _ = fit(tr, grp, False, s2=s2)
        pr = eta_parts(lay, mode.beta, te)
        y = te["y"].to_numpy().astype(float)
        # raw smoothed win log-odds from the training draws
        w = (
            pl.concat(
                [
                    tr.select(pl.col("home_taker").alias("t"), pl.col("y").alias("w")),
                    tr.select(pl.col("away_taker").alias("t"), (1 - pl.col("y")).alias("w")),
                ]
            )
            .group_by("t")
            .agg(pl.col("w").sum().alias("W"), pl.len().alias("n"))
            .with_columns(np.log((pl.col("W") + 1) / (pl.col("n") - pl.col("W") + 1)).alias("raw"))
        )
        rmap = dict(zip(w["t"].to_list(), w["raw"].to_list()))
        rh = np.array([rmap.get(t, 0.0) for t in te["home_taker"].to_list()])
        ra = np.array([rmap.get(t, 0.0) for t in te["away_taker"].to_list()])
        u_h = np.array([mode.beta[lay.u[t]] if t in lay.u else 0.0 for t in te["home_taker"].to_list()])
        u_a = np.array([mode.beta[lay.u[t]] if t in lay.u else 0.0 for t in te["away_taker"].to_list()])
        eta = {"M0": pr["fixed"], "Mraw": pr["fixed"] + rh - ra, "Mshr": pr["fixed"] + u_h - u_a}
        P = {k: expit(v) for k, v in eta.items()}
        per_fold[target] = {k: float(_ll(y, p).mean()) for k, p in P.items()}
        for k, p in P.items():
            L[k].append(_ll(y, p))
        for m in slopes:
            for side, ids in (("home", te["home_taker"].to_numpy()), ("away", te["away_taker"].to_numpy())):
                win = y if side == "home" else 1 - y
                p_m = P[m] if side == "home" else 1 - P[m]
                p_0 = P["M0"] if side == "home" else 1 - P["M0"]
                slopes[m][0].append(pl.DataFrame({"t": ids, "pred": p_m - p_0, "real": win - p_0, "season": target}))
        log.info("faceoff fold %d: %s", target, {k: round(v, 5) for k, v in per_fold[target].items()})
    Lc = {k: np.concatenate(v) for k, v in L.items()}
    out = {
        "per_fold_logloss": per_fold,
        "pooled_logloss": {k: float(v.mean()) for k, v in Lc.items()},
        "n_draws": int(len(Lc["M0"])),
    }
    out["paired"] = {
        "Mshr_vs_M0": common.paired_z(Lc["M0"] - Lc["Mshr"]),
        "Mshr_vs_Mraw": common.paired_z(Lc["Mraw"] - Lc["Mshr"]),
        "Mraw_vs_M0": common.paired_z(Lc["M0"] - Lc["Mraw"]),
    }
    out["slope"] = {}
    rng = np.random.default_rng(seed)
    for m, (frames, _) in slopes.items():
        g = (
            pl.concat(frames)
            .group_by("t", "season")
            .agg(pl.len().alias("n"), pl.col("pred").sum(), pl.col("real").sum())
            .filter(pl.col("n") >= min_draws)
        )
        pr, re = g["pred"].to_numpy(), g["real"].to_numpy()
        s = float((pr * re).sum() / (pr**2).sum())
        bs = [
            float((pr[k] * re[k]).sum() / (pr[k] ** 2).sum())
            for k in (rng.integers(0, len(pr), len(pr)) for _ in range(400))
        ]
        out["slope"][m] = {
            "slope": s,
            "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)),
            "corr": float(np.corrcoef(pr, re)[0, 1]),
            "taker_seasons": int(len(pr)),
        }
    return out


def effects(lay: Layout, mode, s2, d: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    """Per taker-season value in goals above the league average and above the fringe pool, with delta-method SD."""
    beta = mode.beta
    pr = eta_parts(lay, beta, d)
    eta = pr["fixed"] + pr["t_home"] - pr["t_away"]
    f = float(beta[lay.fr_col]) if lay.fringe else 0.0
    rows = []
    for side, sgn in (("home", 1.0), ("away", -1.0)):
        fr = d[f"fr_{side}"].to_numpy()
        T = pr[f"t_{side}"] + f * fr
        rows.append(
            pl.DataFrame(
                {
                    "player_id": d[f"{side}_taker"],
                    "season": d["season"],
                    "S": d["S"],
                    "eta": sgn * eta,
                    "T": T,
                    "fr": fr,
                    "win": d["y"].to_numpy() if side == "home" else 1 - d["y"].to_numpy(),
                }
            )
        )
    r = pl.concat(rows)
    rho_avg = float(r["T"].mean())
    rho_rep = float(r.filter(pl.col("fr") > 0.5)["T"].mean())
    e, T, S = r["eta"].to_numpy(), r["T"].to_numpy(), r["S"].to_numpy()
    p = expit(e)
    r = r.with_columns(
        pl.Series("v_avg", S * (p - expit(e - T + rho_avg))),
        pl.Series("v_rep", S * (p - expit(e - T + rho_rep))),
        pl.Series("w", S * p * (1 - p)),
        pl.Series("p_avg", expit(e - T + rho_avg)),
    )
    a = r.group_by("player_id", "season").agg(
        pl.len().alias("fo_draws"),
        pl.col("win").sum().alias("fo_wins"),
        pl.col("v_avg").sum().alias("fo_gaa"),
        pl.col("v_rep").sum().alias("fo_gar"),
        pl.col("w").sum().alias("wsum"),
        pl.col("p_avg").sum().alias("exp_wins_avg"),
        pl.col("T").mean().alias("fo_shift"),
    )
    cov = mode.posterior_cov()
    sd = []
    for pid, s in zip(a["player_id"].to_list(), a["season"].to_list()):
        idx = [lay.u[pid]] if pid in lay.u else []
        k = pid * 10000 + s
        if k in lay.d:
            idx.append(lay.d[k])
        sd.append(float(np.sqrt(cov[np.ix_(idx, idx)].sum())) if idx else 0.0)
    a = (
        a.with_columns(pl.Series("sd_T", sd))
        .with_columns(
            (pl.col("sd_T") * pl.col("wsum")).alias("fo_gar_sd"),
            (pl.col("fo_wins") - pl.col("exp_wins_avg")).alias("fo_wins_above_avg_raw"),
        )
        .drop("wsum", "sd_T")
    )
    meta_g = meta.select("player_id", "season", "grp").unique(subset=["player_id", "season"])
    return a.join(meta_g, on=["player_id", "season"], how="left").with_columns(
        pl.lit(rho_avg).alias("rho_avg"), pl.lit(rho_rep).alias("rho_rep")
    )


def run() -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    fc, cc = cfg.finishing, cfg.components
    meta = taker_meta(proc)
    grp = dict(meta.unique(subset=["player_id"]).select("player_id", "grp").iter_rows())
    d = _fr(draws(proc), meta)
    diag: dict = {"n_draws": d.height}
    diag["draws_by_season"] = {
        int(r["season"]): r["len"] for r in d.group_by("season").agg(pl.len()).sort("season").to_dicts()
    }
    v = dot_values(proc)
    diag["dot_values"] = {f"{DOTS[i][0]:+.0f},{DOTS[i][1]:+.0f}": float(v[i]) for i in range(len(DOTS))}
    diag["gate_FO4_pass"] = bool(np.all(v > 0))
    diag["mean_swing"] = float(d["S"].mean())
    diag["home_win_rate"] = float(d["y"].mean())
    # variance components on the training seasons only, then forward evaluation
    tr = d.filter(pl.col("season").is_in(fc.sigma_seasons))
    lay0, mode0, s2, trace = fit(tr, grp, False, em_iters=fc.em_iters, em_tol=fc.em_tol_sd)
    diag["sd_train"] = {k: float(np.sqrt(x)) for k, x in s2.items()}
    ev = forward_eval(d, grp, s2, fc.test_folds, cc.min_draws_next, cfg.seed)
    diag["forward_test"] = ev
    diag["gate_FO1_pass"] = bool(ev["paired"]["Mshr_vs_M0"]["z"] > fc.se_threshold)
    diag["gate_FO2_pass"] = bool(ev["paired"]["Mshr_vs_Mraw"]["z"] > fc.se_threshold)
    diag["gate_FO3_pass"] = bool(0.7 <= ev["slope"]["Mshr"]["slope"] <= 1.4)
    # production fit
    lay, mode, s2p, trace = fit(d, grp, True, em_iters=fc.em_iters, em_tol=fc.em_tol_sd)
    diag["sd_production"] = {k: float(np.sqrt(x)) for k, x in s2p.items()}
    diag["side_coefficient"] = float(mode.beta[lay.side_col])
    diag["fringe_effect_logit"] = float(mode.beta[lay.fr_col])
    diag["em_converged"] = bool(trace[-1]["converged"])
    eff = effects(lay, mode, s2p, d, meta)
    eff.write_parquet(proc / "fo_player_season.parquet")
    q = eff.join(
        common.base(proc).select("player_id", "season", "toi_ev", "pos_group"), on=["player_id", "season"], how="left"
    )
    diag["sd_gar_takers_300plus"] = float(q.filter(pl.col("fo_draws") >= 300)["fo_gar"].std())
    diag["n_takers_300plus_draws"] = int(q.filter(pl.col("fo_draws") >= 300).height)
    (met / "fo_component.json").write_text(json.dumps(diag, indent=2, default=float))
    log.info(
        "FO: swing %.4f; forward z vs none %.1f vs raw %.1f; slope %.2f; SD gar (300+ draws) %.2f",
        diag["mean_swing"],
        ev["paired"]["Mshr_vs_M0"]["z"],
        ev["paired"]["Mshr_vs_Mraw"]["z"],
        ev["slope"]["Mshr"]["slope"],
        diag["sd_gar_takers_300plus"],
    )
    return eff


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
