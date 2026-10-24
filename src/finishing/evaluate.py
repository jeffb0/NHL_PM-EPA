"""Pre-registered evaluation of the finishing model (methodology §6-§7).

* ``forward_eval``   fit on seasons <= Y, predict Y+1 (variance components fixed from the training seasons);
                     compare M0 / Mg / Mraw / Mshr(A) / MshrB on held-out log-loss with paired SEs (D-type, F1-F3).
* ``single_season``  independent single-season fits -> honest YOY and split-half (a pooled model's persistent
                     effect is shared across seasons, so a naive YOY of it would be circular).
* ``stabilization``  attempts needed for reliability 0.5, analytic and empirical.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.special import expit

from src.finishing import model
from src.finishing.model import FinFit, Spec

log = logging.getLogger(__name__)

MODEL_ORDER = ["M0", "Mg", "Mraw", "MshrA", "MshrB"]


def _ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def raw_shift(train: pl.DataFrame) -> pl.DataFrame:
    """Minimally regularized raw finishing: log((G + 1) / (xG + 1)) over the training seasons."""
    return (
        train.group_by("shooter")
        .agg(pl.col("goal").sum().alias("G"), pl.col("xg_f").sum().alias("X"))
        .select("shooter", (((pl.col("G") + 1) / (pl.col("X") + 1)).log()).alias("raw"))
    )


def fold_probabilities(train: pl.DataFrame, test: pl.DataFrame, fits: dict[str, FinFit]) -> dict[str, np.ndarray]:
    """Held-out probabilities per model. Season/strength intercepts of the held-out season are set so that
    each strength cell's predicted goals equal its observed goals (identically for every model)."""
    y = test["goal"].to_numpy().astype(float)
    base = test["offset"].to_numpy()
    cell = test["strength_g"].replace_strict(model.SCODE, default=9).to_numpy()
    fa = fits["A"]
    pa = fa.parts(test)
    goalie = pa.get("v", np.zeros(test.height))
    raw = (
        test.select(pl.col("shooter")).join(raw_shift(train), on="shooter", how="left")["raw"].fill_null(0.0).to_numpy()
    )
    eta = {
        "M0": base,
        "Mg": base + goalie,
        "Mraw": base + goalie + raw,
        "MshrA": base + goalie + pa.get("gamma", 0) + pa["u"],
    }
    if "B" in fits:
        pb = fits["B"].parts(test)
        eta["MshrB"] = base + pb.get("v", np.zeros(test.height)) + pb.get("gamma", 0) + pb["u"] + pb.get("w", 0)
    return {k: expit(e + model.calibrate_cell_shift(e, y, cell)) for k, e in eta.items()}


def paired(l0: np.ndarray, l1: np.ndarray) -> dict:
    d = l0 - l1
    return {
        "mean_improvement": float(d.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "z": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
        "n": int(len(d)),
    }


def player_slope(
    test: pl.DataFrame, p_model: np.ndarray, p_ref: np.ndarray, min_attempts: int, reps: int = 400, seed: int = 0
) -> dict:
    """Calibration slope of realized next-season excess goals on predicted excess goals (through the origin)."""
    d = test.select("shooter", "goal").with_columns(pl.Series("pred", p_model - p_ref), pl.Series("ref", p_ref))
    g = d.group_by("shooter").agg(
        pl.len().alias("n"),
        pl.col("goal").sum().alias("goals"),
        pl.col("pred").sum().alias("pred"),
        pl.col("ref").sum().alias("ref"),
    )
    g = g.filter(pl.col("n") >= min_attempts).with_columns((pl.col("goals") - pl.col("ref")).alias("real"))
    return _slope_stats(g["pred"].to_numpy(), g["real"].to_numpy(), reps, seed)


def _slope_stats(pred: np.ndarray, real: np.ndarray, reps: int, seed: int) -> dict:
    slope = float((pred * real).sum() / (pred**2).sum())
    rng = np.random.default_rng(seed)
    n = len(pred)
    bs = [(pred[k] * real[k]).sum() / (pred[k] ** 2).sum() for k in (rng.integers(0, n, n) for _ in range(reps))]
    return {
        "slope": slope,
        "lo": float(np.quantile(bs, 0.025)),
        "hi": float(np.quantile(bs, 0.975)),
        "corr": float(np.corrcoef(pred, real)[0, 1]),
        "players": int(n),
    }


def forward_eval(
    df: pl.DataFrame,
    folds: list[int],
    prior,
    s2: dict[str, dict],
    prefit: dict[int, dict[str, FinFit]] | None,
    min_attempts: int,
    seed: int,
) -> dict:
    """Fit on seasons < fold, predict the fold. ``s2`` = fixed variance components per variant ('A', 'B')."""
    ll = {m: [] for m in MODEL_ORDER}
    per_fold, slopes = {}, {m: [] for m in ("Mraw", "MshrA", "MshrB")}
    for target in folds:
        train, test = df.filter(pl.col("season") < target), df.filter(pl.col("season") == target)
        fits = (prefit or {}).get(target)
        if fits is None:
            fits = {
                "A": model.fit_finishing(train, Spec.A(fringe=False), prior, s2=s2["A"]),
                "B": model.fit_finishing(train, Spec.B(fringe=False), prior, s2=s2["B"]),
            }
        P = fold_probabilities(train, test, fits)
        y = test["goal"].to_numpy().astype(float)
        per_fold[target] = {m: float(_ll(y, p).mean()) for m, p in P.items()}
        for m, p in P.items():
            ll[m].append(_ll(y, p))
        for m in slopes:
            if m in P:
                slopes[m].append((test, P[m], P["Mg"]))
        log.info("forward fold %d: %s", target, {k: round(v, 5) for k, v in per_fold[target].items()})
    L = {m: np.concatenate(v) for m, v in ll.items() if v}
    out = {
        "per_fold_logloss": per_fold,
        "pooled_logloss": {m: float(v.mean()) for m, v in L.items()},
        "n": int(len(L["M0"])),
    }
    out["paired"] = {
        f"{a}_vs_{b}": paired(L[b], L[a])
        for a, b in [
            ("Mg", "M0"),
            ("Mraw", "M0"),
            ("MshrA", "M0"),
            ("MshrA", "Mraw"),
            ("MshrA", "Mg"),
            ("MshrB", "M0"),
            ("MshrB", "Mraw"),
            ("MshrB", "MshrA"),
        ]
        if a in L and b in L
    }
    out["player_slope"] = {}
    for m, items in slopes.items():
        if not items:
            continue
        pred, real = [], []
        for test, p, pref in items:
            d = test.select("shooter", "goal").with_columns(pl.Series("pred", p - pref), pl.Series("ref", pref))
            g = d.group_by("shooter").agg(
                pl.len().alias("n"),
                pl.col("goal").sum().alias("goals"),
                pl.col("pred").sum().alias("pred"),
                pl.col("ref").sum().alias("ref"),
            )
            g = g.filter(pl.col("n") >= min_attempts)
            pred.append(g["pred"].to_numpy())
            real.append((g["goals"] - g["ref"]).to_numpy())
        out["player_slope"][m] = _slope_stats(np.concatenate(pred), np.concatenate(real), 400, seed)
    return out


# ------------------------------------------------------------------ single-season diagnostics
def single_season(df: pl.DataFrame, prior, em_iters: int, em_tol_sd: float, min_attempts: int) -> dict:
    """Independent single-season fits: per-season estimates, YOY and split-half of model vs raw finishing."""
    est, s2s, halves = [], {}, []
    for s in sorted(df["season"].unique().to_list()):
        d = df.filter(pl.col("season") == s)
        fit = model.fit_finishing(d, Spec.single(), prior, em_iters=em_iters, em_tol_sd=em_tol_sd)
        s2s[s] = {k: float(np.sqrt(v)) for k, v in fit.s2.items()}
        est.append(_season_table(d, fit, s))
        # split-half: odd / even game_id, variance components fixed from the full-season fit
        hs = []
        for par in (0, 1):
            dh = d.filter(pl.col("game_id") % 2 == par)
            fh = model.fit_finishing(dh, Spec.single(), prior, s2=fit.s2)
            hs.append(_season_table(dh, fh, s))
        j = hs[0].join(hs[1], on=["shooter", "season"], suffix="_b")
        tot = est[-1].select("shooter", pl.col("n").alias("n_all"))
        halves.append(j.join(tot, on="shooter"))
        log.info("single-season fit %d done", s)
    E = pl.concat(est)
    H = pl.concat(halves).filter(pl.col("n_all") >= min_attempts)
    out = {"single_season_sd": s2s}
    nxt = E.select(
        "shooter",
        (pl.col("season") - 1).alias("season"),
        pl.col("shr").alias("shr_n"),
        pl.col("raw_rate").alias("raw_n"),
        pl.col("raw_total").alias("tot_n"),
        pl.col("n").alias("n_n"),
    )
    Y = E.join(nxt, on=["shooter", "season"]).filter((pl.col("n") >= min_attempts) & (pl.col("n_n") >= min_attempts))
    out["yoy"] = {
        "n_pairs": int(Y.height),
        "model_shift": float(np.corrcoef(Y["shr"], Y["shr_n"])[0, 1]),
        "raw_rate": float(np.corrcoef(Y["raw_rate"], Y["raw_n"])[0, 1]),
        "raw_total_g_minus_xg": float(np.corrcoef(Y["raw_total"], Y["tot_n"])[0, 1]),
    }
    for g in ("F", "D"):
        Yg = Y.filter(pl.col("pos_group") == g)
        out["yoy"][g] = {
            "n_pairs": int(Yg.height),
            "model_shift": float(np.corrcoef(Yg["shr"], Yg["shr_n"])[0, 1]),
            "raw_rate": float(np.corrcoef(Yg["raw_rate"], Yg["raw_n"])[0, 1]),
        }
    r_m, r_r = float(np.corrcoef(H["shr"], H["shr_b"])[0, 1]), float(np.corrcoef(H["raw_rate"], H["raw_rate_b"])[0, 1])
    out["split_half"] = {
        "n_player_seasons": int(H.height),
        "model_r": r_m,
        "raw_r": r_r,
        "model_r_spearman_brown": 2 * r_m / (1 + r_m),
        "raw_r_spearman_brown": 2 * r_r / (1 + r_r),
    }
    return out


def _season_table(d: pl.DataFrame, fit: FinFit, season: int) -> pl.DataFrame:
    lay = fit.layout
    u = pl.DataFrame(
        {"shooter": list(lay.shooter), "shr": [float(fit.beta[c]) for c in lay.shooter.values()]},
        schema={"shooter": pl.Int64, "shr": pl.Float64},
    )
    raw = (
        d.group_by("shooter", "pos_group")
        .agg(pl.len().alias("n"), pl.col("goal").sum().alias("G"), pl.col("xg_f").sum().alias("X"))
        .with_columns(
            ((pl.col("G") - pl.col("X")) / pl.col("n")).alias("raw_rate"),
            (pl.col("G") - pl.col("X")).alias("raw_total"),
        )
    )
    return raw.join(u, on="shooter").with_columns(pl.lit(season, pl.Int64).alias("season"))


def stabilization(df: pl.DataFrame, single_sd: dict, persistent_sd: dict | None, seed: int = 0) -> dict:
    """Attempts for reliability 0.5: analytic (1 / (I sigma^2)) and empirical (Spearman-Brown on random splits)."""
    p = np.clip(df["xg_f"].to_numpy(), 1e-4, 1 - 1e-4)
    info = float(np.mean(p * (1 - p)))
    out = {"mean_bernoulli_information": info, "analytic": {}}
    for g in ("F", "D"):
        sds = [v[f"u_{g}"] for v in single_sd.values() if f"u_{g}" in v]
        if sds:
            s2 = float(np.mean(np.square(sds)))
            out["analytic"][f"single_season_{g}"] = {
                "sd_logit": float(np.sqrt(s2)),
                "attempts_for_r50": 1.0 / (info * s2),
            }
        if persistent_sd and f"u_{g}" in persistent_sd:
            s2 = persistent_sd[f"u_{g}"] ** 2 + persistent_sd.get(f"d_{g}", 0) ** 2
            out["analytic"][f"total_{g}"] = {"sd_logit": float(np.sqrt(s2)), "attempts_for_r50": 1.0 / (info * s2)}
    rng = np.random.default_rng(seed)
    emp = {}
    byp = {k[0]: (g["goal"].to_numpy(), g["xg_f"].to_numpy()) for k, g in df.group_by(["shooter"])}
    for n in (100, 200, 400, 800, 1600):
        pool = [v for v in byp.values() if len(v[0]) >= n]
        if len(pool) < 25:
            continue
        rs = []
        for _ in range(20):
            a, b = [], []
            for goals, xg in pool:
                ix = rng.permutation(len(goals))[:n]
                h1, h2 = ix[: n // 2], ix[n // 2 :]
                a.append((goals[h1].sum() - xg[h1].sum()) / len(h1))
                b.append((goals[h2].sum() - xg[h2].sum()) / len(h2))
            r = np.corrcoef(a, b)[0, 1]
            rs.append(2 * r / (1 + r))
        emp[n] = {"players": len(pool), "reliability": float(np.mean(rs))}
    out["empirical"] = emp
    ns = sorted(emp)
    xs, ys = np.log(ns), [emp[n]["reliability"] for n in ns]
    out["empirical_attempts_for_r50"] = float(np.exp(np.interp(0.5, ys, xs))) if min(ys) < 0.5 < max(ys) else None
    return out
