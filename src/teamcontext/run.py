"""Team context ("system / chemistry") analyses:  python -m src.teamcontext.run   (docs/teamcontext/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.projection import data as pd_
from src.projection import run as pr
from src.projection.model import Spec
from src.teamcontext.data import add_context
from src.wholemodel import build as wb
from src.wholemodel.run import paired

log = logging.getLogger("teamcontext")
COMPONENTS = {
    "total": "r_total",
    "possession": "r_pm",
    "defense": "r_def",
    "finishing": "r_fin",
    "turnover": "r_to",
    "penalties": "r_pen",
    "power play": "r_pp",
    "penalty kill": "r_pk",
}
MIN_BOTH = 300.0
CTX = ("ctx_dest", "ctx_origin", "mover")


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def transitions(p: pl.DataFrame) -> pl.DataFrame:
    """Consecutive-season pairs with >= 300 modeled minutes in both, with a mover flag; every component rate at t and t
    + 1."""
    a = p.select("player_id", "season", "pos_group", "team", "m", *COMPONENTS.values())
    b = p.select(
        "player_id",
        (pl.col("season") - 1).alias("season"),
        pl.col("team").alias("team_next"),
        pl.col("m").alias("m_next"),
        *[pl.col(c).alias(c + "_next") for c in COMPONENTS.values()],
    )
    j = a.join(b, on=["player_id", "season"]).filter((pl.col("m") >= MIN_BOTH) & (pl.col("m_next") >= MIN_BOTH))
    return j.with_columns((pl.col("team") != pl.col("team_next")).alias("mover"))


def _centered_slope(d: pl.DataFrame, xc: str, yc: str) -> tuple[float, float]:
    """Season x position centred, next-season-minute-weighted slope through the origin, and the Pearson correlation."""
    t = d.with_columns(
        (pl.col(xc) - pl.col(xc).mean().over("season", "pos_group")).alias("_x"),
        (pl.col(yc) - pl.col(yc).mean().over("season", "pos_group")).alias("_y"),
    )
    x, y, w = t["_x"].to_numpy(), t["_y"].to_numpy(), t["m_next"].to_numpy()
    return float((w * x * y).sum() / (w * x * x).sum()), float(np.corrcoef(x, y)[0, 1])


def analysis_a(p: pl.DataFrame, reps: int = 300, seed: int = 0) -> dict:
    tr = transitions(p)
    rng = np.random.default_rng(seed)
    ids = tr["player_id"].unique().to_numpy()
    out = {
        "n_transitions": tr.height,
        "n_movers": int(tr["mover"].sum()),
        "movers_by_season": {
            int(s): int(n) for s, n in tr.filter(pl.col("mover")).group_by("season").len().sort("season").iter_rows()
        },
    }
    for name, col in COMPONENTS.items():
        st, mv = tr.filter(~pl.col("mover")), tr.filter(pl.col("mover"))
        s_slope, s_r = _centered_slope(st, col, col + "_next")
        m_slope, m_r = _centered_slope(mv, col, col + "_next")
        boots = []
        for _ in range(reps):
            pick = pl.DataFrame({"player_id": rng.choice(ids, len(ids)).astype(np.int64)}).group_by("player_id").len()
            bt = tr.join(pick, on="player_id").with_columns(pl.col("m_next") * pl.col("len"))
            b_s, _ = _centered_slope(bt.filter(~pl.col("mover")), col, col + "_next")
            b_m, _ = _centered_slope(bt.filter(pl.col("mover")), col, col + "_next")
            boots.append((b_s, b_m, b_m / b_s if b_s else np.nan))
        bb = np.array(boots)
        out[name] = {
            "stayer_slope": s_slope,
            "stayer_r": s_r,
            "mover_slope": m_slope,
            "mover_r": m_r,
            "stayer_minus_mover": s_slope - m_slope,
            "stayer_minus_mover_ci95": [
                float(np.quantile(bb[:, 0] - bb[:, 1], 0.025)),
                float(np.quantile(bb[:, 0] - bb[:, 1], 0.975)),
            ],
            "share_that_travels": m_slope / s_slope,
            "share_that_travels_ci95": [float(np.nanquantile(bb[:, 2], 0.025)), float(np.nanquantile(bb[:, 2], 0.975))],
        }
    return out


def _wls_cluster(X: np.ndarray, y: np.ndarray, w: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(A, X.T @ (w * y))
    u = y - X @ beta
    meat = np.zeros_like(A)
    for k in np.unique(g):
        m = g == k
        s = X[m].T @ (w[m] * u[m])
        meat += np.outer(s, s)
    Ai = np.linalg.inv(A)
    return beta, np.sqrt(np.diag(Ai @ meat @ Ai))


def analysis_b(p2: pl.DataFrame, bio: pl.DataFrame) -> dict:
    """Next-season total rate on the shrunk history feature, age terms and the context features, for movers and stayers
    (target rows >= 300 minutes)."""
    d = pd_.design_frame(p2, bio, 1, floor=MIN_BOTH).filter(pl.col("no_history") < 0.5)
    from src.projection.model import age_basis

    res = {}
    groups = (
        ("movers", d.filter(pl.col("mover") > 0.5)),
        ("stayers", d.filter(pl.col("mover") < 0.5)),
        ("movers_F", d.filter((pl.col("mover") > 0.5) & (pl.col("pos_group") == "F"))),
        ("movers_D", d.filter((pl.col("mover") > 0.5) & (pl.col("pos_group") == "D"))),
    )
    for name, sub in groups:
        A = age_basis(sub["age"].to_numpy())
        cols = ["rt_total"] + (["ctx_dest", "ctx_origin"] if name.startswith("movers") else ["ctx_dest"])
        X = np.column_stack([np.ones(sub.height), *[sub[c].to_numpy() for c in cols], A])
        beta, se = _wls_cluster(X, sub["r_total"].to_numpy(), sub["m"].to_numpy(), sub["player_id"].to_numpy())
        res[name] = {
            "n": sub.height,
            **{
                c: {"coef": float(beta[1 + i]), "se": float(se[1 + i]), "t": float(beta[1 + i] / se[1 + i])}
                for i, c in enumerate(cols)
            },
        }
    return res


def analysis_c(proc, p2: pl.DataFrame, bio: pl.DataFrame) -> dict:
    base, ctx, ctx_gd = Spec(), Spec(extra=CTX), Spec(extra=(*CTX, "ctx_gd"))
    ft, fa = pd_.design_frame(p2, bio, 1, floor=pr.REVISED_FLOOR), pd_.design_frame(p2, bio, 1, targets_only=False)
    parts = []
    for T in pr.TEST_SEASONS:
        fb = (
            pr.forecast(fa, ft, 1, T, base, p2)
            .filter(pl.col("m") >= pr.EVAL_MIN)
            .select("player_id", "season", "m", "y", pl.col("proj").alias("base"))
        )
        fc = pr.forecast(fa, ft, 1, T, ctx, p2).select("player_id", "season", pl.col("proj").alias("ctx"))
        fg = pr.forecast(fa, ft, 1, T, ctx_gd, p2).select("player_id", "season", pl.col("proj").alias("ctx_gd"))
        parts.append(fb.join(fc, on=["player_id", "season"]).join(fg, on=["player_id", "season"]))
    e = pl.concat(parts).join(p2.select("player_id", "season", "mover"), on=["player_id", "season"])
    out = {"n_rows": e.height, "n_mover_rows": int((e["mover"] > 0.5).sum())}
    for name, sub in (
        ("all", e),
        ("movers", e.filter(pl.col("mover") > 0.5)),
        ("stayers", e.filter(pl.col("mover") < 0.5)),
    ):
        mse = {
            c: float((sub["m"] * (sub["y"] - sub[c]) ** 2).sum() / sub["m"].sum()) for c in ("base", "ctx", "ctx_gd")
        }
        out[name] = {
            "mse": mse,
            "ctx_vs_base_z": pr.paired_cluster(sub, "ctx", "base")["z"],
            "ctx_gd_vs_base_z": pr.paired_cluster(sub, "ctx_gd", "base")["z"],
        }
    # team level: roster-weighted sums of the projections, M protocol, test seasons 2021-2025
    wm = pl.read_parquet(proc / "wm_team_table.parquet").filter(pl.col("season") >= 2018)
    cols: dict[str, list] = {"S_base": [], "S_ctx": [], "S_ctx_gd": []}
    for T in range(2019, 2026):
        for key, spec in (("S_base", base), ("S_ctx", ctx), ("S_ctx_gd", ctx_gd)):
            f = pr.forecast(fa, ft, 1, T, spec, p2)
            cols[key].append(
                f.with_columns((pl.col("proj") / 60.0 * pl.col("m")).alias("s"))
                .group_by("team")
                .agg(pl.col("s").sum().alias(key))
                .with_columns(pl.lit(T, pl.Int64).alias("season"))
            )
    for frames in cols.values():
        wm = wm.join(
            pl.concat(frames).with_columns(pl.col("season").cast(wm["season"].dtype)),
            on=["season", "team"],
            how="inner",
        )
    seasons = [2021, 2022, 2023, 2024, 2025]
    q = {k: wb.expanding_fit(wm, ["G", k], seasons) for k in ("S_base", "S_ctx", "S_ctx_gd", "S_rolling_refit")}
    q["B1"] = wb.expanding_fit(wm, ["G", "B_gd"], seasons)
    rm = {k: float(np.sqrt(((v["y"] - v["pred"]) ** 2).mean())) for k, v in q.items()}
    out["team"] = {
        "rmse": rm,
        "ctx_vs_base": paired(q["S_ctx"], q["S_base"]),
        "ctx_gd_vs_base": paired(q["S_ctx_gd"], q["S_base"]),
        "ctx_vs_rolling_last_season": paired(q["S_ctx"], q["S_rolling_refit"]),
        "ctx_gd_vs_rolling_last_season": paired(q["S_ctx_gd"], q["S_rolling_refit"]),
        "ctx_gd_vs_prior_team_gd": paired(q["S_ctx_gd"], q["B1"]),
        "test_seasons": seasons,
    }
    return out


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    gd = (
        wb.recon.team_gd_all(proc)
        .join(wb.games_played(proc), on=["season", "team"])
        .select("season", "team", (pl.col("gd_all") / pl.col("gp")).alias("gd_pg"))
    )
    p2 = add_context(p, gd.with_columns(pl.col("season").cast(pl.Int64)))
    res = {"A": analysis_a(p), "B": analysis_b(p2, bio), "C": analysis_c(proc, p2, bio)}
    cf = pl.read_parquet(proc / "chainfree_values.parquet")
    # defense and special-teams RAPM without the chained prior
    res["A_chainfree"] = analysis_a(pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"), cf))
    tot = res["A"]["total"]
    res["gates"] = {
        "TC1": {
            "stayer_minus_mover": tot["stayer_minus_mover"],
            "ci95": tot["stayer_minus_mover_ci95"],
            "pass": bool(tot["stayer_minus_mover_ci95"][0] > 0),
        },
        "TC2": {
            "t": res["B"]["movers"]["ctx_dest"]["t"],
            "coef": res["B"]["movers"]["ctx_dest"]["coef"],
            "pass": bool(res["B"]["movers"]["ctx_dest"]["t"] > 2),
        },
        "TC3": {
            "movers_z": res["C"]["movers"]["ctx_vs_base_z"],
            "all_z": res["C"]["all"]["ctx_vs_base_z"],
            "pass": bool(res["C"]["movers"]["ctx_vs_base_z"] > 2 and res["C"]["all"]["ctx_vs_base_z"] > -2),
        },
        "TC4": {"z": res["C"]["team"]["ctx_vs_base"]["z"], "pass": bool(res["C"]["team"]["ctx_vs_base"]["z"] > 2)},
    }
    _save(met / "tc_results.json", res)
    log.info(
        "gates %s",
        {
            k: (round(v.get("z", v.get("t", v.get("movers_z", v.get("stayer_minus_mover", 0)))), 2), v["pass"])
            for k, v in res["gates"].items()
        },
    )
    from src.teamcontext.report import run as report

    report()
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
