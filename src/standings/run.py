"""Standings backtest end to end:  python -m src.standings.run   (docs/standings/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.standings import data as sd
from src.standings import forecast as fc

log = logging.getLogger("standings")
FORECAST_SEASONS = list(range(2019, 2026))
TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
Z80 = 1.2815515655446004


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]


def _pred(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def table(proc, cfg) -> pl.DataFrame:
    res = sd.results(proc)
    f = fc.build(proc, cfg, FORECAST_SEASONS)
    df = res.join(sd.prior(res), on=["season", "team_id"], how="left").join(
        f.drop("team"), on=["season", "team_id"], how="left"
    )
    # league-relative (demeaned within season) sums in goals per 82; goalie and skater terms are on different reference
    # levels, so the demeaning is applied to their total
    df = df.with_columns(
        (pl.col("S") + pl.col("G")).alias("SG"),
        (pl.col("S_R") + pl.col("G")).alias("SG_R"),
        (pl.col("S_oracle") + pl.col("G_oracle")).alias("SG_oracle"),
    )
    for c in ("SG", "SG_R", "SG_oracle", "S_res", "S", "G"):
        df = df.with_columns((pl.col(c) - pl.col(c).mean().over("season")).alias(f"{c}_dm"))
    lg = df.group_by("season").agg(
        pl.col("pts82_prev").mean().alias("_a"),
        pl.col("gd82_prev").mean().alias("_b"),
        pl.col("xgd82_prev").mean().alias("_c"),
    )
    df = (
        df.join(lg, on="season")
        .with_columns(
            pl.col("pts82_prev").fill_null(pl.col("_a")).alias("pts82_prev_f"),
            pl.col("gd82_prev").fill_null(0.0).alias("gd82_prev_f"),
            pl.col("xgd82_prev").fill_null(0.0).alias("xgd82_prev_f"),
        )
        .drop("_a", "_b", "_c")
    )
    return df.sort("season", "team_id")


def _z(a: np.ndarray, b: np.ndarray) -> float:
    """Paired z that model a beats model b: mean of (err_b^2 - err_a^2) over its standard error."""
    d = b**2 - a**2
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def _rank_stats(df: pl.DataFrame, col: str) -> dict:
    from scipy.stats import spearmanr

    out = {}
    for T in TEST_SEASONS:
        s = df.filter(pl.col("season") == T)
        out[T] = float(spearmanr(s[col].to_numpy(), s["pts82"].to_numpy())[0])
    acc = []
    for T in TEST_SEASONS:
        s = df.filter(pl.col("season") == T)
        top_a = set(s.sort("pts82", descending=True).head(16)["team_id"].to_list())
        top_p = set(s.sort(col, descending=True).head(16)["team_id"].to_list())
        acc.append(len(top_a & top_p) / 16)
    return {
        "spearman_by_season": out,
        "spearman_mean": float(np.mean(list(out.values()))),
        "top16_overlap": float(np.mean(acc)),
    }


def evaluate(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Expanding-window predictions of points per 82 for every model and baseline (fitted on seasons < T only)."""
    act = df.filter(pl.col("gp") > 0)
    rows = []
    for T in TEST_SEASONS:
        # actual pairs, 2017 onward (goal differential -> points)
        tr_all = act.filter(pl.col("season") < T)
        # seasons with forecasts (2019 onward)
        tr_f = act.filter((pl.col("season") < T) & pl.col("SG").is_not_null())
        te = act.filter(pl.col("season") == T)
        a, b = ols(tr_all["gd82"].to_numpy()[:, None], tr_all["pts82"].to_numpy())
        pts_of = lambda gd, a=a, b=b: a + b * gd
        tp = tr_all.filter(pl.col("pts82_prev").is_not_null())
        out = te.select("season", "team_id", "team", "pts82", "gd82")
        preds = {"B0": np.full(te.height, float(tr_all["pts82"].mean()))}
        for name, col in (("B1", "pts82_prev_f"), ("B2", "gd82_prev_f"), ("B3", "xgd82_prev_f")):
            tp_ = (
                tp.join(
                    tr_all.select("season", "team_id").with_columns(pl.lit(1).alias("_k")), on=["season", "team_id"]
                )
                .select(
                    "pts82",
                    pl.col("pts82_prev").alias("pts82_prev_f"),
                    pl.col("gd82_prev").alias("gd82_prev_f"),
                    pl.col("xgd82_prev").alias("xgd82_prev_f"),
                )
                .drop_nulls()
            )
            beta = ols(tp_[col].to_numpy()[:, None], tp_["pts82"].to_numpy())
            preds[name] = _pred(beta, te[col].to_numpy()[:, None])
        preds["M1"] = pts_of(te["SG_dm"].to_numpy())
        preds["M1_R"] = pts_of(te["SG_R_dm"].to_numpy())
        preds["M1_oracle"] = pts_of(te["SG_oracle_dm"].to_numpy())
        X2 = lambda d: np.column_stack([d["S_dm"].to_numpy(), d["G_dm"].to_numpy()])
        preds["M2"] = _pred(ols(X2(tr_f), tr_f["pts82"].to_numpy()), X2(te))
        X3 = lambda d: np.column_stack([d["SG_dm"].to_numpy(), d["gd82_prev_f"].to_numpy()])
        preds["M3"] = _pred(ols(X3(tr_f), tr_f["pts82"].to_numpy()), X3(te))
        X4 = lambda d: np.column_stack([d["SG_R_dm"].to_numpy(), d["gd82_prev_f"].to_numpy()])
        preds["M3_R"] = _pred(ols(X4(tr_f), tr_f["pts82"].to_numpy()), X4(te))
        Xr = lambda d: np.column_stack([d["S_res_dm"].to_numpy(), d["G_dm"].to_numpy()])
        preds["P_res"] = _pred(ols(Xr(tr_f), tr_f["pts82"].to_numpy()), Xr(te))
        # residual spread of M1 on its training seasons gives the interval
        resid = tr_f["pts82"].to_numpy() - pts_of(tr_f["SG_dm"].to_numpy())
        out = out.with_columns(
            *[pl.Series(k, v) for k, v in preds.items()],
            pl.lit(float(resid.std(ddof=1)) if len(resid) > 2 else float("nan")).alias("sd_M1"),
        )
        rows.append(out)
    e = pl.concat(rows)
    y = e["pts82"].to_numpy()
    names = [
        c for c in e.columns if c in ("B0", "B1", "B2", "B3", "M1", "M1_R", "M1_oracle", "M2", "M3", "M3_R", "P_res")
    ]
    err = {k: y - e[k].to_numpy() for k in names}
    summary = {
        "n_team_seasons": e.height,
        "test_seasons": TEST_SEASONS,
        "rmse": {k: float(np.sqrt((v**2).mean())) for k, v in err.items()},
        "mae": {k: float(np.abs(v).mean()) for k, v in err.items()},
    }
    hist = min(("B1", "B2", "B3"), key=lambda k: summary["rmse"][k])
    summary["best_history"] = hist
    summary["z"] = {
        "M1_vs_B0": _z(err["M1"], err["B0"]),
        "M1_vs_best_history": _z(err["M1"], err[hist]),
        "M3_vs_best_history": _z(err["M3"], err[hist]),
        "M1_vs_M1_R": _z(err["M1"], err["M1_R"]),
        "M1_oracle_vs_M1": _z(err["M1_oracle"], err["M1"]),
        "M3_R_vs_best_history": _z(err["M3_R"], err[hist]),
        "P_res_vs_best_history": _z(err["P_res"], err[hist]),
        "M2_vs_best_history": _z(err["M2"], err[hist]),
        "M1_vs_M3": _z(err["M1"], err["M3"]),
    }
    x = e["M1"].to_numpy()
    A = np.column_stack([np.ones(len(x)), x])
    bc = np.linalg.lstsq(A, y, rcond=None)[0]
    r_ = y - A @ bc
    se = float(np.sqrt((r_**2).sum() / (len(x) - 2) * np.linalg.inv(A.T @ A)[1, 1]))
    sd_ = e["sd_M1"].to_numpy()
    ok = ~np.isnan(sd_)
    cover = float((np.abs(err["M1"][ok]) <= Z80 * sd_[ok]).mean())
    summary["calibration"] = {"slope": float(bc[1]), "se": se, "intercept": float(bc[0])}
    summary["coverage80"] = cover
    summary["rank"] = {k: _rank_stats(e, k) for k in ("M1", "M3", "B1", "B2", "M1_R", "P_res")}
    summary["rmse_by_season"] = {
        str(T): {k: float(np.sqrt((err[k][e["season"].to_numpy() == T] ** 2).mean())) for k in ("M1", "M3", "B2")}
        for T in TEST_SEASONS
    }
    g = {
        "ST1": {"z": summary["z"]["M1_vs_B0"], "pass": bool(summary["z"]["M1_vs_B0"] > 2)},
        "ST2": {"z": summary["z"]["M1_vs_best_history"], "pass": bool(summary["z"]["M1_vs_best_history"] > -2)},
        "ST3": {"z": summary["z"]["M3_vs_best_history"], "pass": bool(summary["z"]["M3_vs_best_history"] > 2)},
        "ST4": {**summary["calibration"], "pass": bool(0.8 <= summary["calibration"]["slope"] <= 1.2)},
        "ST5": {"coverage_80": cover, "pass": bool(0.70 <= cover <= 0.90)},
    }
    summary["gates"] = g
    return e, summary


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    outdir = cfg.path(cfg.data.output_dir)
    df = table(proc, cfg)
    df.write_parquet(proc / "standings_table.parquet")
    e, summary = evaluate(df)
    e.write_csv(outdir / "standings_backtest.csv")
    (met / "standings_backtest.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log.info("RMSE %s", {k: round(v, 2) for k, v in summary["rmse"].items()})
    log.info("gates %s", {k: v["pass"] for k, v in summary["gates"].items()})
    from src.standings.report import run as report

    report()
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
