"""On-ice offense / transition analyses:  python -m src.transition.run   (docs/transition/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.calibration import player as cp
from src.calibration import team as tm
from src.calibration import unshrunk
from src.config import load_config
from src.transition import opm as om

log = logging.getLogger("transition")


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def yoy(df: pl.DataFrame, col: str, min_toi: float = 500.0) -> dict:
    x = df.filter(pl.col("toi_5v5") >= min_toi).with_columns((pl.col(col) / pl.col("toi_5v5") * 60).alias("_r")).with_columns((pl.col("_r") - pl.col("_r").mean().over("season", "pos_group")).alias("_c"))
    n = x.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("_c").alias("_n"))
    z = x.join(n, on=["player_id", "season"])
    return {g: {"r": float(np.corrcoef(s["_c"], s["_n"])[0, 1]), "n": s.height} for (g,), s in z.group_by(["pos_group"])}


def player_level(proc, t: pl.DataFrame, tr: pl.DataFrame) -> dict:
    """Next-season deployment-adjusted goal differential on the season-t position-weighted total rate, with and without the TR rate (methodology §3.2)."""
    m_all = pl.col("toi_5v5") + sum((pl.col(f"st_{c}_min") for c in ("pp", "pk", "4v4", "3v3")), pl.lit(0.0))
    d0 = t.with_columns(m_all.alias("_m")).filter(pl.col("_m") > 0).join(tr.select(pl.col("player_id").cast(t["player_id"].dtype), pl.col("season").cast(t["season"].dtype), "tr_gar"), on=["player_id", "season"], how="left").with_columns(pl.col("tr_gar").fill_null(0.0))
    rates = d0.select("player_id", "season", (pl.col("total_gar_all_ridge") / pl.col("_m") * 60).alias("r_uncal"), (pl.col("total_gar_all_ridge") / pl.col("_m") * 60).alias("r_cal"), (pl.col("tr_gar") / pl.col("_m") * 60).alias("r_tr"))
    oi = cp.onice_forward_all(proc, rates)
    own = rates.select("player_id", (pl.col("season") + 1).alias("season"), "r_uncal", "r_tr")
    team = t.select("player_id", "season", "team", "pos_group")
    d = oi.join(own, on=["player_id", "season"], how="inner").join(team, on=["player_id", "season"], how="inner").filter(pl.col("minutes") >= 250.0)
    d = d.with_columns((pl.col("gd_adj") / pl.col("minutes") * 60).alias("y"), (pl.col("season").cast(pl.Utf8) + "_" + pl.col("team")).alias("tsk"))
    out: dict = {}
    for name, sub in (("all", d), ("D", d.filter(pl.col("pos_group") == "D")), ("F", d.filter(pl.col("pos_group") == "F"))):
        w, g, y = sub["minutes"].to_numpy(), sub["tsk"].to_numpy(), sub["y"].to_numpy()
        base = cp.wls_fe(y, sub.select("r_uncal", "tm_u", "sh_pp", "sh_pk", "sh_o").to_numpy(), w, g)
        full = cp.wls_fe(y, sub.select("r_uncal", "tm_u", "sh_pp", "sh_pk", "sh_o", "r_tr").to_numpy(), w, g)
        ks = sorted(base["sse_by_group"])
        diff = np.array([base["sse_by_group"][k] - full["sse_by_group"][k] for k in ks])         # positive = the model with TR is better
        out[name] = {"n": sub.height, "tr_coef": float(full["beta"][-1]), "tr_se": float(full["se"][-1]), "own_total_coef": float(full["beta"][0]), "paired_z_with_tr_better": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))}
    return out


def team_level(cfg, proc, tr: pl.DataFrame) -> dict:
    """The position-specific identity (docs/calibration §5d) with TR added as forwards' and defensemen's team sums; held-out error against the identity without it."""
    t = unshrunk.build(proc, all_strength=True)
    j = tm.team_table(t, proc, tm.COMPS_POS, target="gd_all")
    trt = t.select("player_id", "season", "team", "pos_group").join(tr.select(pl.col("player_id").cast(t["player_id"].dtype), pl.col("season").cast(t["season"].dtype), "tr_gar"), on=["player_id", "season"], how="left").with_columns(pl.col("tr_gar").fill_null(0.0))
    sums = trt.group_by("season", "team", "pos_group").agg(pl.col("tr_gar").sum()).pivot(on="pos_group", index=["season", "team"], values="tr_gar").rename({"F": "tr_F", "D": "tr_D"}).fill_null(0.0)
    j = j.with_columns(pl.col("season").cast(pl.Int64)).join(sums.with_columns(pl.col("season").cast(pl.Int64)), on=["season", "team"], how="left").with_columns(pl.col("tr_F").fill_null(0.0), pl.col("tr_D").fill_null(0.0)).sort("season", "team")
    y, season = j["gd_all"].to_numpy().astype(float), j["season"].to_numpy()
    base_cols = [tm.RAW[c] for c in tm.COMPS_POS]
    X0 = j.select(base_cols).to_numpy()
    X1 = np.column_stack([X0, j["tr_F"].to_numpy(), j["tr_D"].to_numpy()])
    cv0, cv1 = tm.loso_cv(X0, y, season, tm.LAMBDAS), tm.loso_cv(X1, y, season, tm.LAMBDAS)
    b0, b1 = int(np.argmin(cv0["total"][:-1])), int(np.argmin(cv1["total"][:-1]))
    d = cv0["sq_err"][b0] - cv1["sq_err"][b1]
    z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    _, w = tm.ridge_to_one(X1, y, tm.LAMBDAS[b1])
    rng = np.random.default_rng(cfg.seed)
    bw = np.array([tm.ridge_to_one(X1[k], y[k], tm.LAMBDAS[b1])[1] for k in (rng.integers(0, len(y), len(y)) for _ in range(1000))])
    from src.valuation import recon
    ols = recon.regress(j.with_columns(pl.Series("trF", j["tr_F"]), pl.Series("trD", j["tr_D"])), [*base_cols, "trF", "trD"], y="gd_all")
    return {"cv_mse_without": float(cv0["sq_err"][b0].mean()), "cv_mse_with": float(cv1["sq_err"][b1].mean()), "paired_z": z, "lambda_without": float(tm.LAMBDAS[b0]), "lambda_with": float(tm.LAMBDAS[b1]),
            "weight_tr_F": float(w[-2]), "weight_tr_D": float(w[-1]), "ci_tr_F": [float(np.quantile(bw[:, -2], 0.025)), float(np.quantile(bw[:, -2], 0.975))], "ci_tr_D": [float(np.quantile(bw[:, -1], 0.025)), float(np.quantile(bw[:, -1], 0.975))],
            "ols_slope_tr_F": ols["slopes"]["trF"], "ols_slope_tr_D": ols["slopes"]["trD"]}


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    outdir = cfg.path(cfg.data.output_dir)
    opm = om.opm_table(cfg, proc)
    t = pl.read_parquet(proc / "player_value_calibrated_all_pos.parquet")
    tr, beta = om.add_transition(opm, t)
    res: dict = {"beta_pos": beta, "yoy": {"opm": yoy(tr, "opm_gar"), "tr": yoy(tr, "tr_gar"), "pm": yoy(tr.join(t.select(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64)), on=["player_id", "season"]), "pm_gar")}}
    res["sd_per60"] = {g: {c: float((s[c] / s["toi_5v5"] * 60).std()) for c in ("opm_gar", "tr_gar", "pm_gar")} for g in ("D", "F") for s in [tr.filter((pl.col("pos_group") == g) & (pl.col("toi_5v5") >= 500))]}
    res["player_level"] = player_level(proc, t, tr)
    res["team_level"] = team_level(cfg, proc, tr)
    g = {"TR1": {"yoy_D": res["yoy"]["tr"]["D"]["r"], "yoy_F": res["yoy"]["tr"]["F"]["r"], "pass": bool(res["yoy"]["tr"]["D"]["r"] >= 0.20)},
         "TR2": {"z_all": res["player_level"]["all"]["paired_z_with_tr_better"], "z_D": res["player_level"]["D"]["paired_z_with_tr_better"], "pass": bool(res["player_level"]["all"]["paired_z_with_tr_better"] > 2 and res["player_level"]["D"]["paired_z_with_tr_better"] > 2)},
         "TR3": {"z": res["team_level"]["paired_z"], "pass": bool(res["team_level"]["paired_z"] > 2)}}
    res["gates"] = g
    # application (only computed here; adoption needs all gates, docs/transition §4)
    wF, wD = (float(np.clip(res["team_level"][k], 0, 3)) for k in ("weight_tr_F", "weight_tr_D"))
    tt = t.join(tr.select(pl.col("player_id").cast(t["player_id"].dtype), pl.col("season").cast(t["season"].dtype), "opm_gar", "tr_gar"), on=["player_id", "season"], how="left").with_columns(pl.col("opm_gar").fill_null(0.0), pl.col("tr_gar").fill_null(0.0))
    tt = tt.with_columns((pl.col("total_gar_all_ridge") + pl.when(pl.col("pos_group") == "D").then(wD).otherwise(wF) * pl.col("tr_gar")).alias("total_gar_all_tr"))
    tt = tt.with_columns((pl.col("total_gar_all_tr") - pl.col("total_gar_all_ridge")).alias("tr_change_gar"))
    # TR-4: the deployment-adjusted check for the total with TR against the position-weighted total
    m_all = pl.col("toi_5v5") + sum((pl.col(f"st_{c}_min") for c in ("pp", "pk", "4v4", "3v3")), pl.lit(0.0))
    rates = tt.with_columns(m_all.alias("_m")).filter(pl.col("_m") > 0).select("player_id", "season", (pl.col("total_gar_all_ridge") / pl.col("_m") * 60).alias("r_uncal"), (pl.col("total_gar_all_tr") / pl.col("_m") * 60).alias("r_cal"))
    oi = cp.onice_forward_all(proc, rates)
    own = rates.select("player_id", (pl.col("season") + 1).alias("season"), "r_uncal", "r_cal")
    d = oi.join(own, on=["player_id", "season"], how="inner").join(tt.select("player_id", "season", "team"), on=["player_id", "season"], how="inner").filter(pl.col("minutes") >= 250.0).with_columns(
        (pl.col("gd_adj") / pl.col("minutes") * 60).alias("y_gd_adj"), (pl.col("xgd_adj") / pl.col("minutes") * 60).alias("y_xgd_adj"), (pl.col("season").cast(pl.Utf8) + "_" + pl.col("team")).alias("tsk"))
    ev = cp.evaluate(d, [], ys=("y_gd_adj", "y_xgd_adj"), controls=("sh_pp", "sh_pk", "sh_o"))
    res["TR4_detail"] = {"goals_z": ev["y_gd_adj"]["paired_z_calibrated_better"], "xg_z": ev["y_xgd_adj"]["paired_z_calibrated_better"]}
    g["TR4"] = {"z": ev["y_gd_adj"]["paired_z_calibrated_better"], "pass": bool(ev["y_gd_adj"]["paired_z_calibrated_better"] > -2)}
    res["adopt"] = bool(all(v["pass"] for v in g.values()))
    res["weights_applied"] = {"F": wF, "D": wD}
    tt.write_parquet(proc / "player_value_all_tr.parquet")
    tt.select("player_id", "player_name", "position", "season", "team", "toi_5v5", "pm_gar", "opm_gar", "tr_gar", "total_gar_all", "total_gar_all_ridge", "total_gar_all_tr", "tr_change_gar").write_csv(outdir / "player_transition.csv")
    _save(met / "tr_results.json", res)
    log.info("gates %s | adopt %s", {k: v["pass"] for k, v in g.items()}, res["adopt"])
    from src.transition.report import run as report
    report()
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
