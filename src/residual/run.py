"""Decomposition of the portable team residual:  python -m src.residual.run   (docs/residual/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.projection import data as pd_
from src.residual import decomp as dc
from src.standings import candidates as cd
from src.standings import data as sd
from src.standings import run as sr

log = logging.getLogger("residual")
FIRST = 2018                       # first season with a previous season to carry from
RHO = [f"rho_{b}" for b in dc.BUCKETS]


def describe(t: pl.DataFrame, fr: pl.DataFrame) -> dict:
    R = t["R"].to_numpy()
    out: dict = {"n_team_seasons": t.height, "sd_goal_differential": float(t["gd"].std()), "sd_residual": float(R.std(ddof=1)), "buckets": {}}
    nxt = t.select((pl.col("season") + 1).alias("season"), "team_id", *[pl.col(f"R_{b}").alias(f"p_{b}") for b in dc.BUCKETS], pl.col("R").alias("p_R"))
    yy = t.join(nxt, on=["season", "team_id"], how="inner")
    for b in dc.BUCKETS:
        o, e, r = t[f"o_{b}"].to_numpy(), t[f"e_{b}"].to_numpy(), t[f"R_{b}"].to_numpy()
        slope = float((o * e).sum() / (e * e).sum())
        out["buckets"][b] = {"sd_observed": float(o.std(ddof=1)), "sd_explained": float(e.std(ddof=1)), "sd_residual": float(r.std(ddof=1)), "corr_observed_explained": float(np.corrcoef(o, e)[0, 1]),
                             "slope_observed_on_explained": slope, "share_of_residual_variance": float(np.cov(r, R)[0, 1] / R.var(ddof=1)), "yoy_corr_same_team": float(np.corrcoef(yy[f"R_{b}"], yy[f"p_{b}"])[0, 1])}
    out["yoy_corr_total_residual"] = float(np.corrcoef(yy["R"], yy["p_R"])[0, 1])
    y = fr["R"].to_numpy()
    b, se = dc.ols0(fr.select([f"c_{c}" for c in RHO]).to_numpy(), y)
    out["carry_by_bucket"] = {k: {"coef": float(bi), "se": float(si)} for k, bi, si in zip(dc.BUCKETS, b, se)}
    b, se = dc.ols0(fr["c_rho"].to_numpy()[:, None], y)
    out["carry_lump"] = {"coef": float(b[0]), "se": float(se[0]), "n": fr.height}
    b, se = dc.ols0(fr.select("c_F", "c_D", "left_behind").to_numpy(), y)
    out["who_carries"] = {k: {"coef": float(bi), "se": float(si)} for k, bi, si in zip(("forwards", "defensemen", "left_with_team"), b, se)}
    out["who_carries"]["sd"] = {k: float(fr[c].std()) for k, c in (("forwards", "c_F"), ("defensemen", "c_D"), ("left_with_team", "left_behind"))}
    return out


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    res = sd.results(proc)
    t = dc.table(proc, p, res)
    lump = cd.team_residual(proc, p, res)
    chk = t.join(lump.select("season", "team", pl.col("R").alias("R_lump")), on=["season", "team"])
    car = dc.carried(p, t, [*RHO, "rho"])
    prev = t.select((pl.col("season") + 1).alias("season"), "team_id", pl.col("R").alias("R_prev"), pl.col("M").alias("M_prev"))
    fr = (t.filter(pl.col("season") >= FIRST).join(car, on=["season", "team"], how="left").join(prev, on=["season", "team_id"], how="inner")
          .with_columns((pl.col("R_prev") * pl.col("M") / pl.col("M_prev") - pl.col("c_F") - pl.col("c_D")).alias("left_behind")))
    out: dict = {"identity_max_abs_diff_vs_lump": float((chk["R"] - chk["R_lump"]).abs().max()), "describe": describe(t, fr)}
    t.write_parquet(proc / "residual_buckets.parquet")
    t.select("season", "team", "gd", *[f"o_{b}" for b in dc.BUCKETS], *[f"e_{b}" for b in dc.BUCKETS], *[f"R_{b}" for b in dc.BUCKETS], "R").write_csv(cfg.path(cfg.data.output_dir) / "team_residual_buckets.csv")
    log.info("identity diff %.2e; shares %s", out["identity_max_abs_diff_vs_lump"], {b: round(v["share_of_residual_variance"], 2) for b, v in out["describe"]["buckets"].items()})
    log.info("carry by bucket %s; lump %.2f", {b: round(v["coef"], 2) for b, v in out["describe"]["carry_by_bucket"].items()}, out["describe"]["carry_lump"]["coef"])

    # ---- forecast test: expanding carry coefficients, then the game model's preseason ratings with the split, the fitted lump and the published lump
    seasons = grun.TUNE + grun.TEST
    full_fr = t.filter(pl.col("season") >= FIRST).join(car, on=["season", "team"], how="left")
    coefs, rows_split, rows_lump = {}, [], []
    for T in seasons:
        tr = full_fr.filter(pl.col("season") < T)
        if tr["season"].n_unique() >= 2:
            cb = dc.ols0(tr.select([f"c_{c}" for c in RHO]).to_numpy(), tr["R"].to_numpy())[0]
            cl = float(dc.ols0(tr["c_rho"].to_numpy()[:, None], tr["R"].to_numpy())[0][0])
        else:
            cb, cl = np.ones(len(RHO)), 1.0
        coefs[str(T)] = {"lump": cl, **dict(zip(dc.BUCKETS, cb.tolist()))}
        src = t.filter(pl.col("season") == T - 1)
        rows_split.append(src.select("season", "team", sum(float(c) * pl.col(r) for c, r in zip(cb, RHO)).alias("rho")))
        rows_lump.append(src.select("season", "team", (cl * pl.col("rho")).alias("rho")))
    out["expanding_coefficients"] = coefs
    split, lumpf = pl.concat(rows_split), pl.concat(rows_lump)
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    tg = games.table.filter(pl.col("season") >= grun.TUNE[0])
    te = tg["season"].is_in(grun.TEST).to_numpy()
    y, dates = tg["home_win"].to_numpy()[te], tg["date"].to_numpy()[te]
    kws = {"P_pre": {"lam": 0.0, "alpha": 0.0, "n0": hp["n0"], "update": False}, "P": {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}}
    ll: dict = {}
    for tag, rs in (("published", None), ("lump_fitted", lumpf), ("split", split)):
        pri = gd_.build_priors(proc, cfg, seasons, games, resid=rs)
        for name, kw in kws.items():
            mu = np.concatenate([gm.run_player_season(games, pri, T, **kw) for T in seasons])
            ll[(tag, name)] = gm.logloss(grun.predict(tg, mu)[te], y)
    cmp = lambda a, b, m: grun.paired(ll[(a, m)], ll[(b, m)], dates)  # noqa: E731
    out["game"] = {"n_games": int(te.sum()), "logloss": {f"{a}|{m}": float(v.mean()) for (a, m), v in ll.items()},
                   "split_vs_published": {m: cmp("split", "published", m) for m in kws}, "split_vs_lump_fitted": {m: cmp("split", "lump_fitted", m) for m in kws},
                   "lump_fitted_vs_published": {m: cmp("lump_fitted", "published", m) for m in kws}}
    log.info("game P-pre: split vs published z %.2f, split vs fitted lump z %.2f, fitted lump vs published z %.2f", out["game"]["split_vs_published"]["P_pre"]["z"], out["game"]["split_vs_lump_fitted"]["P_pre"]["z"],
             out["game"]["lump_fitted_vs_published"]["P_pre"]["z"])

    # ---- standings (reported): the published configuration with the split residual
    pj = json.loads((met / "prospects.json").read_text()) if (met / "prospects.json").exists() else {}
    yp = pl.read_parquet(proc / "prospect_features.parquet") if pj.get("implement") else None
    e0, _ = cd.evaluate(cd.assemble(proc, cfg, yp)[0])
    e1, _ = cd.evaluate(cd.assemble(proc, cfg, yp, resid=split)[0])
    j = e0.select("season", "team_id", "pts82", pl.col("C1_C3").alias("base")).join(e1.select("season", "team_id", "C1_C3"), on=["season", "team_id"])
    yy = j["pts82"].to_numpy()
    out["standings"] = {"rmse_published": float(np.sqrt(((yy - j["base"].to_numpy()) ** 2).mean())), "rmse_split": float(np.sqrt(((yy - j["C1_C3"].to_numpy()) ** 2).mean())), "z": sr._z(yy - j["C1_C3"].to_numpy(), yy - j["base"].to_numpy())}
    z1, z2, z3 = out["game"]["split_vs_published"]["P_pre"]["z"], out["game"]["split_vs_lump_fitted"]["P_pre"]["z"], out["game"]["split_vs_published"]["P"]["z"]
    out["gates"] = {"RD1": {"z": z1, "pass": bool(z1 > 2)}, "RD2": {"z": z2, "pass": bool(z2 > 2)}, "RD3": {"z": z3, "pass": bool(z3 > -2)}}
    out["implement"] = bool(z1 > 2 and z3 > -2)
    (met / "residual_decomp.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log.info("standings %.2f -> %.2f (z %.2f); gates %s implement=%s", out["standings"]["rmse_published"], out["standings"]["rmse_split"], out["standings"]["z"], {k: v["pass"] for k, v in out["gates"].items()}, out["implement"])
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
