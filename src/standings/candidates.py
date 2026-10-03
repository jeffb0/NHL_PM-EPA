"""Candidate improvements to the player-to-team forecast:  python -m src.standings.candidates   (docs/standings/improvements.md).

C1 portable team residual, C2 aggregation-consistent fine-tuning, C3 deployment-aware minutes; scored on the standings backtest against M1."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.prospects import features as pf
from src.standings import data as sd
from src.standings import forecast as fc
from src.standings import run as sr

log = logging.getLogger("standings")
AGG = [f"rt_{c}" for c in pd_.COMPS] + ["no_history", "age_lin", "age_h24", "age_h28", "age_h32"]
AGG_COLS = [f"{g}_{f}" for g in ("F", "D") for f in AGG]


# ---------------------------------------------------------------- C1: portable team residual
def team_residual(proc, p: pl.DataFrame, res: pl.DataFrame) -> pl.DataFrame:
    """Per (season, team): R = actual goal differential minus the season's own skater value and realised goals saved above expected (demeaned within season);
    rho = R per skater-minute. Uses only that season's data."""
    sk = p.group_by("season", "team").agg((pl.col("r_total") * pl.col("m") / 60).sum().alias("Sc"), pl.col("m").sum().alias("M"))
    gs = pl.read_parquet(proc / "gk_shots.parquet", columns=["season", "team_id", "goal", "xg_f"]).group_by("season", "team_id").agg((pl.col("xg_f") - pl.col("goal")).sum().alias("Gc"))
    r = (res.select("season", "team_id", "team", "gd").join(sk, on=["season", "team"], how="left").join(gs.with_columns(pl.col("team_id").cast(res["team_id"].dtype)), on=["season", "team_id"], how="left")
         .with_columns(pl.col("Gc").fill_null(0.0), pl.col("Sc").fill_null(0.0)))
    r = r.with_columns((pl.col("Sc") + pl.col("Gc")).alias("SGc")).with_columns((pl.col("gd") - (pl.col("SGc") - pl.col("SGc").mean().over("season"))).alias("R"))
    return r.select("season", "team", "R", "M", (pl.col("R") / pl.col("M")).alias("rho"))


def carried_residual(alloc: pl.DataFrame, p: pl.DataFrame, resid: pl.DataFrame, T: int) -> pl.DataFrame:
    """Per team: sum over the roster of (last season's team residual per skater-minute of the player's team) x projected minutes."""
    prev = p.filter(pl.col("season") == T - 1).select("player_id", "team").join(resid.filter(pl.col("season") == T - 1).select("team", "rho"), on="team", how="left")
    a = alloc.join(prev.select("player_id", "rho"), on="player_id", how="left").with_columns(pl.col("rho").fill_null(0.0))
    return a.group_by("team_id").agg((pl.col("rho") * pl.col("mins")).sum().alias("Rc"))


# ---------------------------------------------------------------- C2: aggregated features
def aggregate(alloc: pl.DataFrame) -> pl.DataFrame:
    ab = pm.age_basis(alloc["age"].to_numpy())
    a = alloc.with_columns(*[pl.Series(n, ab[:, i]) for i, n in enumerate(("age_lin", "age_h24", "age_h28", "age_h32"))])
    long = a.group_by("team_id", "pos_group").agg(*[(pl.col(f) * pl.col("mins") / 60).sum().alias(f) for f in AGG])
    parts = [long.filter(pl.col("pos_group") == g).select("team_id", *[pl.col(f).alias(f"{g}_{f}") for f in AGG]) for g in ("F", "D")]
    return parts[0].join(parts[1], on="team_id", how="full", coalesce=True).with_columns([pl.col(c).fill_null(0.0) for c in AGG_COLS])


# ---------------------------------------------------------------- C3: deployment-aware minutes
def _mx3(d: pl.DataFrame) -> np.ndarray:
    return np.column_stack([pr._minutes_x(d), d["rt_total"].to_numpy(), d["rt_dev"].to_numpy()])


def minutes_fit_c3(tr: pl.DataFrame) -> dict:
    tr = tr.with_columns((pl.col("rt_total") - pl.col("rt_total").mean().over("season", "team", "pos_group")).alias("rt_dev"))
    out = {}
    for g in ("F", "D"):
        d = tr.filter(pl.col("pos_group") == g)
        X, y, w = _mx3(d), d["m"].to_numpy(), np.ones(d.height)
        out[g] = pm.wridge(X, y, w, pm.choose_lambda(X, y, w, d["season"].to_numpy()))
    return out


def minutes_pred_c3(models: dict, d: pl.DataFrame) -> np.ndarray:
    d = d.with_columns((pl.col("rt_total") - pl.col("rt_total").mean().over("team_id", "pos_group")).alias("rt_dev"))
    out = np.zeros(d.height)
    for g in ("F", "D"):
        mk = (d["pos_group"] == g).to_numpy()
        if mk.any():
            out[mk] = np.maximum(models[g].predict(_mx3(d.filter(pl.col("pos_group") == g))), 0.0)
    return out


def share_error(alloc: pl.DataFrame, p: pl.DataFrame, T: int, ab: pl.DataFrame) -> float:
    """Mean absolute error of each kept player's projected share of his team-position minutes against his actual share (0 if he did not play for that team)."""
    act = p.filter(pl.col("season") == T).select("player_id", "team", pl.col("m").alias("m_act"))
    a = alloc.join(ab, on="team_id").join(act, on=["player_id", "team"], how="left").with_columns(pl.col("m_act").fill_null(0.0))
    a = a.with_columns((pl.col("mins") / pl.col("mins").sum().over("team_id", "pos_group")).alias("s_hat"), (pl.col("m_act") / pl.col("m_act").sum().over("team_id", "pos_group")).alias("s_act"))
    return float((a["s_hat"] - a["s_act"]).abs().mean())


# ---------------------------------------------------------------- build and evaluate
def build(proc, cfg, seasons: list[int], yp: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict]:
    roll = pl.read_parquet(proc / "wm_rolling_values.parquet")
    p = pd_.panel(proc, roll)
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    res = sd.results(proc)
    sk, gk = sd.openers(proc)
    per, lg = sd.goalie_usage(proc)
    rates = pl.read_parquet(proc / "gk_rolling_rates.parquet")
    resid = team_residual(proc, p, res)
    M = pf.spec(yp)
    fr_min = pd_.design_frame(p, bio, 1)
    fr_val = pf.with_yp(pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR), yp)
    rows, diag = [], {"share_mae_base": {}, "share_mae_c3": {}}
    for T in seasons:
        ab = res.filter(pl.col("season") == T).select("team_id", "team")
        ros = sk.filter(pl.col("season") == T).select("team_id", "player_id", "pos_group")
        d = pf.with_yp(fc.roster_frame(p, bio, ros, T), yp)
        models = pm.fit_models(pr.train_rows(fr_val, 1, T), M)
        mm = pr._minutes_fit(pr.train_rows(fr_min, 1, T))
        target = fc.team_minutes(p, res, T - 1)
        alloc = fc.allocate_minutes(d, pr._minutes_pred(mm, d), target)
        S = fc._skater_sum(alloc, pm.predict_models(models, alloc, M), "S")
        G = fc.goalie_value(gk, T, rates, per, lg)
        Rc = carried_residual(alloc, p, resid, T)
        A = aggregate(alloc)
        mm3 = minutes_fit_c3(pr.train_rows(fr_min, 1, T))
        alloc3 = fc.allocate_minutes(d, minutes_pred_c3(mm3, d), target)
        S3 = fc._skater_sum(alloc3, pm.predict_models(models, alloc3, M), "S_c3")
        Rc3 = carried_residual(alloc3, p, resid, T).rename({"Rc": "Rc3"})
        diag["share_mae_base"][T] = share_error(alloc, p, T, ab)
        diag["share_mae_c3"][T] = share_error(alloc3, p, T, ab)
        t = ab.with_columns(pl.lit(T, pl.Int64).alias("season"))
        for x in (S, G, Rc, A, S3, Rc3):
            t = t.join(x, on="team_id", how="left")
        rows.append(t.with_columns(pl.col("team_id").cast(pl.Int64)))
        log.info("season %d built", T)
    return pl.concat(rows, how="diagonal_relaxed"), diag


def _ridge_cv(X: np.ndarray, y: np.ndarray, s: np.ndarray) -> pm.Ridge:
    w = np.ones(len(y))
    return pm.wridge(X, y, w, pm.choose_lambda(X, y, w, s))


def evaluate(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    act = df.filter(pl.col("gp") > 0)
    rows, fitted = [], {"c1": {}, "c13": {}, "c2_lambda": {}}
    for T in sr.TEST_SEASONS:
        tr_all = act.filter(pl.col("season") < T)
        tr_f = act.filter((pl.col("season") < T) & pl.col("SG_dm").is_not_null())
        te = act.filter(pl.col("season") == T)
        a, b = sr.ols(tr_all["gd82"].to_numpy()[:, None], tr_all["pts82"].to_numpy())
        pts = lambda gd, a=a, b=b: a + b * gd                                           # noqa: E731
        out = te.select("season", "team_id", "team", "pts82")
        # C1: one coefficient on the carried residual, fitted on the residual of M1 (in goals) in earlier forecast seasons
        def c_fit(base: str, rc: str, tr=tr_f, a=a, b=b) -> float:
            y = (tr["pts82"].to_numpy() - (a + b * tr[base].to_numpy())) / b
            x = tr[rc].to_numpy()
            return float((x * y).sum() / (x * x).sum())
        c1 = c_fit("SG_dm", "Rc_dm")
        c13 = c_fit("SG3_dm", "Rc3_dm")
        fitted["c1"][T], fitted["c13"][T] = c1, c13
        # C2: ridge on aggregated player features toward the player model (target: goal differential residual per 82)
        Xtr, Xte = tr_f.select([f"{c}_dm" for c in AGG_COLS]).to_numpy(), te.select([f"{c}_dm" for c in AGG_COLS]).to_numpy()
        rg = _ridge_cv(Xtr, tr_f["gd82"].to_numpy() - tr_f["SG_dm"].to_numpy(), tr_f["season"].to_numpy())
        fitted["c2_lambda"][T] = rg.lam
        corr = rg.predict(Xte) - rg.ybar                                                  # the correction is centred: the season's mean correction is zero
        preds = {"M1": pts(te["SG_dm"].to_numpy()), "C1": pts(te["SG_dm"].to_numpy() + c1 * te["Rc_dm"].to_numpy()), "C2": pts(te["SG_dm"].to_numpy() + corr),
                 "C3": pts(te["SG3_dm"].to_numpy()), "C1_C3": pts(te["SG3_dm"].to_numpy() + c13 * te["Rc3_dm"].to_numpy()), "C1_fixed1": pts(te["SG_dm"].to_numpy() + te["Rc_dm"].to_numpy())}
        rows.append(out.with_columns(*[pl.Series(k, v) for k, v in preds.items()]))
    return pl.concat(rows), fitted


def score(e: pl.DataFrame) -> dict:
    y = e["pts82"].to_numpy()
    names = ["M1", "C1", "C2", "C3", "C1_C3", "C1_fixed1", "B2", "M3"]
    err = {k: y - e[k].to_numpy() for k in names}
    out = {"rmse": {k: float(np.sqrt((v**2).mean())) for k, v in err.items()}}
    out["z_vs_M1"] = {k: sr._z(err[k], err["M1"]) for k in names if k != "M1"}
    out["z_C1_C3_vs_C1"], out["z_C1_C3_vs_C3"] = sr._z(err["C1_C3"], err["C1"]), sr._z(err["C1_C3"], err["C3"])
    out["z_vs_B2"] = {k: sr._z(err[k], err["B2"]) for k in names if k != "B2"}
    out["z_vs_M3"] = {k: sr._z(err[k], err["M3"]) for k in names if k != "M3"}
    seas = e["season"].to_numpy()
    out["rmse_by_season"] = {str(T): {k: float(np.sqrt((err[k][seas == T] ** 2).mean())) for k in names} for T in sr.TEST_SEASONS}
    out["seasons_beating_M1"] = {k: int(sum(out["rmse_by_season"][str(T)][k] < out["rmse_by_season"][str(T)]["M1"] for T in sr.TEST_SEASONS)) for k in names if k != "M1"}
    cal = {}
    for k in names:
        x = e[k].to_numpy()
        A = np.column_stack([np.ones(len(x)), x])
        cal[k] = float(np.linalg.lstsq(A, y, rcond=None)[0][1])
    out["calibration_slope"] = cal
    return out


SOUND = {"C1": True, "C2": False, "C3": True}


def decide(z: float, sound: bool) -> bool:
    """docs/standings/improvements.md §4."""
    if z > 2:
        return True
    if z > -1:
        return sound
    return False


def assemble(proc, cfg, yp: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict]:
    """Team-season table with every candidate's inputs. With ``yp`` (docs/prospects) the skater sums use the rate model with pre-NHL columns, and M1's sum is rebuilt from them."""
    tab = pl.read_parquet(proc / "standings_table.parquet")
    cand, diag = build(proc, cfg, sr.FORECAST_SEASONS, yp)
    df = tab.select("season", "team_id", "team", "gp", "pts82", "gd82", "S", "G", "SG_dm").join(cand.drop("team"), on=["season", "team_id"], how="left", suffix="_c")
    chk = df.filter(pl.col("S").is_not_null())
    diag["max_abs_S_diff_vs_backtest"] = float((chk["S"] - chk["S_c"]).abs().max())          # zero without yp: the reference M1 is reproduced exactly
    if yp is not None:
        df = df.with_columns((pl.col("S_c") + pl.col("G")).alias("_sg")).with_columns((pl.col("_sg") - pl.col("_sg").mean().over("season")).alias("SG_dm")).drop("_sg")
    df = df.with_columns((pl.col("S_c3") + pl.col("G")).alias("SG3"))
    for c in ["SG3", "Rc", "Rc3", *AGG_COLS]:
        df = df.with_columns((pl.col(c) - pl.col(c).mean().over("season")).alias(f"{c}_dm"))
    return df, diag


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    outdir = cfg.path(cfg.data.output_dir)
    df, diag = assemble(proc, cfg)
    df.write_parquet(proc / "standings_candidates_table.parquet")                 # the 2026 forecast fits C1's coefficient on it
    e, fitted = evaluate(df)
    base = pl.read_csv(outdir / "standings_backtest.csv").select("season", "team_id", "B2", "M3", pl.col("M1").alias("M1_ref"))
    e = e.join(base, on=["season", "team_id"])
    diag["max_abs_M1_diff_vs_backtest"] = float((e["M1"] - e["M1_ref"]).abs().max())
    s = score(e)
    s["fitted"], s["diagnostics"] = fitted, diag
    s["decision"] = {k: {"z_vs_M1": s["z_vs_M1"][k], "sound_a_priori": SOUND[k], "implement": decide(s["z_vs_M1"][k], SOUND[k])} for k in ("C1", "C2", "C3")}
    e.drop("M1_ref").write_csv(outdir / "standings_candidates_backtest.csv")
    (met / "standings_candidates.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    log.info("RMSE %s", {k: round(v, 2) for k, v in s["rmse"].items()})
    log.info("z vs M1 %s", {k: round(v, 2) for k, v in s["z_vs_M1"].items()})
    log.info("decisions %s", {k: v["implement"] for k, v in s["decision"].items()})
    return s


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
