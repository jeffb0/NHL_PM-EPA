"""Projection and aging layer end to end:  python -m src.projection.run   (docs/projection/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.projection import data as pd_
from src.projection import model as pm
from src.projection.data import MIN_TARGET_MIN
from src.projection.model import Spec

log = logging.getLogger("projection")
REVISED_FLOOR = 20.0             # training floor after the team-level result (methodology §9, deviation 1); the pre-registered floor is 200
TEST_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
EVAL_MIN = 300.0
FIRST_TARGET = 2018            # earliest target season with any history (panel starts in 2017)
Z80 = 1.2815515655446004


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def train_rows(frame: pl.DataFrame, k: int, T: int) -> pl.DataFrame:
    """Training transitions for a forecast of season T at horizon k: target seasons with history and known before T - k + 1 ... i.e. targets <= T - k."""
    return frame.filter((pl.col("season") >= FIRST_TARGET + k - 1) & (pl.col("season") <= T - k))


def paired_cluster(df: pl.DataFrame, a: str, b: str, reps: int = 500, seed: int = 0) -> dict:
    """Minute-weighted squared-error difference (b - a; positive favors a) with a player-cluster bootstrap z."""
    d = df.with_columns((pl.col("m") * ((pl.col("y") - pl.col(b)) ** 2 - (pl.col("y") - pl.col(a)) ** 2)).alias("_d"))
    g = d.group_by("player_id").agg(pl.col("_d").sum().alias("D"), pl.col("m").sum().alias("W")).sort("player_id")   # group_by order is not stable: sort so the bootstrap is reproducible
    D, W = g["D"].to_numpy(), g["W"].to_numpy()
    rng = np.random.default_rng(seed)
    stat = float(D.sum() / W.sum())
    boots = [float(D[k].sum() / W[k].sum()) for k in (rng.integers(0, len(D), len(D)) for _ in range(reps))]
    se = float(np.std(boots))
    return {"mean_diff": stat, "z": stat / se if se > 0 else float("nan"), "n_rows": d.height, "n_players": len(D)}


def oof_resid(train: pl.DataFrame, spec: Spec, lam: dict[str, float], target: str = "r_total") -> pl.DataFrame:
    """Out-of-fold (leave-one-target-season-out) residuals on the training rows, for the interval model."""
    parts = []
    for s in sorted(train["season"].unique().to_list()):
        tr, te = train.filter(pl.col("season") != s), train.filter(pl.col("season") == s)
        if tr.height == 0:
            continue
        models = pm.fit_models(tr, spec, target=target, lam=None if not lam else float(np.mean(list(lam.values()))))
        parts.append(te.select("m").with_columns(pl.Series("res", te[target].to_numpy() - pm.predict_models(models, te, spec))))
    return pl.concat(parts)


def forecast(frame_all: pl.DataFrame, frame_targets: pl.DataFrame, k: int, T: int, spec: Spec, p: pl.DataFrame, target: str = "r_total") -> pl.DataFrame:
    """Rate forecasts for season T at horizon k with models trained on transitions known at the time."""
    tr = train_rows(frame_targets, k, T)
    models = pm.fit_models(tr, spec, target=target)
    test = frame_all.filter(pl.col("season") == T)
    pred = pm.predict_models(models, test, spec)
    oof = oof_resid(tr, spec, {}, target) if tr["season"].n_unique() >= 2 else None
    a, b = pm.variance_model(oof["m"].to_numpy(), oof["res"].to_numpy(), oof["m"].to_numpy()) if oof is not None else (0.25, 50.0)
    out = test.select("player_id", "season", "pos_group", "team", "m", pl.col(target).alias("r_total"), "no_history", "rt_total", "m_eff", "age").with_columns(pl.Series("pred", pred), pl.Series("sig_a", [a] * test.height), pl.Series("sig_b", [b] * test.height))
    out = out.rename({"r_total": "y"})
    pos_mean = tr.group_by("pos_group").agg(((pl.col(target) * pl.col("m")).sum() / pl.col("m").sum()).alias("pos_mean"))
    out = out.join(pos_mean, on="pos_group", how="left")
    shrink = pl.col("m_eff") / (pl.col("m_eff") + pd_.M0)
    last = p.select("player_id", (pl.col("season") + k).alias("season"), pl.col(target).alias("r_last"))
    out = out.join(last, on=["player_id", "season"], how="left").with_columns(pl.col("r_last").fill_null(pl.col("pos_mean")), pl.col("pos_mean").alias("b_mean"), pl.col("r_last").alias("b_last"),
                                                                                (pl.col("rt_total") + pl.col("pos_mean") * (1 - shrink)).alias("b_marcel"), pl.col("pred").alias("proj"))
    return out


def _evaluate(cfg, proc, p: pl.DataFrame, bio: pl.DataFrame, floor: float, comp_col: str = "S_rolling_refit") -> dict:
    """All gates and tables for one training floor (target-season minutes required of a training row)."""
    res: dict = {"floor": floor}
    M = Spec()
    specs = {"M": M, "total_only": Spec(comps="total"), "no_age": Spec(age=False), "no_hw": Spec(hw=False)}

    # ---- horizon 1, 2, 3 evaluation of the main spec plus baselines
    ev: dict[int, pl.DataFrame] = {}
    for k in (1, 2, 3):
        fr_t, fr_all = pd_.design_frame(p, bio, k, floor=floor), pd_.design_frame(p, bio, k, targets_only=False)
        parts = []
        for T in [t for t in TEST_SEASONS if t >= 2018 + 2 * k]:       # at least two training target seasons
            f = forecast(fr_all, fr_t, k, T, M, p)
            parts.append(f.filter(pl.col("m") >= EVAL_MIN))
        ev[k] = pl.concat(parts)
        if k == 1:
            frames1 = (fr_t, fr_all)
    e1 = ev[1]
    res["horizon"] = {}
    for k, e in ev.items():
        mse = lambda c, e=e: float((e["m"] * (e["y"] - e[c]) ** 2).sum() / e["m"].sum())  # noqa: E731
        res["horizon"][str(k)] = {"n_rows": e.height, "seasons": sorted(e["season"].unique().to_list()), **{f"mse_{c}": mse(c) for c in ("proj", "b_mean", "b_last", "b_marcel")},
                                  "z_vs_mean": paired_cluster(e, "proj", "b_mean")["z"], "z_vs_marcel": paired_cluster(e, "proj", "b_marcel")["z"], "z_vs_last": paired_cluster(e, "proj", "b_last")["z"]}
    # ---- variants at horizon 1
    var = {}
    fr_t, fr_all = frames1
    for name, spec in specs.items():
        if name == "M":
            continue
        parts = [forecast(fr_all, fr_t, 1, T, spec, p).filter(pl.col("m") >= EVAL_MIN).select("player_id", "season", "m", "y", pl.col("proj").alias(name)) for T in TEST_SEASONS]
        var[name] = pl.concat(parts)
    for m0 in (500.0, 2000.0):
        ft, fa = pd_.design_frame(p, bio, 1, m0=m0, floor=floor), pd_.design_frame(p, bio, 1, m0=m0, targets_only=False)
        parts = [forecast(fa, ft, 1, T, M, p).filter(pl.col("m") >= EVAL_MIN).select("player_id", "season", "m", "y", pl.col("proj").alias(f"m0_{int(m0)}")) for T in TEST_SEASONS]
        var[f"m0_{int(m0)}"] = pl.concat(parts)
    j = e1.select("player_id", "season", "m", "y", "proj")
    variants = {}
    for name, v in var.items():
        jj = j.join(v.select("player_id", "season", name), on=["player_id", "season"])
        variants[name] = {"mse": float((jj["m"] * (jj["y"] - jj[name]) ** 2).sum() / jj["m"].sum()), "M_vs_variant_z": paired_cluster(jj, "proj", name)["z"]}
    res["variants"] = variants
    # ---- gates PR-1..PR-5
    h1 = res["horizon"]["1"]
    w, y, x = e1["m"].to_numpy(), e1["y"].to_numpy(), e1["proj"].to_numpy()
    A = np.column_stack([np.ones(len(x)), x])
    bcoef = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
    r_ = y - A @ bcoef
    cov = np.linalg.inv(A.T @ (w[:, None] * A)) * float((w * r_**2).sum() / (len(x) - 2))
    slope = {"slope": float(bcoef[1]), "se": float(np.sqrt(cov[1, 1])), "intercept": float(bcoef[0])}
    sd = np.sqrt(e1["sig_a"].to_numpy() + e1["sig_b"].to_numpy() / e1["m"].to_numpy())
    cover = float((np.abs(y - x) <= Z80 * sd).mean())
    res["gates"] = {"PR1": {"z": h1["z_vs_mean"], "pass": bool(h1["z_vs_mean"] > 2)}, "PR2": {"z": h1["z_vs_marcel"], "pass": bool(h1["z_vs_marcel"] > 2)},
                    "PR3": {"z": variants["no_age"]["M_vs_variant_z"], "pass": bool(variants["no_age"]["M_vs_variant_z"] > 2)},
                    "PR4": {**slope, "pass": bool(0.8 <= slope["slope"] <= 1.2)}, "PR5": {"coverage_80": cover, "pass": bool(0.70 <= cover <= 0.90)}}

    # ---- team level (PR-6): fixed slope 1 on the projected GAR versus the rolling-refit last-season GAR
    wm = pl.read_parquet(proc / "wm_team_table.parquet").filter(pl.col("season") >= 2018)
    fr_t, fr_all = frames1
    sp = []
    for T in range(2019, 2026):
        f = forecast(fr_all, fr_t, 1, T, M, p)
        sp.append(f.with_columns((pl.col("proj") / 60.0 * pl.col("m")).alias("s")).group_by("team").agg(pl.col("s").sum().alias("S_proj")).with_columns(pl.lit(T, pl.Int64).alias("season")))
    wm = wm.join(pl.concat(sp).with_columns(pl.col("season").cast(wm["season"].dtype)), on=["season", "team"], how="inner")
    from src.wholemodel import build as wb
    from src.wholemodel.run import paired as team_paired
    seasons = [2021, 2022, 2023, 2024, 2025]
    tp = wb.expanding_fit(wm, ["G"], seasons, fixed_slope="S_proj")
    tr_ = wb.expanding_fit(wm, ["G"], seasons, fixed_slope=comp_col)
    tb = wb.expanding_fit(wm, ["G"], seasons)
    tb1 = wb.expanding_fit(wm, ["G", "B_gd"], seasons)
    rm = lambda q: float(np.sqrt(((q["y"] - q["pred"]) ** 2).mean()))  # noqa: E731
    tpf = wb.expanding_fit(wm, ["G", "S_proj"], seasons)                                   # the M protocol of the methodology: slope fitted on earlier seasons
    trf = wb.expanding_fit(wm, ["G", comp_col], seasons)
    tpb = wb.expanding_fit(wm, ["G", "S_proj", "B_gd"], seasons)
    trb = wb.expanding_fit(wm, ["G", comp_col, "B_gd"], seasons)
    res["team"] = {"rmse": {"projected": rm(tpf), "rolling_last_season": rm(trf), "projected_plus_prior_gd": rm(tpb), "rolling_plus_prior_gd": rm(trb), "goalie_only": rm(tb), "prior_team_gd": rm(tb1),
                            "projected_fixed_slope": rm(tp), "rolling_fixed_slope": rm(tr_)},
                   "projected_vs_rolling": team_paired(tpf, trf), "projected_vs_goalie_only": team_paired(tpf, tb), "projected_vs_prior_team_gd": team_paired(tpf, tb1),
                   "projected_plus_gd_vs_rolling_plus_gd": team_paired(tpb, trb), "fixed_slope_projected_vs_rolling": team_paired(tp, tr_), "test_seasons": seasons}
    res["gates"]["PR6"] = {"z": res["team"]["projected_vs_rolling"]["z"], "pass": bool(res["team"]["projected_vs_rolling"]["z"] > 2)}

    # ---- minutes model (reported)
    mins = []
    mfr = pd_.design_frame(p, bio, 1)                        # minutes are modeled conditional on playing at least 200 minutes
    for T in TEST_SEASONS:
        tr = train_rows(mfr, 1, T)
        te = frames1[1].filter((pl.col("season") == T) & (pl.col("m") >= MIN_TARGET_MIN) & (pl.col("m_last") > 0))
        mm = _minutes_fit(tr)
        mins.append(te.select("m", "m_last").with_columns(pl.Series("m_hat", _minutes_pred(mm, te))))
    mn = pl.concat(mins)
    res["minutes"] = {"mae_model": float((mn["m"] - mn["m_hat"]).abs().mean()), "mae_last_season": float((mn["m"] - mn["m_last"]).abs().mean()), "n": mn.height}

    return res


def run(chainfree: bool = False) -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    outdir = cfg.path(cfg.data.output_dir)
    roll = pl.read_parquet(proc / "wm_rolling_values.parquet")
    cf = pl.read_parquet(proc / "chainfree_values.parquet") if chainfree else None
    p = pd_.panel(proc, roll, cf)
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    res: dict = {"coverage": {"skater_seasons": p.height, "with_birth_date": int(p.join(bio.filter(pl.col("birth_date").is_not_null()), on="player_id", how="inner").height)}}
    if chainfree:
        # docs/chainfree/methodology.md: the revised-floor evaluation and the results-target diagnostic on chain-free component rates; published outputs are not touched
        comp = "S_rolling_rho0"
        res["revised"] = _evaluate(cfg, proc, p, bio, REVISED_FLOOR, comp)
        res["results_target"] = _evaluate_results(proc, p, bio, REVISED_FLOOR, comp)
        _save(met / "proj_validation_chainfree.json", res)
        log.info("chain-free gates %s", {k: (round(v.get("z", v.get("slope", v.get("coverage_80", 0))), 2), v["pass"]) for k, v in res["revised"]["gates"].items()})
        return res
    res["as_preregistered"] = _evaluate(cfg, proc, p, bio, pd_.MIN_TARGET_MIN)
    res["revised"] = _evaluate(cfg, proc, p, bio, REVISED_FLOOR)
    res["results_target"] = _evaluate_results(proc, p, bio, REVISED_FLOOR)
    res.update(_final(p, bio, cfg, proc, outdir, res))
    _save(met / "proj_validation.json", res)
    for tag in ("as_preregistered", "revised"):
        log.info("%s gates %s", tag, {k: (round(v.get("z", v.get("slope", v.get("coverage_80", 0))), 2), v["pass"]) for k, v in res[tag]["gates"].items()})
    from src.projection.report import run as report
    report()
    return res


def results_outcome(proc) -> pl.DataFrame:
    """Per player-season deployment-adjusted on-ice goal differential per 60 (all modeled strengths) and the minute shares by state."""
    from src.calibration.player import onice_forward_all
    dummy = pl.DataFrame({"player_id": [0], "season": [0], "r_uncal": [0.0], "r_cal": [0.0]}, schema={"player_id": pl.Int64, "season": pl.Int64, "r_uncal": pl.Float64, "r_cal": pl.Float64})
    o = onice_forward_all(proc, dummy).with_columns(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64))
    return o.select("player_id", "season", (pl.col("gd_adj") / pl.col("minutes") * 60).alias("y_res"), "sh_pp", "sh_pk", "sh_o")


def _rmse(q: pl.DataFrame) -> float:
    return float(np.sqrt(((q["y"] - q["pred"]) ** 2).mean()))


def _evaluate_results(proc, p: pl.DataFrame, bio: pl.DataFrame, floor: float, comp_col: str = "S_rolling_refit") -> dict:
    """Exploratory (methodology section 9, deviation 2): the same ridge trained on next-season *results* (adjusted on-ice goal differential per 60)
    instead of next-season value estimates, with the season's state shares as (oracle) usage controls."""
    p2 = p.join(results_outcome(proc), on=["player_id", "season"], how="left").with_columns(pl.col("y_res").fill_null(0.0), pl.col("sh_pp").fill_null(0.0), pl.col("sh_pk").fill_null(0.0), pl.col("sh_o").fill_null(0.0))
    spec = Spec(extra=("sh_pp", "sh_pk", "sh_o"))
    spec_tot = Spec(comps="total", extra=("sh_pp", "sh_pk", "sh_o"))
    ft, fa = pd_.design_frame(p2, bio, 1, floor=floor), pd_.design_frame(p2, bio, 1, targets_only=False)
    parts, part_t = [], []
    for T in TEST_SEASONS:
        parts.append(forecast(fa, ft, 1, T, spec, p2, target="y_res").filter(pl.col("m") >= EVAL_MIN))
        part_t.append(forecast(fa, ft, 1, T, spec_tot, p2, target="y_res").filter(pl.col("m") >= EVAL_MIN).select("player_id", "season", pl.col("proj").alias("total_only")))
    e = pl.concat(parts).join(pl.concat(part_t), on=["player_id", "season"])

    def mse(c: str) -> float:
        return float((e["m"] * (e["y"] - e[c]) ** 2).sum() / e["m"].sum())

    out = {"n_rows": e.height, **{f"mse_{c}": mse(c) for c in ("proj", "b_mean", "b_last", "total_only")}, "z_vs_mean": paired_cluster(e, "proj", "b_mean")["z"],
           "z_vs_last": paired_cluster(e, "proj", "b_last")["z"], "z_vs_total_only": paired_cluster(e, "proj", "total_only")["z"]}
    models = pm.fit_models(ft, spec, target="y_res")
    out["component_weights"] = {g: dict(zip(pm.feature_names(spec), models[g].raw_coef().tolist())) for g in ("F", "D")}
    wm = pl.read_parquet(proc / "wm_team_table.parquet").filter(pl.col("season") >= 2018)
    sp = []
    for T in range(2019, 2026):
        f = forecast(fa, ft, 1, T, spec, p2, target="y_res")
        sp.append(f.with_columns((pl.col("proj") / 60.0 * pl.col("m")).alias("s")).group_by("team").agg(pl.col("s").sum().alias("S_res")).with_columns(pl.lit(T, pl.Int64).alias("season")))
    wm = wm.join(pl.concat(sp).with_columns(pl.col("season").cast(wm["season"].dtype)), on=["season", "team"], how="inner")
    from src.wholemodel import build as wb
    from src.wholemodel.run import paired as team_paired
    seasons = [2021, 2022, 2023, 2024, 2025]
    t_res = wb.expanding_fit(wm, ["G", "S_res"], seasons)
    t_roll = wb.expanding_fit(wm, ["G", comp_col], seasons)
    t_g = wb.expanding_fit(wm, ["G"], seasons)
    t_b1 = wb.expanding_fit(wm, ["G", "B_gd"], seasons)
    out["team"] = {"rmse": {"results_target": _rmse(t_res), "rolling_last_season": _rmse(t_roll), "goalie_only": _rmse(t_g), "prior_team_gd": _rmse(t_b1)},
                   "results_vs_rolling": team_paired(t_res, t_roll), "results_vs_goalie_only": team_paired(t_res, t_g), "results_vs_prior_team_gd": team_paired(t_res, t_b1), "test_seasons": seasons}
    return out


def _minutes_x(d: pl.DataFrame) -> np.ndarray:
    return np.column_stack([d["m_last"].to_numpy(), d["m_mean3"].to_numpy(), d["no_history"].to_numpy(), pm.age_basis(d["age"].to_numpy())])


def _minutes_fit(tr: pl.DataFrame) -> dict:
    out = {}
    for g in ("F", "D"):
        d = tr.filter(pl.col("pos_group") == g)
        X, y, w = _minutes_x(d), d["m"].to_numpy(), np.ones(d.height)
        out[g] = pm.wridge(X, y, w, pm.choose_lambda(X, y, w, d["season"].to_numpy()))
    return out


def _minutes_pred(models: dict, d: pl.DataFrame) -> np.ndarray:
    out = np.zeros(d.height)
    for g in ("F", "D"):
        mk = (d["pos_group"] == g).to_numpy()
        if mk.any():
            out[mk] = np.maximum(models[g].predict(_minutes_x(d.filter(pl.col("pos_group") == g))), 0.0)
    return out


def future_frame(p: pl.DataFrame, bio: pl.DataFrame, target_season: int, k: int) -> pl.DataFrame:
    """Rows for players who played in season t = target_season - k, with features from seasons <= t and age at the target season."""
    h = pd_.history(p).filter(pl.col("t") == target_season - k)
    base = p.filter(pl.col("season") == target_season - k).select("player_id", "pos_group", "team", pl.col("m").alias("m_prev"))
    d = base.join(h, left_on="player_id", right_on="player_id", how="inner").join(bio, on="player_id", how="left")
    d = d.with_columns(pl.lit(target_season, pl.Int64).alias("season"), pl.lit(0.0).alias("no_history"), pd_.age_at(pl.lit(target_season), pl.col("birth_date")).alias("age"), pl.lit(0.0).alias("m"), pl.lit(0.0).alias("r_total"))
    for c in ("height_in", "weight_lb", "age"):
        med = d.group_by("pos_group").agg(pl.col(c).median().alias("_med"))
        d = d.join(med, on="pos_group", how="left").with_columns(pl.col(c).fill_null(pl.col("_med"))).drop("_med")
    return d


def _final(p: pl.DataFrame, bio: pl.DataFrame, cfg, proc, outdir, res: dict) -> dict:
    M = Spec()
    last = int(p["season"].max())
    out: dict = {}
    proj = None
    names = pl.read_parquet(proc / "players.parquet").select(pl.col("player_id").cast(pl.Int64), (pl.col("first_name") + " " + pl.col("last_name")).alias("player_name"))
    gpw = json.loads((cfg.path(cfg.data.metrics_dir) / "val_gpw.json").read_text())["goals_per_win"]
    for k in (1, 2, 3):
        fr_t = pd_.design_frame(p, bio, k, floor=REVISED_FLOOR)
        tr = fr_t.filter(pl.col("season") >= FIRST_TARGET + k - 1)
        models = pm.fit_models(tr, M)
        oof = oof_resid(tr, M, {})
        a, b = pm.variance_model(oof["m"].to_numpy(), oof["res"].to_numpy(), oof["m"].to_numpy())
        fut = future_frame(p, bio, last + k, k)              # features from history at t = last, age at the target season
        pred = pm.predict_models(models, fut, M)
        # the fixed reference sd at 1,500 minutes stands for a full-season player
        sd_full = np.sqrt(a + b / 1500.0)
        cols = fut.select("player_id", "pos_group", "team", "age", "m_prev").with_columns(pl.Series(f"rate_h{k}", pred), pl.Series(f"rate_h{k}_lo80", pred - Z80 * sd_full), pl.Series(f"rate_h{k}_hi80", pred + Z80 * sd_full))
        proj = cols if proj is None else proj.join(cols.select("player_id", f"rate_h{k}", f"rate_h{k}_lo80", f"rate_h{k}_hi80"), on="player_id", how="left")
        if k == 1:
            mm = _minutes_fit(pd_.design_frame(p, bio, 1).filter(pl.col("season") >= FIRST_TARGET))
            fut_m = fut.join(pd_.history(p).filter(pl.col("t") == last).select("player_id", "m_last", "m_mean3"), on="player_id", how="left", suffix="_h")
            fut_m = fut_m.with_columns(pl.col("m_last").fill_null(0.0), pl.col("m_mean3").fill_null(0.0))
            m_hat = _minutes_pred(mm, fut_m)
            proj = proj.with_columns(pl.Series("minutes_h1", m_hat)).with_columns((pl.col("rate_h1") * pl.col("minutes_h1") / 60).alias("gar_h1"), (pl.col("rate_h1_lo80") * pl.col("minutes_h1") / 60).alias("gar_h1_lo80"),
                                                                                    (pl.col("rate_h1_hi80") * pl.col("minutes_h1") / 60).alias("gar_h1_hi80")).with_columns((pl.col("gar_h1") / gpw).alias("wins_h1"))
            # aging curves with a player-cluster bootstrap and the fitted component weights
            ages = np.arange(20.0, 39.0)
            base_curve = pm.age_curve(models, M, ages)
            rng = np.random.default_rng(cfg.seed)
            pid = tr["player_id"].unique().to_list()
            boots = {"F": [], "D": []}
            for _ in range(200):
                pick = rng.choice(pid, len(pid))
                cnt = pl.DataFrame({"player_id": pick.astype(np.int64)}).group_by("player_id").len()
                bt = tr.join(cnt, on="player_id").with_columns((pl.col("m") * pl.col("len")).alias("m_w"))
                mb = pm.fit_models(bt, M, lam={g: models[g].lam for g in ("F", "D")}, weight="m_w")
                cb = pm.age_curve(mb, M, ages)
                for g in boots:
                    boots[g].append(cb[g])
            rows = []
            for g in ("F", "D"):
                bb = np.array(boots[g])
                for i, a_ in enumerate(ages):
                    rows.append({"pos_group": g, "age": float(a_), "effect_per60": float(base_curve[g][i]), "lo95": float(np.quantile(bb[:, i], 0.025)), "hi95": float(np.quantile(bb[:, i], 0.975))})
            pl.DataFrame(rows).write_csv(cfg.path("reports/projection/aging_curves.csv"))
            out["component_weights"] = {g: dict(zip(pm.feature_names(M), models[g].raw_coef().tolist())) for g in ("F", "D")}
            out["lambda_h1"] = {g: models[g].lam for g in ("F", "D")}
    proj = proj.join(names, on="player_id", how="left").rename({"pos_group": "pos_group"})
    proj.sort("gar_h1", descending=True).write_csv(outdir / "player_projection.csv")
    proj.write_parquet(proc / "player_projection.parquet")
    out["projection_rows"] = proj.height
    return out


def main():
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(chainfree="--chainfree" in sys.argv)


if __name__ == "__main__":
    main()
