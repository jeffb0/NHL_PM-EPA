"""Finishing talent end to end:  python -m src.finishing.run [--from STEP | --only STEP ...]

Steps: shots -> xg -> tune -> final -> stability -> report.
Decision rules are pre-registered in docs/finishing/methodology.md §7 and applied mechanically here.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import polars as pl

from src.config import load_config
from src.finishing import evaluate as ev
from src.finishing import model
from src.finishing.model import Spec
from src.valuation.replacement import status

log = logging.getLogger("finishing")
STEPS = ["shots", "xg", "tune", "final", "stability", "report"]


def _paths(cfg):
    return cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.output_dir), cfg.path(cfg.data.metrics_dir)


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_table(cfg) -> pl.DataFrame:
    """Finishing shots + xG_f + same-season fringe status + logit offset."""
    proc, _, _ = _paths(cfg)
    shots = pl.read_parquet(proc / "fin_shots.parquet")
    xg = pl.read_parquet(proc / "fin_xg.parquet").select("game_id", "event_idx", "xg_f")
    ps = pl.read_parquet(proc / "player_season.parquet")
    st = status(ps, cfg.valuation.regular_forwards, cfg.valuation.regular_defense).select(pl.col("player_id").alias("shooter"), "season", "fringe")
    df = shots.join(xg, on=["game_id", "event_idx"], how="inner").join(st, on=["shooter", "season"], how="left")
    df = df.with_columns(pl.col("fringe").fill_null(True))
    assert df.height == shots.height, "xG_f join dropped rows"
    return model.with_offset(df.select("game_id", "event_idx", "season", "shooter", "goalie", "pos_group", "goal", "xg_f", "strength_g", "tg", "fringe", "type", "shot_type_c"))


def step_tune(cfg):
    proc, _, met = _paths(cfg)
    fc = cfg.finishing
    df = load_table(cfg)
    train = df.filter(pl.col("season").is_in(fc.sigma_seasons))
    res: dict = {"n_train_shots": train.height}
    prefit_val, s2, traces = {}, {}, {}
    for v, spec in (("A", Spec.A(fringe=False)), ("B", Spec.B(fringe=False))):
        fit = model.fit_finishing(train, spec, em_iters=fc.em_iters, em_tol_sd=fc.em_tol_sd)
        s2[v], traces[v] = fit.s2, fit.trace
        prefit_val.setdefault(fc.validation_fold, {})[v] = fit
        log.info("EM variant %s: sd %s", v, {k: round(float(np.sqrt(x)), 4) for k, x in fit.s2.items()})
    res["s2"], res["em_trace"] = s2, traces
    res["sd"] = {v: {k: float(np.sqrt(x)) for k, x in d.items()} for v, d in s2.items()}
    val = ev.forward_eval(df, [fc.validation_fold], None, s2, prefit_val, fc.min_attempts_next, cfg.seed)
    res["validation"] = val
    pb = val["paired"]["MshrB_vs_MshrA"]
    use_b = bool(pb["mean_improvement"] > fc.se_threshold * pb["se"])
    res["D_type"] = {"variant": "B" if use_b else "A", **pb}
    log.info("D-type: variant %s (B vs A z=%.2f)", res["D_type"]["variant"], pb["z"])
    res["test"] = ev.forward_eval(df, fc.test_folds, None, s2, None, fc.min_attempts_next, cfg.seed)
    key = "MshrB" if use_b else "MshrA"
    t = res["test"]
    g = {"F1_beats_M0": t["paired"][f"{key}_vs_M0"], "F2_beats_Mraw": t["paired"][f"{key}_vs_Mraw"], "F3_slope": t["player_slope"][key]}
    g["F1_pass"] = bool(g["F1_beats_M0"]["z"] > fc.se_threshold)
    g["F2_pass"] = bool(g["F2_beats_Mraw"]["z"] > fc.se_threshold)
    g["F3_pass"] = bool(0.7 <= g["F3_slope"]["slope"] <= 1.4)
    res["gates"] = g
    _save(met / "fin_tune.json", res)
    log.info("gates: F1 z=%.1f F2 z=%.1f F3 slope=%.2f", g["F1_beats_M0"]["z"], g["F2_beats_Mraw"]["z"], g["F3_slope"]["slope"])


def step_final(cfg):
    proc, out_dir, met = _paths(cfg)
    fc = cfg.finishing
    tune = json.loads((met / "fin_tune.json").read_text())
    variant = tune["D_type"]["variant"]
    spec = Spec.B(fringe=True) if variant == "B" else Spec.A(fringe=True)
    df = load_table(cfg)
    fit = model.fit_finishing(df, spec, em_iters=fc.em_iters, em_tol_sd=fc.em_tol_sd)
    log.info("production fit (%s): sd %s", variant, {k: round(float(np.sqrt(x)), 4) for k, x in fit.s2.items()})
    cov = fit.mode.posterior_cov()
    # cache the fit (coefficients, layout ids, variance components): enough to score new shots or refit with a new prior
    joblib.dump({"layout": fit.layout, "beta": fit.beta, "s2": fit.s2, "spec": spec, "variant": variant, "em_trace": fit.trace}, cfg.path(f"{cfg.data.output_dir}/models") / "fin_fit.joblib")
    eff = model.player_effects(fit, df, fc.draws, cfg.seed, cov)
    meta = pl.read_parquet(proc / "players.parquet").select("player_id", (pl.col("first_name") + " " + pl.col("last_name")).alias("player_name"), "position")
    ps = pl.read_parquet(proc / "player_season.parquet").select("player_id", "season", pl.col("primary_team").alias("team"), "toi_ev")
    st = status(pl.read_parquet(proc / "player_season.parquet"), cfg.valuation.regular_forwards, cfg.valuation.regular_defense).select("player_id", "season", "fringe")
    eff = eff.join(meta, on="player_id", how="left").join(ps, on=["player_id", "season"], how="left").join(st, on=["player_id", "season"], how="left")
    eff = eff.with_columns((pl.col("goals_5v5") - pl.col("xg_f_5v5")).alias("raw_g_minus_xg_5v5")).sort("season", "fin_gaa_5v5", "player_id", descending=[False, True, False])  # player_id breaks ties (group_by order is not stable)
    eff.write_parquet(proc / "fin_player_season.parquet")
    eff.write_csv(out_dir / "fin_player_season.csv")
    # shot-level shifts, so the valuation step can re-reference replacement (sensitivity) without refitting
    parts = fit.parts(df)
    zero = np.zeros(df.height)
    l_i = df["offset"].to_numpy() + parts.get("cell", zero) + parts.get("v", zero) + parts.get("e", zero)
    T_i = parts.get("gamma", zero) + parts.get("u", zero) + parts.get("d", zero) + parts.get("w", zero) + parts.get("fringe", zero)
    df.select("shooter", "season", "pos_group", "strength_g").with_columns(pl.Series("l", l_i), pl.Series("T", T_i)).write_parquet(proc / "fin_shot_shifts.parquet")
    # persistent talent and the goalie by-product
    lay = fit.layout
    diag = np.diag(cov)
    talent = pl.DataFrame({"player_id": list(lay.shooter), "u_logit": [float(fit.beta[c] + 0) for c in lay.shooter.values()],
                           "u_sd": [float(np.sqrt(diag[c])) for c in lay.shooter.values()]}, schema={"player_id": pl.Int64, "u_logit": pl.Float64, "u_sd": pl.Float64})
    talent = talent.join(meta, on="player_id", how="left").join(df.group_by("shooter").agg(pl.len().alias("attempts"), pl.col("goal").sum().alias("goals"), pl.col("xg_f").sum().alias("xg_f")).rename({"shooter": "player_id"}), on="player_id")
    talent.sort("u_logit", "player_id", descending=[True, False]).write_csv(out_dir / "fin_player_talent.csv")
    goalies = pl.DataFrame({"player_id": list(lay.goalie), "v_logit": [float(fit.beta[c]) for c in lay.goalie.values()], "v_sd": [float(np.sqrt(diag[c])) for c in lay.goalie.values()]},
                           schema={"player_id": pl.Int64, "v_logit": pl.Float64, "v_sd": pl.Float64}).join(meta, on="player_id", how="left")
    goalies.sort("v_logit", "player_id").write_csv(out_dir / "fin_goalie_effects.csv")
    fr_lo, fr_hi = lay.blocks["fringe"]
    _save(met / "fin_final.json", {"variant": variant, "sd": {k: float(np.sqrt(x)) for k, x in fit.s2.items()}, "em_trace": fit.trace,
                                   "fringe_effect_logit": {"F": float(fit.beta[fr_lo]), "D": float(fit.beta[fr_lo + 1])},
                                   "fringe_effect_sd": {"F": float(np.sqrt(diag[fr_lo])), "D": float(np.sqrt(diag[fr_lo + 1]))},
                                   "n_shots": df.height, "n_cols": lay.n_cols, "newton_converged": fit.mode.converged})
    log.info("final: %d player-seasons; fringe effect logit F %.3f D %.3f", eff.height, fit.beta[fr_lo], fit.beta[fr_lo + 1])


def step_stability(cfg):
    _, _, met = _paths(cfg)
    fc = cfg.finishing
    df = load_table(cfg)
    ss = ev.single_season(df, None, fc.em_iters, fc.em_tol_sd, fc.min_attempts_yoy)
    final = json.loads((met / "fin_final.json").read_text())
    ss["stabilization"] = ev.stabilization(df, ss["single_season_sd"], final["sd"])
    _save(met / "fin_stability.json", ss)
    log.info("YOY model %.3f raw %.3f | split-half model %.3f raw %.3f", ss["yoy"]["model_shift"], ss["yoy"]["raw_rate"], ss["split_half"]["model_r"], ss["split_half"]["raw_r"])


def run_steps(steps: list[str]) -> None:
    cfg = load_config()
    for s in steps:
        log.info("=== finishing:%s ===", s)
        if s == "shots":
            from src.finishing.shots import run
            run()
        elif s == "xg":
            from src.finishing.xg2 import run
            run()
        elif s == "report":
            from src.finishing.report import run
            run()
        else:
            globals()[f"step_{s}"](cfg)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STEPS, default="shots")
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args()
    run_steps(args.only or STEPS[STEPS.index(args.start):])


if __name__ == "__main__":
    main()
