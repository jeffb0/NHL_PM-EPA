"""Value integration end to end:  python -m src.valuation.run [--only STEP ...]

Steps: gpw -> combine -> report.  Needs the finishing step's outputs (fin_player_season, fin_shot_shifts)
and the defense stints. Methodology: docs/finishing/methodology.md §8-§9.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.valuation import combine as cb
from src.valuation import gpw as gp
from src.valuation.replacement import status

log = logging.getLogger("valuation")
STEPS = ["gpw", "combine", "report"]


def _paths(cfg):
    return cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.output_dir), cfg.path(cfg.data.metrics_dir)


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float))


def step_gpw(cfg):
    proc, _, met = _paths(cfg)
    ts = gp.team_seasons(proc)
    res = gp.goals_per_win(ts, cfg.valuation.gpw_bootstrap, cfg.seed)
    _save(met / "val_gpw.json", res)
    log.info("goals per win %.2f (95%% CI %.2f-%.2f), R2 %.3f, n=%d", res["goals_per_win"], *res["ci95"], res["r2"], res["n_team_seasons"])


def load_extras(proc) -> dict[str, pl.DataFrame]:
    """Turnover, faceoff and penalty components (src/components), when they have been built."""
    spec = {"to": ("to_player_season.parquet", ["to_gar", "to_gar_sd", "to_gar_unadj", "to_rate_per60", "to_rep_per60"]), "fo": ("fo_player_season.parquet", ["fo_gar", "fo_gar_sd", "fo_draws", "fo_gaa"]),
            "pen": ("pen_player_season.parquet", ["pen_gar", "pen_gar_sd", "drawn_per60_post", "taken_per60_post"])}
    return {k: pl.read_parquet(proc / f).select("player_id", "season", *cols) for k, (f, cols) in spec.items() if (proc / f).exists()}


def build_table(cfg, n_f: int, n_d: int, ps, dpm, shifts, fin_ps, sel, gpw: float, refit_cache: dict):
    st = status(ps, n_f, n_d)
    pm, rep_pm = cb.pm_component(ps, st, cfg.valuation.min_toi)
    key = (n_f, n_d)
    if key not in refit_cache:
        stints = pl.read_parquet(_paths(cfg)[0] / "def_stints.parquet")
        refit_cache[key] = cb.dpm_refit(stints, st, sorted(cfg.data.seasons), sel["alphas"]["xg"], sel["D2"]["rho"])
    refit, fr = refit_cache[key]
    dp = cb.dpm_component(refit, fr, st, ps, dpm)
    fin_alt = cb.fin_from_shifts(shifts, st)
    base = (n_f, n_d) == (cfg.valuation.regular_forwards, cfg.valuation.regular_defense)
    if base:  # the model's own values (with posterior draws); must agree with the shift-based recomputation
        chk = fin_ps.select("player_id", "season", "fin_gar_5v5").join(fin_alt, on=["player_id", "season"], how="left").with_columns((pl.col("fin_gar_5v5") - pl.col("fin_gar").fill_null(0)).abs().alias("d"))
        log.info("finishing recomputation check: max abs diff %.2e", chk["d"].max())
        assert chk["d"].max() < 1e-6, "finishing shifts file inconsistent with the fitted player effects"
        fin = fin_ps.select("player_id", "season", pl.col("fin_gar_5v5").alias("fin_gar"))
        sd = fin_ps.select("player_id", "season", "fin_gar_sd", "fin_gar_lo", "fin_gar_hi")
    else:
        fin, sd = fin_alt, None
    memo = fin_ps.select("player_id", "season", pl.col("attempts_5v5").alias("fin_attempts_5v5"), pl.col("raw_g_minus_xg_5v5"), pl.col("fin_gaa_5v5"),
                         pl.col("fin_gar_other").alias("fin_gar_other_strengths_memo"), pl.col("attempts_other").alias("fin_attempts_other_memo"))
    t = cb.combine(ps.join(st.select("player_id", "season", "fringe"), on=["player_id", "season"], how="left"), pm, dp, fin, sd, memo, gpw, load_extras(_paths(cfg)[0]))
    t = t.join(st.select("player_id", "season", "fringe"), on=["player_id", "season"], how="left") if "fringe" not in t.columns else t
    ra_path = _paths(cfg)[0] / "to_ra_player_season.parquet"
    if ra_path.exists() and "to_gar" in t.columns:   # role-adjusted turnover variant: memo columns, and the total with it substituted for TO
        ra = pl.read_parquet(ra_path).select("player_id", "season", "to_ra_gar", "to_ra_gar_sd", "inv60")
        t = t.join(ra, on=["player_id", "season"], how="left").with_columns(pl.col("to_ra_gar").fill_null(0.0), pl.col("to_ra_gar_sd").fill_null(0.0))
        t = t.with_columns((pl.col("total_gar") - pl.col("to_gar") + pl.col("to_ra_gar")).alias("total_gar_to_ra"))
    return t, rep_pm, fr


def step_combine(cfg):
    proc, out_dir, met = _paths(cfg)
    ps = pl.read_parquet(proc / "player_season.parquet")
    dpm = pl.read_parquet(proc / "dpm_player_season.parquet")
    shifts = pl.read_parquet(proc / "fin_shot_shifts.parquet")
    fin_ps = pl.read_parquet(proc / "fin_player_season.parquet")
    sel = json.loads((met / "dpm_selection.json").read_text())
    gpw = json.loads((met / "val_gpw.json").read_text())["goals_per_win"]
    vc = cfg.valuation
    cache: dict = {}
    t, rep_pm, fr = build_table(cfg, vc.regular_forwards, vc.regular_defense, ps, dpm, shifts, fin_ps, sel, gpw, cache)
    keep = ["player_id", "player_name", "team", "position", "pos_group", "season", "toi_5v5", "fringe",
            "pm_gar", "def_gar", "fin_gar", "to_gar", "fo_gar", "pen_gar", "total_gar", "total_gar_lo", "total_gar_hi", "total_war", "total_war_sd",
            "pm_wins", "def_wins", "fin_wins", "to_wins", "fo_wins", "pen_wins", "pm_gar_sd", "def_gar_sd", "fin_gar_sd", "to_gar_sd", "fo_gar_sd", "pen_gar_sd", "total_gar_sd", "to_gar_unadj", "to_ra_gar", "to_ra_gar_sd", "total_gar_to_ra", "inv60", "fo_draws", "drawn_per60_post", "taken_per60_post", "def_gar_perseason", "total_gar_perseason",
            "pm_post_per60", "pm_rep_per60", "dpm_var_per60", "dpm_rep_per60", "pm_shrink_B",
            "fin_attempts_5v5", "raw_g_minus_xg_5v5", "fin_gaa_5v5", "fin_gar_other_strengths_memo", "fin_attempts_other_memo",
            "pm_epa_per60_ev", "dpm_val_per60", "qualified_ev"]
    t = t.with_columns(((pl.col("pm_epa_per60_ev") - pl.col("pm_rep_per60")) * pl.col("toi_5v5") / 60).fill_null(0.0).alias("pm_raw_gar"),
                       ((pl.col("to_rate_per60") - pl.col("to_rep_per60")) * pl.col("toi_5v5") / 60).fill_null(0.0).alias("to_raw_gar"))   # unshrunk memo values
    t = t.select([c for c in keep if c in t.columns] + ["pm_raw_gar", "to_raw_gar"])
    t.write_parquet(proc / "player_value_season.parquet")
    t.write_csv(out_dir / "player_value_season.csv")
    log.info("player_value_season: %d rows", t.height)
    # replacement levels actually used
    dpm_rep = fr.select("season", pl.col("dpm_rep_per60_F").alias("dpm_rep_F"), pl.col("dpm_rep_per60_D").alias("dpm_rep_D"))
    pm_rep = rep_pm.pivot(on="pos_group", index="season", values="pm_rep_per60").rename({"F": "pm_rep_F", "D": "pm_rep_D"})
    reps = pm_rep.join(dpm_rep, on="season").sort("season")
    # sensitivity to the regular/fringe cutoffs
    sens = {}
    base_tot = t.select("player_id", "season", "total_gar")
    for n_f, n_d in vc.sensitivity:
        ta, _, _ = build_table(cfg, n_f, n_d, ps, dpm, shifts, fin_ps, sel, gpw, cache)
        j = base_tot.join(ta.select("player_id", "season", pl.col("total_gar").alias("alt"), "toi_5v5"), on=["player_id", "season"])
        q = j.filter(pl.col("toi_5v5") >= vc.min_toi)
        sens[f"{n_f}F_{n_d}D"] = {"corr_total_gar": float(np.corrcoef(q["total_gar"], q["alt"])[0, 1]),
                                  "mean_shift_goals_per_1000min": float(((q["alt"] - q["total_gar"]) / q["toi_5v5"] * 1000).mean()),
                                  "rank_corr_spearman": float(q.select(pl.corr(pl.col("total_gar").rank(), pl.col("alt").rank()))[0, 0])}
    q = t.filter(pl.col("toi_5v5") >= vc.min_toi)
    comps = [c for c in ("pm_gar", "def_gar", "fin_gar", "to_gar", "fo_gar", "pen_gar") if c in q.columns]
    comp_sd = {c: float(q[c].std()) for c in (*comps, "total_gar")}
    comp_corr = {f"{a}~{b}": float(np.corrcoef(q[a], q[b])[0, 1]) for i, a in enumerate(comps) for b in comps[i + 1:]}
    rec = cb.reconcile(t, proc)
    pooled = {"F": float(reps["dpm_rep_F"].mean()), "D": float(reps["dpm_rep_D"].mean()), "sd_across_seasons_F": float(reps["dpm_rep_F"].std()), "sd_across_seasons_D": float(reps["dpm_rep_D"].std()),
              "goals_per_1000min_per_sd_F": float(reps["dpm_rep_F"].std() * 1000 / 60), "goals_per_1000min_per_sd_D": float(reps["dpm_rep_D"].std() * 1000 / 60)}
    _save(met / "val_combine.json", {"replacement_rates": reps.to_dicts(), "dpm_pooled_replacement": pooled, "dpm_fringe_effects": fr.to_dicts(), "sensitivity": sens, "component_sd_qualified": comp_sd,
                                     "component_corr_qualified": comp_corr, "reconciliation": rec, "goals_per_win": gpw,
                                     "mean_total_gar_qualified": float(q["total_gar"].mean()), "n_qualified": q.height})
    log.info("reconciliation joint slopes %s; sd GAR %s", rec["joint"], comp_sd)


def run_steps(steps: list[str]) -> None:
    cfg = load_config()
    for s in steps:
        log.info("=== valuation:%s ===", s)
        if s == "report":
            from src.valuation.report import run
            run()
        else:
            globals()[f"step_{s}"](cfg)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args()
    run_steps(args.only or STEPS)


if __name__ == "__main__":
    main()
