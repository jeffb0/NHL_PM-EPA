"""Joint calibration end to end:  python -m src.calibration.run [--only STEP ...]

Steps: team -> player -> team_all -> player_all -> report (team_all and player_all are the all-strength extension, methodology §5b; team_gk and player_gk add the goalie, §5c). Needs the integrated table (``python -m src.valuation.run``).
Writes reports/metrics/cal_*.json and data/outputs/player_value_calibrated.csv (methodology in docs/calibration/).
"""

from __future__ import annotations

import argparse
import json
import logging

import polars as pl

from src.calibration import player as pl_
from src.calibration import team as tm
from src.calibration import unshrunk
from src.config import load_config

log = logging.getLogger("calibration")
STEPS = ["team", "player", "team_all", "player_all", "team_gk", "player_gk", "report"]


def _paths(cfg):
    return cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.output_dir), cfg.path(cfg.data.metrics_dir)


def step_team(cfg):
    proc, out_dir, met = _paths(cfg)
    t = unshrunk.build(proc)
    j = tm.team_table(t, proc)
    res = tm.calibrate(j, tm.COMPS, boot=1000, seed=cfg.seed)
    res["shrunk_ols"] = tm.shrunk_ols(j, tm.COMPS)
    res["xg_to_goals_factor"] = tm.xg_to_goals_factor(proc)
    res["n_team_seasons"] = j.height
    gpw = json.loads((met / "val_gpw.json").read_text())["goals_per_win"]
    w = res["weights_applied"]
    res["gate_CAL4_pass"] = bool(all(0.5 <= v <= 2.0 for v in w.values()))
    tc = tm.apply_weights(t, w, gpw)
    tc = tm.apply_weights(tc, res["weights_ridge"], gpw, tag="ridge")      # the ridge weights, reported and validated whether or not adopted
    res["mean_abs_change_gar_ridge_qualified"] = float(tc.filter(pl.col("toi_5v5") >= cfg.valuation.min_toi)["ridge_change_gar"].abs().mean())
    res["mean_abs_change_gar_qualified"] = float(tc.filter(pl.col("toi_5v5") >= cfg.valuation.min_toi)["cal_change_gar"].abs().mean())
    res["gate_CAL5_pass"] = bool((not res["adopt"]) == (tc["cal_change_gar"].abs().max() < 1e-9))
    tc.write_parquet(proc / "player_value_calibrated.parquet")
    tc.write_csv(out_dir / "player_value_calibrated.csv")
    (met / "cal_team.json").write_text(json.dumps(res, indent=2, default=float))
    log.info("calibration: lambda=%.3g weights %s; paired z vs ones %.2f -> %s", res["lambda"], {k: round(v, 2) for k, v in res["weights_ridge"].items()}, res["paired_z_calibrated_vs_ones"],
             "ADOPTED" if res["adopt"] else "weights set to 1")


def step_player(cfg):
    proc, _, met = _paths(cfg)
    t = pl.read_parquet(proc / "player_value_calibrated.parquet")
    rates = pl_.rate_table(t, tm.COMPS)
    oi = pl_.onice_forward(proc, rates)
    d = pl_.dataset(t, oi, rates, tm.COMPS, min_minutes=200.0)
    res = pl_.evaluate(d, tm.COMPS)
    res["gate_CAL3_pass"] = bool(res["y_gd"]["paired_z_calibrated_better"] > -2.0)
    (met / "cal_player.json").write_text(json.dumps(res, indent=2, default=float))
    log.info("player-level forward check: %d player-seasons; own slope (goals) %.2f; calibrated-better z %.2f", res["n_player_seasons"], res["y_gd"]["calibrated"]["own_slope"], res["y_gd"]["paired_z_calibrated_better"])


def _team_all(cfg, gk: bool = False):
    """All-strength calibration (methodology §5b); with ``gk`` the team's goalie value is an eleventh regressor (§5c) and outputs carry the suffix ``_gk``."""
    proc, out_dir, met = _paths(cfg)
    sfx = "_gk" if gk else ""
    comps = tm.COMPS_ALL_GK if gk else tm.COMPS_ALL
    if not (proc / "st_rapm_player_season.parquet").exists() or (gk and not (proc / "goalie_team_season.parquet").exists()):
        log.warning("required outputs missing: skipping the all-strength calibration%s", " with the goalie" if gk else "")
        return
    t = unshrunk.build(proc, all_strength=True)
    j = tm.team_table(t, proc, comps, target="gd_all")
    res = tm.calibrate(j, comps, boot=1000, seed=cfg.seed, target="gd_all")
    res["shrunk_ols"] = tm.shrunk_ols(j, comps, target="gd_all")
    res["n_team_seasons"] = j.height
    res["sd_team_gd_all"] = float(j["gd_all"].std())
    gpw = json.loads((met / "val_gpw.json").read_text())["goals_per_win"]
    w = res["weights_applied"]
    res["gate_CAL4_pass"] = bool(all(0.5 <= v <= 2.0 for v in w.values()))
    sk = lambda d: {k: v for k, v in d.items() if k != "gk"}   # noqa: E731  skater weights apply to skater columns
    tc = tm.apply_weights_all(t, sk(w), gpw, "cal")
    tc = tm.apply_weights_all(tc, sk(res["weights_ridge"]), gpw, "ridge")
    q = tc.filter(pl.col("toi_5v5") >= cfg.valuation.min_toi)
    res["mean_abs_change_gar_ridge_qualified"] = float(q["ridge_change_gar"].abs().mean())
    res["gate_CAL5_pass"] = bool((not res["adopt"]) == (tc["cal_change_gar"].abs().max() < 1e-9))
    if gk:
        g = pl.read_parquet(proc / "goalie_player_season.parquet").with_columns((pl.col("goalie_gar") * w["gk"]).alias("goalie_gar_cal"), (pl.col("goalie_gar") * res["weights_ridge"]["gk"]).alias("goalie_gar_ridge"))
        g.write_parquet(proc / "goalie_value_calibrated.parquet")
        g.write_csv(out_dir / "goalie_value_calibrated.csv")
        # one table for every player: skaters (all-strength total) and goalies (goalie value), calibrated columns follow the decision (weights stay 1 unless adopted)
        sk_rows = tc.select("player_id", "player_name", "position", "season", "team", pl.lit("skater").alias("role"), (pl.col("toi_5v5") + sum((pl.col(f"st_{c}_min") for c in ("pp", "pk", "4v4", "3v3")), pl.lit(0.0))).alias("usage"),
                            pl.lit("minutes").alias("usage_unit"), pl.col("total_gar_all").alias("gar_uncalibrated"), pl.col("total_gar_all_cal").alias("gar"), pl.col("total_gar_all_sd").alias("gar_sd"))
        gk_rows = g.select("player_id", "player_name", pl.lit("G").alias("position"), "season", "team", pl.lit("goalie").alias("role"), pl.col("shots").cast(pl.Float64).alias("usage"), pl.lit("shots faced").alias("usage_unit"),
                           pl.col("goalie_gar").alias("gar_uncalibrated"), pl.col("goalie_gar_cal").alias("gar"), pl.col("gsaa_sd").alias("gar_sd"))
        allin = pl.concat([sk_rows.with_columns(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64)), gk_rows.with_columns(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64))]).with_columns(
            (pl.col("gar") / gpw).alias("wins")).sort("season", "gar", descending=[False, True])
        allin.write_parquet(proc / "player_value_all_in.parquet")
        allin.write_csv(out_dir / "player_value_all_in.csv")
    tc.write_parquet(proc / f"player_value_calibrated_all{sfx}.parquet")
    tc.write_csv(out_dir / f"player_value_calibrated_all{sfx}.csv")
    (met / f"cal_team_all{sfx}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("all-strength calibration%s: lambda=%.3g weights %s; paired z vs ones %.2f -> %s", " with the goalie" if gk else "", res["lambda"], {k: round(v, 2) for k, v in res["weights_ridge"].items()},
             res["paired_z_calibrated_vs_ones"], "ADOPTED" if res["adopt"] else "weights set to 1")


def _player_all(cfg, gk: bool = False):
    proc, _, met = _paths(cfg)
    sfx = "_gk" if gk else ""
    if not (proc / f"player_value_calibrated_all{sfx}.parquet").exists():
        return
    t = pl.read_parquet(proc / f"player_value_calibrated_all{sfx}.parquet")
    rates = pl_.rate_table_all(t, tm.COMPS_ALL)
    oi = pl_.onice_forward_all(proc, rates)
    d = pl_.dataset(t, oi, rates, tm.COMPS_ALL, min_minutes=250.0)
    res = pl_.evaluate(d, tm.COMPS_ALL)                         # as pre-registered: raw all-strength on-ice differential per 60
    res["gate_CAL3_pass"] = bool(res["y_gd"]["paired_z_calibrated_better"] > -2.0)
    adj = pl_.evaluate(d, tm.COMPS_ALL, ys=("y_gd_adj", "y_xgd_adj"), controls=("sh_pp", "sh_pk", "sh_o"))   # deployment-adjusted (methodology §7 deviation 4)
    res["deployment_adjusted"] = adj
    res["gate_CAL3_pass_adjusted"] = bool(adj["y_gd_adj"]["paired_z_calibrated_better"] > -2.0)
    (met / f"cal_player_all{sfx}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("all-strength player check%s: %d player-seasons; own slope (goals) %.2f; ridge-better z %.2f", " with the goalie" if gk else "", res["n_player_seasons"], res["y_gd"]["calibrated"]["own_slope"], res["y_gd"]["paired_z_calibrated_better"])


def step_team_all(cfg):
    _team_all(cfg, False)


def step_player_all(cfg):
    _player_all(cfg, False)


def step_team_gk(cfg):
    _team_all(cfg, True)


def step_player_gk(cfg):
    _player_all(cfg, True)


def run_steps(steps: list[str]) -> None:
    cfg = load_config()
    for s in steps:
        log.info("=== calibration:%s ===", s)
        if s == "report":
            from src.calibration.report import run
            run()
            from src.calibration.report_all import run as run_all
            run_all()
            run_all(gk=True)
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
