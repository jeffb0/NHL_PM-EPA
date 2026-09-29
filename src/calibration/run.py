"""Joint calibration end to end:  python -m src.calibration.run [--only STEP ...]

Steps: team -> player -> report. Needs the integrated table (``python -m src.valuation.run``).
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
STEPS = ["team", "player", "report"]


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


def run_steps(steps: list[str]) -> None:
    cfg = load_config()
    for s in steps:
        log.info("=== calibration:%s ===", s)
        if s == "report":
            from src.calibration.report import run
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
