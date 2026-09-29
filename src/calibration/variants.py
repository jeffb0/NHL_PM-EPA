"""Exploratory (post-hoc, not pre-registered): next-season within-team prediction of alternative totals.

    python -m src.calibration.variants

Each variant replaces the total in the player-level forward check of methodology §4 and is compared with the
current `total_gar` (paired z over team-seasons; positive favors the variant). Nothing here changes a default.
"""

from __future__ import annotations

import json

import polars as pl

from src.calibration import player as P
from src.calibration import team as tm
from src.config import load_config

VARIANTS = {
    "without turnover cost": pl.col("total_gar") - pl.col("to_gar"),
    "with original (not role-adjusted) turnover": pl.col("total_gar_to_orig"),
    "without faceoffs": pl.col("total_gar") - pl.col("fo_gar"),
    "without penalties": pl.col("total_gar") - pl.col("pen_gar"),
    "without turnover and penalties": pl.col("total_gar") - pl.col("to_gar") - pl.col("pen_gar"),
    "without turnover, faceoffs, penalties": pl.col("total_gar") - pl.col("to_gar") - pl.col("fo_gar") - pl.col("pen_gar"),
}


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    t = pl.read_parquet(proc / "player_value_calibrated.parquet")
    out = {}
    for name, expr in VARIANTS.items():
        alt = t.with_columns(expr.alias("total_gar_ridge"))  # the 'calibrated' slot of the forward check carries the variant
        r = P.rate_table(alt, tm.COMPS)
        d = P.dataset(alt, P.onice_forward(proc, r), r, tm.COMPS, 200.0)
        res = P.evaluate(d, tm.COMPS)
        out[name] = {"own_slope_goals": res["y_gd"]["calibrated"]["own_slope"], "own_se": res["y_gd"]["calibrated"]["own_se"],
                     "paired_z_goals": res["y_gd"]["paired_z_calibrated_better"], "paired_z_xg": res["y_xgd"]["paired_z_calibrated_better"]}
    met = cfg.path(cfg.data.metrics_dir)
    (met / "cal_variants.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:42s} slope {v['own_slope_goals']:.2f} ± {v['own_se']:.2f}  z goals {v['paired_z_goals']:6.2f}  z xG {v['paired_z_xg']:6.2f}")
