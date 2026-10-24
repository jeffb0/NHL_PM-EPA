"""Supplementary V(s) evaluation on the *control* surface (the surface behind movement deltas).

The bake-off's headline MSE is dominated by faceoff and loose-puck starts, whose value
depends mostly on kind + coarse zone. Movement deltas mostly difference `control` values at
fine locations, so we separately report, per candidate:
  * out-of-sample MSE / R^2 on validation-season takeaway rows (the only alive `control` rows)
  * season-to-season stability of the control surface on a fixed 2-ft rink grid (5v5)
  * spatial spread (SD of V over the 5v5 attacking zone) — does the spec resolve space at all?
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.value.fit import alive_frame, make, parse_spec, score, terminal_values

log = logging.getLogger(__name__)
SPECS = ["mc/zones/k=auto", "mc/grid20/k=auto", "mc/grid10/k=auto", "gbm/cont"]


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    poss = terminal_values(pl.read_parquet(proc / "possessions.parquet"), pl.read_parquet(proc / "shots_xg.parquet"))
    alive = alive_frame(pl.read_parquet(proc / "observations.parquet"), poss, cfg.epa.exclude_empty_net)
    tr = alive.filter(pl.col("season").is_in(cfg.splits.train))
    va = alive.filter(pl.col("season").is_in(cfg.splits.validation) & (pl.col("obs_kind") == "takeaway"))
    xs, ys = np.meshgrid(np.arange(-99, 100, 2.0), np.arange(-42, 42.5, 2.0))
    grid = pl.DataFrame({"obs_kind": "takeaway", "x": xs.ravel(), "y": ys.ravel(), "strength": "5v5", "score_diff": 0})
    oz = (grid["x"] >= 25).to_numpy()
    out = {"n_valid_control_rows": va.height}
    seasons = cfg.data.scoring_regimes[0]
    for name in SPECS:
        spec = parse_spec(name)
        m = make(spec, cfg.seed).fit(tr)
        p = m.predict(va)
        s = {k: v for k, v in score(p, va["G"].to_numpy(), va["goal"].to_numpy()).items() if k != "reliability"}
        surf = m.predict(grid)
        cors = []
        for a, b in zip(seasons[:-1], seasons[1:]):
            sa = make(spec, cfg.seed).fit(alive.filter(pl.col("season") == a)).predict(grid)
            sb = make(spec, cfg.seed).fit(alive.filter(pl.col("season") == b)).predict(grid)
            cors.append(float(np.corrcoef(sa, sb)[0, 1]))
        out[name] = {
            **s,
            "surface_stability_mean_r": float(np.mean(cors)),
            "surface_stability_pairs": cors,
            "oz_spatial_sd": float(surf[oz].std()),
            "max_v": float(surf.max()),
            "min_v": float(surf.min()),
        }
        log.info(
            "control %-18s mse=%.6f r2=%.4f stab=%.3f oz_sd=%.4f",
            name,
            s["mse"],
            s["r2_vs_const"],
            np.mean(cors),
            surf[oz].std(),
        )
    (cfg.path(cfg.data.metrics_dir) / "value_control_eval.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
