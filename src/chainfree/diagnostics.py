"""Exploratory diagnostics on why projected rates lose to last season's rates at the team level
(docs/chainfree/methodology.md §7).

    python -m src.chainfree.diagnostics

Each team-level number is the M protocol of docs/wholemodel (goalie proxy plus the row's roster-weighted sum, slope
fitted on earlier
seasons) over test seasons 2021-2025, compared with last season's rolling-refit rates. Nothing here is a gate.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.projection import data as pd_
from src.projection import run as pr
from src.projection.model import Spec
from src.wholemodel import build as wb
from src.wholemodel.run import paired

log = logging.getLogger("cf_diagnostics")
SEASONS = [2021, 2022, 2023, 2024, 2025]


def _rmse(q: pl.DataFrame) -> float:
    return float(np.sqrt(((q["y"] - q["pred"]) ** 2).mean()))


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    wm0 = pl.read_parquet(proc / "wm_team_table.parquet").filter(pl.col("season") >= 2019)
    base = wb.expanding_fit(wm0, ["G", "S_rolling_refit"], SEASONS)
    res: dict = {"rolling_last_season_rmse": _rmse(base)}

    # (a) how much history: sums of last season's rates, of two seasons, of the 5/4/3 Marcel weights, of a 70/30 blend
    fa = pd_.design_frame(p, bio, 1, m0=1e-6, targets_only=False)
    p1 = p.select(
        "player_id", (pl.col("season") + 1).alias("season"), pl.col("r_total").alias("r1"), pl.col("m").alias("m1")
    )
    p2 = p.select(
        "player_id", (pl.col("season") + 2).alias("season"), pl.col("r_total").alias("r2"), pl.col("m").alias("m2")
    )
    d = (
        fa.join(p1, on=["player_id", "season"], how="left")
        .join(p2, on=["player_id", "season"], how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in ("r1", "r2", "m1", "m2")])
    )
    two = (pl.col("r1") * pl.col("m1") + pl.col("r2") * pl.col("m2")) / (pl.col("m1") + pl.col("m2")).clip(1e-9, None)
    d = d.with_columns(
        (pl.col("r1") / 60 * pl.col("m")).alias("s_last"),
        ((pl.col("r1") * 0.7 + pl.col("r2") * 0.3) / 60 * pl.col("m")).alias("s_blend_70_30"),
        (two / 60 * pl.col("m")).alias("s_two_seasons"),
        (pl.col("rbar_total") / 60 * pl.col("m")).alias("s_marcel_5_4_3"),
    )
    s = d.group_by("season", "team").agg(
        *[pl.col(c).sum() for c in ("s_last", "s_blend_70_30", "s_two_seasons", "s_marcel_5_4_3")]
    )
    wm = wm0.join(s.with_columns(pl.col("season").cast(wm0["season"].dtype)), on=["season", "team"], how="inner")
    res["history_length"] = {}
    for c in ("s_last", "s_blend_70_30", "s_two_seasons", "s_marcel_5_4_3"):
        q = wb.expanding_fit(wm, ["G", c], SEASONS)
        res["history_length"][c] = {"rmse": _rmse(q), "z_vs_rolling": paired(q, base)["z"]}

    # (b) the extra ballast M0 of the projection features
    res["ballast_sweep"] = {}
    for m0 in (1e-6, 250.0, 500.0, 1000.0):
        ft, fb = pd_.design_frame(p, bio, 1, m0=m0, floor=20.0), pd_.design_frame(p, bio, 1, m0=m0, targets_only=False)
        rows = [
            pr.forecast(fb, ft, 1, T, Spec(), p)
            .with_columns((pl.col("proj") / 60 * pl.col("m")).alias("s"))
            .group_by("team")
            .agg(pl.col("s").sum().alias("S_x"))
            .with_columns(pl.lit(T, pl.Int64).alias("season"))
            for T in range(2019, 2026)
        ]
        w = wm0.join(
            pl.concat(rows).with_columns(pl.col("season").cast(wm0["season"].dtype)), on=["season", "team"], how="inner"
        )
        q = wb.expanding_fit(w, ["G", "S_x"], SEASONS)
        res["ballast_sweep"][f"{m0:.0f}"] = {"team_rmse": _rmse(q), "z_vs_rolling": paired(q, base)["z"]}

    # (c) within-season dispersion and correlation of the summed values with team goal differential
    ft, fb = pd_.design_frame(p, bio, 1, floor=20.0), pd_.design_frame(p, bio, 1, targets_only=False)
    rows = [
        pr.forecast(fb, ft, 1, T, Spec(), p)
        .with_columns((pl.col("proj") / 60 * pl.col("m")).alias("s"))
        .group_by("team")
        .agg(pl.col("s").sum().alias("S_proj"))
        .with_columns(pl.lit(T, pl.Int64).alias("season"))
        for T in range(2019, 2026)
    ]
    w = wm0.join(
        pl.concat(rows).with_columns(pl.col("season").cast(wm0["season"].dtype)), on=["season", "team"], how="inner"
    )
    z = w.with_columns(
        [(pl.col(c) - pl.col(c).mean().over("season")).alias(c + "_c") for c in ("S_proj", "S_rolling_refit", "gd_all")]
    )
    res["within_season"] = {
        "corr_projected": float(np.corrcoef(z["S_proj_c"], z["gd_all_c"])[0, 1]),
        "corr_last_season": float(np.corrcoef(z["S_rolling_refit_c"], z["gd_all_c"])[0, 1]),
        "sd_projected": float(z["S_proj_c"].std()),
        "sd_last_season": float(z["S_rolling_refit_c"].std()),
    }
    (met / "cf_diagnostics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("history length %s", {k: round(v["rmse"], 2) for k, v in res["history_length"].items()})
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
