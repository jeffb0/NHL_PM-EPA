"""Arena normalization variants for the turnover charge (docs/components/methodology.md §9, deviation 7).

Compares, out of sample on visiting players (fit on odd game ids, score on even and vice versa):
  none            each visitor's game rate = his own away rate elsewhere that season
  additive        + an arena x season shift per 60                       (the adopted adjustment)
  multiplicative  x an arena x season factor (count-proportional bias)
and reports whether the direct (recorded giveaway) and pooled (unrecorded loss) parts have different arena effects.

    python -m src.components.arena_variants
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.components import common, turnover
from src.config import load_config

log = logging.getLogger(__name__)


def _visitors(proc, gt: pl.DataFrame) -> pl.DataFrame:
    toi = pl.read_parquet(proc / "toi_game.parquet").select("game_id", "player_id", "team_id", "toi_5v5").join(common.good_game_ids(proc), on="game_id")
    g = pl.read_parquet(proc / "games.parquet").select("game_id", "season", "home_id")
    d = toi.join(gt.drop("season"), on=["game_id", "player_id"], how="left").join(g, on="game_id").with_columns(
        pl.col("to_total").fill_null(0.0), (pl.col("team_id") != pl.col("home_id")).alias("away"))
    v = d.filter(pl.col("away") & (pl.col("toi_5v5") > 120)).with_columns(pl.col("to_total").sum().over("player_id", "season").alias("sx"), pl.col("toi_5v5").sum().over("player_id", "season").alias("st"))
    return v.filter(pl.col("st") - pl.col("toi_5v5") > 1800).with_columns(
        (pl.col("to_total") / pl.col("toi_5v5") * 3600).alias("y"), ((pl.col("sx") - pl.col("to_total")) / (pl.col("st") - pl.col("toi_5v5")) * 3600).alias("base"))


def holdout_error(v: pl.DataFrame) -> dict:
    sse = {"none": 0.0, "additive": 0.0, "multiplicative": 0.0}
    n = 0.0
    for par in (0, 1):
        tr, te = v.filter(pl.col("game_id") % 2 == par), v.filter(pl.col("game_id") % 2 != par)
        tr = tr.with_columns((pl.col("y") - pl.col("base")).alias("r"))
        a = tr.group_by("season", "home_id").agg(((pl.col("r") * pl.col("toi_5v5")).sum() / pl.col("toi_5v5").sum()).alias("a"),
                                                 ((pl.col("base") * pl.col("r") * pl.col("toi_5v5")).sum() / (pl.col("base") ** 2 * pl.col("toi_5v5")).sum()).alias("m"))
        t = te.join(a, on=["season", "home_id"], how="left").with_columns(pl.col("a").fill_null(0.0), pl.col("m").fill_null(0.0))
        w, y, b = t["toi_5v5"].to_numpy(), t["y"].to_numpy(), t["base"].to_numpy()
        sse["none"] += float((w * (y - b) ** 2).sum())
        sse["additive"] += float((w * (y - (b + t["a"].to_numpy())) ** 2).sum())
        sse["multiplicative"] += float((w * (y - b * (1 + t["m"].to_numpy())) ** 2).sum())
        n += float(w.sum())
    mse = {k: x / n for k, x in sse.items()}
    return {"mse": mse, "gain_additive_pct": 100 * (1 - mse["additive"] / mse["none"]), "gain_multiplicative_pct": 100 * (1 - mse["multiplicative"] / mse["none"])}


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    gt = turnover.game_terms(proc)
    v = _visitors(proc, gt)
    out = {"holdout": {"2017-2023": holdout_error(v.filter(pl.col("season") <= 2023)), "2024-25": holdout_error(v.filter(pl.col("season") >= 2024))}}
    parts = {c: turnover.arena_effects(proc, gt, c) for c in ("to_direct", "to_pooled", "to_total")}
    out["excess_arena_sd"] = {c: {s: v_["arena_sd_excess"] for s, v_ in summ.items()} for c, (_, summ) in parts.items()}
    e = {c: eff.rename({"arena_effect": c}) for c, (eff, _) in parts.items()}
    d = e["to_direct"].join(e["to_pooled"], on=["season", "home_id"]).join(e["to_total"], on=["season", "home_id"])
    pre = d.filter(pl.col("season") <= 2023)
    out["corr_direct_pooled_2017_2023"] = float(np.corrcoef(pre["to_direct"], pre["to_pooled"])[0, 1])
    persist = {}
    for c in ("to_direct", "to_pooled", "to_total"):
        n = d.select("home_id", (pl.col("season") - 1).alias("season"), pl.col(c).alias("n"))
        j = d.join(n, on=["home_id", "season"]).filter(pl.col("season") <= 2022)
        persist[c] = float(np.corrcoef(j[c], j["n"])[0, 1])
    out["persistence_r_2017_2023"] = persist
    (cfg.path(cfg.data.metrics_dir) / "to_arena_variants.json").write_text(json.dumps(out, indent=2, default=float))
    log.info("arena variants: holdout gain additive %.2f%% multiplicative %.2f%% (2017-23)", out["holdout"]["2017-2023"]["gain_additive_pct"], out["holdout"]["2017-2023"]["gain_multiplicative_pct"])
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
