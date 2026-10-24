"""Rolling refits of the finishing and faceoff models (docs/wholemodel/methodology.md §5b).

For each season E the model is refit on the seasons up to E only (variance components fixed at the values of the
original
training seasons, as in the forward folds of those components), and the season-E values are computed from that fit.
These
values contain no information from later seasons and replace the full-data values in the ``rolling_refit`` total.

    python -m src.wholemodel.rolling [--only fin fo] [--seasons 2017 2018 ...]

Each fold is cached in data/processed/wm_roll/, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import polars as pl

from src.components import faceoff as fo
from src.config import load_config
from src.finishing import model as fm
from src.finishing.model import Spec
from src.finishing.run import load_table
from src.specialteams.finish import class_gar

log = logging.getLogger("wm_rolling")
FOLDS = list(range(2017, 2026))  # 2025 is the latest season: its fold is what the projection layer forecasts from


def _dir(cfg):
    d = cfg.path(cfg.data.processed_dir) / "wm_roll"
    d.mkdir(exist_ok=True)
    return d


def roll_fin(cfg, seasons: list[int]) -> None:
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    tune = json.loads((met / "fin_tune.json").read_text())
    variant = tune["D_type"]["variant"]
    spec = Spec.B(fringe=True) if variant == "B" else Spec.A(fringe=True)
    s2 = tune["s2"][variant]
    df = load_table(cfg)
    shots = pl.read_parquet(proc / "fin_shots.parquet", columns=["game_id", "event_idx", "own_skaters", "opp_skaters"])
    df = df.join(shots, on=["game_id", "event_idx"], how="left")
    for E in seasons:
        out = _dir(cfg) / f"fin_{E}.parquet"
        if out.exists():
            continue
        t0 = time.time()
        train = df.filter(pl.col("season") <= E)
        fit = fm.fit_finishing(train, spec, s2=s2, em_iters=0)
        cur = train.filter(pl.col("season") == E)
        eff = fm.player_effects(fit, cur, 0, cfg.seed, None).select(
            "player_id", "season", pl.col("fin_gar_5v5").alias("roll_fin_gar")
        )
        parts = fit.parts(cur)
        zero = np.zeros(cur.height)
        l_i = cur["offset"].to_numpy() + parts.get("cell", zero) + parts.get("v", zero) + parts.get("e", zero)
        T_i = (
            parts.get("gamma", zero)
            + parts.get("u", zero)
            + parts.get("d", zero)
            + parts.get("w", zero)
            + parts.get("fringe", zero)
        )
        d = cur.select(
            pl.col("shooter"), "season", "pos_group", "strength_g", "own_skaters", "opp_skaters", "fringe"
        ).with_columns(pl.Series("l", l_i), pl.Series("T", T_i))
        st = (
            class_gar(d)
            .filter(pl.col("cls") != "other")
            .pivot(on="cls", index=["player_id", "season"], values="fin_gar_st")
        )
        st = st.rename({c: f"roll_st_{c}_fin" for c in st.columns if c in ("pp", "pk", "4v4", "3v3")})
        res = eff.with_columns(pl.col("season").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)).join(
            st, on=["player_id", "season"], how="full", coalesce=True
        )
        res.write_parquet(out)
        log.info("finishing fold %d: %d rows in %.0fs (train shots %d)", E, res.height, time.time() - t0, train.height)


def roll_fo(cfg, seasons: list[int]) -> None:
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    diag = json.loads((met / "fo_component.json").read_text())
    s2 = {k: v**2 for k, v in diag["sd_train"].items()}
    meta = fo.taker_meta(proc)
    grp = dict(meta.unique(subset=["player_id"]).select("player_id", "grp").iter_rows())
    d = fo._fr(fo.draws(proc), meta)
    for E in seasons:
        out = _dir(cfg) / f"fo_{E}.parquet"
        if out.exists():
            continue
        t0 = time.time()
        tr = d.filter(pl.col("season") <= E)
        lay, mode, s2f, _ = fo.fit(tr, grp, True, s2=s2)
        eff = (
            fo.effects(lay, mode, s2f, tr, meta)
            .filter(pl.col("season") == E)
            .select("player_id", "season", pl.col("fo_gar").alias("roll_fo_gar"))
        )
        eff.with_columns(pl.col("season").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)).write_parquet(out)
        log.info("faceoff fold %d: %d rows in %.0fs", E, eff.height, time.time() - t0)


def combine(cfg) -> pl.DataFrame | None:
    """Stack the cached folds into ``wm_rolling_values.parquet`` (player_id, season, roll_fin_gar, roll_fo_gar,
    roll_st_<cls>_fin)."""
    proc = cfg.path(cfg.data.processed_dir)
    fin = [pl.read_parquet(p) for p in sorted(_dir(cfg).glob("fin_*.parquet"))]
    fos = [pl.read_parquet(p) for p in sorted(_dir(cfg).glob("fo_*.parquet"))]
    if not fin or not fos:
        return None
    out = pl.concat(fin, how="diagonal").join(pl.concat(fos), on=["player_id", "season"], how="full", coalesce=True)
    for c in ("roll_fin_gar", "roll_fo_gar", *[f"roll_st_{k}_fin" for k in ("pp", "pk", "4v4", "3v3")]):
        out = out.with_columns((pl.col(c) if c in out.columns else pl.lit(0.0)).fill_null(0.0).alias(c))
    out.write_parquet(proc / "wm_rolling_values.parquet")
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["fin", "fo"], default=["fo", "fin"])
    ap.add_argument("--seasons", nargs="+", type=int, default=FOLDS)
    args = ap.parse_args()
    cfg = load_config()
    if "fo" in args.only:
        roll_fo(cfg, args.seasons)
    if "fin" in args.only:
        roll_fin(cfg, args.seasons)
    out = combine(cfg)
    log.info("combined rolling values: %s", None if out is None else out.height)


if __name__ == "__main__":
    main()
