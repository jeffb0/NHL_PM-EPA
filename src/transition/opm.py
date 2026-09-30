"""On-ice offense (OPM) and the transition residual TR (docs/transition/methodology.md §2)."""

from __future__ import annotations

import dataclasses
import json
import logging

import numpy as np
import polars as pl
from scipy import sparse

from src.defense import ridge
from src.valuation import combine as cb
from src.valuation.replacement import status

log = logging.getLogger(__name__)
TARGETS = cb.DPM_TARGETS      # xg, start_turnover, start_faceoff, start_unobserved
W = np.array([1.0, -1.0, -1.0, -1.0])


def refit_offense(stints: pl.DataFrame, st: pl.DataFrame, seasons: list[int], alpha: float) -> pl.DataFrame:
    """Independent-season (rho = 0) offense refit with fringe-count covariates. Returns player-season rows with the valuation-ready
    offense coefficient ``off_val`` (per 60), the 5v5 minutes, and the pooled-by-season fringe attacker effects ``att_fringe_F/D``."""
    rows = []
    for s in seasons:
        d0 = stints.filter(pl.col("season") == s)
        d = ridge.build_design(d0, TARGETS)
        extra = cb._fringe_columns(d0, st.filter(pl.col("season") == s))
        d2 = dataclasses.replace(d, X=sparse.hstack([d.X, sparse.csr_matrix(extra)]).tocsr(), n_cov=d.n_cov + extra.shape[1])
        beta = ridge.fit(d2, {t: alpha for t in TARGETS}, None)
        coefs = ridge.coef_frame(d2, beta)
        f = beta[-4:, :]                                              # [defF, defD, attF, attD] x targets
        off_val = coefs["off_xg"] - coefs["off_start_turnover"] - coefs["off_start_faceoff"] - coefs["off_start_unobserved"]
        mins = ridge.player_minutes(d2)
        rows.append(coefs.select("player_id").with_columns(pl.Series("off_val", off_val), pl.lit(s, pl.Int64).alias("season"), pl.lit(float(f[2] @ W)).alias("att_fringe_F"), pl.lit(float(f[3] @ W)).alias("att_fringe_D"))
                    .join(mins, on="player_id"))
        log.info("OPM refit %d: attacker fringe effect (value per 60 per fringe skater) F %.4f D %.4f", s, float(f[2] @ W), float(f[3] @ W))
    return pl.concat(rows)


def opm_table(cfg, proc) -> pl.DataFrame:
    ps = pl.read_parquet(proc / "player_season.parquet")
    sel = json.loads((cfg.path(cfg.data.metrics_dir) / "dpm_selection.json").read_text())
    st = status(ps, cfg.valuation.regular_forwards, cfg.valuation.regular_defense)
    stints = pl.read_parquet(proc / "def_stints.parquet")
    r = refit_offense(stints, st, sorted(cfg.data.seasons), sel["alphas"]["xg"])
    stat = st.select(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64), "pos_group", "fringe")
    r = r.with_columns(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64)).join(stat, on=["player_id", "season"], how="left").with_columns(pl.col("fringe").fill_null(True))
    fx = r.group_by("season").agg(pl.col("att_fringe_F").first(), pl.col("att_fringe_D").first())
    pooled = {"F": float(fx["att_fringe_F"].mean()), "D": float(fx["att_fringe_D"].mean())}         # pooled over seasons (the DPM default)
    f_att = pl.when(pl.col("pos_group") == "D").then(pooled["D"]).otherwise(pooled["F"])
    r = r.with_columns((pl.col("off_val") - (~pl.col("fringe")).cast(pl.Float64) * f_att).alias("opm_rate60")).with_columns((pl.col("opm_rate60") * pl.col("toi_5v5") / 60).alias("opm_gar"))
    return r.select("player_id", "season", "pos_group", "fringe", "toi_5v5", "off_val", "opm_rate60", "opm_gar")


def add_transition(opm: pl.DataFrame, t: pl.DataFrame, min_toi: float = 500.0) -> tuple[pl.DataFrame, dict]:
    """tr_gar = opm_gar - beta_pos x pm_gar, beta_pos from the minute-weighted within-position regression through the origin on qualified skaters."""
    j = opm.join(t.select(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64), "pm_gar"), on=["player_id", "season"], how="left").with_columns(pl.col("pm_gar").fill_null(0.0))
    beta = {}
    for g in ("F", "D"):
        q = j.filter((pl.col("pos_group") == g) & (pl.col("toi_5v5") >= min_toi))
        x, y, w = q["pm_gar"].to_numpy(), q["opm_gar"].to_numpy(), q["toi_5v5"].to_numpy()
        beta[g] = float((w * x * y).sum() / (w * x * x).sum())
    b = pl.when(pl.col("pos_group") == "D").then(beta["D"]).otherwise(beta["F"])
    return j.with_columns((pl.col("opm_gar") - b * pl.col("pm_gar")).alias("tr_gar")), beta
