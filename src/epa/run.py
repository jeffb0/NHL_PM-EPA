"""Compute PM-EPA transitions and credits for all seasons with cross-fitted V models."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import joblib
import numpy as np
import polars as pl

from src.config import load_config
from src.epa.engine import attribute, counterfactual_frames, flip_strength, possession_identity_residual
from src.states.mapping import STRENGTH_GROUP

log = logging.getLogger(__name__)


def value_obs(frame: pl.DataFrame, models: dict, key: str = "season") -> np.ndarray:
    out = np.full(frame.height, np.nan)
    seasons = frame[key].to_numpy()
    for s in np.unique(seasons):
        m = seasons == s
        model = models.get(int(s), models["all"])
        out[m] = model.predict(frame.filter(pl.Series(m)))
    return out


DECISION_KINDS = ["shot", "giveaway", "lost_takeaway"]


def prepare(cfg, models: dict, dec_models: dict | None = None, att_models: dict | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Value every observation. Starts/mids use V_start (alive-state value); terminal decision
    points (shot / giveaway / lost puck) use V_dec, the average possession outcome given the
    terminal decision happens at that spot (methodology §4.2). ``dec_models=None`` values
    decision points with V_start as well (the pre-v1.0 behaviour, kept for ablation)."""
    dec_models = dec_models or models
    proc = cfg.path(cfg.data.processed_dir)
    obs = pl.read_parquet(proc / "observations.parquet")
    poss = pl.read_parquet(proc / "possessions_valued.parquet")
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "home_id", "away_id")
    onice = pl.read_parquet(proc / "onice.parquet")

    is_dec = pl.col("obs_kind").is_in(DECISION_KINDS)
    v_start = value_obs(obs, models)
    v_dec = value_obs(obs, dec_models)
    obs = obs.with_columns(pl.Series("v", np.where(obs.select(is_dec).to_series().to_numpy(), v_dec, v_start)))
    # expected xG of an attempt from this spot (shot rows only): splits the shot into attempt vs quality
    is_shot = (obs["obs_kind"] == "shot").to_numpy()
    v_att = value_obs(obs, att_models) if att_models else np.full(obs.height, np.nan)
    obs = obs.with_columns(pl.Series("v_att", np.where(is_shot & np.isfinite(v_att), v_att, np.nan)).fill_nan(None))
    # opponent counterfactual values for turnovers
    cf = counterfactual_frames(obs)
    gv = cf["gv_opp"].with_columns(pl.Series("v_cf", value_obs(cf["gv_opp"], models))).select("poss_id", "seq", "v_cf")
    # the takeaway victim was at a decision point (their lost-puck spot): value it with V_dec
    tk = cf["tk_opp"].with_columns(pl.Series("v_cf", value_obs(cf["tk_opp"], dec_models))).select("poss_id", "seq", "v_cf")
    obs = obs.join(pl.concat([gv, tk]), on=["poss_id", "seq"], how="left")
    unobs = obs.select("poss_id", "seq", "season", "x", "y", "score_diff", flip_strength(pl.col("strength")).alias("strength"),
                       pl.lit("loose_unobserved").alias("obs_kind"))
    obs = obs.with_columns(pl.Series("v_cf_unobs", value_obs(unobs, models)))
    # on-ice lists for the possessing team and its opponent
    obs = obs.join(games, on="game_id").join(onice, on=["game_id", "event_idx"], how="left")
    home = pl.col("team_id") == pl.col("home_id")
    obs = obs.with_columns(
        pl.when(home).then(pl.col("home_on")).otherwise(pl.col("away_on")).fill_null([]).alias("onice"),
        pl.when(home).then(pl.col("away_on")).otherwise(pl.col("home_on")).fill_null([]).alias("opp_onice"),
        pl.when(home).then(pl.col("away_id")).otherwise(pl.col("home_id")).alias("opp_team_id"),
    ).drop("home_on", "away_on")
    obs = obs.join(poss.select("poss_id", "start_link"), on="poss_id", how="left")
    end_oi = (
        poss.select("poss_id", "game_id", "team_id", pl.col("end_event_idx").alias("event_idx"))
        .join(onice, on=["game_id", "event_idx"], how="left").join(games.select("game_id", "home_id"), on="game_id")
        .select("poss_id", pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.col("home_on")).otherwise(pl.col("away_on")).fill_null([]).alias("end_onice"))
    )
    obs = obs.join(end_oi, on="poss_id", how="left")
    # exclusions: empty-net (either goalie pulled) or unclassifiable strength anywhere in the possession
    bad = (
        obs.with_columns(pl.col("strength").replace_strict(STRENGTH_GROUP, default="other").alias("_sg"))
        .group_by("poss_id").agg(pl.col("_sg").is_in(["EN", "other"]).any().alias("excluded"))
    )
    poss = poss.join(bad, on="poss_id", how="left").with_columns(pl.col("excluded").fill_null(False))
    if not cfg.epa.exclude_empty_net:
        poss = poss.with_columns(pl.lit(False).alias("excluded"))
    poss = propagate_exclusion(poss)
    return obs, poss


def propagate_exclusion(poss: pl.DataFrame) -> pl.DataFrame:
    """A takeaway swing books credit on the taker's possession and debit on the loser's; keep or
    drop the pair together (else the loser's possession no longer reconciles to G_p)."""
    links = poss.filter((pl.col("start_type") == "takeaway") & pl.col("start_link").is_not_null()).select("poss_id", "start_link")
    pairs = list(zip(links["poss_id"].to_list(), links["start_link"].to_list()))
    excluded = set(poss.filter(pl.col("excluded"))["poss_id"].to_list())
    changed = True
    while changed:
        changed = False
        for a, b in pairs:
            if (a in excluded) != (b in excluded):
                excluded.update((a, b))
                changed = True
    return poss.with_columns(pl.col("poss_id").is_in(list(excluded)).alias("excluded"))


def run(lam: float | None = None, epa_overrides: dict | None = None, suffix: str = "", write: bool = True):
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    lam = cfg.value.lambda_opponent if lam is None else lam
    epa_cfg = SimpleNamespace(**{**cfg.epa.model_dump(), "faceoff_start_share": 0.0, **(epa_overrides or {})})
    bundle = joblib.load(cfg.path(cfg.data.value_dir) / "v_models.joblib")
    obs_path = proc / "obs_valued.parquet"
    if obs_path.exists() and suffix:
        obs, poss = pl.read_parquet(obs_path), pl.read_parquet(proc / "poss_valued_epa.parquet")
    else:
        obs, poss = prepare(cfg, bundle["models"], bundle.get("dec_models"), bundle.get("att_models"))
        obs.write_parquet(obs_path)
        poss.write_parquet(proc / "poss_valued_epa.parquet")
    transitions, credits = attribute(obs, poss, epa_cfg, cfg.possession, lam)
    if write:
        transitions.write_parquet(proc / f"transitions{suffix}.parquet")
        credits.write_parquet(proc / f"credits{suffix}.parquet")
    log.info("transitions=%d credit rows=%d (lam=%.2f, overrides=%s)", transitions.height, credits.height, lam, epa_overrides)
    return transitions, credits, poss


def identity_check() -> dict:
    """lam=0 accounting identity: per-possession components sum to terminal xG."""
    _, credits, poss = run(lam=0.0, suffix="_lam0", write=False)
    res = possession_identity_residual(credits, poss)
    return {"n_possessions": res.height, "max_abs_residual": float(res["residual"].abs().max()),
            "mean_abs_residual": float(res["residual"].abs().mean())}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
