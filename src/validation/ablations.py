"""Ablation studies (brief §9.3): vary one design choice at a time, recompute PM-EPA/60 and
report stability / validity deltas vs the baseline configuration.

Each row: YOY r (F, D; within scoring regime), team-changer YOY r, r with G+A1/60
(convergent), r with individual SH% (discriminant), and for V ablations the test-season
V MSE from the bake-off.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import polars as pl

from src.config import load_config
from src.epa.engine import PM_COMPONENTS
from src.epa.run import prepare
from src.metrics.aggregate import MOVEMENT, good_games
from src.validation.validate import _r, regime

log = logging.getLogger(__name__)


def pm_per60(credits: pl.DataFrame, base: pl.DataFrame, proc, components: list[str]) -> pl.DataFrame:
    c = credits.filter(
        pl.col("player_id").is_not_null() & (pl.col("strength") == "5v5") & pl.col("component").is_in(components)
    )
    c = c.join(good_games(proc), on="game_id")
    s = c.group_by("player_id", "season").agg(pl.col("value").sum().alias("pm_ev"))
    return base.join(s, on=["player_id", "season"], how="left").with_columns(
        (pl.col("pm_ev").fill_null(0) * 60 / pl.col("toi_ev")).alias("m")
    )


def summarize(d: pl.DataFrame, cfg) -> dict:
    q = d.filter(pl.col("toi_ev") >= cfg.metrics.min_toi_ev)
    nx = q.select(
        "player_id",
        (pl.col("season") - 1).alias("season"),
        pl.col("m").alias("m_next"),
        pl.col("primary_team").alias("team_next"),
    )
    j = q.join(nx, on=["player_id", "season"]).filter(
        pl.col("season").map_elements(lambda s: regime(s, cfg) == regime(s + 1, cfg), return_dtype=pl.Boolean)
    )
    # season-centred copies: a whole season shifting in level (e.g. mis-valued regime) must not read as "stability"
    j = j.with_columns(
        (pl.col("m") - pl.col("m").mean().over("season", "pos_group")).alias("mc"),
        (pl.col("m_next") - pl.col("m_next").mean().over("season", "pos_group")).alias("mc_next"),
    )
    out = {}
    for pos in ["F", "D"]:
        a = j.filter(pl.col("pos_group") == pos)
        ch = a.filter(pl.col("primary_team") != pl.col("team_next"))
        qq = q.filter(pl.col("pos_group") == pos)
        out[pos] = {
            "yoy_r": _r(a["m"].to_numpy(), a["m_next"].to_numpy()),
            "yoy_r_changers": _r(ch["m"].to_numpy(), ch["m_next"].to_numpy()),
            "yoy_r_centered": _r(a["mc"].to_numpy(), a["mc_next"].to_numpy()),
            "yoy_r_changers_centered": _r(ch["mc"].to_numpy(), ch["mc_next"].to_numpy()),
            "r_p1": _r(qq["m"].to_numpy(), qq["p1_per60_ev"].to_numpy().astype(float)),
            "r_ind_sh": _r(qq["m"].to_numpy().astype(float), qq["ind_sh_pct"].to_numpy().astype(float)),
            "sd_per60": float(qq["m"].std()),
        }
    return out


def run() -> None:
    from src.epa.engine import attribute
    from src.value.fit import alive_frame, crossfit_train_seasons, make, parse_spec

    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    base = pl.read_parquet(proc / "player_season.parquet").select(
        "player_id", "season", "toi_ev", "pos_group", "primary_team", "p1_per60_ev", "ind_sh_pct"
    )
    obs = pl.read_parquet(proc / "obs_valued.parquet")
    poss = pl.read_parquet(proc / "poss_valued_epa.parquet")
    vb = json.loads((cfg.path(cfg.data.metrics_dir) / "value_bakeoff.json").read_text())
    epa0 = {**cfg.epa.model_dump(), "faceoff_start_share": 0.0}
    lam0 = cfg.value.lambda_opponent
    results = {}

    def one(name, epa=None, lam=None, obs_=None, comps=PM_COMPONENTS, note=""):
        e = SimpleNamespace(**{**epa0, **(epa or {})})
        _, cr = attribute(obs_ if obs_ is not None else obs, poss, e, cfg.possession, lam0 if lam is None else lam)
        results[name] = {"note": note, **summarize(pm_per60(cr, base, proc, comps), cfg)}
        log.info("ablation %-34s F r=%.3f D r=%.3f", name, results[name]["F"]["yoy_r"], results[name]["D"]["yoy_r"])

    one(
        "baseline",
        note=f"V={vb['chosen']}, lambda={lam0}, shares {epa0['start_share']}/{epa0['end_share']}, on-ice pool",
    )
    for lam in (0.0, 0.5, 1.0):
        if lam != lam0:
            one(f"lambda={lam:g}", lam=lam, note="opponent possession value after turnovers weighted by lambda")
    one("no_onice_pool", epa={"onice_pool": False}, note="residual shares unattributed (individual-only credit)")
    one("shares_0.5/0.5", epa={"start_share": 0.5, "end_share": 0.5}, note="all split credit to endpoints")
    one("shares_0.2/0.2", epa={"start_share": 0.2, "end_share": 0.2})
    one(
        "shares_0/0_all_pool", epa={"start_share": 0.0, "end_share": 0.0}, note="all unobserved movement to on-ice pool"
    )
    one("shares_0.15/0.55", epa={"start_share": 0.15, "end_share": 0.55}, note="end-weighted")
    one("movement_only", comps=MOVEMENT, note="exclude turnovers/stoppages (include_turnovers=false)")
    one(
        "direct_only",
        comps=["carry", "move_start", "move_end", "takeaway", "giveaway"],
        note="drop all pooled components",
    )
    one("turnovers_only", comps=["takeaway", "giveaway", "takeaway_against", "unobserved_loss", "stoppage_loss"])

    # V-model ablations: refit cross-fitted V, re-value observations, re-attribute
    alive = alive_frame(
        pl.read_parquet(proc / "observations.parquet"),
        pl.read_parquet(proc / "possessions_valued.parquet"),
        cfg.epa.exclude_empty_net,
    )
    import joblib

    bundle = joblib.load(cfg.path(cfg.data.value_dir) / "v_models.joblib")
    dec_models, att_models = bundle["dec_models"], bundle["att_models"]
    # decision-point value off: value shot/giveaway spots with V_start (the pre-v1.0 design)
    o_nodec, _ = prepare(cfg, bundle["models"], None, None)
    one(
        "no_decision_value(pre-v1.0)",
        obs_=o_nodec,
        note="decision points valued by V_start (takeaway-based control surface); no shot-attempt split",
    )
    one(
        "no_shot_attempt_credit",
        comps=[c for c in PM_COMPONENTS if c != "shot_attempt"],
        note="decision failures (giveaways) charged, successes (attempts) not",
    )
    for spec_name in [
        "mc/zones/k=0",
        "mc/grid20/k=auto",
        "mc/grid10/k=auto",
        "mc/zones/k=auto",
        "gbm/cont",
        "mc/zones/k=auto/score",
    ]:
        if spec_name == vb["chosen"]:
            continue
        spec = parse_spec(spec_name)
        models = {
            s: make(spec, cfg.seed).fit(alive.filter(pl.col("season").is_in(crossfit_train_seasons(s, cfg))))
            for s in cfg.data.seasons
        }
        models["all"] = models[cfg.data.seasons[-1]]
        o2, _ = prepare(cfg, models, dec_models, att_models)
        mse = next((r["mse"] for r in vb["bakeoff"]["rows"] if r["spec"] == spec_name), None)
        one(f"V:{spec_name}", obs_=o2, note=f"validation V MSE {mse}")
    # regime-naive cross-fitting
    if cfg.value.crossfit_within_regime:
        spec = parse_spec(vb["chosen"])
        models = {s: make(spec, cfg.seed).fit(alive.filter(pl.col("season") != s)) for s in cfg.data.seasons}
        models["all"] = models[cfg.data.seasons[-1]]
        o2, _ = prepare(cfg, models, dec_models, att_models)
        one("V:crossfit_regime_naive", obs_=o2, note="season Y valued by all seasons != Y regardless of scoring regime")
    path = cfg.path(cfg.data.metrics_dir) / "ablations.json"
    path.write_text(json.dumps(results, indent=2))
    log.info("ablations written to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
