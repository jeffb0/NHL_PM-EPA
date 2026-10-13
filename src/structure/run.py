"""Structural additions to the game model, tested:  python -m src.structure.run   (docs/structure/methodology.md)."""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import polars as pl
from scipy.stats import norm

from src.angles.run import bh, record
from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.schedule.run import BASE, logit_table, oos
from src.structure import features as sf

log = logging.getLogger("structure")
LAMS, LAM_TEAMS, WS = [0.02, 0.04, 0.06], [0.0, 0.005, 0.01, 0.02, 0.04], [0.0, 0.25, 0.5, 0.75, 1.0]
TERMS = ["team_rating", "gk_tired", "gk_rusty", "home_ice_team", "low_scoring", "early_season", "stars"]
SIGN = {"team_rating": 1, "gk_tired": -1, "gk_rusty": -1, "home_ice_team": 1, "low_scoring": -1, "early_season": -1, "stars": 1}


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    tuned = json.loads((met / "game_model.json").read_text())["tuned"]
    hp, g2 = tuned["P"], tuned["G2"]
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games)
    keep = (games.table["season"] >= grun.TUNE[0]).to_numpy()
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    tt = t.filter(pl.col("season").is_in(grun.TUNE))
    y, s = t["home_win"].to_numpy(), t["season"].to_numpy()
    te = np.isin(s, grun.TEST)
    dates = t["date"].to_numpy()[te]

    def P(g: gd_.Games, ss, **kw) -> np.ndarray:
        return np.concatenate([gm.run_player_season(g, pri, T, **{"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"], **kw}) for T in ss])

    # the model margin and the star feature (three highest preseason ratings among the dressed skaters) in one pass
    pairs = [gm.run_player_season(games, pri, T, lam=hp["lam"], alpha=hp["alpha"], n0=hp["n0"],
                                  feature=lambda ids, m, th=pri.theta0[T]: float(sum(sorted(th.get(i, 0.0) for i in ids.tolist())[-3:]))) for T in seasons]
    mu, stars = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
    ll_p = gm.logloss(grun.predict(t, mu)[te], y[te])
    res: dict = {"n_test_games": int(te.sum()), "n_fit_games": int(len(y)), "logloss_P": float(ll_p.mean()), "accuracy_P": float(((grun.predict(t, mu)[te] > 0.5) == (y[te] > 0.5)).mean()), "updates": {}, "terms": {}}

    # ---- U1: a team layer beside the skater update
    grid = {(lam, lt): grun.tune_score(tt, P(games, grun.TUNE, lam=lam, lam_team=lt)) for lam, lt in itertools.product(LAMS, LAM_TEAMS)}
    lam, lt = min(grid, key=grid.get)
    u1 = {"tuned": {"lam": lam, "lam_team": lt}, "tune_logloss": grid[(lam, lt)], "tune_logloss_P": grid[(hp["lam"], 0.0)], "tune_grid": {f"{a}_{b}": v for (a, b), v in grid.items()}, "null_selected": lt == 0.0}
    ll = gm.logloss(grun.predict(t, P(games, seasons, lam=lam, lam_team=lt))[te], y[te])
    u1.update({"logloss": float(ll.mean()), "paired": grun.paired(ll, ll_p, dates), "by_season": {str(T): float((ll - ll_p)[s[te] == T].mean()) for T in grun.TEST}})
    res["updates"]["U1_team_layer"] = u1

    # ---- U2: shot volume in the update target
    shots = pl.read_parquet(proc / "fin_shots.parquet", columns=["game_id", "season", "team_id", "xg_v1"])
    vol = sf.volume_target(shots, games.table.select("game_id", "home_id", "away_id"))
    tv = games.table.join(vol, on="game_id", how="left").with_columns(pl.coalesce("vol_gd", "xgd").alias("vol_gd"))
    res["volume"] = {"corr_with_xgd": float(np.corrcoef(tv["xgd"], tv["vol_gd"])[0, 1]), "sd_xgd": float(tv["xgd"].std()), "sd_volume": float(tv["vol_gd"].std())}

    def blended(w: float) -> gd_.Games:
        tb = tv.with_columns(((1 - w) * pl.col("xgd") + w * pl.col("vol_gd")).alias("xgd")).select(games.table.columns)
        return gd_.Games(tb, games.lineups, games.goalies, games.stats)

    gw = {w: blended(w) for w in WS}
    grid = {(lam, w): grun.tune_score(tt, P(gw[w], grun.TUNE, lam=lam)) for lam, w in itertools.product(LAMS, WS)}
    lam, w = min(grid, key=grid.get)
    u2 = {"tuned": {"lam": lam, "w": w}, "tune_logloss": grid[(lam, w)], "tune_logloss_P": grid[(hp["lam"], 0.0)], "tune_grid": {f"{a}_{b}": v for (a, b), v in grid.items()}, "null_selected": w == 0.0}
    ll = gm.logloss(grun.predict(t, P(gw[w], seasons, lam=lam))[te], y[te])
    u2.update({"logloss": float(ll.mean()), "paired": grun.paired(ll, ll_p, dates), "by_season": {str(T): float((ll - ll_p)[s[te] == T].mean()) for T in grun.TEST}})
    res["updates"]["U2_shot_volume"] = u2
    for v in res["updates"].values():
        z = v["paired"]["z"]
        v["verdict"] = "not adopted (tuning selects the null value)" if v["null_selected"] else "adopt" if z > 2 else "follow up" if z > 0 else "not adopted"

    # ---- added terms
    raw = pl.read_parquet(proc / "games.parquet").with_columns(pl.col("game_date").str.to_date("%Y-%m-%d").alias("date"))
    so = pl.col("last_period_type") == "SO"
    xt = shots.group_by("game_id").agg(pl.col("xg_v1").sum().alias("xg_total"))
    sched = raw.with_columns(pl.when(so).then(0).otherwise(pl.col("home_final") - pl.col("away_final")).cast(pl.Float64).alias("gd")).join(xt, on="game_id", how="left")
    starters = {k: v[0][0] for k, v in games.goalies.items()}
    env = sf.environment(sched)
    f = (t.with_columns(pl.Series("mu", mu), pl.Series("stars", stars), pl.Series("team_rating", gm.run_team_rating(games, g2["k"], g2["rho"], g2["alpha"], pri.home_adv)[keep]))
         .join(sf.goalie_flags(sched, starters), on="game_id", how="left").join(sf.home_ice_team(sched), on="game_id", how="left").join(env, on="game_id", how="left")
         .join(raw.select("game_id", "last_period_type"), on="game_id", how="left"))
    f = f.with_columns(((pl.col("exp_total") - pl.col("exp_total").mean()) / pl.col("exp_total").std() * pl.col("mu")).alias("low_scoring"), (pl.col("mu") * (pl.col("min_played") < sf.EARLY_GAMES)).alias("early_season"),
                       (pl.col("stars") / pl.col("stars").std()).alias("stars"))
    assert f.select(pl.all_horizontal(pl.col(TERMS).is_not_null()).all()).item()
    X = lambda cols: f.select(["mu", *BASE, *cols]).to_numpy()            # noqa: E731
    p0 = gm.logit_pred(gm.logit_fit(X([]), y), X([]))
    ll0 = gm.logloss(oos(X([]), y, s)[te], y[te])
    for a in TERMS:
        c = logit_table(X([a]), y, ["mu", *BASE, a])[0][a]
        ll = gm.logloss(oos(X([a]), y, s)[te], y[te])
        res["terms"][a] = {"coef": c["coef"], "se": c["se"], "t": c["t"], "p": float(2 * norm.sf(abs(c["t"]))), "expected_sign": SIGN[a], "sign_as_expected": bool(np.sign(c["coef"]) == SIGN[a]),
                           "sd": float(f[a].std()), "share_nonzero": float((f[a] != 0).mean()), "logloss": float(ll.mean()), "paired": grun.paired(ll, ll0, dates),
                           "by_season": {str(T): float((ll - ll0)[s[te] == T].mean()) for T in grun.TEST}}
    rej = bh({a: v["p"] for a, v in res["terms"].items()})
    for a, v in res["terms"].items():
        v["bh_10pct"], v["oos_z_gt_2"] = rej[a], bool(v["paired"]["z"] > 2)
        v["verdict"] = "adopt" if v["bh_10pct"] and v["oos_z_gt_2"] else "follow up" if v["bh_10pct"] or v["oos_z_gt_2"] else "not adopted"

    # ---- reported detail
    b2h, b2a, th, ta = (f[c].to_numpy() == 1 for c in ("b2b_home", "b2b_away", "tired_h", "tired_a"))
    both = lambda fh, fa: record(np.concatenate([fh, fa]), np.concatenate([y, 1 - y]), np.concatenate([p0, 1 - p0]))      # noqa: E731
    res["back_to_back_by_goalie"] = {"tired_goalie": both(b2h & th, b2a & ta), "rested_goalie": both(b2h & ~th, b2a & ~ta)}
    otm = (f["last_period_type"] != "REG").to_numpy()
    res["stars_past_regulation"] = {"n": int(otm.sum()), "share_of_games": float(otm.mean()), **logit_table(f.select("mu", "stars").to_numpy()[otm], y[otm], ["mu", "stars"])[0]}
    ll = gm.logloss(oos(X(TERMS), y, s)[te], y[te])
    res["all_terms"] = {"logloss": float(ll.mean()), "paired": grun.paired(ll, ll0, dates)}
    res["implement"] = [k for k, v in {**res["updates"], **res["terms"]}.items() if v["verdict"] == "adopt"]
    res["follow_up"] = [k for k, v in {**res["updates"], **res["terms"]}.items() if v["verdict"] == "follow up"]
    (met / "structure.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for k, v in res["updates"].items():
        log.info("%-16s tuned %s log loss %.5f z %+.2f %s", k, v["tuned"], v["logloss"], v["paired"]["z"], v["verdict"])
    for a, v in res["terms"].items():
        log.info("%-16s coef %+.3f t %+.2f p %.3f oos z %+.2f %s", a, v["coef"], v["t"], v["p"], v["paired"]["z"], v["verdict"])
    log.info("P log loss %.5f accuracy %.3f; adopt %s; follow up %s", res["logloss_P"], res["accuracy_P"], res["implement"], res["follow_up"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
