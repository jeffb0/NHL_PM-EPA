"""Possession under-scaling test:  python -m src.possession.run   (docs/possession/methodology.md)."""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.standings import candidates as cd
from src.standings import run as sr

log = logging.getLogger("possession")
KF = [1.0, 1.5, 2.0, 3.0]
KD = [1.0, 1.5, 2.0, 3.0, 4.0]


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    t_all = games.table.filter(pl.col("season") >= grun.TUNE[0])
    tt = t_all.filter(pl.col("season").is_in(grun.TUNE))
    pre = {"lam": 0.0, "alpha": 0.0, "n0": hp["n0"], "update": False}
    full = {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}

    # ---- tuning on 2019-2020: preseason-only game model
    surface = {}
    for kf, kd in itertools.product(KF, KD):
        pri = gd_.build_priors(proc, cfg, grun.TUNE, games, kappa={"F": kf, "D": kd})
        mu = np.concatenate([gm.run_player_season(games, pri, T, **pre) for T in grun.TUNE])
        surface[f"{kf:g}|{kd:g}"] = grun.tune_score(tt, mu)
    best = min(surface, key=surface.get)
    kf, kd = (float(x) for x in best.split("|"))
    log.info("tuned kappa F=%g D=%g (tuning log loss %.5f against %.5f at 1|1)", kf, kd, surface[best], surface["1|1"])
    res: dict = {"tuned": {"F": kf, "D": kd}, "tuning_surface": surface}

    # ---- test seasons
    seasons = grun.TUNE + grun.TEST
    te = t_all["season"].is_in(grun.TEST).to_numpy()
    y, dates, seas = t_all["home_win"].to_numpy()[te], t_all["date"].to_numpy()[te], t_all["season"].to_numpy()[te]

    def losses(kappa: dict | None) -> dict:
        pri = gd_.build_priors(proc, cfg, seasons, games, kappa=kappa)
        out = {}
        for name, kw in (("P_pre", pre), ("P", full)):
            mu = np.concatenate([gm.run_player_season(games, pri, T, **kw) for T in seasons])
            out[name] = gm.logloss(grun.predict(t_all, mu)[te], y)
        return out

    base = losses(None)
    variants = {"tuned": {"F": kf, "D": kd}, "earlier_estimate_1.5_1.5": {"F": 1.5, "D": 1.5}, "defense_only_1_3": {"F": 1.0, "D": 3.0}}
    game = {"n_games": int(te.sum()), "logloss_base": {k: float(v.mean()) for k, v in base.items()}}
    for name, kappa in variants.items():
        ll = base if all(v == 1.0 for v in kappa.values()) else losses(kappa)
        game[name] = {"kappa": kappa, **{m: {"logloss": float(ll[m].mean()), **grun.paired(ll[m], base[m], dates)} for m in ("P_pre", "P")}}
        game[name]["by_season"] = {str(T): {m: float((base[m][seas == T] - ll[m][seas == T]).mean()) for m in ("P_pre", "P")} for T in grun.TEST}      # positive: the rescaled model is better
        log.info("%s %s: P-pre z %.2f, P z %.2f", name, kappa, game[name]["P_pre"]["z"], game[name]["P"]["z"])
    res["game"] = game

    # ---- standings (reported): the published configuration with and without the rescaling
    pj = json.loads((met / "prospects.json").read_text()) if (met / "prospects.json").exists() else {}
    yp = pl.read_parquet(proc / "prospect_features.parquet") if pj.get("implement") else None
    nothing = kf == 1.0 and kd == 1.0
    ks = variants["earlier_estimate_1.5_1.5"] if nothing else variants["tuned"]      # with nothing tuned, the context variant is shown instead (reported only)
    e0, _ = cd.evaluate(cd.assemble(proc, cfg, yp)[0])
    df1, _ = cd.assemble(proc, cfg, yp, ks)
    if not nothing:
        df1.write_parquet(proc / "standings_candidates_table_kappa.parquet")
    e1, fitted = cd.evaluate(df1)
    j = e0.select("season", "team_id", "pts82", pl.col("C1_C3").alias("base"), pl.col("M1").alias("base_M1")).join(e1.select("season", "team_id", "C1_C3", "M1"), on=["season", "team_id"])
    yy = j["pts82"].to_numpy()
    r = lambda c: float(np.sqrt(((yy - j[c].to_numpy()) ** 2).mean()))  # noqa: E731
    res["standings"] = {"kappa": ks, "rmse_published": r("base"), "rmse_rescaled": r("C1_C3"), "z": sr._z(yy - j["C1_C3"].to_numpy(), yy - j["base"].to_numpy()), "rmse_M1": r("base_M1"), "rmse_M1_rescaled": r("M1"),
                        "z_M1": sr._z(yy - j["M1"].to_numpy(), yy - j["base_M1"].to_numpy()), "c_residual_by_fold": fitted["c13"]}
    log.info("standings: %.2f -> %.2f (z %.2f); M1 alone %.2f -> %.2f (z %.2f)", *[res["standings"][k] for k in ("rmse_published", "rmse_rescaled", "z", "rmse_M1", "rmse_M1_rescaled", "z_M1")])
    z1, z2 = (0.0, 0.0) if nothing else (game["tuned"]["P_pre"]["z"], game["tuned"]["P"]["z"])      # the tuned model is the baseline itself when nothing was selected
    g = {"PS1": {"z": z1, "pass": bool(z1 > 2)}, "PS2": {"z": z2, "pass": bool(z2 > 2)}, "PS3": {"pass": bool(z1 > -2 and z2 > -2)}}
    res["tuning_selected_no_rescaling"] = nothing
    res["gates"] = g
    res["implement"] = bool(not nothing and g["PS3"]["pass"] and z1 > -1 and z2 > -1)
    (met / "possession_scale.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("gates %s implement=%s", {k: v["pass"] for k, v in g.items()}, res["implement"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
