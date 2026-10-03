"""Pre-NHL priors: tests and decision:  python -m src.prospects.run   (docs/prospects/methodology.md §5-§6)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.projection.model import Spec
from src.prospects import features as pf
from src.standings import candidates as cd
from src.standings import run as sr

log = logging.getLogger("prospects")
PLAYER_TEST = [2020, 2021, 2022, 2023, 2024, 2025]
THIN = 1000.0


def player_level(p: pl.DataFrame, bio: pl.DataFrame, yp: pl.DataFrame) -> dict:
    fr_t = pf.with_yp(pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR), yp)
    fr_a = pf.with_yp(pd_.design_frame(p, bio, 1, targets_only=False), yp)
    M, MY = Spec(), pf.spec(yp)
    parts, coefs = [], {}
    for T in PLAYER_TEST:
        tr, te = pr.train_rows(fr_t, 1, T), fr_a.filter(pl.col("season") == T)
        m0, m1 = pm.fit_models(tr, M), pm.fit_models(tr, MY)
        parts.append(te.select("player_id", "season", "pos_group", "m", "m_eff", pl.col("r_total").alias("y")).with_columns(pl.Series("M", pm.predict_models(m0, te, M)), pl.Series("MY", pm.predict_models(m1, te, MY))))
        coefs[T] = {g: dict(zip(pm.feature_names(MY)[-4:], m1[g].raw_coef()[-4:].tolist())) for g in ("F", "D")}
    e = pl.concat(parts)

    def cmp(d: pl.DataFrame) -> dict:
        mse = lambda c: float((d["m"] * (d["y"] - d[c]) ** 2).sum() / d["m"].sum())  # noqa: E731
        return {"n": d.height, "mse_M": mse("M"), "mse_MY": mse("MY"), "z": pr.paired_cluster(d, "MY", "M")["z"]}

    thin = e.filter((pl.col("m_eff") < THIN) & (pl.col("m") >= 200))
    return {"thin": cmp(thin), "thin_F": cmp(thin.filter(pl.col("pos_group") == "F")), "thin_D": cmp(thin.filter(pl.col("pos_group") == "D")),
            "no_history": cmp(e.filter((pl.col("m_eff") == 0) & (pl.col("m") >= 200))), "all": cmp(e.filter(pl.col("m") >= 300)), "coefficients_last_fold": coefs[PLAYER_TEST[-1]]}


def game_level(proc, cfg, met, yp: pl.DataFrame) -> dict:
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    te = t["season"].is_in(grun.TEST).to_numpy()
    y, dates = t["home_win"].to_numpy()[te], t["date"].to_numpy()[te]
    ll = {}
    for tag, feats in (("M", None), ("MY", yp)):
        pri = gd_.build_priors(proc, cfg, seasons, games, feats)
        mu = lambda **kw: np.concatenate([gm.run_player_season(games, pri, T, **kw) for T in seasons])  # noqa: E731, B023
        ll[f"P_pre_{tag}"] = gm.logloss(grun.predict(t, mu(lam=0.0, alpha=0.0, n0=hp["n0"], update=False))[te], y)
        ll[f"P_{tag}"] = gm.logloss(grun.predict(t, mu(lam=hp["lam"], alpha=hp["alpha"], n0=hp["n0"]))[te], y)
    return {"n_games": int(te.sum()), "logloss": {k: float(v.mean()) for k, v in ll.items()}, "P_pre": grun.paired(ll["P_pre_MY"], ll["P_pre_M"], dates), "P": grun.paired(ll["P_MY"], ll["P_M"], dates)}


def standings_level(proc, cfg, outdir, yp: pl.DataFrame) -> dict:
    df, _ = cd.assemble(proc, cfg, yp)
    df.write_parquet(proc / "standings_candidates_table_yp.parquet")              # the forecast fits C1's coefficient on it when the priors are implemented
    e, _ = cd.evaluate(df)
    old = pl.read_csv(outdir / "standings_candidates_backtest.csv").select("season", "team_id", pl.col("C1_C3").alias("old"), pl.col("M1").alias("old_M1"))
    j = e.join(old, on=["season", "team_id"])
    y = j["pts82"].to_numpy()
    r = lambda c: float(np.sqrt(((y - j[c].to_numpy()) ** 2).mean()))  # noqa: E731
    return {"rmse_C1_C3_M": r("old"), "rmse_C1_C3_MY": r("C1_C3"), "z": sr._z(y - j["C1_C3"].to_numpy(), y - j["old"].to_numpy()), "rmse_M1_M": r("old_M1"), "rmse_M1_MY": r("M1"),
            "z_M1": sr._z(y - j["M1"].to_numpy(), y - j["old_M1"].to_numpy())}


def run() -> dict:
    cfg = load_config()
    proc, raw, met, outdir = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.raw_dir), cfg.path(cfg.data.metrics_dir), cfg.path(cfg.data.output_dir)
    yp = pf.table(raw, list(range(2018, 2027)))
    yp.write_parquet(proc / "prospect_features.parquet")
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    bio = pd_.player_bio(proc, raw)
    st, _ = pf.load_landing(raw)
    tr = pf.transitions(st)
    f, pooled = pf.factors(tr, 2026)
    cnt = dict(tr.group_by("league").len().iter_rows())
    res: dict = {"coverage": {"players_with_page": int(yp["player_id"].n_unique()), "skaters_in_panel": int(p["player_id"].n_unique()), "panel_skaters_with_page": int(p.select("player_id").unique().join(yp.select("player_id").unique(), on="player_id").height),
                              "transitions": tr.height},
                 "factors_2026": {"pooled": pooled, **{k: {"factor": f[k], "n": cnt[k]} for k in sorted(cnt, key=cnt.get, reverse=True)[:12]}}}
    res["player"] = player_level(p, bio, yp)
    log.info("player level: thin z %.2f, all z %.2f", res["player"]["thin"]["z"], res["player"]["all"]["z"])
    res["game"] = game_level(proc, cfg, met, yp)
    log.info("game level: P-pre z %.2f, P z %.2f", res["game"]["P_pre"]["z"], res["game"]["P"]["z"])
    res["standings"] = standings_level(proc, cfg, outdir, yp)
    log.info("standings: C1+C3 %.2f -> %.2f (z %.2f)", res["standings"]["rmse_C1_C3_M"], res["standings"]["rmse_C1_C3_MY"], res["standings"]["z"])
    z1, z2, za, zp = res["player"]["thin"]["z"], res["game"]["P_pre"]["z"], res["player"]["all"]["z"], res["game"]["P"]["z"]
    g = {"YP1": {"z": z1, "pass": bool(z1 > 2)}, "YP2": {"z": z2, "pass": bool(z2 > 2)}, "YP3": {"z_all_players": za, "z_full_game_model": zp, "pass": bool(za > -2 and zp > -2)}}
    res["gates"] = g
    res["implement"] = bool(g["YP3"]["pass"] and z1 > -1 and z2 > -1)                      # sound a priori (methodology section 6)
    (met / "prospects.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("gates %s implement=%s", {k: v["pass"] for k, v in g.items()}, res["implement"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
