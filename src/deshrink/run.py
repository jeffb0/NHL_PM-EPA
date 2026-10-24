"""De-shrinking the 5v5 play block of the player ratings:  python -m src.deshrink.run
(docs/deshrink/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.possession.scale import rescale
from src.projection import data as pd_
from src.residual import decomp as dc
from src.standings import candidates as cd
from src.standings import data as sd
from src.standings import run as sr

log = logging.getLogger("deshrink")
EARLY = 20


def kappa_before(t: pl.DataFrame, T: int | None = None) -> float:
    """Slope through the origin of observed on explained 5v5 play (centred within season) over team-seasons before ``T``
    (all seasons when None)."""
    d = t if T is None else t.filter(pl.col("season") < T)
    o, e = d["o_play5"].to_numpy(), d["e_play5"].to_numpy()
    return float((o * e).sum() / (e * e).sum())


def priors_by_season(proc, cfg, seasons: list[int], games: gd_.Games, kappas: dict[int, float]) -> gd_.Priors:
    """Game-model priors with each season's ratings built from the panel scaled by that season's factor."""
    out = None
    for T in seasons:
        pr_ = gd_.build_priors(proc, cfg, [T], games, kappa={"play5": kappas[T]})
        if out is None:
            out = pr_
        else:
            for a in ("theta0", "toi_last", "goalie0", "shots_pg", "career", "m_eff"):
                getattr(out, a).update(getattr(pr_, a))
    return out


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    res = sd.results(proc)
    t0 = dc.table(proc, p, res)
    seasons = grun.TUNE + grun.TEST
    kap = {T: kappa_before(t0, T) for T in seasons}
    k_all = kappa_before(t0)
    t1 = dc.table(proc, rescale(p, {"play5": k_all}), res)
    out: dict = {
        "kappa_by_season": {str(T): v for T, v in kap.items()},
        "kappa_all_seasons": k_all,
        "kappa_next_season": k_all,
        "descriptive_all_seasons": {
            "slope_before": kappa_before(t0),
            "slope_after": kappa_before(t1),
            "sd_residual_before": float(t0["R"].std()),
            "sd_residual_after": float(t1["R"].std()),
            "sd_play5_residual_before": float(t0["R_play5"].std()),
            "sd_play5_residual_after": float(t1["R_play5"].std()),
        },
    }
    log.info(
        "kappa by season %s; all seasons %.3f; residual SD %.1f -> %.1f",
        {T: round(v, 3) for T, v in kap.items()},
        k_all,
        out["descriptive_all_seasons"]["sd_residual_before"],
        out["descriptive_all_seasons"]["sd_residual_after"],
    )

    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    tg = games.table.filter(pl.col("season") >= grun.TUNE[0])
    te = tg["season"].is_in(grun.TEST).to_numpy()
    y, dates, seas = tg["home_win"].to_numpy()[te], tg["date"].to_numpy()[te], tg["season"].to_numpy()[te]
    kws = {
        "P_pre": {"lam": 0.0, "alpha": 0.0, "n0": hp["n0"], "update": False},
        "P": {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]},
    }
    pris = {
        "published": gd_.build_priors(proc, cfg, seasons, games),
        "deshrunk": priors_by_season(proc, cfg, seasons, games, kap),
    }
    ll = {
        (tag, name): gm.logloss(
            grun.predict(tg, np.concatenate([gm.run_player_season(games, pri, T, **kw) for T in seasons]))[te], y
        )
        for tag, pri in pris.items()
        for name, kw in kws.items()
    }
    # each team's first games of a season, where the prior matters most
    long = pl.concat(
        [
            tg.select("game_id", "season", "date", pl.col("home_id").alias("team")),
            tg.select("game_id", "season", "date", pl.col("away_id").alias("team")),
        ]
    ).sort("season", "team", "date", "game_id")
    n = (
        long.with_columns(pl.col("game_id").cum_count().over("season", "team").alias("n"))
        .group_by("game_id")
        .agg(pl.col("n").min())
    )
    early = (tg.select("game_id").join(n, on="game_id", how="left")["n"].to_numpy() <= EARLY)[te]
    out["game"] = {
        "n_games": int(te.sum()),
        "n_early_games": int(early.sum()),
        "logloss": {f"{a}|{m}": float(v.mean()) for (a, m), v in ll.items()},
        "deshrunk_vs_published": {m: grun.paired(ll[("deshrunk", m)], ll[("published", m)], dates) for m in kws},
        "early_games": {
            m: {
                "logloss_published": float(ll[("published", m)][early].mean()),
                "logloss_deshrunk": float(ll[("deshrunk", m)][early].mean()),
                **grun.paired(ll[("deshrunk", m)][early], ll[("published", m)][early], dates[early]),
            }
            for m in kws
        },
        "by_season": {
            str(T): {m: float((ll[("deshrunk", m)] - ll[("published", m)])[seas == T].mean()) for m in kws}
            for T in grun.TEST
        },
    }
    th0, th1 = pris["published"].theta0[grun.TEST[-1]], pris["deshrunk"].theta0[grun.TEST[-1]]
    ids = sorted(set(th0) & set(th1))
    a, b = np.array([th0[i] for i in ids]), np.array([th1[i] for i in ids])
    out["ratings_last_test_season"] = {
        "n": len(ids),
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "sd_published": float(a.std()),
        "sd_deshrunk": float(b.std()),
        "mean_abs_change": float(np.abs(b - a).mean()),
    }

    # ---- standings (reported): one factor for the whole pipeline, from seasons before the first forecast season
    pj = json.loads((met / "prospects.json").read_text()) if (met / "prospects.json").exists() else {}
    yp = pl.read_parquet(proc / "prospect_features.parquet") if pj.get("implement") else None
    k_st = kappa_before(t0, min(sr.FORECAST_SEASONS))
    e0, _ = cd.evaluate(cd.assemble(proc, cfg, yp)[0])
    e1, _ = cd.evaluate(cd.assemble(proc, cfg, yp, kappa={"play5": k_st})[0])
    j = e0.select("season", "team_id", "pts82", pl.col("C1_C3").alias("base")).join(
        e1.select("season", "team_id", "C1_C3"), on=["season", "team_id"]
    )
    yy = j["pts82"].to_numpy()
    out["standings"] = {
        "kappa": k_st,
        "rmse_published": float(np.sqrt(((yy - j["base"].to_numpy()) ** 2).mean())),
        "rmse_deshrunk": float(np.sqrt(((yy - j["C1_C3"].to_numpy()) ** 2).mean())),
        "z": sr._z(yy - j["C1_C3"].to_numpy(), yy - j["base"].to_numpy()),
    }
    z1, z2 = out["game"]["deshrunk_vs_published"]["P_pre"]["z"], out["game"]["deshrunk_vs_published"]["P"]["z"]
    out["gates"] = {
        "DS1": {"z": z1, "improves": bool(z1 > 2), "no_harm": bool(z1 > -1)},
        "DS2": {"z": z2, "improves": bool(z2 > 2), "no_harm": bool(z2 > -1)},
    }
    out["implement"] = bool(z1 > -1 and z2 > -1)  # the rule's output
    # not switched on: see docs/deshrink/methodology.md §7
    out["implemented"] = False
    (met / "deshrink.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log.info("log loss %s", {k: round(v, 5) for k, v in out["game"]["logloss"].items()})
    log.info(
        "DS-1 z %.2f, DS-2 z %.2f; standings %.2f -> %.2f (z %.2f); implement=%s",
        z1,
        z2,
        out["standings"]["rmse_published"],
        out["standings"]["rmse_deshrunk"],
        out["standings"]["z"],
        out["implement"],
    )
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
