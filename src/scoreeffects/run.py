"""Score-effects adjustment: factors and the game-model test:  python -m src.scoreeffects.run
(docs/scoreeffects/methodology.md)."""

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
from src.scoreeffects import factors as sf

log = logging.getLogger("scoreeffects")


def adjusted_xgd(proc, rates: pl.DataFrame) -> pl.DataFrame:
    """Per game: home-minus-away expected goals with 5v5 shots weighted by score state, the weights for season T fitted
    on seasons before T (none before the first season)."""
    sh = pl.read_parquet(
        proc / "fin_shots.parquet",
        columns=["game_id", "season", "team_id", "period", "score_diff", "strength", "xg_v1"],
    )
    parts = []
    for T in sorted(sh["season"].unique().to_list()):
        w = sf.weights(rates, [s for s in sorted(rates["season"].unique().to_list()) if s < T])
        parts.append(sf.apply(sh.filter(pl.col("season") == T), w, "xg_v1"))
    a = pl.concat(parts).group_by("game_id", "team_id").agg(pl.col("xg_adj").sum())
    g = pl.read_parquet(proc / "games.parquet").select("game_id", "home_id", "away_id")
    return (
        g.join(a.rename({"team_id": "home_id", "xg_adj": "h"}), on=["game_id", "home_id"])
        .join(a.rename({"team_id": "away_id", "xg_adj": "a"}), on=["game_id", "away_id"])
        .select("game_id", (pl.col("h") - pl.col("a")).alias("xgd_adj"))
    )


def run() -> dict:
    cfg = load_config()
    proc, met, outdir = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir), cfg.path(cfg.data.output_dir)
    rates = sf.state_rates(proc)
    w_all = sf.weights(rates, sorted(rates["season"].unique().to_list()))
    # all completed seasons: used by the live reports and forecasts
    w_all.write_csv(outdir / "score_effect_weights.csv", float_precision=4)
    res: dict = {
        "weights_all_seasons": {f"p{r['period']}_d{r['d']:+d}": round(r["w"], 4) for r in w_all.iter_rows(named=True)},
        "rate60_all_seasons": {
            f"p{r['period']}_d{r['d']:+d}": round(r["rate60"], 3) for r in w_all.iter_rows(named=True)
        },
    }
    log.info("weights (all seasons): %s", res["weights_all_seasons"])

    tune = json.loads((met / "game_model.json").read_text())["tuned"]
    hp, g2 = tune["P"], tune["G2"]
    games = gd_.load_games(proc)
    adj = adjusted_xgd(proc, rates)
    t2 = games.table.join(adj, on="game_id", how="left").with_columns(pl.coalesce("xgd_adj", "xgd").alias("xgd_adj"))
    res["differential"] = {
        "corr_raw_adjusted": float(np.corrcoef(t2["xgd"], t2["xgd_adj"])[0, 1]),
        "sd_raw": float(t2["xgd"].std()),
        "sd_adjusted": float(t2["xgd_adj"].std()),
        "mean_abs_change": float((t2["xgd"] - t2["xgd_adj"]).abs().mean()),
        "mean_change_for_winner": float(
            t2.with_columns(
                (
                    pl.when(pl.col("home_win") == 1)
                    .then(pl.col("xgd_adj") - pl.col("xgd"))
                    .otherwise(pl.col("xgd") - pl.col("xgd_adj"))
                ).alias("c")
            )["c"].mean()
        ),
    }
    games_adj = gd_.Games(
        t2.drop("xgd").rename({"xgd_adj": "xgd"}).select(games.table.columns), games.lineups, games.goalies, games.stats
    )
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games)
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    keep = (games.table["season"] >= grun.TUNE[0]).to_numpy()
    te = t["season"].is_in(grun.TEST).to_numpy()
    tt = t["season"].is_in(grun.TUNE).to_numpy()
    y, dates, seas = t["home_win"].to_numpy()[te], t["date"].to_numpy()[te], t["season"].to_numpy()[te]
    kw = {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}

    def P(g: gd_.Games, **k) -> np.ndarray:
        return np.concatenate([gm.run_player_season(g, pri, T, **k) for T in seasons])

    ll = {
        "P_raw": gm.logloss(grun.predict(t, P(games, **kw))[te], y),
        "P_adj": gm.logloss(grun.predict(t, P(games_adj, **kw))[te], y),
        "G2_raw": gm.logloss(
            grun.predict(t, gm.run_team_rating(games, g2["k"], g2["rho"], g2["alpha"], pri.home_adv)[keep])[te], y
        ),
        "G2_adj": gm.logloss(
            grun.predict(t, gm.run_team_rating(games_adj, g2["k"], g2["rho"], g2["alpha"], pri.home_adv)[keep])[te], y
        ),
    }
    # re-tuned on 2019-2020 with adjusted differentials (reported)
    best = None
    tune_t = t.filter(pl.col("season").is_in(grun.TUNE))
    for lam, alpha in itertools.product(grun.LAMS[1:], grun.ALPHAS):
        sc = grun.tune_score(
            tune_t,
            np.concatenate(
                [gm.run_player_season(games_adj, pri, T, lam=lam, alpha=alpha, n0=hp["n0"]) for T in grun.TUNE]
            ),
        )
        if best is None or sc < best[0]:
            best = (sc, lam, alpha)
    ll["P_adj_retuned"] = gm.logloss(grun.predict(t, P(games_adj, lam=best[1], alpha=best[2], n0=hp["n0"]))[te], y)
    _ = tt
    res["logloss"] = {k: float(v.mean()) for k, v in ll.items()}
    res["retuned"] = {"lam": best[1], "alpha": best[2]}
    res["paired"] = {
        "P_adj_vs_P_raw": grun.paired(ll["P_adj"], ll["P_raw"], dates),
        "G2_adj_vs_G2_raw": grun.paired(ll["G2_adj"], ll["G2_raw"], dates),
        "P_adj_retuned_vs_P_raw": grun.paired(ll["P_adj_retuned"], ll["P_raw"], dates),
        "P_adj_vs_G2_adj": grun.paired(ll["P_adj"], ll["G2_adj"], dates),
        "P_raw_vs_G2_raw": grun.paired(ll["P_raw"], ll["G2_raw"], dates),
    }
    res["by_season"] = {
        str(T): {k: float(v[seas == T].mean()) for k, v in ll.items() if k in ("P_raw", "P_adj")} for T in grun.TEST
    }
    z = res["paired"]["P_adj_vs_P_raw"]["z"]
    res["gates"] = {"SE1": {"z": z, "pass": bool(z > 2)}, "SE2": {"z": z, "pass": bool(z > -2)}}
    res["implement"] = bool(z > -1)
    res["n_games"] = int(te.sum())
    (met / "score_effects.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("log loss %s", {k: round(v, 5) for k, v in res["logloss"].items()})
    log.info(
        "P adjusted vs raw z %.2f; G2 adjusted vs raw z %.2f; implement=%s",
        z,
        res["paired"]["G2_adj_vs_G2_raw"]["z"],
        res["implement"],
    )
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
