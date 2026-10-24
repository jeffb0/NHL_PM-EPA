"""Game-level model end to end:  python -m src.gamemodel.run   (docs/gamemodel/methodology.md)."""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm

log = logging.getLogger("gamemodel")
TUNE = [2019, 2020]
TEST = [2021, 2022, 2023, 2024, 2025]
LAMS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.10]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N0S = [500.0, 1500.0, 3000.0]
KS = [0.01, 0.02, 0.03, 0.05, 0.08]
RHOS = [0.5, 0.7, 0.85, 1.0]


def _X(t: pl.DataFrame, mu: np.ndarray | None) -> np.ndarray:
    if mu is None:
        return np.empty((t.height, 0))
    return np.column_stack([mu, t["b2b_home"].to_numpy(), t["b2b_away"].to_numpy()])


def tune_score(t: pl.DataFrame, mu: np.ndarray) -> float:
    """In-sample log loss of the logistic step on the tuning seasons (the same treatment for every configuration)."""
    y = t["home_win"].to_numpy()
    X = _X(t, mu)
    return float(gm.logloss(gm.logit_pred(gm.logit_fit(X, y), X), y).mean())


def predict(t: pl.DataFrame, mu: np.ndarray | None) -> np.ndarray:
    """Out-of-sample win probabilities for the test seasons: the logistic step is fitted on seasons 2019 .. T - 1."""
    s = t["season"].to_numpy()
    y = t["home_win"].to_numpy()
    X = _X(t, mu)
    p = np.full(t.height, np.nan)
    for T in TEST:
        tr, te = (s >= TUNE[0]) & (s < T), s == T
        p[te] = gm.logit_pred(gm.logit_fit(X[tr], y[tr]), X[te])
    return p


def paired(ll_a: np.ndarray, ll_b: np.ndarray, dates: np.ndarray) -> dict:
    """z that model a beats model b on per-game log loss; also with standard errors clustered by date."""
    d = ll_b - ll_a
    z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    _, inv = np.unique(dates, return_inverse=True)
    sc, nc = np.bincount(inv, weights=d), np.bincount(inv)
    var = float(((sc - nc * d.mean()) ** 2).sum()) / len(d) ** 2
    return {"mean_diff": float(d.mean()), "z": z, "z_cluster_date": float(d.mean() / np.sqrt(var))}


def cal_slope(p: np.ndarray, y: np.ndarray) -> float:
    lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    return float(gm.logit_fit(lg[:, None], y)[1])


def run() -> dict:
    cfg = load_config()
    proc, met, outdir = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir), cfg.path(cfg.data.output_dir)
    games = gd_.load_games(proc)
    pri = gd_.build_priors(proc, cfg, TUNE + TEST, games)
    t_all = games.table
    t = t_all.filter(pl.col("season") >= TUNE[0])
    tt = t.filter(pl.col("season").is_in(TUNE))
    log.info(
        "%d games from 2019 (%d tuning, %d test)", t.height, tt.height, t.filter(pl.col("season").is_in(TEST)).height
    )

    def player_mu(seasons, **kw) -> np.ndarray:
        return np.concatenate([gm.run_player_season(games, pri, T, **kw) for T in seasons])

    # ---- tuning on 2019-2020 only
    tune: dict = {}
    for name, attr in (("P", "player"), ("P_team", "team")):
        best = None
        for lam, alpha, n0 in itertools.product(LAMS, ALPHAS, N0S):
            if lam == 0 and alpha != ALPHAS[0]:
                # alpha is irrelevant without a skater update
                continue
            sc = tune_score(tt, player_mu(TUNE, lam=lam, alpha=alpha, n0=n0, attribution=attr))
            if best is None or sc < best[0]:
                best = (sc, lam, alpha, n0)
        tune[name] = {"logloss_tune": best[0], "lam": best[1], "alpha": best[2], "n0": best[3]}
        log.info("tuned %s: %s", name, tune[name])
    keep = (t_all["season"] >= TUNE[0]).to_numpy()
    tmask = t_all.filter(pl.col("season") >= TUNE[0])["season"].is_in(TUNE).to_numpy()
    for name, alphas in (("G1", [1.0]), ("G2", ALPHAS)):
        best = None
        for k, rho, alpha in itertools.product(KS, RHOS, alphas):
            mu = gm.run_team_rating(games, k, rho, alpha, pri.home_adv)[keep]
            sc = tune_score(tt, mu[tmask])
            if best is None or sc < best[0]:
                best = (sc, k, rho, alpha)
        tune[name] = {"logloss_tune": best[0], "k": best[1], "rho": best[2], "alpha": best[3]}
        log.info("tuned %s: %s", name, tune[name])

    # ---- frozen hyperparameters, all seasons from 2019, out-of-sample logistic step
    hp, ht = tune["P"], tune["P_team"]
    seasons = TUNE + TEST
    mus = {
        "P": player_mu(seasons, lam=hp["lam"], alpha=hp["alpha"], n0=hp["n0"]),
        "P_pre": player_mu(seasons, lam=0.0, alpha=0.0, n0=hp["n0"], update=False),
        "P_team": player_mu(seasons, lam=ht["lam"], alpha=ht["alpha"], n0=ht["n0"], attribution="team"),
        "P_prev": player_mu(seasons, lam=hp["lam"], alpha=hp["alpha"], n0=hp["n0"], lineup="previous"),
        "G1": gm.run_team_rating(games, tune["G1"]["k"], tune["G1"]["rho"], 1.0, pri.home_adv)[keep],
        "G2": gm.run_team_rating(games, tune["G2"]["k"], tune["G2"]["rho"], tune["G2"]["alpha"], pri.home_adv)[keep],
        "G0": None,
    }
    te = t["season"].is_in(TEST).to_numpy()
    y = t["home_win"].to_numpy()[te]
    dates = t["date"].to_numpy()[te]
    seas = t["season"].to_numpy()[te]
    probs = {k: predict(t, v)[te] for k, v in mus.items()}
    ll = {k: gm.logloss(p, y) for k, p in probs.items()}
    res: dict = {
        "n_test_games": int(te.sum()),
        "home_advantage_goals": pri.home_adv,
        "tuned": tune,
        "logloss": {k: float(v.mean()) for k, v in ll.items()},
        "brier": {k: float(((p - y) ** 2).mean()) for k, p in probs.items()},
        "accuracy": {k: float(((p > 0.5) == (y > 0.5)).mean()) for k, p in probs.items()},
        "calibration_slope": {k: cal_slope(p, y) for k, p in probs.items() if k != "G0"},
        "logloss_by_season": {str(T): {k: float(v[seas == T].mean()) for k, v in ll.items()} for T in TEST},
    }
    bench = "G1" if res["logloss"]["G1"] <= res["logloss"]["G2"] else "G2"
    res["best_team_benchmark"] = bench
    res["paired"] = {
        "P_vs_G0": paired(ll["P"], ll["G0"], dates),
        "P_vs_G1": paired(ll["P"], ll["G1"], dates),
        "P_vs_G2": paired(ll["P"], ll["G2"], dates),
        "P_vs_P_pre": paired(ll["P"], ll["P_pre"], dates),
        "P_vs_P_team": paired(ll["P"], ll["P_team"], dates),
        "P_vs_P_prev": paired(ll["P"], ll["P_prev"], dates),
        "P_pre_vs_best_team": paired(ll["P_pre"], ll[bench], dates),
        "P_team_vs_best_team": paired(ll["P_team"], ll[bench], dates),
    }
    z = lambda k: res["paired"][k]["z"]
    res["gates"] = {
        "GM1": {"z": z("P_vs_G0"), "pass": bool(z("P_vs_G0") > 2)},
        "GM2": {"z": z(f"P_vs_{bench}"), "benchmark": bench, "pass": bool(z(f"P_vs_{bench}") > 2)},
        "GM3": {"z": z("P_vs_P_pre"), "pass": bool(z("P_vs_P_pre") > 2)},
        "GM4": {"z": z("P_vs_P_team"), "pass": bool(z("P_vs_P_team") > 2)},
        "GM5": {"slope": res["calibration_slope"]["P"], "pass": bool(0.8 <= res["calibration_slope"]["P"] <= 1.2)},
    }
    out = (
        t.filter(pl.col("season").is_in(TEST))
        .select(
            "game_id",
            "season",
            "date",
            pl.col("home_abbrev").alias("home"),
            pl.col("away_abbrev").alias("away"),
            "home_win",
        )
        .with_columns(*[pl.Series(f"p_{k}", v) for k, v in probs.items()])
    )
    out.write_csv(outdir / "game_predictions.csv")
    res["market"] = market(cfg, out)
    (met / "game_model.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("log loss %s", {k: round(v, 4) for k, v in res["logloss"].items()})
    log.info("gates %s", {k: (round(v.get("z", v.get("slope")), 2), v["pass"]) for k, v in res["gates"].items()})
    from src.gamemodel.report import run as report

    report()
    return res


def market(cfg, out: pl.DataFrame) -> dict | None:
    """Comparison with de-vigged closing probabilities when data/external/odds.csv (game_date, home, away, p_home_close)
    is supplied."""
    path = cfg.path("data/external/odds.csv")
    if not path.exists():
        return None
    o = pl.read_csv(path).with_columns(pl.col("game_date").str.to_date("%Y-%m-%d").alias("date"))
    j = out.join(o.select("date", "home", "away", "p_home_close"), on=["date", "home", "away"], how="inner")
    y = j["home_win"].to_numpy()
    lm, lp = gm.logloss(j["p_home_close"].to_numpy(), y), gm.logloss(j["p_P"].to_numpy(), y)
    return {
        "n_games": j.height,
        "logloss_market": float(lm.mean()),
        "logloss_P": float(lp.mean()),
        "P_vs_market": paired(lp, lm, j["date"].to_numpy()),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
