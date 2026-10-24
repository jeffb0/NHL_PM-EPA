"""Rest, travel and schedule effects tested in the game model:  python -m src.schedule.run
(docs/schedule/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
from scipy.stats import chi2

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.schedule import features as sf

log = logging.getLogger("schedule")
BASE = ["b2b_home", "b2b_away"]
# predicted sign on home win (docs §2)
SIGN = {"rest_adv": 1, "km_h": -1, "km_a": 1, "east_a": 1, "west_a": 1, "trip_a": 1, "dense_h": -1, "dense_a": 1}


def oos(X: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Out-of-sample win probabilities: the logistic step for season T is fitted on seasons 2019 .. T - 1."""
    p = np.full(len(y), np.nan)
    for T in grun.TEST:
        tr, te = s < T, s == T
        p[te] = gm.logit_pred(gm.logit_fit(X[tr], y[tr]), X[te])
    return p


def logit_table(X: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[dict, float]:
    """Coefficients, standard errors and the log likelihood of one logistic fit."""
    b = gm.logit_fit(X, y)
    A = np.column_stack([np.ones(len(y)), X])
    p = 1 / (1 + np.exp(-(A @ b)))
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ (A * (p * (1 - p))[:, None]))))
    return {
        n: {"coef": float(b[i + 1]), "se": float(se[i + 1]), "t": float(b[i + 1] / se[i + 1])}
        for i, n in enumerate(names)
    }, float(-gm.logloss(p, y).sum())


def ols_table(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * (r @ r) / (len(y) - A.shape[1]))
    return {
        n: {"coef": float(b[i + 1]), "se": float(se[i + 1]), "t": float(b[i + 1] / se[i + 1])}
        for i, n in enumerate(names)
    }


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games)
    mu = np.concatenate(
        [gm.run_player_season(games, pri, T, lam=hp["lam"], alpha=hp["alpha"], n0=hp["n0"]) for T in seasons]
    )
    # features from the full schedule (including games the model table drops), so rest and travel are measured from the
    # true previous game
    sched = pl.read_parquet(proc / "games.parquet").with_columns(
        pl.col("game_date").str.to_date("%Y-%m-%d").alias("date")
    )
    t = (
        games.table.filter(pl.col("season") >= grun.TUNE[0])
        .with_columns(pl.Series("mu", mu))
        .join(sf.game_features(sched), on="game_id", how="left")
    )
    allf = sum(sf.BLOCKS.values(), [])
    assert t.select(pl.all_horizontal(pl.col(allf).is_not_null()).all()).item()
    y, s, gd, xgd = t["home_win"].to_numpy(), t["season"].to_numpy(), t["gd"].to_numpy(), t["xgd"].to_numpy()
    te = np.isin(s, grun.TEST)
    dates = t["date"].to_numpy()[te]
    X = lambda cols: t.select(["mu", *BASE, *cols]).to_numpy()

    cands = {**sf.BLOCKS, "ALL": allf}
    ll = {"base": gm.logloss(oos(X([]), y, s)[te], y[te])}
    ll.update({k: gm.logloss(oos(X(c), y, s)[te], y[te]) for k, c in cands.items()})
    res: dict = {
        "n_test_games": int(te.sum()),
        "n_fit_games": int(len(y)),
        "logloss": {k: float(v.mean()) for k, v in ll.items()},
        "paired": {k: grun.paired(ll[k], ll["base"], dates) for k in cands},
        "by_season": {str(T): {k: float(v[s[te] == T].mean()) for k, v in ll.items()} for T in grun.TEST},
    }

    # one fit on all of 2019-2025: coefficients, block likelihood-ratio tests, and the same features on goal and
    # expected-goal differential
    _, l0 = logit_table(X([]), y, ["mu", *BASE])
    full, lfull = logit_table(X(allf), y, ["mu", *BASE, *allf])
    res["coef_win_all_features"] = full
    res["coef_goal_diff_all_features"] = ols_table(X(allf), gd, ["mu", *BASE, *allf])
    res["coef_xg_diff_all_features"] = ols_table(X(allf), xgd, ["mu", *BASE, *allf])
    res["blocks"] = {}
    for k, c in cands.items():
        tab, lk = logit_table(X(c), y, ["mu", *BASE, *c])
        lr = 2 * (lk - l0)
        strong = [n for n in c if n in SIGN and abs(tab[n]["t"]) > 2]
        signs_ok = all(np.sign(tab[n]["coef"]) == SIGN[n] for n in strong)
        z = res["paired"][k]["z"]
        p = float(chi2.sf(lr, len(c)))
        adopt = bool(z > 2 or (z > -1 and p < 0.05 and signs_ok)) if k != "ALL" else False
        res["blocks"][k] = {
            "features": c,
            "coef_win": {n: tab[n] for n in c},
            "lr": float(lr),
            "df": len(c),
            "lr_p": p,
            "strong_predicted": strong,
            "signs_as_predicted": bool(signs_ok),
            "z": z,
            "SC1": bool(z > 2),
            "SC2": bool(z > -2),
            "adopt": adopt,
        }
    res["usage"] = {
        n: {"mean": float(t[n].mean()), "sd": float(t[n].std()), "share_nonzero": float((t[n] != 0).mean())}
        for n in [*BASE, *allf]
    }
    res["lr_all_vs_base"] = float(2 * (lfull - l0))
    res["implement"] = [k for k, v in res["blocks"].items() if v["adopt"]]
    (met / "schedule_effects.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("log loss %s", {k: round(v, 5) for k, v in res["logloss"].items()})
    log.info(
        "paired z %s; LR p %s; adopt %s",
        {k: round(v["z"], 2) for k, v in res["paired"].items()},
        {k: round(v["lr_p"], 3) for k, v in res["blocks"].items()},
        res["implement"],
    )
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
