"""Priors for goalies the rolling rates do not cover:  python -m src.goaltending.unknown
(docs/gamemodel/methodology.md §10 item 2).

The game model gives a goalie with no rolling rate the replacement rate. Two groups fall there:

* **returning** goalies: a fitted career effect, but no shots in the last training season (injury, a year elsewhere);
* **debut** goalies: no NHL shots before the season.

For each forward fold (fit on seasons <= T, look at T + 1) this compares what those goalies did with the replacement
rate and with the alternative: the career effect for returning goalies, the mean of earlier debut seasons for debut
goalies. It then runs the game model with both alternatives against the current priors. Writes
reports/metrics/gk_unknown.json; the live forecasts read ``debut_rate`` and ``implement`` from it."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gmm
from src.gamemodel import run as grun
from src.goaltending import model as gm
from src.goaltending.run import load_shots

log = logging.getLogger("goaltending")


def folds(proc, met) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Per forward fold: the returning goalies' rates for season T + 1, and what returning and debut goalies did."""
    tune = json.loads((met / "gk_tune.json").read_text(encoding="utf-8"))
    team, s2 = tune["variant"] == "G2", tune["s2"][tune["variant"]]
    df = load_shots(proc)
    seasons = sorted(df["season"].unique().to_list())
    rates, ret, deb = [], [], []
    for T in seasons[:-1]:
        tr = df.filter(pl.col("season") <= T)
        lay = gm.build_layout(tr, team, fringe=True)
        mode, _, _ = gm.fit(tr, lay, s2=s2)
        cur = tr.filter(pl.col("season") == T)
        ptr = gm.shot_probs(cur, lay, mode.beta)
        rep = float((ptr["base"] - ptr["rep"]).mean())
        back = gm.returning_rates(cur, lay, mode.beta)
        rates.append(back.with_columns(pl.lit(T + 1, pl.Int64).alias("season")))
        te = df.filter(pl.col("season") == T + 1)
        pte = gm.shot_probs(te, lay, mode.beta, cell_season=pl.lit(T))
        real = (
            te.with_columns(pl.Series("saved", pte["cell_only"] - te["goal"].to_numpy()))
            .group_by("goalie")
            .agg(pl.col("saved").sum(), pl.len().alias("n"))
            .with_columns(pl.lit(T + 1, pl.Int64).alias("season"), pl.lit(rep).alias("rep"))
        )
        ret.append(real.join(back, on="goalie", how="inner"))
        deb.append(real.filter(~pl.col("goalie").is_in(list(lay.goalie))))
    return pl.concat(rates), pl.concat(ret), pl.concat(deb)


def _rate(d: pl.DataFrame, col: str | None = None) -> float:
    """Shot-weighted goals saved per shot: realized, or of a prior column."""
    n = d["n"].sum()
    return float((d["saved"].sum() if col is None else (d[col] * d["n"]).sum()) / n) if n else float("nan")


def _interval(d: pl.DataFrame, reps: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of the realized rate, goalie-seasons resampled."""
    rng = np.random.default_rng(seed)
    s, n = d["saved"].to_numpy(), d["n"].to_numpy()
    bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(s), len(s)) for _ in range(reps))]
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def debut_rate(deb: pl.DataFrame, before: int | None = None) -> float | None:
    """Mean goals saved per shot of debut goalie-seasons (those before season ``before``); None if there are none."""
    d = deb if before is None else deb.filter(pl.col("season") < before)
    return _rate(d) if d.height else None


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    rates, ret, deb = folds(proc, met)
    sq = lambda d, col: float(((d["saved"] - d[col] * d["n"]) ** 2).sum())
    res: dict = {
        "returning": {
            "goalie_seasons": ret.height,
            "shots": int(ret["n"].sum()),
            "realized_per_shot": _rate(ret),
            "interval95": _interval(ret),
            "career_effect_prior": _rate(ret, "rate_v"),
            "replacement_prior": _rate(ret, "rep"),
            "sq_error_career_effect": sq(ret, "rate_v"),
            "sq_error_replacement": sq(ret, "rep"),
        },
        "debut": {
            "goalie_seasons": deb.height,
            "shots": int(deb["n"].sum()),
            "realized_per_shot": _rate(deb),
            "interval95": _interval(deb),
            "replacement_prior": _rate(deb, "rep"),
            "realized_under_300_shots": _rate(deb.filter(pl.col("n") < 300)),
            "realized_300_plus_shots": _rate(deb.filter(pl.col("n") >= 300)),
        },
        "debut_rate": debut_rate(deb),
    }
    log.info("returning: %s", {k: round(v, 5) for k, v in res["returning"].items() if isinstance(v, float)})
    log.info("debut: %s", {k: round(v, 5) for k, v in res["debut"].items() if isinstance(v, float)})

    # ---- game model: current priors against the two alternatives (hyperparameters frozen)
    hp = json.loads((met / "game_model.json").read_text(encoding="utf-8"))["tuned"]["P"]
    kw = {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games)
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    te = t["season"].is_in(grun.TEST).to_numpy()
    y, dates = t["home_win"].to_numpy()[te], t["date"].to_numpy()[te]

    def loss(use_returning: bool, use_debut: bool) -> tuple[np.ndarray, np.ndarray]:
        saved = dict(pri.goalie0)
        for T in seasons:
            g0, rep = saved[T]
            add = dict(rates.filter(pl.col("season") == T).select("goalie", "rate_v").iter_rows())
            dr = debut_rate(deb, before=T)
            pri.goalie0[T] = (
                {**add, **g0} if use_returning else g0,
                dr if use_debut and dr is not None else rep,
            )
        mu = np.concatenate([gmm.run_player_season(games, pri, T, **kw) for T in seasons])
        pri.goalie0.update(saved)
        return gmm.logloss(grun.predict(t, mu)[te], y), mu[te]

    l0, m0 = loss(False, False)
    game = {"n_games": int(te.sum()), "logloss_current": float(l0.mean())}
    for name, a, b in (("returning", True, False), ("debut", False, True), ("both", True, True)):
        l1, m1 = loss(a, b)
        ch = m1 != m0
        game[name] = {
            "logloss": float(l1.mean()),
            "games_changed": int(ch.sum()),
            "logloss_changed_games_before": float(l0[ch].mean()),
            "logloss_changed_games_after": float(l1[ch].mean()),
            **grun.paired(l1, l0, dates),
        }
    res["game"] = game
    log.info("game model: %s", {k: round(v["z"], 2) for k, v in game.items() if isinstance(v, dict)})
    # used live when the goalies' own results reject the replacement rate and the game model is not worse
    lo_r, lo_d = res["returning"]["interval95"][0], res["debut"]["interval95"][0]
    res["implement"] = bool(
        lo_r > res["returning"]["replacement_prior"]
        and lo_d > res["debut"]["replacement_prior"]
        and game["both"]["z"] > 0
    )
    (met / "gk_unknown.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("implement=%s debut_rate=%.5f", res["implement"], res["debut_rate"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
