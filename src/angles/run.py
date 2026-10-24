"""Situational angles tested against the game model:  python -m src.angles.run   (docs/angles/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
from scipy.stats import chi2, norm

from src.angles import features as af
from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.schedule.run import BASE, logit_table, ols_table, oos

log = logging.getLogger("angles")
FDR = 0.10


def bh(p: dict[str, float], q: float = FDR) -> dict[str, bool]:
    """Benjamini-Hochberg: which hypotheses are rejected at false-discovery rate q."""
    order = sorted(p, key=p.get)
    k = max((i + 1 for i, n in enumerate(order) if p[n] <= q * (i + 1) / len(order)), default=0)
    return {n: i < k for i, n in enumerate(order)}


def record(flag: np.ndarray, won: np.ndarray, p: np.ndarray) -> dict:
    """Win-loss record of the teams in a situation against the wins the baseline expected."""
    n, w, e = int(flag.sum()), float(won[flag].sum()), float(p[flag].sum())
    return {
        "n": n,
        "wins": int(w),
        "losses": n - int(w),
        "expected_wins": e,
        "win_pct": w / n if n else None,
        "expected_pct": e / n if n else None,
        "z": float((w - e) / np.sqrt((p[flag] * (1 - p[flag])).sum())) if n else None,
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
    # angles from the full schedule (every season, including games the model table drops), so "previous game" is the
    # true previous game
    sched = (
        pl.read_parquet(proc / "games.parquet")
        .with_columns(pl.col("game_date").str.to_date("%Y-%m-%d").alias("date"))
        .join(games.table.select("game_id", "xgd"), on="game_id", how="left")
    )
    ab = dict(
        pl.concat(
            [
                sched.select(pl.col("home_id").alias("i"), pl.col("home_abbrev").alias("t")),
                sched.select(pl.col("away_id").alias("i"), pl.col("away_abbrev").alias("t")),
            ]
        )
        .unique()
        .iter_rows()
    )
    season_of = dict(sched.select("game_id", "season").iter_rows())
    starters = {(gid, ab[tid]): (v[0][0], v[0][2]) for (gid, tid), v in games.goalies.items() if gid in season_of}
    t = (
        games.table.filter(pl.col("season") >= grun.TUNE[0])
        .with_columns(pl.Series("mu", mu))
        .join(af.game_angles(sched, starters), on="game_id", how="left")
    )
    assert t.select(pl.all_horizontal(pl.col(af.ANGLES).is_not_null()).all()).item()
    sd = {c: float(t[c].std()) for c in af.CONTINUOUS}
    # continuous angles per standard deviation of the home-minus-away value
    t = t.with_columns(*[(pl.col(c) / sd[c]).alias(c) for c in af.CONTINUOUS])
    y, s, gd = t["home_win"].to_numpy(), t["season"].to_numpy(), t["gd"].to_numpy()
    te = np.isin(s, grun.TEST)
    dates = t["date"].to_numpy()[te]
    X = lambda cols: t.select(["mu", *BASE, *cols]).to_numpy()

    b0 = gm.logit_fit(X([]), y)
    # baseline probability of a home win, fitted on all games (three parameters)
    p0 = gm.logit_pred(b0, X([]))
    _, l0 = logit_table(X([]), y, ["mu", *BASE])
    ll0 = gm.logloss(oos(X([]), y, s)[te], y[te])
    res: dict = {
        "n_fit_games": int(len(y)),
        "n_test_games": int(te.sum()),
        "logloss_base": float(ll0.mean()),
        "sd_continuous": sd,
        "angles": {},
    }
    for a in af.ANGLES:
        tab, _ = logit_table(X([a]), y, ["mu", *BASE, a])
        c = tab[a]
        ll = gm.logloss(oos(X([a]), y, s)[te], y[te])
        d = {
            "coef": c["coef"],
            "se": c["se"],
            "t": c["t"],
            "p": float(2 * norm.sf(abs(c["t"]))),
            "folk": af.FOLK[a],
            "agrees_with_folk": bool(np.sign(c["coef"]) == af.FOLK[a]),
            "pts_win_prob": float(25 * c["coef"]),
            "goal_diff": ols_table(X([a]), gd, ["mu", *BASE, a])[a],
            "logloss": float(ll.mean()),
            "paired": grun.paired(ll, ll0, dates),
        }
        # the record as the sayings quote it: teams in the situation, both sides pooled
        if a not in af.CONTINUOUS:
            fh = t[f"{a}_H"].to_numpy() == 1 if a not in af.AWAY_ONLY else np.zeros(len(y), bool)
            fa = t[f"{a}_A"].to_numpy() == 1 if a not in af.HOME_ONLY else np.zeros(len(y), bool)
            d["record"] = record(np.concatenate([fh, fa]), np.concatenate([y, 1 - y]), np.concatenate([p0, 1 - p0]))
            d["record_home"], d["record_away"] = record(fh, y, p0), record(fa, 1 - y, 1 - p0)
        res["angles"][a] = d
    rej = bh({a: v["p"] for a, v in res["angles"].items()})
    for a, v in res["angles"].items():
        v["bh_10pct"], v["oos_z_gt_2"] = rej[a], bool(v["paired"]["z"] > 2)
        v["verdict"] = (
            "adopt"
            if v["bh_10pct"] and v["oos_z_gt_2"]
            else "follow up"
            if v["bh_10pct"] or v["oos_z_gt_2"]
            else "not adopted"
        )
    _, lall = logit_table(X(af.ANGLES), y, ["mu", *BASE, *af.ANGLES])
    lla = gm.logloss(oos(X(af.ANGLES), y, s)[te], y[te])
    res["joint"] = {
        "lr": float(2 * (lall - l0)),
        "df": len(af.ANGLES),
        "p": float(chi2.sf(2 * (lall - l0), len(af.ANGLES))),
        "logloss_all": float(lla.mean()),
        "paired_all_vs_base": grun.paired(lla, ll0, dates),
    }
    res["implement"] = [a for a, v in res["angles"].items() if v["verdict"] == "adopt"]
    res["follow_up"] = [a for a, v in res["angles"].items() if v["verdict"] == "follow up"]
    (met / "angles.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for a, v in sorted(res["angles"].items(), key=lambda kv: kv[1]["p"]):
        r = v.get("record")
        log.info(
            "%-18s coef %+.3f t %+.2f p %.3f oos z %+.2f %s %s",
            a,
            v["coef"],
            v["t"],
            v["p"],
            v["paired"]["z"],
            f"{r['wins']}-{r['losses']} exp {r['expected_wins']:.0f} z {r['z']:+.2f}" if r else "",
            v["verdict"],
        )
    log.info(
        "joint LR p %.3f; all-angles oos z %.2f; adopt %s; follow up %s",
        res["joint"]["p"],
        res["joint"]["paired_all_vs_base"]["z"],
        res["implement"],
        res["follow_up"],
    )
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
