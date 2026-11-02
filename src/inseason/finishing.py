"""Update F, in-season finishing:  python -m src.inseason.finishing   (docs/inseason/methodology.md §2, §5, §6).

For season T the finishing model (variant A, the variance components of fin_tune.json) is fitted on seasons < T. Each
shooter's effect m0 and its spread s0 enter the season; after each game the effect is re-estimated from his attempts so
far with one Laplace step, with the prior widened by the season deviation:

    m1 = m0 + sum(goal - p) / (1 / (s0^2 + sd_d^2) + sum p(1 - p)),   p = each attempt's probability at m0

and the rating moves by (m1 - m0) x his expected-goal sensitivity per 60 (sum p(1 - p) per minute, last season and this
season so far, x 60).

s0^2 is the persistent effect's posterior variance with the season deviations integrated out, from each training
season's information I_s: 1 / s0^2 = 1 / sd_u^2 + sum_s I_s / (1 + sd_d^2 I_s). The dense posterior covariance of the
fit is not formed (about 12,000 columns per fold).

Within the test season the attempts' strength-cell intercepts are set so each cell's predicted goals equal its observed
goals at m0 (the league-level calibration of docs/finishing §7; the same for both arms of every comparison). The
fringe block is left out of both arms: replacement status is a same-season label."""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np
import polars as pl
from scipy.special import expit

from src.config import load_config
from src.finishing import model as fm
from src.finishing.model import Spec
from src.finishing.run import load_table
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun

log = logging.getLogger("inseason")
CHECKPOINTS = [100, 200]
FIT_SEASONS = grun.TUNE + grun.TEST


def season_shots(df: pl.DataFrame, fit: fm.FinFit, T: int, s2: dict) -> pl.DataFrame:
    """Season T's attempts in game order with p0 (probability at the shooter's entering effect m0) and the prior
    precision of his effect this season."""
    lay = fit.layout
    train = df.filter(pl.col("season") < T)
    # each training season's information about the shooter, at the fitted mode
    parts = fit.parts(train)
    zero = np.zeros(train.height)
    eta = train["offset"].to_numpy() + sum(parts.get(b, zero) for b in ("cell", "fringe", "u", "d", "v", "e"))
    p = expit(eta)
    info = (
        train.select("shooter", "season", "pos_group")
        .with_columns(pl.Series("w", p * (1 - p)))
        .group_by("shooter", "season", "pos_group")
        .agg(pl.col("w").sum().alias("I"))
    )
    var_d = pl.when(pl.col("pos_group") == "D").then(s2["d_D"]).otherwise(s2["d_F"])
    var_u = pl.when(pl.col("pos_group") == "D").then(s2["u_D"]).otherwise(s2["u_F"])
    prec = (
        info.with_columns((pl.col("I") / (1 + var_d * pl.col("I"))).alias("eff"), var_u.alias("su"))
        .group_by("shooter")
        .agg((1 / pl.col("su").first() + pl.col("eff").sum()).alias("prec_u"))
    )
    m0 = pl.DataFrame(
        {
            "shooter": pl.Series(list(lay.shooter.keys()), dtype=pl.Int64),
            "m0": fit.beta[np.array(list(lay.shooter.values()), dtype=np.int64)],
        }
    )
    cur = df.filter(pl.col("season") == T).sort("game_id", "event_idx")
    goalie = fit.parts(cur).get("v", np.zeros(cur.height))
    cur = (
        cur.join(m0, on="shooter", how="left")
        .join(prec, on="shooter", how="left")
        .with_columns(
            pl.col("m0").fill_null(0.0),
            # no history: the population prior
            pl.col("prec_u").fill_null(1 / var_u),
        )
        .with_columns((1 / (1 / pl.col("prec_u") + var_d)).alias("prec"))
    )
    base = cur["offset"].to_numpy() + goalie
    y = cur["goal"].to_numpy().astype(float)
    cell = cur["strength_g"].replace_strict(fm.SCODE, default=9).to_numpy()
    base = base + fm.calibrate_cell_shift(base + cur["m0"].to_numpy(), y, cell)
    p0 = expit(base + cur["m0"].to_numpy())
    return cur.with_columns(pl.Series("base", base), pl.Series("p0", p0))


def player_level(cur: pl.DataFrame) -> pl.DataFrame:
    """Per shooter and checkpoint n: the rest-of-season log loss at m0 and at m1 (from his first n attempts)."""
    d = cur.with_columns(pl.int_range(pl.len()).over("shooter").alias("k"))
    out = []
    for n in CHECKPOINTS:
        first = (
            d.filter(pl.col("k") < n)
            .group_by("shooter")
            .agg(
                pl.len().alias("n"),
                (pl.col("goal") - pl.col("p0")).sum().alias("sg"),
                (pl.col("p0") * (1 - pl.col("p0"))).sum().alias("sw"),
                pl.col("prec").first(),
                pl.col("m0").first(),
                pl.col("pos_group").first(),
            )
            .filter(pl.col("n") == n)
            .with_columns((pl.col("sg") / (pl.col("prec") + pl.col("sw"))).alias("dm"))
        )
        rest = d.filter(pl.col("k") >= n).join(first.select("shooter", "dm"), on="shooter", how="inner")
        y = rest["goal"].to_numpy().astype(float)
        p1 = expit(rest["base"].to_numpy() + rest["m0"].to_numpy() + rest["dm"].to_numpy())
        out.append(
            rest.select("shooter", "season", "pos_group", "goal", "p0", "dm").with_columns(
                pl.lit(n).alias("checkpoint"),
                pl.Series("p1", p1),
                pl.Series("d", gm.logloss(rest["p0"].to_numpy(), y) - gm.logloss(p1, y)),
            )
        )
    return pl.concat(out)


def cluster_z(d: pl.DataFrame) -> dict:
    """Per-attempt log-loss improvement (m1 over m0) with standard errors clustered by player-season."""
    g = d.group_by("shooter", "season").agg(pl.col("d").sum().alias("D"), pl.len().alias("n"))
    D, n = g["D"].to_numpy(), g["n"].to_numpy()
    mean = D.sum() / n.sum()
    se = float(np.sqrt(((D - n * mean) ** 2).sum()) / n.sum())
    # rest-of-season excess goals: realized against predicted by the update
    r = d.group_by("shooter", "season").agg(
        (pl.col("goal") - pl.col("p0")).sum().alias("real"), (pl.col("p1") - pl.col("p0")).sum().alias("pred")
    )
    pred, real = r["pred"].to_numpy(), r["real"].to_numpy()
    return {
        "player_seasons": int(g.height),
        "attempts": int(n.sum()),
        "mean_improvement": float(mean),
        "se": se,
        "z": float(mean / se),
        "calibration_slope": float((pred * real).sum() / (pred**2).sum()),
        "sd_predicted_rest_goals": float(pred.std()),
        "sd_dm": float(d.unique(["shooter", "season"])["dm"].std()),
    }


class FinUpdate:
    """The rating change from update F for one season, re-estimated after every game (the game-model hook)."""

    def __init__(self, cur: pl.DataFrame, last: dict, prec: dict, minutes_last: dict):
        g = cur.group_by("game_id", "shooter").agg(
            (pl.col("goal") - pl.col("p0")).sum().alias("sg"), (pl.col("p0") * (1 - pl.col("p0"))).sum().alias("sw")
        )
        self.by_game: dict = defaultdict(dict)
        for gid, sid, sg, sw in g.iter_rows():
            self.by_game[gid][sid] = (sg, sw)
        self.prec, self.last, self.min_last = prec, last, minutes_last
        self.sg: dict = defaultdict(float)
        self.sw: dict = defaultdict(float)
        self.mins: dict = defaultdict(float)
        self.cache: dict = {}

    def adj(self, i: int) -> float:
        if i not in self.cache:
            if i not in self.prec or self.sw[i] == 0.0:
                self.cache[i] = 0.0
            else:
                dm = self.sg[i] / (self.prec[i] + self.sw[i])
                m = self.min_last.get(i, 0.0) + self.mins[i]
                self.cache[i] = dm * (self.last.get(i, 0.0) + self.sw[i]) * 60.0 / m if m > 0 else 0.0
        return self.cache[i]

    def observe(self, gid: int, ids: np.ndarray, mins: np.ndarray) -> None:
        shots = self.by_game.get(gid, {})
        for i, mi in zip(ids.tolist(), mins.tolist()):
            self.mins[i] += mi
            if i in shots:
                sg, sw = shots[i]
                self.sg[i] += sg
                self.sw[i] += sw
            self.cache.pop(i, None)


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    tune = json.loads((met / "fin_tune.json").read_text())
    s2 = tune["s2"]["A"]
    df = load_table(cfg)
    pg = pl.read_parquet(proc / "player_game.parquet", columns=["player_id", "season", "toi_all"])
    mins = pg.group_by("player_id", "season").agg((pl.col("toi_all").sum() / 60.0).alias("m"))
    rows, updates = [], {}
    for T in FIT_SEASONS:
        fit = fm.fit_finishing(df.filter(pl.col("season") < T), Spec.A(fringe=True), s2=s2)
        cur = season_shots(df, fit, T, s2)
        if T in grun.TEST:
            rows.append(player_level(cur))
        # last season's expected-goal sensitivity, at the entering effect
        prev = season_shots(df, fit, T - 1, s2)
        last = dict(prev.group_by("shooter").agg((pl.col("p0") * (1 - pl.col("p0"))).sum()).iter_rows())
        prec = dict(cur.select("shooter", "prec").unique("shooter").iter_rows())
        ml = dict(mins.filter(pl.col("season") == T - 1).select("player_id", "m").iter_rows())
        updates[T] = (cur, last, prec, ml)
        log.info("season %d: %d attempts, %d shooters", T, cur.height, len(prec))
    pl_ = pl.concat(rows)
    res: dict = {"variance_components_sd": {k: float(np.sqrt(v)) for k, v in s2.items()}}
    res["player"] = {
        str(n): {
            "all": cluster_z(pl_.filter(pl.col("checkpoint") == n)),
            "forwards": cluster_z(pl_.filter((pl.col("checkpoint") == n) & (pl.col("pos_group") == "F"))),
            "defensemen": cluster_z(pl_.filter((pl.col("checkpoint") == n) & (pl.col("pos_group") == "D"))),
        }
        for n in CHECKPOINTS
    }
    log.info("player level: %s", {n: round(v["all"]["z"], 2) for n, v in res["player"].items()})

    # ---- game level
    hp = json.loads((met / "game_model.json").read_text(encoding="utf-8"))["tuned"]["P"]
    kw = {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}
    games = gd_.load_games(proc)
    pri = gd_.build_priors(proc, cfg, FIT_SEASONS, games)
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    te = t["season"].is_in(grun.TEST).to_numpy()
    y, dates = t["home_win"].to_numpy()[te], t["date"].to_numpy()[te]
    mu0 = np.concatenate([gm.run_player_season(games, pri, T, **kw) for T in FIT_SEASONS])
    mu1 = np.concatenate([gm.run_player_season(games, pri, T, fin=FinUpdate(*updates[T]), **kw) for T in FIT_SEASONS])
    p0, p1 = grun.predict(t, mu0)[te], grun.predict(t, mu1)[te]
    l0, l1 = gm.logloss(p0, y), gm.logloss(p1, y)
    res["game"] = {
        "n_games": int(te.sum()),
        "logloss_without": float(l0.mean()),
        "logloss_with": float(l1.mean()),
        "calibration_slope_with": grun.cal_slope(p1, y),
        "sd_mu_change": float((mu1 - mu0)[te].std()),
        "max_abs_mu_change": float(np.abs(mu1 - mu0)[te].max()),
        **grun.paired(l1, l0, dates),
    }
    log.info(
        "game level: %.5f -> %.5f z %.2f", res["game"]["logloss_without"], res["game"]["logloss_with"], res["game"]["z"]
    )
    z100, z200, zg = res["player"]["100"]["all"]["z"], res["player"]["200"]["all"]["z"], res["game"]["z"]
    slope = res["game"]["calibration_slope_with"]
    res["gates"] = {
        "IS1": {"z_100": z100, "z_200": z200, "pass": bool(z100 > 2 and z200 > 2)},
        "IS4_F": {"z": zg, "calibration_slope": slope, "pass": bool(zg > -2 and 0.8 <= slope <= 1.2)},
    }
    res["implement"] = bool(res["gates"]["IS1"]["pass"] and zg > 0 and res["gates"]["IS4_F"]["pass"])
    (met / "inseason_finishing.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("gates %s implement=%s", {k: v["pass"] for k, v in res["gates"].items()}, res["implement"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
