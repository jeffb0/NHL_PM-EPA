"""Update P, individual results for established skaters:  python -m src.inseason.individual
(docs/inseason/methodology.md §3, §5, §6; implementation choices §9 item 5).

The thin-history adjustment of docs/prospects/hybrid.md extended to skaters with m_eff >= 1,000: after his n-th game a
skater's rating is his prior plus a ridge prediction of how far his full-season value will land from the prior, from
seven signals over his games so far. The models are fitted by position, age group (31+ at season start or younger) and
checkpoint on earlier seasons."""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np
import polars as pl

from src.config import load_config
from src.finishing import model as fm
from src.finishing.model import Spec
from src.finishing.run import load_table
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.inseason import finishing as uf
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.prospects import features as pf
from src.prospects import hybrid as hy

log = logging.getLogger("inseason")
CHECK = [5, 10, 20, 40]
RECENT = 20
OLD = 31.0
SIGNALS = [*hy.SIGNALS, "rel_xgd60", "toi_change"]
# cumulative line: games, minutes, 5v5 minutes, puck value, ixG, points, on-ice xGF, xGA, off-ice xGD, off-ice 5v5 min
LINE = ["m", "m5", "pm", "ixg", "pts", "xgf", "xga", "off_xgd", "off_m5"]
PLAYER_TEST = grun.TEST


def signal_row(c: np.ndarray, toi_last: float | None) -> np.ndarray:
    """A cumulative line [games, *LINE] -> the seven signals."""
    g, m, m5 = max(c[0], 1.0), max(c[1], 1e-9), c[2]
    on = (c[6] - c[7]) / m5 * 60 if m5 > 0 else 0.0
    off = c[8] / c[9] * 60 if c[9] > 0 else 0.0
    return np.array(
        [m / g, c[3] / m * 60, c[4] / m * 60, c[5] / m * 60, on, on - off, m / g - toi_last if toi_last else 0.0]
    )


def game_lines(proc) -> pl.DataFrame:
    """Per player-game, in date order within player-season: the LINE quantities and the game index k (``proc``: the
    historical or the live tables)."""
    g = pl.read_parquet(proc / "games.parquet").select("game_id", pl.col("game_date").alias("date"))
    pg = (
        pl.read_parquet(
            proc / "player_game.parquet",
            columns=["game_id", "player_id", "team_id", "season", "toi_all", *gd_.STAT_SRC],
        )
        .join(
            pl.read_parquet(proc / "players.parquet").select(pl.col("player_id").cast(pl.Int64), "position"),
            on="player_id",
            how="left",
        )
        .filter((pl.col("toi_all") > 0) & (pl.col("position") != "G"))
        .join(g, on="game_id")
    )
    team = pg.group_by("game_id", "team_id").agg(
        (pl.col("oi_xgf").sum() / 5).alias("t_xf"),
        (pl.col("oi_xga").sum() / 5).alias("t_xa"),
        (pl.col("toi_5v5").sum() / 5 / 60).alias("t_m5"),
    )
    pg = pg.join(team, on=["game_id", "team_id"])
    return (
        pg.select(
            "game_id",
            "team_id",
            "date",
            pl.col("player_id").cast(pl.Int64),
            pl.col("season").cast(pl.Int64),
            (pl.col("toi_all") / 60.0).alias("m"),
            (pl.col("toi_5v5") / 60.0).alias("m5"),
            pl.col("pm_epa_total").cast(pl.Float64).alias("pm"),
            pl.col("ixg").cast(pl.Float64),
            (pl.col("goals") + pl.col("a1") + pl.col("a2")).cast(pl.Float64).alias("pts"),
            pl.col("oi_xgf").cast(pl.Float64).alias("xgf"),
            pl.col("oi_xga").cast(pl.Float64).alias("xga"),
            ((pl.col("t_xf") - pl.col("oi_xgf")) - (pl.col("t_xa") - pl.col("oi_xga"))).alias("off_xgd"),
            (pl.col("t_m5") - pl.col("toi_5v5") / 60.0).clip(0.0).alias("off_m5"),
        )
        .sort("player_id", "season", "date", "game_id")
        .with_columns(
            pl.int_range(pl.len()).over("player_id", "season").alias("k"),
            pl.len().over("player_id", "season").alias("gp"),
        )
    )


def toi_last(lines: pl.DataFrame) -> pl.DataFrame:
    """Last season's minutes per game, keyed by (player, season)."""
    return (
        lines.group_by("player_id", "season")
        .agg(pl.col("m").mean().alias("toi_last"))
        .with_columns(pl.col("season") + 1)
    )


def checkpoints(lines: pl.DataFrame, recent: bool = False) -> pl.DataFrame:
    """Per player-season and checkpoint n: the seven signals over his first n games (``recent``: over games n - 20 to
    n, at n = 40 only)."""
    tl = toi_last(lines)
    out = []
    for n in [40] if recent else CHECK:
        lo = n - RECENT if recent else 0
        a = (
            lines.filter((pl.col("k") >= lo) & (pl.col("k") < n) & (pl.col("gp") >= n))
            .group_by("player_id", "season")
            .agg(pl.len().alias("g"), *[pl.col(c).sum() for c in LINE])
            .join(tl, on=["player_id", "season"], how="left")
        )
        X = np.array(
            [signal_row(np.array([r[0], *r[1:-1]]), r[-1]) for r in a.select("g", *LINE, "toi_last").iter_rows()]
        )
        out.append(
            a.select("player_id", "season").with_columns(
                pl.lit(n).alias("n"), *[pl.Series(s, X[:, j]) for j, s in enumerate(SIGNALS)]
            )
        )
    return pl.concat(out)


def residuals(p: pl.DataFrame, bio: pl.DataFrame, yp: pl.DataFrame) -> pl.DataFrame:
    """Established player-seasons (m_eff >= 1,000) from 2019: full-season rate, the prior from earlier seasons (rate
    model v3), the miss, and age at season start."""
    fr_t = pf.with_yp(pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR), yp)
    fr_a = pf.with_yp(pd_.design_frame(p, bio, 1, targets_only=False), yp)
    M = pf.spec(yp)
    parts = []
    for T in range(hy.FIRST_RESID, int(p["season"].max()) + 1):
        te = fr_a.filter((pl.col("season") == T) & (pl.col("m_eff") >= hy.THIN))
        prior = pm.predict_models(pm.fit_models(pr.train_rows(fr_t, 1, T), M), te, M)
        parts.append(
            te.select("player_id", "season", "pos_group", "age", "m", pl.col("r_total").alias("y")).with_columns(
                pl.Series("prior", prior)
            )
        )
    return pl.concat(parts).with_columns(
        (pl.col("y") - pl.col("prior")).alias("resid"),
        pl.when(pl.col("age") >= OLD).then(pl.lit("old")).otherwise(pl.lit("young")).alias("agegrp"),
    )


def fit(train: pl.DataFrame) -> dict:
    """Ridge models {(position, age group, checkpoint): model} of the prior's miss (rows with >= 200 minutes)."""
    out: dict = {}
    for (g, a, n), d in train.filter(pl.col("m") >= 200).group_by("pos_group", "agegrp", "n"):
        if d.height < 40:
            continue
        X, y, w = d.select(SIGNALS).to_numpy(), d["resid"].to_numpy(), d["m"].to_numpy()
        out[(g, a, n)] = pm.wridge(X, y, w, pm.choose_lambda(X, y, w, d["season"].to_numpy()))
    return out


def predict(models: dict, d: pl.DataFrame) -> np.ndarray:
    X = d.select(SIGNALS).to_numpy()
    out = np.zeros(d.height)
    for i, key in enumerate(zip(d["pos_group"].to_list(), d["agegrp"].to_list(), d["n"].to_list())):
        if key in models:
            out[i] = float(models[key].predict(X[i][None, :])[0])
    return out


class Individual:
    """Update P for one season: the adjustment of each established skater after his games so far (game-model hook)."""

    def __init__(self, models: dict, who: dict, lines: dict, last: dict):
        # who: player -> (position, age group); lines: (game, player) -> LINE values; last: player -> toi last season
        self.models, self.who, self.lines, self.last = models, who, lines, last
        self.cum: dict = defaultdict(lambda: np.zeros(1 + len(LINE)))
        self.cache: dict = {}

    def adj(self, i: int) -> float:
        if i not in self.cache:
            c = self.cum.get(i)
            v = 0.0
            if i in self.who and c is not None and c[0] >= CHECK[0]:
                g, a = self.who[i]
                m = self.models.get((g, a, max(k for k in CHECK if k <= c[0])))
                if m is not None:
                    v = float(m.predict(signal_row(c, self.last.get(i))[None, :])[0])
            self.cache[i] = v
        return self.cache[i]

    def observe(self, gid: int, team: int, ids: np.ndarray, mins: np.ndarray) -> None:
        for i in ids.tolist():
            line = self.lines.get((gid, i))
            if line is not None:
                c = self.cum[i]
                c[0] += 1.0
                c[1:] += line
            self.cache.pop(i, None)


def live_lines(live_dir) -> dict:
    """(game, player) -> LINE values for the season in the live tables."""
    return {
        (gid, pid): np.array(v, dtype=float)
        for gid, pid, *v in game_lines(live_dir).select("game_id", "player_id", *LINE).iter_rows()
    }


def established(m_eff: dict, age: dict, pos: dict) -> dict:
    """player -> (position, age group) for skaters with m_eff >= 1,000 entering the season."""
    return {
        i: (pos.get(i, "F"), "old" if (age.get(i) or 0.0) >= OLD else "young")
        for i, v in m_eff.items()
        if v is not None and v >= hy.THIN
    }


class Both:
    """Update F and update P together (IS-3)."""

    def __init__(self, *parts):
        self.parts = parts

    def adj(self, i: int) -> float:
        return sum(p.adj(i) for p in self.parts)

    def observe(self, gid, team, ids, mins) -> None:
        for p in self.parts:
            p.observe(gid, team, ids, mins)


def run() -> dict:
    cfg = load_config()
    proc, raw, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.raw_dir), cfg.path(cfg.data.metrics_dir)
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    bio = pd_.player_bio(proc, raw)
    yp = pl.read_parquet(proc / "elite_features.parquet")
    lines = game_lines(proc)
    resid = residuals(p, bio, yp)
    data = resid.join(checkpoints(lines), on=["player_id", "season"], how="inner")
    # the live forecasts fit their models on this (every completed season)
    data.write_parquet(proc / "inseason_training.parquet")
    res: dict = {
        "rows_by_checkpoint": {
            str(n): int(data.filter((pl.col("n") == n) & (pl.col("m") >= 200)).height) for n in CHECK
        }
    }

    # ---- player level, expanding fits
    parts = []
    for T in PLAYER_TEST:
        models = fit(data.filter(pl.col("season") < T))
        te = data.filter((pl.col("season") == T) & (pl.col("m") >= 200))
        parts.append(te.with_columns(pl.Series("adj", predict(models, te))))
    e = pl.concat(parts).with_columns((pl.col("prior") + pl.col("adj")).alias("new"))

    def cmp(d: pl.DataFrame) -> dict:
        if d.height < 10:
            return {"n": d.height}
        mse = lambda c: float((d["m"] * (d["y"] - d[c]) ** 2).sum() / d["m"].sum())
        a, b = mse("prior"), mse("new")
        return {
            "n": d.height,
            "mse_prior": a,
            "mse_new": b,
            "share_of_error_removed": 1 - b / a,
            "z": pr.paired_cluster(d, "new", "prior")["z"],
            "sd_adjustment": float(d["adj"].std()),
        }

    res["player"] = {
        str(n): {
            "all": cmp(e.filter(pl.col("n") == n)),
            "forwards": cmp(e.filter((pl.col("n") == n) & (pl.col("pos_group") == "F"))),
            "defensemen": cmp(e.filter((pl.col("n") == n) & (pl.col("pos_group") == "D"))),
            "age_31_plus": cmp(e.filter((pl.col("n") == n) & (pl.col("agegrp") == "old"))),
            "forwards_31_plus": cmp(
                e.filter((pl.col("n") == n) & (pl.col("agegrp") == "old") & (pl.col("pos_group") == "F"))
            ),
            "under_31": cmp(e.filter((pl.col("n") == n) & (pl.col("agegrp") == "young"))),
        }
        for n in CHECK
    }
    # variant P-recent at n = 40: the last 20 games instead of all 40
    rec = resid.join(checkpoints(lines, recent=True), on=["player_id", "season"], how="inner")
    rparts = []
    for T in PLAYER_TEST:
        models = fit(rec.filter(pl.col("season") < T))
        te = rec.filter((pl.col("season") == T) & (pl.col("m") >= 200))
        rparts.append(te.with_columns(pl.Series("adj", predict(models, te))))
    er = pl.concat(rparts).with_columns((pl.col("prior") + pl.col("adj")).alias("new"))
    res["player_recent_n40"] = cmp(er)
    both = (
        e.filter(pl.col("n") == 40)
        .select("player_id", "season", "y", "m", pl.col("new").alias("season_so_far"))
        .join(er.select("player_id", "season", pl.col("new").alias("recent")), on=["player_id", "season"])
    )
    res["recent_vs_season_so_far_n40"] = {
        "n": both.height,
        "z": pr.paired_cluster(both, "recent", "season_so_far")["z"],
    }
    log.info("player level z: %s", {n: round(v["all"]["z"], 2) for n, v in res["player"].items()})

    # ---- game level: the current model (rate model v3 priors + the thin-history adjustment) with and without U-P
    hp = json.loads((met / "game_model.json").read_text(encoding="utf-8"))["tuned"]["P"]
    kw = {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games, yp)
    hdata = hy.residuals(p, bio, yp).join(hy.checkpoints(proc), on=["player_id", "season"], how="inner")
    early = {
        T: hy.Early(
            {i for i, v in pri.m_eff[T].items() if v < hy.THIN}, hy.fit(hdata.filter(pl.col("season") < T)), pri.pos
        )
        for T in seasons
    }
    tl = dict(((r[0], r[1]), r[2]) for r in toi_last(lines).iter_rows())
    line_map = {
        (gid, pid): np.array(v, dtype=float) for gid, pid, *v in lines.select("game_id", "player_id", *LINE).iter_rows()
    }
    ind = {}
    for T in seasons:
        models = fit(data.filter(pl.col("season") < T))
        r = resid.filter(pl.col("season") == T)
        who = {i: (g, a) for i, g, a in r.select("player_id", "pos_group", "agegrp").iter_rows()}
        last = {i: tl[(i, T)] for i in who if (i, T) in tl}
        ind[T] = (models, who, last)

    def individual(T: int) -> Individual:
        models, who, last = ind[T]
        return Individual(models, who, line_map, last)

    # update F's season inputs, for the combined model
    s2 = json.loads((met / "fin_tune.json").read_text())["s2"]["A"]
    df = load_table(cfg)
    mins = (
        pl.read_parquet(proc / "player_game.parquet", columns=["player_id", "season", "toi_all"])
        .group_by("player_id", "season")
        .agg((pl.col("toi_all").sum() / 60.0).alias("m"))
    )
    fin_inputs = {}
    for T in seasons:
        ffit = fm.fit_finishing(df.filter(pl.col("season") < T), Spec.A(fringe=True), s2=s2)
        cur = uf.season_shots(df, ffit, T, s2)
        prev = uf.season_shots(df, ffit, T - 1, s2)
        fin_inputs[T] = (
            cur,
            dict(prev.group_by("shooter").agg((pl.col("p0") * (1 - pl.col("p0"))).sum()).iter_rows()),
            dict(cur.select("shooter", "prec").unique("shooter").iter_rows()),
            dict(mins.filter(pl.col("season") == T - 1).select("player_id", "m").iter_rows()),
        )

    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    te_ = t["season"].is_in(grun.TEST).to_numpy()
    y, dates = t["home_win"].to_numpy()[te_], t["date"].to_numpy()[te_]
    mu = {
        "current": [gm.run_player_season(games, pri, T, early=early[T], **kw) for T in seasons],
        "P": [gm.run_player_season(games, pri, T, early=early[T], inseason=individual(T), **kw) for T in seasons],
        "F_and_P": [
            gm.run_player_season(
                games, pri, T, early=early[T], inseason=Both(individual(T), uf.FinUpdate(*fin_inputs[T])), **kw
            )
            for T in seasons
        ],
    }
    prob = {k: grun.predict(t, np.concatenate(v))[te_] for k, v in mu.items()}
    ll = {k: gm.logloss(v, y) for k, v in prob.items()}
    res["game"] = {
        "n_games": int(te_.sum()),
        "logloss": {k: float(v.mean()) for k, v in ll.items()},
        "calibration_slope": {k: grun.cal_slope(v, y) for k, v in prob.items()},
        "P_vs_current": grun.paired(ll["P"], ll["current"], dates),
        "F_and_P_vs_current": grun.paired(ll["F_and_P"], ll["current"], dates),
        "sd_mu_change_P": float((np.concatenate(mu["P"]) - np.concatenate(mu["current"]))[te_].std()),
    }
    log.info("game level log loss %s", {k: round(v, 5) for k, v in res["game"]["logloss"].items()})
    z10, z20 = res["player"]["10"]["all"]["z"], res["player"]["20"]["all"]["z"]
    zp, zb = res["game"]["P_vs_current"]["z"], res["game"]["F_and_P_vs_current"]["z"]
    slope = res["game"]["calibration_slope"]["P"]
    res["gates"] = {
        "IS2": {"z_n10": z10, "z_n20": z20, "pass": bool(z10 > 2 and z20 > 2)},
        "IS3": {"z": zb, "pass": bool(zb > 2)},
        "IS4_P": {"z": zp, "calibration_slope": slope, "pass": bool(zp > -2 and 0.8 <= slope <= 1.2)},
    }
    res["implement"] = bool(res["gates"]["IS2"]["pass"] and zp > 0 and res["gates"]["IS4_P"]["pass"])
    (met / "inseason_individual.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("gates %s implement=%s", {k: v["pass"] for k, v in res["gates"].items()}, res["implement"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
