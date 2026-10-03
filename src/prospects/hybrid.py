"""Early-sample updating for thin-history players:  python -m src.prospects.hybrid   (docs/prospects/hybrid.md)."""

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
from src.prospects import features as pf

log = logging.getLogger("prospects")
CHECK = [2, 5, 10, 20, 40]
SIGNALS = ["toi_pg", "pm60", "ixg60", "pts60", "oi_xgd60"]
THIN = 1000.0
PLAYER_TEST = [2020, 2021, 2022, 2023, 2024, 2025]
FIRST_RESID = 2019          # first season with a prior projection (the projection needs one earlier target season)


def signal_row(c: np.ndarray) -> np.ndarray:
    """[games, minutes, 5v5 minutes, puck value, ixG, points, on-ice xGF, xGA] -> the five signals."""
    g, m, m5 = max(c[0], 1.0), max(c[1], 1e-9), c[2]
    return np.array([m / g, c[3] / m * 60, c[4] / m * 60, c[5] / m * 60, (c[6] - c[7]) / m5 * 60 if m5 > 0 else 0.0])


class Early:
    """The adjustment for one season: ridge models by position and checkpoint, applied to thin-history players."""

    def __init__(self, thin: set, models: dict, pos: dict):
        self.thin, self.models, self.pos = thin, models, pos

    def adj(self, i: int, c: np.ndarray | None) -> float:
        if c is None or i not in self.thin or c[0] < CHECK[0]:
            return 0.0
        m = self.models.get(self.pos.get(i, "F"), {}).get(max(k for k in CHECK if k <= c[0]))
        return float(m.predict(signal_row(c)[None, :])[0]) if m is not None else 0.0


def checkpoints(proc) -> pl.DataFrame:
    """Per player-season and checkpoint n: the five signals over his first n games (players with at least n games)."""
    g = pl.read_parquet(proc / "games.parquet").select("game_id", pl.col("game_date").alias("date"))
    pg = pl.read_parquet(proc / "player_game.parquet", columns=["game_id", "player_id", "season", "toi_all", *gd_.STAT_SRC]).filter(pl.col("toi_all") > 0).join(g, on="game_id")
    pg = pg.sort("player_id", "season", "date", "game_id").with_columns(pl.int_range(pl.len()).over("player_id", "season").alias("k"), pl.len().over("player_id", "season").alias("gp"))
    out = []
    for n in CHECK:
        a = pg.filter((pl.col("k") < n) & (pl.col("gp") >= n)).group_by("player_id", "season").agg(
            (pl.col("toi_all").sum() / 60.0).alias("m"), (pl.col("toi_5v5").sum() / 60.0).alias("m5"), pl.col("pm_epa_total").sum().alias("pm"), pl.col("ixg").sum().alias("ixg"),
            (pl.col("goals") + pl.col("a1") + pl.col("a2")).sum().cast(pl.Float64).alias("pts"), (pl.col("oi_xgf") - pl.col("oi_xga")).sum().alias("xgd"))
        out.append(a.select(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64), pl.lit(n).alias("n"), (pl.col("m") / n).alias("toi_pg"), (pl.col("pm") / pl.col("m") * 60).alias("pm60"), (pl.col("ixg") / pl.col("m") * 60).alias("ixg60"),
                            (pl.col("pts") / pl.col("m") * 60).alias("pts60"), pl.when(pl.col("m5") > 0).then(pl.col("xgd") / pl.col("m5") * 60).otherwise(0.0).alias("oi_xgd60")))
    return pl.concat(out)


def residuals(p: pl.DataFrame, bio: pl.DataFrame, yp: pl.DataFrame | None) -> pl.DataFrame:
    """Thin-history player-seasons from 2019: the full-season rate, the prior projection made from earlier seasons, and their difference."""
    fr_t = pf.with_yp(pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR), yp)
    fr_a = pf.with_yp(pd_.design_frame(p, bio, 1, targets_only=False), yp)
    M = pf.spec(yp)
    parts = []
    for T in range(FIRST_RESID, int(p["season"].max()) + 1):
        te = fr_a.filter((pl.col("season") == T) & (pl.col("m_eff") < THIN))
        prior = pm.predict_models(pm.fit_models(pr.train_rows(fr_t, 1, T), M), te, M)
        parts.append(te.select("player_id", "season", "pos_group", "m", "m_eff", pl.col("r_total").alias("y")).with_columns(pl.Series("prior", prior)))
    return pl.concat(parts).with_columns((pl.col("y") - pl.col("prior")).alias("resid"))


def fit(train: pl.DataFrame) -> dict:
    """Ridge models {position: {checkpoint: model}} of the prior's miss on the five signals (rows: thin-history player-seasons with >= 200 minutes)."""
    out: dict = {}
    for g in ("F", "D"):
        for n in CHECK:
            d = train.filter((pl.col("pos_group") == g) & (pl.col("n") == n) & (pl.col("m") >= 200))
            if d.height < 40:
                continue
            X, y, w = d.select(SIGNALS).to_numpy(), d["resid"].to_numpy(), d["m"].to_numpy()
            out.setdefault(g, {})[n] = pm.wridge(X, y, w, pm.choose_lambda(X, y, w, d["season"].to_numpy()))
    return out


def predict(models: dict, d: pl.DataFrame) -> np.ndarray:
    out = np.zeros(d.height)
    X = d.select(SIGNALS).to_numpy()
    pos, n = d["pos_group"].to_list(), d["n"].to_list()
    for i in range(d.height):
        m = models.get(pos[i], {}).get(n[i])
        if m is not None:
            out[i] = float(m.predict(X[i][None, :])[0])
    return out


def run() -> dict:
    cfg = load_config()
    proc, raw, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.raw_dir), cfg.path(cfg.data.metrics_dir)
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    bio = pd_.player_bio(proc, raw)
    yp = pl.read_parquet(proc / "elite_features.parquet")
    data = residuals(p, bio, yp).join(checkpoints(proc), on=["player_id", "season"], how="inner").join(yp.select("player_id", "season", "nhl_gp_before", "overall"), on=["player_id", "season"], how="left")
    data.write_parquet(proc / "hybrid_training.parquet")
    res: dict = {"n_rows_by_checkpoint": {str(n): int(data.filter((pl.col("n") == n) & (pl.col("m") >= 200)).height) for n in CHECK}}

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
        mse = lambda c: float((d["m"] * (d["y"] - d[c]) ** 2).sum() / d["m"].sum())  # noqa: E731
        a, b = mse("prior"), mse("new")
        return {"n": d.height, "mse_prior": a, "mse_new": b, "share_of_error_removed": 1 - b / a, "z": pr.paired_cluster(d, "new", "prior")["z"], "sd_adjustment": float(d["adj"].std())}

    res["player"] = {str(n): {"thin": cmp(e.filter(pl.col("n") == n)), "true_rookies": cmp(e.filter((pl.col("n") == n) & (pl.col("nhl_gp_before") == 0))),
                              "first_round_first_two_seasons": cmp(e.filter((pl.col("n") == n) & (pl.col("overall") <= 32) & (pl.col("nhl_gp_before") < 100)))} for n in CHECK}
    final = fit(data)
    res["coefficients_per_sd"] = {g: {str(n): {**dict(zip(SIGNALS, [float(x) for x in m.beta])), "lambda": m.lam} for n, m in mm.items()} for g, mm in final.items()}
    res["signal_sd"] = {g: {str(n): dict(zip(SIGNALS, [float(x) for x in m.sd])) for n, m in mm.items()} for g, mm in final.items()}
    log.info("player level z by checkpoint (thin): %s", {n: round(v["thin"]["z"], 2) for n, v in res["player"].items()})
    log.info("share of the prior's error removed: %s", {n: round(v["thin"]["share_of_error_removed"], 3) for n, v in res["player"].items()})

    # ---- game level: thin-history players' ratings carry the adjustment (models for season T fitted on seasons < T)
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games, yp)
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    te_ = t["season"].is_in(grun.TEST).to_numpy()
    y, dates = t["home_win"].to_numpy()[te_], t["date"].to_numpy()[te_]
    kw = {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}
    early = {T: Early({i for i, v in pri.m_eff[T].items() if v < THIN}, fit(data.filter(pl.col("season") < T)), pri.pos) for T in seasons}
    mu0 = np.concatenate([gm.run_player_season(games, pri, T, **kw) for T in seasons])
    mu1 = np.concatenate([gm.run_player_season(games, pri, T, early=early[T], **kw) for T in seasons])
    l0, l1 = gm.logloss(grun.predict(t, mu0)[te_], y), gm.logloss(grun.predict(t, mu1)[te_], y)
    res["game"] = {"n_games": int(te_.sum()), "logloss_without": float(l0.mean()), "logloss_with": float(l1.mean()), **grun.paired(l1, l0, dates)}
    log.info("game level: %.5f -> %.5f z %.2f", res["game"]["logloss_without"], res["game"]["logloss_with"], res["game"]["z"])
    z5, z10, z2, zg = res["player"]["5"]["thin"]["z"], res["player"]["10"]["thin"]["z"], res["player"]["2"]["thin"]["z"], res["game"]["z"]
    res["gates"] = {"HY1": {"z_n5": z5, "z_n10": z10, "pass": bool(z5 > 2 and z10 > 2)}, "HY2": {"z": zg, "pass": bool(zg > -2)}, "HY3": {"z_n2": z2, "pass": bool(z2 > 2)}}
    res["implement"] = bool(zg > -2 and z5 > -1 and z10 > -1)
    (met / "hybrid_prospects.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("gates %s implement=%s", {k: v["pass"] for k, v in res["gates"].items()}, res["implement"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
