"""Check D0, target consistency:  python -m src.inseason.d0   (docs/inseason/methodology.md §1).

Preseason ratings include 5v5 finishing (goals beyond what the attempts were worth); the in-season update target is
expected-goal differential, which excludes it. Two measurements on the test seasons, game model P with its frozen
hyperparameters:

1. Per game, the update error e = xgd - mu regressed on the home-minus-away lineup finishing dF (each dressed skater's
   finishing part of his preseason rating x his minutes / 60) and on the rest of mu (mu - dF). If expected-goal
   differential follows the non-finishing part of the rating at slope k and ignores finishing, e = (k - 1)(mu - dF) - dF:
   the finishing slope b_F is -1 and the gap b_F - b_rest is -k. If finishing showed up in xgd like the rest of the
   rating, the two slopes would be equal.
2. Per player-season, the change in rating over the season (end minus start) regressed on his finishing part and on the
   rest of his preseason rating, weighted by minutes: 1 + slope on finishing = the share of it left at season's end.

Special-teams finishing sits inside the special-teams rates of the projection and is not separated here."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel.run import TEST
from src.possession.scale import rescale
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.standings import forecast as fc

log = logging.getLogger("inseason")
EARLY_GAMES = 20


def finishing_part(proc, cfg, seasons: list[int]) -> dict:
    """season -> {player_id: the finishing term of his preseason projected rate, goals per 60 relative to the training
    mean}: the fitted coefficient on rt_fin times his rt_fin minus its weighted mean (the same fits as build_priors)."""
    p = rescale(pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet")), None)
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    fr = pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR)
    pg = pl.read_parquet(proc / "player_game.parquet", columns=["player_id", "season", "toi_all"]).filter(
        pl.col("toi_all") > 0
    )
    posdf = pl.read_parquet(proc / "players.parquet").select(
        pl.col("player_id").cast(pl.Int64),
        pl.when(pl.col("position") == "D").then(pl.lit("D")).otherwise(pl.lit("F")).alias("pos_group"),
    )
    spec = pm.Spec()
    k = pm.feature_names(spec).index("rt_fin")
    out = {}
    for T in seasons:
        who = (
            pg.filter(pl.col("season") == T)
            .select(pl.col("player_id").cast(pl.Int64))
            .unique()
            .join(posdf, on="player_id", how="left")
            .with_columns(pl.col("pos_group").fill_null("F"), pl.lit(0, pl.Int64).alias("team_id"))
        )
        d = fc.roster_frame(p, bio, who, T)
        models = pm.fit_models(pr.train_rows(fr, 1, T), spec)
        part = np.zeros(d.height)
        for g in ("F", "D"):
            m = (d["pos_group"] == g).to_numpy()
            r = models[g]
            part[m] = r.raw_coef()[k] * (d.filter(pl.col("pos_group") == g)["rt_fin"].to_numpy() - r.mu[k])
        out[T] = dict(zip(d["player_id"].to_list(), part.tolist()))
    return out


def ols(y: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares with an intercept: coefficients and heteroskedasticity-robust (HC1) covariance."""
    A = np.column_stack([np.ones(len(y)), X])
    w = np.ones(len(y)) if w is None else w
    Aw = A * w[:, None]
    inv = np.linalg.inv(A.T @ Aw)
    b = inv @ (Aw.T @ y)
    r = y - A @ b
    meat = (Aw * r[:, None]).T @ (Aw * r[:, None])
    return b, inv @ meat @ inv * len(y) / (len(y) - A.shape[1])


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    hp = json.loads((met / "game_model.json").read_text(encoding="utf-8"))["tuned"]["P"]
    games = gd_.load_games(proc)
    pri = gd_.build_priors(proc, cfg, TEST, games)
    fin = finishing_part(proc, cfg, TEST)

    # the update keeps its theta in a local dict: record it through the skater update
    seen: dict = {}
    orig = gm.skater_update

    def spy(theta, ids, mins, step, gain=None):
        seen["theta"] = theta
        orig(theta, ids, mins, step, gain)

    gm.skater_update = spy
    rows, players = [], []
    try:
        for T in TEST:
            seen.clear()
            mu = gm.run_player_season(games, pri, T, lam=hp["lam"], alpha=hp["alpha"], n0=hp["n0"])
            t = games.table.filter(pl.col("season") == T)
            f = fin[T]
            n_played: dict = {}
            for k, (gid, hid, aid, xgd) in enumerate(t.select("game_id", "home_id", "away_id", "xgd").iter_rows()):
                side = []
                for team in (hid, aid):
                    ids, mins = games.lineups[(gid, team)]
                    side.append(sum(f.get(i, 0.0) * m for i, m in zip(ids.tolist(), mins.tolist())) / 60.0)
                    n_played[team] = n_played.get(team, 0) + 1
                g_no = min(n_played[hid], n_played[aid])
                rows.append((T, gid, g_no, xgd - mu[k], mu[k], side[0] - side[1]))
            theta_end = seen.get("theta", {})
            mins = (
                pl.read_parquet(proc / "player_game.parquet", columns=["player_id", "season", "toi_all"])
                .filter(pl.col("season") == T)
                .group_by("player_id")
                .agg((pl.col("toi_all").sum() / 60.0).alias("m"))
            )
            for pid, m in mins.iter_rows():
                if pid in pri.theta0[T]:
                    t0 = pri.theta0[T][pid]
                    players.append((T, pid, m, t0, f.get(pid, 0.0), theta_end.get(pid, t0) - t0))
    finally:
        gm.skater_update = orig

    g = pl.DataFrame(rows, schema=["season", "game_id", "game_no", "e", "mu", "dF"], orient="row")
    pp = pl.DataFrame(players, schema=["season", "player_id", "m", "theta0", "fin", "change"], orient="row")
    res: dict = {"hyperparameters": hp, "test_seasons": TEST, "games": g.height}

    def game_fit(d: pl.DataFrame) -> dict:
        e, dF, mu = d["e"].to_numpy(), d["dF"].to_numpy(), d["mu"].to_numpy()
        b, V = ols(e, np.column_stack([mu - dF, dF]))
        gap = b[2] - b[1]
        se_gap = float(np.sqrt(V[1, 1] + V[2, 2] - 2 * V[1, 2]))
        return {
            "games": d.height,
            "slope_rest": float(b[1]),
            "se_rest": float(np.sqrt(V[1, 1])),
            "slope_finishing": float(b[2]),
            "se_finishing": float(np.sqrt(V[2, 2])),
            "gap": float(gap),
            "se_gap": se_gap,
            "z_gap": float(gap / se_gap),
            "sd_dF": float(dF.std()),
            "sd_mu": float(mu.std()),
        }

    res["per_game"] = {
        "all": game_fit(g),
        f"first_{EARLY_GAMES}_games": game_fit(g.filter(pl.col("game_no") <= EARLY_GAMES)),
        f"after_{EARLY_GAMES}_games": game_fit(g.filter(pl.col("game_no") > EARLY_GAMES)),
        **{str(T): game_fit(g.filter(pl.col("season") == T)) for T in TEST},
    }
    q = pp.filter(pl.col("m") >= 200)
    b, V = ols(
        q["change"].to_numpy(),
        np.column_stack([(q["theta0"] - q["fin"]).to_numpy(), q["fin"].to_numpy()]),
        q["m"].to_numpy(),
    )
    res["per_player"] = {
        "player_seasons": q.height,
        "slope_rest": float(b[1]),
        "se_rest": float(np.sqrt(V[1, 1])),
        "slope_finishing": float(b[2]),
        "se_finishing": float(np.sqrt(V[2, 2])),
        "share_left_finishing": float(1 + b[2]),
        "share_left_rest": float(1 + b[1]),
        "sd_fin_per60": float(q["fin"].std()),
        "fin_p90_per60": float(q["fin"].quantile(0.9)),
    }
    pg_ = res["per_game"]["all"]
    res["mismatch"] = bool(pg_["z_gap"] < -2)
    (met / "inseason_d0.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info("per game: %s", {k: round(v, 3) for k, v in pg_.items() if isinstance(v, float)})
    log.info("per player: %s", {k: round(v, 3) for k, v in res["per_player"].items() if isinstance(v, float)})
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
