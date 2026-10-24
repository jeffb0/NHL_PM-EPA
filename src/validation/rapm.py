"""5v5 offensive xG RAPM (Macdonald 2011; Gramacy, Jensen & Taddy 2013) as an external
validation target for PM-EPA.

Stints = maximal intervals within a period with constant on-ice skater sets and 5v5
(both goalies in). Each stint yields two rows (each team on offense):
    y = 60 * xGF / minutes,  weight = minutes
    X = +1 for each offensive skater (offense block), +1 for each defensive skater
        (defense block), + home-offense indicator
Weighted ridge; alpha chosen by game-grouped CV on the first window.
Output: per-player offensive impact (xGF/60 above average teammate context).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

log = logging.getLogger(__name__)


def _game_stints(args) -> list[tuple]:
    gid, sh, ev, shots, home_id = args
    out = []
    for per in np.unique(sh["period"]):
        m = sh["period"] == per
        st, en, pid, home = sh["start"][m], sh["end"][m], sh["player_id"][m], sh["team_id"][m] == home_id
        em = ev["period"] == per
        et = ev["t_period"][em]
        sit_change = et[np.r_[True, np.diff(ev["code"][em]) != 0]] if em.any() else np.array([])
        bounds = np.unique(np.concatenate([st, en, sit_change, [0]]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a <= 0:
                continue
            on = (st <= a) & (en >= b)
            h, aw = pid[on & home], pid[on & ~home]
            if len(h) != 5 or len(aw) != 5:
                continue
            # situation at the interval: last event at or before a
            if em.any():
                k = np.searchsorted(et, a, side="right") - 1
                code = ev["code"][em][max(k, 0)]
                if code != 1551:
                    continue
            sm = (shots["period"] == per) & (shots["t_period"] > a) & (shots["t_period"] <= b)
            hx = shots["xg"][sm & shots["is_home"]].sum()
            ax = shots["xg"][sm & ~shots["is_home"]].sum()
            out.append((gid, (b - a) / 60.0, tuple(h), tuple(aw), hx, ax))
    return out


def build_stints(proc, seasons: list[int], goalie_ids: set[int]) -> pl.DataFrame:
    games = pl.read_parquet(proc / "games.parquet").filter(pl.col("season").is_in(seasons)).select("game_id", "home_id")
    q = pl.read_parquet(proc / "shift_quality.parquet").filter(pl.col("onice_match_rate") >= 0.9).select("game_id")
    games = games.join(q, on="game_id")
    sh = (
        pl.read_parquet(proc / "shifts.parquet")
        .join(games.select("game_id"), on="game_id")
        .filter(~pl.col("player_id").is_in(list(goalie_ids)))
    )
    sh = sh.unique(subset=["game_id", "player_id", "period", "start", "end"])
    ev = pl.read_parquet(
        proc / "events.parquet",
        columns=[
            "game_id",
            "event_idx",
            "period",
            "t_period",
            "home_skaters",
            "away_skaters",
            "home_goalie",
            "away_goalie",
            "type",
            "is_home",
        ],
    ).join(games.select("game_id"), on="game_id")
    ev = ev.with_columns(
        (
            pl.col("away_goalie") * 1000
            + pl.col("away_skaters") * 100
            + pl.col("home_skaters") * 10
            + pl.col("home_goalie")
        ).alias("code")
    ).sort("game_id", "event_idx")
    xg = pl.read_parquet(proc / "shots_xg.parquet").select("game_id", "event_idx", "xg")
    shots = ev.filter(pl.col("type").is_in(["shot-on-goal", "missed-shot", "goal"])).join(
        xg, on=["game_id", "event_idx"]
    )
    shg = {k[0]: g for k, g in shots.group_by(["game_id"])}
    evg = {k[0]: g for k, g in ev.group_by(["game_id"])}
    tasks = []
    for (gid,), g in sh.group_by(["game_id"]):
        e = evg.get(gid)
        s = shg.get(gid)
        if e is None:
            continue
        home_id = games.filter(pl.col("game_id") == gid)["home_id"][0]
        tasks.append(
            (
                gid,
                {c: g[c].to_numpy() for c in ["period", "start", "end", "player_id", "team_id"]},
                {"period": e["period"].to_numpy(), "t_period": e["t_period"].to_numpy(), "code": e["code"].to_numpy()},
                {
                    "period": s["period"].to_numpy(),
                    "t_period": s["t_period"].to_numpy(),
                    "xg": s["xg"].to_numpy(),
                    "is_home": s["is_home"].to_numpy().astype(bool),
                }
                if s is not None
                else {
                    "period": np.array([]),
                    "t_period": np.array([]),
                    "xg": np.array([]),
                    "is_home": np.array([], dtype=bool),
                },
                home_id,
            )
        )
    rows = []
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(_game_stints, tasks, chunksize=32):
            rows.extend(r)
    return pl.DataFrame(rows, schema=["game_id", "minutes", "home_on", "away_on", "home_xg", "away_xg"], orient="row")


def design(stints: pl.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    players = sorted({p for col in ("home_on", "away_on") for lst in stints[col].to_list() for p in lst})
    idx = {p: i for i, p in enumerate(players)}
    P = len(players)
    n = stints.height
    rows, cols = [], []
    home_on, away_on = stints["home_on"].to_list(), stints["away_on"].to_list()
    for r in range(n):
        for p in home_on[r]:  # row r: home offense, away defense
            rows.append(r)
            cols.append(idx[p])
        for p in away_on[r]:
            rows.append(r)
            cols.append(P + idx[p])
        for p in away_on[r]:  # row n + r: away offense, home defense
            rows.append(n + r)
            cols.append(idx[p])
        for p in home_on[r]:
            rows.append(n + r)
            cols.append(P + idx[p])
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * n, 2 * P))
    home_col = sparse.csr_matrix(np.r_[np.ones(n), np.zeros(n)][:, None])
    X = sparse.hstack([X, home_col]).tocsr()
    mins = stints["minutes"].to_numpy()
    y = np.r_[stints["home_xg"].to_numpy(), stints["away_xg"].to_numpy()] * 60 / np.r_[mins, mins]
    w = np.r_[mins, mins]
    groups = np.r_[stints["game_id"].to_numpy(), stints["game_id"].to_numpy()]
    return X, y, w, groups, players


def choose_alpha(X, y, w, groups, grid=(100, 300, 1000, 3000, 10000)) -> tuple[float, dict]:
    res = {}
    for a in grid:
        errs = []
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = Ridge(alpha=a, fit_intercept=True).fit(X[tr], y[tr], sample_weight=w[tr])
            errs.append(np.average((m.predict(X[te]) - y[te]) ** 2, weights=w[te]))
        res[a] = float(np.mean(errs))
    return min(res, key=res.get), res


def fit_rapm(X, y, w, players, alpha) -> pl.DataFrame:
    m = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=w)
    P = len(players)
    toi = np.asarray(X[:, :P].multiply(w[:, None]).sum(0)).ravel()  # offensive minutes
    return pl.DataFrame(
        {"player_id": players, "rapm_off_xgf60": m.coef_[:P], "rapm_def_xga60": m.coef_[P : 2 * P], "rapm_minutes": toi}
    )
