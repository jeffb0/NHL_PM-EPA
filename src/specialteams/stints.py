"""Non-5v5 stints carrying shot xG (methodology §3).

A stint is a maximal interval (a, b] within a period with constant on-ice skaters (from the shift charts) whose
skater counts agree with every event recorded inside it. Faceoff times are also boundaries, so a faceoff starts a
stint and the stint records its zone start. Goalies are not tracked (their shifts are dropped); a pulled goalie
shows up as a sixth skater. Only states outside 5v5 with 3 to 6 skaters a side are kept (methodology §1).

Shots are unblocked attempts (shot-on-goal, missed, goal) with the xG of ``shots_xg.parquet``, the same shots and
xG as the finishing component, assigned to the stint containing the event (start < t <= end).

Output ``st_stints.parquet``: one row per stint: on-ice sets, skater counts, zone start, score state, and each
side's xG and goals (h_ = home attacking, a_ = away attacking).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from src.defense.stints import ZONE_LINE, _locate

log = logging.getLogger(__name__)
UNBLOCKED = ["shot-on-goal", "missed-shot", "goal"]
GROUPS = ["adv", "4v4", "3v3"]
NO_COUNT_EVENTS = ["period-end", "game-end", "shootout-complete"]


def group_of(nh: int, na: int) -> str | None:
    """Stint group by skater counts: 'adv' (mismatched, extra-attacker states included), '4v4', '3v3'; None = not
    modeled."""
    if nh == na:
        return {4: "4v4", 3: "3v3"}.get(nh)
    return "adv" if 3 <= nh <= 6 and 3 <= na <= 6 else None


def _game(args) -> tuple[list[tuple], list[tuple]]:
    gid, sh, ev, shots = args
    rows: list[tuple] = []
    cov: list[tuple] = []
    for per in np.unique(sh["period"]):
        m = sh["period"] == per
        st, en, pid, home = sh["start"][m], sh["end"][m], sh["player_id"][m], sh["home"][m]
        em = ev["period"] == per
        if not em.any():
            continue
        et, hs, aw, is_fo, ex, ehome, hsd = (
            ev["t"][em],
            ev["hs"][em],
            ev["as"][em],
            ev["is_fo"][em],
            ev["x"][em],
            ev["home"][em],
            ev["home_sd"][em],
        )
        code = hs * 10 + aw
        change = et[np.r_[True, np.diff(code) != 0]]
        bounds = np.unique(np.concatenate([st, en, change, et[is_fo], [0]]))
        n_int = len(bounds) - 1
        ek = _locate(bounds, et, is_fo)
        inb = (ek >= 0) & (ek < n_int)
        n_in = np.bincount(ek[inb], minlength=n_int)
        valid = np.zeros(n_int, dtype=bool)
        meta: list[tuple | None] = [None] * n_int
        for k in range(n_int):
            a, b = bounds[k], bounds[k + 1]
            on = (st <= a) & (en >= b)
            h, w = np.sort(pid[on & home]), np.sort(pid[on & ~home])
            nh, na = len(h), len(w)
            if group_of(nh, na) is None:
                continue
            if n_in[k]:
                inside = inb & (ek == k)
                if np.any(hs[inside] != nh) or np.any(aw[inside] != na):
                    continue
            else:
                j = max(np.searchsorted(et, a, side="right") - 1, 0)
                if hs[j] != nh or aw[j] != na:
                    continue
            zone = "fly"
            fo = np.flatnonzero(is_fo & (et == a))
            if len(fo) and np.isfinite(ex[fo[0]]):
                xh = ex[fo[0]] if ehome[fo[0]] else -ex[fo[0]]
                zone = "OZ" if xh > ZONE_LINE else "DZ" if xh < -ZONE_LINE else "NZ"
            nxt = min(np.searchsorted(et, a, side="right"), len(et) - 1)
            valid[k] = True
            meta[k] = (
                gid,
                int(per),
                float(a),
                float(b),
                float(b - a) / 60.0,
                h.tolist(),
                w.tolist(),
                nh,
                na,
                zone,
                int(np.clip(hsd[nxt], -2, 2)),
            )
        vals = np.zeros((n_int, 4))
        sm = shots["period"] == per
        if sm.any():
            k = _locate(bounds, shots["t"][sm], np.zeros(sm.sum(), dtype=bool))
            inb_s = (k >= 0) & (k < n_int)
            hit = inb_s.copy()
            hit[inb_s] = valid[k[inb_s]]
            xg, goal, sh_home = shots["xg"][sm], shots["goal"][sm], shots["home"][sm]
            col = np.where(sh_home, 0, 1)
            np.add.at(vals, (k[hit], col[hit]), xg[hit])
            np.add.at(vals, (k[hit], col[hit] + 2), goal[hit].astype(float))
            # coverage bookkeeping: xG of shots by the state recorded on the shot event, and how much of it landed in a
            # valid stint
            labels = np.array([group_of(int(x), int(y)) or "" for x, y in zip(shots["hs"][sm], shots["as"][sm])])
            for g_name in GROUPS:
                grp = labels == g_name
                cov.append((gid, g_name, float(xg[grp].sum()), float(xg[grp & hit].sum())))
        rows.extend(meta[k] + tuple(vals[k].tolist()) for k in np.flatnonzero(valid))
    return rows, cov


def build(
    proc, seasons: list[int], goalie_ids: set[int], min_match_rate: float, workers: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    games = (
        pl.read_parquet(proc / "games.parquet")
        .filter(pl.col("season").is_in(seasons))
        .select("game_id", "season", "home_id")
    )
    q = (
        pl.read_parquet(proc / "shift_quality.parquet")
        .filter(pl.col("onice_match_rate") >= min_match_rate)
        .select("game_id")
    )
    games = games.join(q, on="game_id")
    sh = (
        pl.read_parquet(proc / "shifts.parquet")
        .join(games, on="game_id")
        .filter(~pl.col("player_id").is_in(list(goalie_ids)))
        .unique(subset=["game_id", "player_id", "period", "start", "end"])
        .with_columns((pl.col("team_id") == pl.col("home_id")).alias("home"))
    )
    ev = (
        pl.read_parquet(
            proc / "events.parquet",
            columns=[
                "game_id",
                "event_idx",
                "period",
                "t_period",
                "type",
                "x",
                "is_home",
                "home_score",
                "away_score",
                "home_skaters",
                "away_skaters",
            ],
        )
        .join(games.select("game_id"), on="game_id")
        .filter(pl.col("home_skaters").is_not_null() & pl.col("away_skaters").is_not_null())
        # bookkeeping events carry placeholder skater counts (e.g. 4v4 at the end of a 3v3 overtime)
        .filter(~pl.col("type").is_in(NO_COUNT_EVENTS))
        .sort("game_id", "event_idx")
        .with_columns((pl.col("home_score") - pl.col("away_score")).fill_null(0).alias("home_sd"))
    )
    xg = pl.read_parquet(proc / "shots_xg.parquet", columns=["game_id", "event_idx", "xg"])
    shots = (
        ev.filter(pl.col("type").is_in(UNBLOCKED))
        .join(xg, on=["game_id", "event_idx"], how="inner")
        .with_columns((pl.col("type") == "goal").alias("goal"), pl.col("is_home").fill_null(False).alias("home"))
    )
    evg = {k[0]: g for k, g in ev.group_by(["game_id"])}
    shg = {k[0]: g for k, g in shots.group_by(["game_id"])}
    tasks = []
    for (gid,), g in sh.group_by(["game_id"]):
        e = evg.get(gid)
        s = shg.get(gid)
        if e is None or s is None:
            continue
        tasks.append(
            (
                gid,
                {
                    "period": g["period"].to_numpy(),
                    "start": g["start"].to_numpy(),
                    "end": g["end"].to_numpy(),
                    "player_id": g["player_id"].to_numpy(),
                    "home": g["home"].to_numpy(),
                },
                {
                    "period": e["period"].to_numpy(),
                    "t": e["t_period"].to_numpy(),
                    "hs": e["home_skaters"].to_numpy().astype(np.int64),
                    "as": e["away_skaters"].to_numpy().astype(np.int64),
                    "is_fo": (e["type"] == "faceoff").to_numpy(),
                    "x": e["x"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                    "home": e["is_home"].fill_null(False).to_numpy(),
                    "home_sd": e["home_sd"].to_numpy(),
                },
                {
                    "period": s["period"].to_numpy(),
                    "t": s["t_period"].to_numpy(),
                    "xg": s["xg"].to_numpy(),
                    "goal": s["goal"].to_numpy(),
                    "home": s["home"].to_numpy(),
                    "hs": s["home_skaters"].to_numpy().astype(np.int64),
                    "as": s["away_skaters"].to_numpy().astype(np.int64),
                },
            )
        )
    out, cov = [], []
    with ProcessPoolExecutor(workers) as pool:
        for r, c in pool.map(_game, tasks, chunksize=16):
            out.extend(r)
            cov.extend(c)
    schema = {
        "game_id": pl.Int64,
        "period": pl.Int64,
        "a": pl.Float64,
        "b": pl.Float64,
        "minutes": pl.Float64,
        "home_on": pl.List(pl.Int64),
        "away_on": pl.List(pl.Int64),
        "nh": pl.Int64,
        "na": pl.Int64,
        "zone_home": pl.Utf8,
        "score_home": pl.Int64,
        "h_xg": pl.Float64,
        "a_xg": pl.Float64,
        "h_goals": pl.Float64,
        "a_goals": pl.Float64,
    }
    st = pl.DataFrame(out, schema=schema, orient="row").join(games.select("game_id", "season"), on="game_id")
    st = st.with_columns(
        pl.struct("nh", "na").map_elements(lambda s: group_of(s["nh"], s["na"]), return_dtype=pl.Utf8).alias("grp")
    )
    cv = pl.DataFrame(
        cov,
        schema={"game_id": pl.Int64, "grp": pl.Utf8, "xg_state": pl.Float64, "xg_assigned": pl.Float64},
        orient="row",
    ).join(games.select("game_id", "season"), on="game_id")
    return st, cv
