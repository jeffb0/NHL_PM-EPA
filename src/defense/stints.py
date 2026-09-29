"""5v5 stints carrying the opponent's PM-EPA ledger (methodology §3–§4).

A stint is a maximal interval (a, b] within a period with constant on-ice skaters, 5v5 with both
goalies in. Faceoff times are also stint boundaries, so every faceoff starts a stint and the stint
records its zone start. Each ledger row (one possession's value booked at one event) is placed in
the stint containing its event under the on-ice convention: start < t <= end, faceoffs
start <= t < end (src/possession/onice.py).

Output ``def_stints.parquet``: one row per stint with the on-ice sets, covariates, and for each side
(h_ = home attacking, a_ = away attacking) the sum of every ledger component plus goals.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

COMPONENTS = ["start_turnover", "start_faceoff", "start_recovery", "start_unobserved", "move", "term", "quality"]
TURNOVER_STARTS = {"takeaway", "loose_opp_giveaway", "loose_unobserved"}
RECOVERY_STARTS = {"loose_own_miss", "loose_own_sog", "loose_own_block", "rebound", "loose_opp_miss", "loose_opp_sog", "loose_opp_block"}
SIT_5V5 = 1551  # situationCode: away goalie, away skaters, home skaters, home goalie
ZONE_LINE = 25.0  # blue lines at |x| = 25 ft
N_VAL = 2 * len(COMPONENTS) + 2  # h_ components, a_ components, h_goals, a_goals


def classify(phase: pl.Expr, component: pl.Expr, start_type: pl.Expr) -> pl.Expr:
    """Map a (phase, PM-EPA component, possession start type) ledger row to its DPM component."""
    return (
        pl.when(phase == "move").then(pl.lit("move"))
        .when(phase == "terminal").then(pl.when(component == "shot_quality").then(pl.lit("quality")).otherwise(pl.lit("term")))
        .when((component == "takeaway") | start_type.is_in(list(TURNOVER_STARTS))).then(pl.lit("start_turnover"))
        .when(start_type == "faceoff").then(pl.lit("start_faceoff"))
        .when(start_type.is_in(list(RECOVERY_STARTS))).then(pl.lit("start_recovery"))
        .otherwise(pl.lit("start_unobserved"))
    )


def ledger(proc, credits_file: str = "credits.parquet") -> pl.DataFrame:
    """Team-level ledger: one row per (possession team, event, DPM component), located in time."""
    cr = pl.read_parquet(proc / credits_file, columns=["game_id", "poss_id", "team_id", "event_idx", "component", "phase", "value"])
    poss = pl.read_parquet(proc / "poss_valued_epa.parquet", columns=["poss_id", "start_type"])
    ev = pl.read_parquet(proc / "events.parquet", columns=["game_id", "event_idx", "period", "t_period", "type"])
    games = pl.read_parquet(proc / "games.parquet", columns=["game_id", "home_id"])
    led = (
        cr.join(poss, on="poss_id", how="left")
        .with_columns(classify(pl.col("phase"), pl.col("component"), pl.col("start_type").fill_null("unobserved")).alias("dcomp"))
        .group_by("game_id", "team_id", "event_idx", "dcomp").agg(pl.col("value").sum())
        .join(ev, on=["game_id", "event_idx"], how="left")
        .join(games, on="game_id")
        .with_columns((pl.col("team_id") == pl.col("home_id")).alias("att_home"), (pl.col("type") == "faceoff").alias("is_fo"))
    )
    n_missing = led.filter(pl.col("t_period").is_null()).height
    if n_missing:
        log.warning("ledger rows without an event time (dropped): %d", n_missing)
    return led.filter(pl.col("t_period").is_not_null()).select("game_id", "event_idx", "period", "t_period", "is_fo", "att_home", "dcomp", "value")


def _locate(bounds: np.ndarray, t: np.ndarray, faceoff: np.ndarray) -> np.ndarray:
    """Interval index k of (bounds[k], bounds[k+1]] for each time; faceoffs use [bounds[k], bounds[k+1])."""
    return np.where(faceoff, np.searchsorted(bounds, t, side="right"), np.searchsorted(bounds, t, side="left")) - 1


def _game(args) -> tuple[list[tuple], dict]:
    gid, sh, ev, led = args
    C = len(COMPONENTS)
    out = []
    tot = ok = 0.0
    for per in np.unique(sh["period"]):
        m = sh["period"] == per
        st, en, pid, home = sh["start"][m], sh["end"][m], sh["player_id"][m], sh["home"][m]
        em = ev["period"] == per
        if not em.any():
            continue
        et, code, is_fo, ex, ehome, hsd = ev["t"][em], ev["code"][em], ev["is_fo"][em], ev["x"][em], ev["home"][em], ev["home_sd"][em]
        sit_change = et[np.r_[True, np.diff(code) != 0]]
        bounds = np.unique(np.concatenate([st, en, sit_change, et[is_fo], [0]]))
        n_int = len(bounds) - 1
        # situation of an interval: the codes of the events inside it (all must be 5v5); an interval with no
        # events inherits the last earlier event's code. (The code of the last event *before* the interval lags
        # a change: after a penalty expires, the first 5v5-coded event would otherwise sit in a rejected interval.)
        ek = _locate(bounds, et, is_fo)
        inb = (ek >= 0) & (ek < n_int)
        n_in = np.bincount(ek[inb], minlength=n_int)
        n_bad = np.bincount(ek[inb & (code != SIT_5V5)], minlength=n_int)
        valid = np.zeros(n_int, dtype=bool)
        meta = [None] * n_int
        for k in range(n_int):
            a, b = bounds[k], bounds[k + 1]
            on = (st <= a) & (en >= b)
            h, aw = np.sort(pid[on & home]), np.sort(pid[on & ~home])
            if len(h) != 5 or len(aw) != 5:
                continue
            if n_in[k]:
                if n_bad[k]:
                    continue
            elif code[max(np.searchsorted(et, a, side="right") - 1, 0)] != SIT_5V5:
                continue
            zone = "fly"
            fo = np.flatnonzero(is_fo & (et == a))
            if len(fo) and np.isfinite(ex[fo[0]]):
                xh = ex[fo[0]] if ehome[fo[0]] else -ex[fo[0]]  # faceoff x is in the winner's attacking frame
                zone = "OZ" if xh > ZONE_LINE else "DZ" if xh < -ZONE_LINE else "NZ"
            nxt = min(np.searchsorted(et, a, side="right"), len(et) - 1)  # score before the next event = score during the stint
            valid[k] = True
            meta[k] = (gid, int(per), float(a), float(b), float(b - a) / 60.0, h.tolist(), aw.tolist(), zone, int(np.clip(hsd[nxt], -2, 2)))
        vals = np.zeros((n_int, N_VAL))
        lm = led["period"] == per
        if lm.any():
            k = _locate(bounds, led["t"][lm], led["is_fo"][lm])
            v, five = led["value"][lm], led["is_5v5"][lm]
            hit = (k >= 0) & (k < n_int)
            hit[hit] = valid[k[hit]]
            tot += np.abs(v[five]).sum()
            ok += np.abs(v[five & hit]).sum()
            col = np.where(led["att_home"][lm], 0, C) + led["comp"][lm]
            np.add.at(vals, (k[hit], col[hit]), v[hit])
        gm = ev["is_goal"][em]
        if gm.any():
            k = _locate(bounds, et[gm], np.zeros(gm.sum(), dtype=bool))
            hit = (k >= 0) & (k < n_int)
            hit[hit] = valid[k[hit]]
            np.add.at(vals, (k[hit], np.where(ehome[gm][hit], 2 * C, 2 * C + 1)), 1.0)
        out.extend(meta[k] + tuple(vals[k].tolist()) for k in np.flatnonzero(valid))
    return out, {"game_id": gid, "ledger_abs_5v5": tot, "ledger_abs_assigned": ok}


def build(proc, seasons: list[int], goalie_ids: set[int], min_match_rate: float, workers: int, credits_file: str = "credits.parquet") -> tuple[pl.DataFrame, pl.DataFrame]:
    games = pl.read_parquet(proc / "games.parquet").filter(pl.col("season").is_in(seasons)).select("game_id", "season", "home_id")
    q = pl.read_parquet(proc / "shift_quality.parquet").filter(pl.col("onice_match_rate") >= min_match_rate).select("game_id")
    games = games.join(q, on="game_id")
    sh = (
        pl.read_parquet(proc / "shifts.parquet").join(games, on="game_id")
        .filter(~pl.col("player_id").is_in(list(goalie_ids)))
        .unique(subset=["game_id", "player_id", "period", "start", "end"])
        .with_columns((pl.col("team_id") == pl.col("home_id")).alias("home"))
    )
    ev = (
        pl.read_parquet(proc / "events.parquet", columns=["game_id", "event_idx", "period", "t_period", "type", "x", "is_home",
                                                          "home_score", "away_score", "home_skaters", "away_skaters", "home_goalie", "away_goalie"])
        .join(games.select("game_id"), on="game_id").sort("game_id", "event_idx")
        .with_columns(
            (pl.col("away_goalie") * 1000 + pl.col("away_skaters") * 100 + pl.col("home_skaters") * 10 + pl.col("home_goalie")).alias("code"),
            (pl.col("home_score") - pl.col("away_score")).fill_null(0).alias("home_sd"),
        )
    )
    led = ledger(proc, credits_file).join(games.select("game_id"), on="game_id")
    # 5v5 flag for the G5 denominator: the situation of the event where the value was booked
    led = led.join(ev.select("game_id", "event_idx", "code"), on=["game_id", "event_idx"], how="left").with_columns(
        (pl.col("code") == SIT_5V5).fill_null(False).alias("is_5v5"),
        pl.col("dcomp").replace_strict({c: i for i, c in enumerate(COMPONENTS)}, return_dtype=pl.Int64).alias("comp"),
    )
    evg = {k[0]: g for k, g in ev.group_by(["game_id"])}
    ledg = {k[0]: g for k, g in led.group_by(["game_id"])}
    tasks = []
    for (gid,), g in sh.group_by(["game_id"]):
        e = evg.get(gid)
        if e is None:
            continue
        lg = ledg.get(gid, led.clear())
        tasks.append((
            gid,
            {"period": g["period"].to_numpy(), "start": g["start"].to_numpy(), "end": g["end"].to_numpy(),
             "player_id": g["player_id"].to_numpy(), "home": g["home"].to_numpy()},
            {"period": e["period"].to_numpy(), "t": e["t_period"].to_numpy(), "code": e["code"].to_numpy(),
             "is_fo": (e["type"] == "faceoff").to_numpy(), "is_goal": (e["type"] == "goal").to_numpy(),
             "x": e["x"].cast(pl.Float64).fill_null(np.nan).to_numpy(), "home": e["is_home"].fill_null(False).to_numpy(),
             "home_sd": e["home_sd"].to_numpy()},
            {"period": lg["period"].to_numpy(), "t": lg["t_period"].to_numpy(), "is_fo": lg["is_fo"].to_numpy(),
             "att_home": lg["att_home"].to_numpy(), "comp": lg["comp"].to_numpy(), "value": lg["value"].to_numpy(),
             "is_5v5": lg["is_5v5"].to_numpy()},
        ))
    rows, qual = [], []
    with ProcessPoolExecutor(workers) as pool:
        for r, qd in pool.map(_game, tasks, chunksize=16):
            rows.extend(r)
            qual.append(qd)
    cols = (["game_id", "period", "a", "b", "minutes", "home_on", "away_on", "zone_home", "score_home"]
            + [f"h_{c}" for c in COMPONENTS] + [f"a_{c}" for c in COMPONENTS] + ["h_goals", "a_goals"])
    schema = {c: pl.Float64 for c in cols} | {"game_id": pl.Int64, "period": pl.Int64, "home_on": pl.List(pl.Int64), "away_on": pl.List(pl.Int64),
                                              "zone_home": pl.Utf8, "score_home": pl.Int64}
    st = pl.DataFrame(rows, schema=schema, orient="row").join(games.select("game_id", "season"), on="game_id")
    return st, pl.DataFrame(qual)
