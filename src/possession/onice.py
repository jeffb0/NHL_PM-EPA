"""On-ice skaters per event and per-player time on ice by strength, from official shift charts.

On-ice convention (standard in public hockey analytics):
  * a player is on ice for an event at time t if  start < t <= end
  * for faceoffs (which begin a shift)             start <= t <  end
Goalies are excluded from on-ice skater lists.

TOI by strength: the PBP ``situationCode`` of the most recent event is forward-filled to a
1-second grid per period; each shift second is labelled with the strength from the
player's team perspective (5v5 / PP / PK / 4v4 / 3v3 / EN / other).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from src.config import load_config

log = logging.getLogger(__name__)

STRENGTHS = ["5v5", "PP", "PK", "4v4", "3v3", "EN", "other"]


def _strength_codes(own_sk: np.ndarray, opp_sk: np.ndarray, own_g: np.ndarray, opp_g: np.ndarray) -> np.ndarray:
    out = np.full(own_sk.shape, STRENGTHS.index("other"), dtype=np.int8)
    even = own_sk == opp_sk
    out[even & (own_sk == 5)] = 0
    out[even & (own_sk == 4)] = 3
    out[even & (own_sk == 3)] = 4
    out[own_sk > opp_sk] = 1
    out[own_sk < opp_sk] = 2
    out[(own_g == 0) | (opp_g == 0)] = 5
    return out


def _game(args: tuple) -> tuple[list[dict], list[dict], dict]:
    gid, ev, sh, home_id = args
    # ---- on-ice per event
    e_per, e_t, e_fo = ev["period"], ev["t_period"], ev["type"] == "faceoff"
    s_per, s_start, s_end, s_pid, s_home = (
        sh["period"],
        sh["start"],
        sh["end"],
        sh["player_id"],
        sh["team_id"] == home_id,
    )
    same = e_per[:, None] == s_per[None, :]
    t = e_t[:, None]
    covered = np.where(e_fo[:, None], (s_start <= t) & (t < s_end), (s_start < t) & (t <= s_end))
    mask = same & covered
    onice = []
    n_ok = n_55 = 0
    for i in range(len(e_per)):
        m = mask[i]
        h = np.sort(s_pid[m & s_home])
        a = np.sort(s_pid[m & ~s_home])
        onice.append(
            {"game_id": gid, "event_idx": int(ev["event_idx"][i]), "home_on": h.tolist(), "away_on": a.tolist()}
        )
        if (
            ev["home_skaters"][i] == 5
            and ev["away_skaters"][i] == 5
            and ev["home_goalie"][i] == 1
            and ev["away_goalie"][i] == 1
        ):
            n_55 += 1
            n_ok += int(len(h) == 5 and len(a) == 5)
    # ---- TOI by strength (1-second grid)
    toi_rows = []
    for per in np.unique(s_per):
        pm = e_per == per
        if not pm.any():
            continue
        length = int(max(s_end[s_per == per].max(), e_t[pm].max(), 1))
        grid_t = np.arange(length)  # second [k, k+1)
        et = e_t[pm]
        # state applying during second k = situation of last event with t <= k
        idx = np.searchsorted(et, grid_t, side="right") - 1
        idx = np.clip(idx, 0, None)
        hs, as_ = ev["home_skaters"][pm][idx], ev["away_skaters"][pm][idx]
        hg, ag = ev["home_goalie"][pm][idx], ev["away_goalie"][pm][idx]
        home_code = _strength_codes(hs, as_, hg, ag)
        away_code = _strength_codes(as_, hs, ag, hg)
        cum_home = np.zeros((len(STRENGTHS), length + 1))
        cum_away = np.zeros((len(STRENGTHS), length + 1))
        for c in range(len(STRENGTHS)):
            cum_home[c, 1:] = np.cumsum(home_code == c)
            cum_away[c, 1:] = np.cumsum(away_code == c)
        sm = s_per == per
        st = np.clip(s_start[sm].astype(int), 0, length)
        en = np.clip(s_end[sm].astype(int), 0, length)
        cums = np.where(s_home[sm][None, :], cum_home[:, en] - cum_home[:, st], cum_away[:, en] - cum_away[:, st])
        for j, pid in enumerate(s_pid[sm]):
            row = {"game_id": gid, "player_id": int(pid), "team_id": int(sh["team_id"][sm][j])}
            row.update({f"toi_{s}": float(cums[c, j]) for c, s in enumerate(STRENGTHS)})
            toi_rows.append(row)
    q = {"game_id": gid, "n_5v5_events": n_55, "onice_match_rate": n_ok / n_55 if n_55 else np.nan}
    return onice, toi_rows, q


def run() -> None:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    events = pl.read_parquet(proc / "events.parquet").sort("game_id", "event_idx")
    players = pl.read_parquet(proc / "players.parquet")
    goalies = set(players.filter(pl.col("position") == "G")["player_id"].to_list())
    shifts = pl.read_parquet(proc / "shifts.parquet").filter(~pl.col("player_id").is_in(list(goalies)))
    shifts = shifts.unique(subset=["game_id", "player_id", "period", "start", "end"])
    ev_cols = ["event_idx", "period", "t_period", "type", "home_skaters", "away_skaters", "home_goalie", "away_goalie"]
    sh_groups = {k[0]: g for k, g in shifts.group_by(["game_id"])}
    tasks = []
    for (gid,), g in events.group_by(["game_id"], maintain_order=True):
        if gid not in sh_groups:
            continue
        s = sh_groups[gid]
        tasks.append(
            (
                gid,
                {c: g[c].to_numpy() for c in ev_cols},
                {c: s[c].to_numpy() for c in ["period", "start", "end", "player_id", "team_id"]},
                int(g["home_id"][0]),
            )
        )
    onice, toi, qual = [], [], []
    with ProcessPoolExecutor(4) as pool:
        for o, t, q in pool.map(_game, tasks, chunksize=32):
            onice.extend(o)
            toi.extend(t)
            qual.append(q)
    oi = pl.DataFrame(
        onice,
        schema={"game_id": pl.Int64, "event_idx": pl.Int64, "home_on": pl.List(pl.Int64), "away_on": pl.List(pl.Int64)},
    )
    oi.write_parquet(proc / "onice.parquet")
    toi_df = (
        pl.DataFrame(toi).group_by("game_id", "player_id", "team_id").agg(pl.col(f"toi_{s}").sum() for s in STRENGTHS)
    )
    toi_df = toi_df.with_columns((pl.sum_horizontal([f"toi_{s}" for s in STRENGTHS])).alias("toi_all"))
    toi_df.write_parquet(proc / "toi_game.parquet")
    q = pl.DataFrame(qual)
    q.write_parquet(proc / "shift_quality.parquet")
    n_games = events["game_id"].n_unique()
    log.info(
        "on-ice: %d events; games with shifts %d/%d; median 5v5 match rate %.4f; games <90%% match: %d",
        oi.height,
        q.height,
        n_games,
        q["onice_match_rate"].median(),
        q.filter(pl.col("onice_match_rate") < 0.9).height,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
