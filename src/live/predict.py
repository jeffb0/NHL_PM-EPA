"""Forecast an upcoming game with the game model:  python -m src.live.predict <AWAY> <HOME> [YYYY-MM-DD] [season]

Preseason ratings for the season (projection + carried team residual, goalie rolling rates through last season), updated
with the season's completed games
(docs/gamemodel: hyperparameters frozen, original rate model), then the logistic step fitted on 2019 to last season. The
lineup is each team's previous game's;
the starting goalie is the one with the most prior work unless named. Run src.live.update first so the season's games
are current."""

from __future__ import annotations

import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.config import ROOT, load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.inseason import individual as ind
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.prospects import features as pf
from src.prospects import hybrid as hy
from src.scoreeffects import factors as sf
from src.standings import candidates as cd
from src.standings import data as sd
from src.standings import forecast as fc
from src.standings.next import load_snapshot, next_goalie_rates

log = logging.getLogger("live")
LIVE = ROOT / "data/live/processed"
UNBLOCKED = ["shot-on-goal", "missed-shot", "goal"]


def live_games(season: int) -> gd_.Games:
    """The season's completed games from the live tables, in the game model's format."""
    g = pl.read_parquet(LIVE / "games.parquet").filter(pl.col("season") == season)
    so = pl.col("last_period_type") == "SO"
    g = g.with_columns(
        pl.when(so).then(0).otherwise(pl.col("home_final") - pl.col("away_final")).cast(pl.Float64).alias("gd"),
        (pl.col("home_final") > pl.col("away_final")).cast(pl.Float64).alias("home_win"),
        pl.col("game_date").str.to_date("%Y-%m-%d").alias("date"),
    )
    ev = pl.read_parquet(LIVE / "events.parquet", columns=["game_id", "event_idx", "player2", "period", "score_diff"])
    sh = (
        pl.read_parquet(LIVE / "shots_xg.parquet")
        .filter(pl.col("type").is_in(UNBLOCKED))
        .join(ev, on=["game_id", "event_idx"])
    )
    flag, wp = ROOT / "reports/metrics/score_effects.json", ROOT / "data/outputs/score_effect_weights.csv"
    # docs/scoreeffects: 5v5 shots weighted by score state in the differential that updates ratings
    if flag.exists() and wp.exists() and json.loads(flag.read_text()).get("implement"):
        sh = sf.apply(sh, pl.read_csv(wp)).with_columns(pl.col("xg_adj").alias("xg_update"))
    else:
        sh = sh.with_columns(pl.col("xg").alias("xg_update"))
    x = sh.group_by("game_id", "team_id").agg(pl.col("xg_update").sum().alias("xg"))
    g = (
        g.join(x.rename({"team_id": "home_id", "xg": "xh"}), on=["game_id", "home_id"])
        .join(x.rename({"team_id": "away_id", "xg": "xa"}), on=["game_id", "away_id"])
        .with_columns((pl.col("xh") - pl.col("xa")).alias("xgd"))
    )
    tg = pl.concat(
        [
            g.select("game_id", "date", pl.col("home_id").alias("team_id")),
            g.select("game_id", "date", pl.col("away_id").alias("team_id")),
        ]
    ).sort("team_id", "date")
    tg = tg.with_columns(
        ((pl.col("date") - pl.col("date").shift(1).over("team_id")).dt.total_days() == 1)
        .fill_null(False)
        .cast(pl.Float64)
        .alias("b2b")
    )
    g = g.join(
        tg.select("game_id", pl.col("team_id").alias("home_id"), pl.col("b2b").alias("b2b_home")),
        on=["game_id", "home_id"],
    ).join(
        tg.select("game_id", pl.col("team_id").alias("away_id"), pl.col("b2b").alias("b2b_away")),
        on=["game_id", "away_id"],
    )
    pg = pl.read_parquet(LIVE / "player_game.parquet").filter((pl.col("toi_all") > 0) & (pl.col("position") != "G"))
    lineups, stats = gd_.lineup_tables(pg)
    # goalies: the starter is the goalie on the most shots on goal; every unblocked attempt against with a goalie in net
    # is booked to him
    teams = g.select("game_id", "home_id", "away_id")
    ag = (
        sh.filter(~pl.col("empty_net"))
        .join(teams, on="game_id")
        .with_columns(
            pl.when(pl.col("team_id") == pl.col("home_id"))
            .then(pl.col("away_id"))
            .otherwise(pl.col("home_id"))
            .alias("def_id")
        )
    )
    start = (
        ag.filter(pl.col("player2").is_not_null() & pl.col("type").is_in(["shot-on-goal", "goal"]))
        .group_by("game_id", "def_id", "player2")
        .len()
        .sort(["len", "player2"], descending=[True, False])
        .unique(["game_id", "def_id"], keep="first", maintain_order=True)
    )
    tot = (
        ag.group_by("game_id", "def_id")
        .agg(pl.len().alias("n"), (pl.col("xg") - pl.col("goal")).sum().alias("saved"))
        .join(start.select("game_id", "def_id", "player2"), on=["game_id", "def_id"])
    )
    goalies = {
        (r["game_id"], r["def_id"]): [(int(r["player2"]), float(r["n"]), float(r["saved"]))]
        for r in tot.iter_rows(named=True)
    }
    g = g.sort("date", "game_id")
    return gd_.Games(
        g.select(
            "game_id",
            "season",
            "date",
            "home_id",
            "away_id",
            "home_abbrev",
            "away_abbrev",
            "gd",
            "xgd",
            "home_win",
            "b2b_home",
            "b2b_away",
        ),
        lineups,
        goalies,
        stats,
    )


def rate_features(cfg, raw, season: int) -> pl.DataFrame | None:
    """The elite-prospect columns (docs/prospects/elite.md) when that test's rule adopted them, rebuilt from the career
    pages on disk so new roster players are covered; else None (the original rate model)."""
    flag = cfg.path(cfg.data.metrics_dir) / "elite_prospects.json"
    if not flag.exists() or not json.loads(flag.read_text()).get("implement_A"):
        return None
    return pf.elite_table(raw, list(range(pr.FIRST_TARGET, season + 1)))


def season_priors(
    proc, cfg, season: int, live: gd_.Games, base: gd_.Priors, yp: pl.DataFrame | None
) -> tuple[gd_.Priors, pl.DataFrame, str]:
    """Preseason ratings for a season not yet in the historical tables: everyone on the published rosters or seen in a
    game this season. ``yp``: the elite-prospect columns of ``rate_features`` (None for the original rate model)."""
    raw = cfg.path(cfg.data.raw_dir)
    snap, stamp = load_snapshot(raw, season)
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    res = sd.results(proc)
    bio0 = pd_.player_bio(proc, raw)
    nb = snap.select("player_id", "birth_date", "height_in", "weight_lb").unique("player_id")
    bio = pl.concat([nb, bio0.filter(~pl.col("player_id").is_in(nb["player_id"]))])
    seen = (
        pl.read_parquet(LIVE / "player_game.parquet")
        .filter(pl.col("position") != "G")
        .select(
            pl.col("player_id").cast(pl.Int64),
            pl.when(pl.col("position") == "D").then(pl.lit("D")).otherwise(pl.lit("F")).alias("pos_group"),
        )
        .unique("player_id")
    )
    who = (
        pl.concat([snap.filter(pl.col("pos_group") != "G").select("player_id", "pos_group"), seen])
        .unique("player_id")
        .with_columns(pl.lit(0, pl.Int64).alias("team_id"))
    )
    d = pf.with_yp(fc.roster_frame(p, bio, who, season), yp)
    M = pf.spec(yp)
    rate = pm.predict_models(
        pm.fit_models(pr.train_rows(pf.with_yp(pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR), yp), 1, season), M),
        d,
        M,
    )
    resid = cd.team_residual(proc, p, res)
    prev = (
        p.filter(pl.col("season") == season - 1)
        .select("player_id", "team")
        .join(resid.filter(pl.col("season") == season - 1).select("team", "rho"), on="team", how="left")
        .select("player_id", "rho")
    )
    rho = d.select("player_id").join(prev, on="player_id", how="left")["rho"].fill_null(0.0).to_numpy()
    tl = (
        pl.read_parquet(proc / "player_game.parquet", columns=["player_id", "season", "toi_all"])
        .filter((pl.col("season") == season - 1) & (pl.col("toi_all") > 0))
        .group_by("player_id")
        .agg((pl.col("toi_all").mean() / 60).alias("m"))
    )
    # goalies without a rolling rate (docs/gamemodel §10 item 2): a returning goalie gets his career effect and a goalie
    # with no NHL history the mean of past debut seasons, when that test adopted them; else the replacement rate
    uj = cfg.path(cfg.data.metrics_dir) / "gk_unknown.json"
    unknown = json.loads(uj.read_text(encoding="utf-8")) if uj.exists() else {}
    rates = next_goalie_rates(proc, cfg, season, returning=bool(unknown.get("implement")))
    no_history = float(unknown["debut_rate"]) if unknown.get("implement") else float(rates["rep_v"][0])
    _, lg = sd.goalie_usage(proc)
    pos = {**base.pos, **dict(who.select("player_id", "pos_group").iter_rows())}
    pri = gd_.Priors(
        {season: dict(zip(d["player_id"].to_list(), (rate + 60.0 * rho).tolist()))},
        pos,
        {season: dict(tl.iter_rows())},
        {season: (dict(rates.select("goalie", "rate_v").iter_rows()), no_history)},
        {season: float(lg.filter(pl.col("season") == season - 1)["shots82"].item()) / 82.0},
        base.home_adv,
        m_eff={season: dict(d.select("player_id", "m_eff").iter_rows())},
        age={season: dict(d.select("player_id", "age").iter_rows())},
    )
    return pri, snap, stamp


def overrides(path: Path | None = None) -> tuple[set, dict, dict]:
    """Manual roster knowledge (data/external/roster_overrides.json): players who will not play, named starting goalies
    and named backups by team.
    Ignored when the file says ``"active": false`` (the default since 2026-10-10): the morning lineup feed is then the only
    source of lineup news."""
    path = path or ROOT / "data/external/roster_overrides.json"
    if not path.exists():
        return set(), {}, {}
    o = json.loads(path.read_text(encoding="utf-8"))
    if not o.get("active", True):
        return set(), {}, {}
    return (
        {int(u["player_id"]) for u in o.get("unavailable", [])},
        {k: int(v) for k, v in o.get("starters", {}).items()},
        {k: int(v) for k, v in o.get("backups", {}).items()},
    )


def historical_mu(
    proc,
    hist: gd_.Games,
    base: gd_.Priors,
    seasons: list[int],
    kw: dict,
    hdata: pl.DataFrame | None,
    idata: pl.DataFrame | None,
) -> np.ndarray:
    """The expected margin of every historical game from the model in its live configuration: for season T the
    early-sample models (``hdata``) and the individual-results models (``idata``) are fitted on seasons before T."""
    lines = last = who = None
    if idata is not None:
        gl = ind.game_lines(proc)
        lines = {
            (gid, pid): np.array(v, dtype=float)
            for gid, pid, *v in gl.select("game_id", "player_id", *ind.LINE).iter_rows()
        }
        last = {(i, s): m for i, s, m in ind.toi_last(gl).iter_rows()}
        who = idata.select("player_id", "season", "pos_group", "agegrp").unique(["player_id", "season"])
    out = []
    for T in seasons:
        early = u = None
        if hdata is not None:
            thin = {i for i, v in base.m_eff[T].items() if v < hy.THIN}
            early = hy.Early(thin, hy.fit(hdata.filter(pl.col("season") < T)), base.pos)
        if idata is not None:
            w = {i: (g, a) for i, g, a in who.filter(pl.col("season") == T).drop("season").iter_rows()}
            u = ind.Individual(
                ind.fit(idata.filter(pl.col("season") < T)), w, lines, {i: last[(i, T)] for i in w if (i, T) in last}
            )
        out.append(gm.run_player_season(hist, base, T, early=early, inseason=u, **kw))
    return np.concatenate(out)


def prepare(season: int = 2026) -> dict:
    """Everything that does not depend on the game: the logistic step and margin calibration from history, this season's
    games and preseason ratings."""
    # the historical tables, not the live ones, feed the priors
    os.environ.pop("PM_EPA_CONFIG", None)
    out_set, named, backups = overrides()
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    kw = {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}
    hist = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    # the adopted extensions: elite-prospect priors (docs/prospects/elite.md), the early-sample adjustment for
    # thin-history players (docs/prospects/hybrid.md) and individual results for established skaters (docs/inseason
    # update P), each when its test's rule adopted it
    yp = rate_features(cfg, cfg.path(cfg.data.raw_dir), season)
    hj, ij = met / "hybrid_prospects.json", met / "inseason_individual.json"
    hdata = (
        pl.read_parquet(proc / "hybrid_training.parquet")
        if hj.exists() and json.loads(hj.read_text()).get("implement") and (proc / "hybrid_training.parquet").exists()
        else None
    )
    idata = (
        pl.read_parquet(proc / "inseason_training.parquet")
        if ij.exists() and json.loads(ij.read_text()).get("implement") and (proc / "inseason_training.parquet").exists()
        else None
    )
    base = gd_.build_priors(proc, cfg, seasons, hist, yp)
    th = hist.table.filter(pl.col("season") >= grun.TUNE[0])
    # the logistic step and the margin calibration are fitted on the model as it runs live: the same priors and
    # adjustments, each season's adjustment models fitted on earlier seasons (docs/market/methodology.md §8 item 16)
    mu_h = historical_mu(proc, hist, base, seasons, kw, hdata, idata)
    beta = gm.logit_fit(
        np.column_stack([mu_h, th["b2b_home"].to_numpy(), th["b2b_away"].to_numpy()]), th["home_win"].to_numpy()
    )
    # actual margin on the model's expected margin
    cal = np.linalg.lstsq(np.column_stack([np.ones(len(mu_h)), mu_h]), th["gd"].to_numpy(), rcond=None)[0]
    live = live_games(season)
    pri, snap, stamp = season_priors(proc, cfg, season, live, base, yp)
    # this season's adjustments, fitted on every completed season
    early = None
    if hdata is not None:
        early = hy.Early({i for i, v in pri.m_eff[season].items() if v < hy.THIN}, hy.fit(hdata), pri.pos)
    individual = None
    if idata is not None:
        individual = (
            ind.fit(idata),
            ind.established(pri.m_eff[season], pri.age.get(season, {}), pri.pos),
            ind.live_lines(LIVE),
            pri.toi_last[season],
        )
    g_last = pl.read_parquet(proc / "games.parquet").filter(pl.col("season") == season - 1)
    ids = dict(
        pl.concat(
            [
                g_last.select(pl.col("home_abbrev").alias("t"), pl.col("home_id").alias("i")),
                live.table.select(pl.col("home_abbrev").alias("t"), pl.col("home_id").alias("i")),
                live.table.select(pl.col("away_abbrev").alias("t"), pl.col("away_id").alias("i")),
            ]
        )
        .unique()
        .iter_rows()
    )
    per, _ = sd.goalie_usage(proc)
    avg = float(g_last.select((pl.col("home_final") + pl.col("away_final")).mean()).item() / 2)
    return {
        "season": season,
        "kw": kw,
        "hp": hp,
        "beta": beta,
        "cal": cal,
        "live": live,
        "pri": pri,
        "snap": snap,
        "ids": ids,
        "per": per,
        "avg": avg,
        "out_set": out_set,
        "named": named,
        "backups": backups,
        "snapshot_stamp": stamp,
        "early": early,
        "individual": individual,
        "names": dict(snap.select("player_id", "name").iter_rows()),
    }


def _team_inputs(
    ctx: dict,
    ab: str,
    tid: int,
    day,
    goalies: dict,
    news: dict | None = None,
    force_b2b: bool = False,
    force_goalie: int | None = None,
) -> tuple[tuple, int, dict]:
    """Lineup (previous game's, or the published roster's top 12 forwards and 6 defensemen by last season's ice time if
    the team has not played), starting goalie, and notes."""
    live, snap, season, out_set = ctx["live"], ctx["snap"], ctx["season"], ctx["out_set"]
    tg = live.table.filter((pl.col("home_id") == tid) | (pl.col("away_id") == tid)).sort("date", "game_id")
    prev_gk = None
    if tg.height:
        gid = tg["game_id"][-1]
        lu_ids, lu_min = live.lineups[(gid, tid)]
        prev_gk = live.goalies[(gid, tid)][0][0]
        b2b = float((day - tg["date"][-1]).days == 1)
    else:
        tl = ctx["pri"].toi_last[season]
        r = (
            snap.filter((pl.col("team") == ab) & (pl.col("pos_group") != "G"))
            .with_columns(pl.col("player_id").replace_strict(tl, default=0.0, return_dtype=pl.Float64).alias("m"))
            .sort("m", descending=True)
        )
        r = pl.concat([r.filter(pl.col("pos_group") == "F").head(12), r.filter(pl.col("pos_group") == "D").head(6)])
        lu_ids = r["player_id"].to_numpy().astype(np.int64)
        lu_min = np.where(
            r["m"].to_numpy() > 0, r["m"].to_numpy(), np.where(r["pos_group"].to_numpy() == "D", 19.0, 14.0)
        )
        b2b = 0.0
    # the team plays the day before, in a game not yet completed
    if force_b2b:
        b2b = 1.0
    # unavailable players: the manual overrides plus the day's injury list; their places are filled from the published
    # roster by last season's ice time
    news = news or {}
    out_all = set(out_set) | set(news.get("out_ids", ()))
    on_team = set(snap.filter(pl.col("team") == ab)["player_id"].to_list())
    # traded, waived or assigned since his last game: no longer on the published roster
    if on_team:
        out_all |= {int(x) for x in lu_ids.tolist() if int(x) not in on_team}
    keep = ~np.isin(lu_ids, list(out_all))
    removed = [int(x) for x in lu_ids[~keep]]
    lu_ids, lu_min = lu_ids[keep], lu_min[keep]
    pos, tl = ctx["pri"].pos, ctx["pri"].toi_last[season]
    bench = (
        snap.filter(
            (pl.col("team") == ab)
            & (pl.col("pos_group") != "G")
            & ~pl.col("player_id").is_in(list(out_all | set(lu_ids.tolist())))
        )
        .with_columns(pl.col("player_id").replace_strict(tl, default=0.0, return_dtype=pl.Float64).alias("m"))
        .sort("m", descending=True)
    )
    added = []
    for grp, want, dflt in (("F", 12, 14.0), ("D", 6, 19.0)):
        have = sum(1 for x in lu_ids.tolist() if pos.get(x, "F") == grp)
        for r in bench.filter(pl.col("pos_group") == grp).head(max(want - have, 0)).iter_rows(named=True):
            lu_ids, lu_min = np.append(lu_ids, r["player_id"]), np.append(lu_min, r["m"] if r["m"] > 0 else dflt)
            added.append(int(r["player_id"]))
    cand = snap.filter(
        (pl.col("team") == ab) & (pl.col("pos_group") == "G") & ~pl.col("player_id").is_in(list(out_all))
    ).select(pl.col("player_id").alias("goalie"))
    per = ctx["per"]
    w = (
        cand.join(
            per.filter(pl.col("season") == season - 1).select("goalie", pl.col("shots").alias("s1")),
            on="goalie",
            how="left",
        )
        .join(
            per.filter(pl.col("season") == season - 2).select("goalie", pl.col("shots").alias("s2")),
            on="goalie",
            how="left",
        )
        .with_columns((pl.col("s1").fill_null(0) + 0.5 * pl.col("s2").fill_null(0)).alias("w"))
        .sort(["w", "goalie"], descending=[True, False])
    )
    listed = news.get("goalie_id")
    order = [int(x) for x in w["goalie"].to_list()]
    # second night of a back-to-back: the backup starts unless a starter is confirmed (docs/market/methodology.md §8).
    # "Yesterday's starter" is the goalie who played,
    # or, when yesterday's game is not completed yet, the team's usual starter.
    usual = goalies.get(ab) or (order[0] if order else prev_gk)
    yesterday = usual if force_b2b else prev_gk
    backup = ctx.get("backups", {}).get(ab)
    if backup is None or backup == yesterday or backup in out_all:
        backup = next((x for x in order if x != yesterday), None)
    # a confirmed starter is a fact: it outranks every rule
    if listed and listed not in out_all and news.get("goalie_status") == "confirmed":
        gk, src = listed, "lineup feed (confirmed)"
    elif b2b and backup is not None:
        gk, src = backup, "back-to-back rule (rested goalie)"
    elif ab in goalies:  # a manual override outranks an expected starter
        gk, src = goalies[ab], "override"
    elif listed and listed not in out_all:
        gk, src = listed, f"lineup feed ({news.get('goalie_status')})"
    else:
        gk, src = (int(w["goalie"][0]) if w.height else prev_gk), "most prior work"
    confirmed = src == "lineup feed (confirmed)"
    # the team's other goalie: who starts if the unconfirmed one does not
    alt = next((x for x in order if x != gk), None)
    # a what-if forecast (the goalie flag in the daily email)
    if force_goalie is not None:
        gk, src = force_goalie, "scenario"
    nm = ctx["names"]
    info = {
        "goalie": nm.get(gk, str(gk)),
        "goalie_id": gk,
        "goalie_source": src,
        "goalie_confirmed": confirmed,
        "goalie_alt_id": alt,
        "goalie_alt": nm.get(alt, str(alt)) if alt is not None else None,
        "previous_game_goalie": nm.get(prev_gk, str(prev_gk)) if prev_gk else None,
        "games_played": tg.height,
        "lineup_source": ("previous game" if tg.height else "published roster") + (" + injury feed" if news else ""),
        "b2b": b2b,
        "removed": [nm.get(x, str(x)) for x in removed],
        "added": [nm.get(x, str(x)) for x in added],
        "listed_out": len(news.get("out_names", [])),
        "unmatched_out": news.get("unmatched", []),
        "questionable": news.get("questionable", []),
    }
    return (lu_ids.astype(np.int64), lu_min), gk, info


def forecast(
    ctx: dict,
    away: str,
    home: str,
    date: str | None = None,
    goalies: dict | None = None,
    news: dict | None = None,
    force_b2b: set | None = None,
    force_goalie: dict | None = None,
) -> dict:
    live, season = ctx["live"], ctx["season"]
    goalies = {**ctx["named"], **(goalies or {})}
    if home not in ctx["ids"] or away not in ctx["ids"]:
        raise ValueError(f"unknown team {away} or {home}")
    hid, aid = ctx["ids"][home], ctx["ids"][away]
    day = pl.Series([date]).str.to_date("%Y-%m-%d")[0] if date else live.table["date"].max()
    fake, info = -1, {}
    for tid, ab in ((hid, home), (aid, away)):
        live.lineups[(fake, tid)], gk, info[ab] = _team_inputs(
            ctx, ab, tid, day, goalies, (news or {}).get(ab), ab in (force_b2b or ()), (force_goalie or {}).get(ab)
        )
        live.goalies[(fake, tid)] = [(gk, 0.0, 0.0)]
    row = pl.DataFrame(
        {
            "game_id": [fake],
            "season": [season],
            "date": [day],
            "home_id": [hid],
            "away_id": [aid],
            "home_abbrev": [home],
            "away_abbrev": [away],
            "gd": [0.0],
            "xgd": [0.0],
            "home_win": [0.0],
            "b2b_home": [info[home]["b2b"]],
            "b2b_away": [info[away]["b2b"]],
        },
        schema=live.table.schema,
    )
    played = live.table
    live.table = pl.concat([played, row])
    try:
        # the adjustment keeps per-season state, so each forecast replays the season with a fresh one
        u_p = ind.Individual(*ctx["individual"]) if ctx.get("individual") else None
        mu = float(
            gm.run_player_season(live, ctx["pri"], season, early=ctx.get("early"), inseason=u_p, **ctx["kw"])[-1]
        )
    finally:
        live.table = played  # the appended game never stays in the season table
        for tid in (hid, aid):
            live.lineups.pop((fake, tid), None)
            live.goalies.pop((fake, tid), None)
    p_home = float(gm.logit_pred(ctx["beta"], np.array([[mu, info[home]["b2b"], info[away]["b2b"]]]))[0])
    margin, avg = float(ctx["cal"][0] + ctx["cal"][1] * mu), ctx["avg"]
    return {
        "away": away,
        "home": home,
        "date": str(day),
        "p_home_win": p_home,
        "p_away_win": 1 - p_home,
        "expected_margin_home": margin,
        "model_mu": mu,
        "avg_goals_per_team": avg,
        "expected_goals_home": avg + margin / 2,
        "expected_goals_away": avg - margin / 2,
        "teams": info,
        "hyperparameters": ctx["hp"],
        "completed_games_this_season": played.height,
    }


def goalie_range(ctx: dict, f: dict, date: str | None, news: dict | None, force_b2b: set | None) -> tuple[float, float]:
    """Lowest and highest home win probability over who might start: every combination of the listed goalie and the
    team's other goalie for each side whose starter is not confirmed."""
    alts = {
        ab: t["goalie_alt_id"]
        for ab, t in f["teams"].items()
        if not t["goalie_confirmed"] and t["goalie_alt_id"] is not None
    }
    ps = [f["p_home_win"]]
    for combo in (c for n in range(1, len(alts) + 1) for c in itertools.combinations(alts, n)):
        ps.append(
            forecast(
                ctx,
                f["away"],
                f["home"],
                date,
                news=news,
                force_b2b=force_b2b,
                force_goalie={ab: alts[ab] for ab in combo},
            )["p_home_win"]
        )
    return min(ps), max(ps)


def run(away: str, home: str, date: str | None = None, season: int = 2026, goalies: dict | None = None) -> dict:
    out = forecast(prepare(season), away, home, date, goalies)
    log.info(
        "%s at %s: P(%s win) %.3f; expected score %s %.2f, %s %.2f",
        away,
        home,
        home,
        out["p_home_win"],
        home,
        out["expected_goals_home"],
        away,
        out["expected_goals_away"],
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    r = run(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else None,
        int(sys.argv[4]) if len(sys.argv) > 4 else 2026,
    )
    print(json.dumps(r, indent=2, default=float))
