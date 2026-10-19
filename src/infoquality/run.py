"""What day-of information is worth to the game model, and how good the live run's is:  python -m src.infoquality.run   (docs/infoquality/methodology.md)."""

from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np
import polars as pl

from src.config import ROOT, load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun

log = logging.getLogger("infoquality")


def goalie_guesses(games: gd_.Games) -> tuple[dict, dict, pl.DataFrame]:
    """Per (game, team): the rule's starter and the previous game's starter, from games before this one; and a row per team-game saying whether each guess was right."""
    rule, prev, rows = {}, {}, []
    starts: dict = {}                 # (season, team) -> Counter of starts by goalie
    last: dict = {}                   # (season, team) -> (date, starter)
    for gid, season, d, hid, aid in games.table.select("game_id", "season", "date", "home_id", "away_id").iter_rows():
        for tid in (hid, aid):
            actual = games.goalies[(gid, tid)][0][0]
            c, lg = starts.setdefault((season, tid), Counter()), last.get((season, tid))
            order = [g for g, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
            b2b = lg is not None and (d - lg[0]).days == 1
            if not order:
                r = actual                                             # a team's first game of a season: no history to guess from
            elif b2b:
                r = next((g for g in order if g != lg[1]), lg[1])
            else:
                r = order[0]
            rule[(gid, tid)], prev[(gid, tid)] = r, (lg[1] if lg else actual)
            rows.append({"game_id": gid, "season": season, "team_id": tid, "first_game": lg is None, "b2b": b2b, "rule_right": r == actual, "prev_right": (lg[1] if lg else actual) == actual})
            c[actual] += 1
            last[(season, tid)] = (d, actual)
    return rule, prev, pl.DataFrame(rows)


def lineup_overlap(games: gd_.Games) -> pl.DataFrame:
    """Per team-game after a team's first of a season: the share of tonight's skaters, and of tonight's minutes, who dressed in the team's previous game."""
    last, rows = {}, []
    for gid, season, hid, aid in games.table.select("game_id", "season", "home_id", "away_id").iter_rows():
        for tid in (hid, aid):
            ids, mins = games.lineups[(gid, tid)]
            p = last.get((season, tid))
            if p is not None:
                inn = np.isin(ids, p)
                rows.append({"game_id": gid, "season": season, "share_players": float(inn.mean()), "share_minutes": float(mins[inn].sum() / mins.sum()), "new_players": int((~inn).sum())})
            last[(season, tid)] = ids
    return pl.DataFrame(rows)


def backtest(proc, cfg, met) -> dict:
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games)
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    te = t["season"].is_in(grun.TEST).to_numpy()
    y, dates = t["home_win"].to_numpy()[te], t["date"].to_numpy()[te]
    rule, prev, gq = goalie_guesses(games)
    actual = {k: v[0][0] for k, v in games.goalies.items()}
    kw = {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}
    variants = {"P": {}, "A1_prev_skaters": {"lineup": "previous", "starter": actual}, "A2_rule_goalie": {"starter": rule}, "A3_prev_goalie": {"starter": prev},
                "A4_prev_skaters_rule_goalie": {"lineup": "previous", "starter": rule}, "P_prev": {"lineup": "previous"}}
    prob = {k: grun.predict(t, np.concatenate([gm.run_player_season(games, pri, T, **kw, **v) for T in seasons]))[te] for k, v in variants.items()}
    ll = {k: gm.logloss(p, y) for k, p in prob.items()}
    out: dict = {"n_games": int(te.sum()), "variants": {}}
    for k in variants:
        d = np.abs(prob[k] - prob["P"])
        out["variants"][k] = {"logloss": float(ll[k].mean()), "mean_abs_change": float(d.mean()), "share_moved_2pts": float((d > 0.02).mean()), "share_moved_5pts": float((d > 0.05).mean()), "max_change": float(d.max()),
                              **({"P_vs_this": grun.paired(ll["P"], ll[k], dates)} if k != "P" else {})}
    g = gq.filter(pl.col("season").is_in(grun.TEST) & ~pl.col("first_game"))
    acc = lambda d: {"n": d.height, "rule_right": float(d["rule_right"].mean()), "previous_starter_right": float(d["prev_right"].mean())}        # noqa: E731
    out["goalie_guess"] = {"all": acc(g), "back_to_back": acc(g.filter(pl.col("b2b"))), "other": acc(g.filter(~pl.col("b2b")))}
    lo = lineup_overlap(games).filter(pl.col("season").is_in(grun.TEST))
    out["previous_lineup"] = {"n": lo.height, "mean_share_players": float(lo["share_players"].mean()), "mean_share_minutes": float(lo["share_minutes"].mean()), "share_identical": float((lo["new_players"] == 0).mean()),
                              "share_one_change": float((lo["new_players"] == 1).mean()), "share_three_or_more": float((lo["new_players"] >= 3).mean())}
    # where the goalie guess is wrong, how far does the forecast move?
    wrong = set(gq.filter(~pl.col("rule_right"))["game_id"].to_list())
    w = t["game_id"].is_in(list(wrong)).to_numpy()[te]
    d = np.abs(prob["A2_rule_goalie"] - prob["P"])
    out["rule_goalie_wrong"] = {"share_of_games": float(w.mean()), "mean_abs_change_when_wrong": float(d[w].mean()), "p90_change_when_wrong": float(np.quantile(d[w], 0.9))}
    return out


GSCALES, N0S = [0.0, 0.25, 0.5, 0.75, 1.0], [3000.0, 6000.0, 12000.0]


def goalie_scale_test(proc, cfg, met) -> dict:
    """Follow-up (docs/infoquality/methodology.md §6): are the model's goalie differences too wide? Tuned on 2019-2020, tested on 2021-2025."""
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games)
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    tt = t.filter(pl.col("season").is_in(grun.TUNE))
    te = t["season"].is_in(grun.TEST).to_numpy()
    y, dates, seas = t["home_win"].to_numpy()[te], t["date"].to_numpy()[te], t["season"].to_numpy()[te]
    mu = lambda ss, gs, n0: np.concatenate([gm.run_player_season(games, pri, T, lam=hp["lam"], alpha=hp["alpha"], n0=n0, gscale=gs) for T in ss])      # noqa: E731
    grid = {(gs, n0): grun.tune_score(tt, mu(grun.TUNE, gs, n0)) for gs in GSCALES for n0 in N0S}
    gs, n0 = min(grid, key=grid.get)
    p0, p1 = grun.predict(t, mu(seasons, 1.0, hp["n0"]))[te], grun.predict(t, mu(seasons, gs, n0))[te]
    l0, l1 = gm.logloss(p0, y), gm.logloss(p1, y)
    z = grun.paired(l1, l0, dates)
    null = gs == 1.0 and n0 == hp["n0"]
    test_grid = {f"{a}_{int(b)}": float(gm.logloss(grun.predict(t, mu(seasons, a, b))[te], y).mean()) for a in GSCALES for b in (hp["n0"],)}      # reported: test log loss along gscale at the published n0
    return {"tuned": {"gscale": gs, "n0": n0}, "null_selected": null, "tune_grid": {f"{a}_{int(b)}": v for (a, b), v in grid.items()}, "tune_logloss_P": grid[(1.0, hp["n0"])], "tune_logloss": grid[(gs, n0)],
            "logloss_P": float(l0.mean()), "logloss": float(l1.mean()), "paired": z, "by_season": {str(T): float((l1 - l0)[seas == T].mean()) for T in grun.TEST}, "test_logloss_by_gscale_at_published_n0": test_grid,
            "mean_abs_change": float(np.abs(p1 - p0).mean()), "share_moved_2pts": float((np.abs(p1 - p0) > 0.02).mean()),
            "verdict": "not adopted (tuning selects the published values)" if null else "adopt" if z["z"] > 2 else "follow up" if z["z"] > 0 else "not adopted"}


def live_audit() -> dict:
    """Was the logged starter the goalie who started? First forecast on the game's date and last forecast before puck drop, by source."""
    from src.live.paths import LOGS
    logs, proc = LOGS / "forecast_log.csv", ROOT / "data/live/processed"
    if not logs.exists():
        return {}
    f = pl.read_csv(logs, infer_schema_length=None).with_columns(pl.col("made_utc").str.to_datetime(time_zone="UTC").alias("made"), pl.col("start_utc").str.to_datetime(time_zone="UTC").alias("start"))
    f = f.with_columns(pl.col("made").dt.convert_time_zone("America/New_York").alias("made_et")).filter(pl.col("made") < pl.col("start"))
    sh = pl.read_parquet(proc / "shots_xg.parquet")
    ev = pl.read_parquet(proc / "events.parquet")
    g = pl.read_parquet(proc / "games.parquet").select("game_id", "home_id", "away_id")
    names = {r["player_id"]: f"{r['first_name']} {r['last_name']}" for r in pl.read_parquet(proc / "players.parquet").iter_rows(named=True)}
    gcol = "player2" if "player2" in sh.columns else None
    src = sh if gcol else ev
    gcol = gcol or next(c for c in ("goalie_id", "player2") if c in ev.columns)
    s = src.filter(pl.col(gcol).is_not_null() & pl.col("type").is_in(["shot-on-goal", "goal"])).join(g, on="game_id").with_columns(pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.lit("away")).otherwise(pl.lit("home")).alias("side"))
    st = s.group_by("game_id", "side", gcol).len().sort(["len", gcol], descending=[True, False]).unique(["game_id", "side"], keep="first", maintain_order=True)
    actual = {(r["game_id"], r["side"]): names.get(r[gcol], str(r[gcol])) for r in st.iter_rows(named=True)}
    same_day = f.filter(pl.col("made_et").dt.date().cast(pl.String) == pl.col("game_date"))
    rows = []
    for tag, d in (("first_on_game_day", same_day.sort("made").unique("game_id", keep="first")), ("last_before_puck_drop", f.sort("made").unique("game_id", keep="last"))):
        for r in d.iter_rows(named=True):
            for side in ("home", "away"):
                a = actual.get((r["game_id"], side))
                if a is not None:
                    rows.append({"which": tag, "game_id": r["game_id"], "side": side, "source": (r[f"{side}_goalie_source"] or "").split(" (rested")[0], "hour_et": r["made_et"].hour, "right": r[f"{side}_goalie"] == a})
    if not rows:
        return {}
    a = pl.DataFrame(rows)
    by = lambda d, c: {str(k): {"n": n, "right": float(x)} for k, n, x in d.group_by(c).agg(pl.len(), pl.col("right").mean()).sort(c).iter_rows()}      # noqa: E731
    out = {}
    for tag in ("first_on_game_day", "last_before_puck_drop"):
        d = a.filter(pl.col("which") == tag)
        out[tag] = {"team_games": d.height, "right": float(d["right"].mean()) if d.height else None, "by_source": by(d, "source"), "by_hour_et": by(d, "hour_et")}
    return out


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    out = {"backtest": backtest(proc, cfg, met), "live": live_audit(), "goalie_scale": goalie_scale_test(proc, cfg, met)}
    log.info("goalie scale: tuned %s, log loss %.5f against %.5f, z %+.2f, %s", out["goalie_scale"]["tuned"], out["goalie_scale"]["logloss"], out["goalie_scale"]["logloss_P"], out["goalie_scale"]["paired"]["z"], out["goalie_scale"]["verdict"])
    (met / "infoquality.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    for k, v in out["backtest"]["variants"].items():
        log.info("%-28s log loss %.5f  mean |dp| %.4f  moved >2pts %.3f  %s", k, v["logloss"], v["mean_abs_change"], v["share_moved_2pts"], f"P better by z {v['P_vs_this']['z']:+.2f}" if "P_vs_this" in v else "")
    log.info("goalie guess %s", out["backtest"]["goalie_guess"])
    log.info("previous lineup %s", out["backtest"]["previous_lineup"])
    log.info("live %s", out["live"])
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
