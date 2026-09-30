"""First-season-ahead forecast from a published roster snapshot:  python -m src.standings.next 2026   (docs/standings/methodology.md §10)."""

from __future__ import annotations

import gzip
import json
import logging
import sys

import numpy as np
import polars as pl

from src.config import load_config
from src.goaltending import model as gm
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.projection.model import Spec
from src.standings import data as sd
from src.standings import forecast as fc
from src.standings import run as sr

log = logging.getLogger("standings")


def load_snapshot(raw, season: int, stamp: str | None = None) -> tuple[pl.DataFrame, str]:
    base = raw / "rosters_next" / str(season)
    stamp = stamp or sorted(p.name for p in base.iterdir() if p.is_dir())[-1]
    rows = []
    for f in sorted((base / stamp).glob("*.json.gz")):
        team = f.name.split(".")[0]
        d = json.load(gzip.open(f))
        for grp, pg in (("forwards", "F"), ("defensemen", "D"), ("goalies", "G")):
            for p in d.get(grp, []):
                rows.append({"team": team, "player_id": int(p["id"]), "name": f"{p['firstName']['default']} {p['lastName']['default']}", "pos_group": pg, "birth_date": p.get("birthDate"),
                             "height_in": p.get("heightInInches"), "weight_lb": p.get("weightInPounds")})
    snap = pl.DataFrame(rows).with_columns(pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False), pl.col("height_in").cast(pl.Float64), pl.col("weight_lb").cast(pl.Float64))
    return snap, stamp


def next_goalie_rates(proc, cfg, season: int) -> pl.DataFrame:
    """Rolling persistent save rates for ``season``: the goaltending model fitted on shots through season - 1 (same settings as src/goaltending forward step)."""
    met = cfg.path(cfg.data.metrics_dir)
    tune = json.loads((met / "gk_tune.json").read_text(encoding="utf-8"))
    team = tune["variant"] == "G2"
    s2 = tune["s2"][tune["variant"]]
    df = pl.read_parquet(proc / "gk_shots.parquet")
    T = season - 1
    tr = df.filter(pl.col("season") <= T)
    lay = gm.build_layout(tr, team, fringe=True)
    mode, _, _ = gm.fit(tr, lay, s2=s2)
    ptr = gm.shot_probs(tr.filter(pl.col("season") == T), lay, mode.beta)
    cur = tr.filter(pl.col("season") == T).with_columns(pl.Series("saved_v", ptr["base"] - ptr["v"]))
    g = cur.group_by("goalie").agg(pl.col("saved_v").sum().alias("saved"), pl.len().alias("n_prev"))
    rep = float((ptr["base"] - ptr["rep"]).mean())
    return g.select("goalie", pl.lit(season, pl.Int64).alias("season"), (pl.col("saved") / pl.col("n_prev")).alias("rate_v"), "n_prev").with_columns(pl.lit(rep).alias("rep_v"))


def forecast_next(season: int = 2026, stamp: str | None = None) -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    raw = cfg.path(cfg.data.raw_dir)
    outdir = cfg.path(cfg.data.output_dir)
    snap, stamp = load_snapshot(raw, season, stamp)
    last = season - 1
    res = sd.results(proc)
    ids = res.filter(pl.col("season") == last).select("team", "team_id")
    snap = snap.join(ids, on="team", how="left")
    assert snap["team_id"].null_count() == 0, "unmapped team"
    roll = pl.read_parquet(proc / "wm_rolling_values.parquet")
    p = pd_.panel(proc, roll)
    bio0 = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    new_bio = snap.select("player_id", "birth_date", "height_in", "weight_lb").unique("player_id")
    bio = pl.concat([new_bio, bio0.filter(~pl.col("player_id").is_in(new_bio["player_id"]))])       # the snapshot's bio wins where both exist
    M = Spec()
    fr_val = pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR)
    fr_min = pd_.design_frame(p, bio, 1)
    ros = snap.filter(pl.col("pos_group") != "G").select("team_id", "player_id", "pos_group")
    d = fc.roster_frame(p, bio, ros, season)
    models = pm.fit_models(pr.train_rows(fr_val, 1, season), M)
    mm = pr._minutes_fit(pr.train_rows(fr_min, 1, season))
    alloc = fc.allocate_minutes(d, pr._minutes_pred(mm, d), fc.team_minutes(p, res, last))
    rate = pm.predict_models(models, alloc, M)
    S = fc._skater_sum(alloc, rate, "S")
    per, lg = sd.goalie_usage(proc)
    rates = next_goalie_rates(proc, cfg, season)
    gk = snap.filter(pl.col("pos_group") == "G").select("team_id", pl.col("player_id").alias("goalie"), pl.lit(season, pl.Int64).alias("season"))
    G = fc.goalie_value(gk, season, rates, per, lg, max_goalies=2)
    # player-level table for auditing
    players = alloc.with_columns(pl.Series("proj_rate60", rate)).with_columns((pl.col("proj_rate60") * pl.col("mins") / 60).alias("proj_gar")).join(snap.select("player_id", "name", "team"), on="player_id", how="left")
    players.select("team", "name", "pos_group", "age", "no_history", "mins", "proj_rate60", "proj_gar").sort("team", "proj_gar", descending=[False, True]).write_csv(outdir / f"standings_{season}_players.csv")

    # points: M1 as pre-registered; references M3 (blend) and B2 (prior goal differential) with fits on all earlier seasons
    tab = pl.read_parquet(proc / "standings_table.parquet")
    act = tab.filter(pl.col("gp") > 0)
    t = ids.join(S, on="team_id", how="left").join(G, on="team_id", how="left")
    t = t.with_columns((pl.col("S") + pl.col("G")).alias("SG")).with_columns((pl.col("SG") - pl.col("SG").mean()).alias("SG_dm"))
    a, b = sr.ols(act["gd82"].to_numpy()[:, None], act["pts82"].to_numpy())
    t = t.with_columns((a + b * pl.col("SG_dm")).alias("pts_M1"), (pl.col("SG_dm")).alias("gd_hat"))
    resid = act.filter(pl.col("SG_dm").is_not_null())
    sd_m1 = float((resid["pts82"] - (a + b * resid["SG_dm"])).std())
    prev = res.filter(pl.col("season") == last).select("team_id", pl.col("pts82").alias("pts82_last"), pl.col("gd82").alias("gd82_last"))
    t = t.join(prev, on="team_id")
    trf = act.filter(pl.col("SG_dm").is_not_null() & pl.col("gd82_prev").is_not_null())
    X3 = lambda d_: np.column_stack([d_["SG_dm"].to_numpy(), d_["gd82_prev"].to_numpy()])   # noqa: E731
    b3 = sr.ols(X3(trf), trf["pts82"].to_numpy())
    t = t.with_columns(pl.Series("pts_M3", sr._pred(b3, np.column_stack([t["SG_dm"].to_numpy(), t["gd82_last"].to_numpy()]))))
    tp = act.filter(pl.col("gd82_prev").is_not_null())
    b2 = sr.ols(tp["gd82_prev"].to_numpy()[:, None], tp["pts82"].to_numpy())
    t = t.with_columns(pl.Series("pts_B2", sr._pred(b2, t["gd82_last"].to_numpy()[:, None])))
    t = t.with_columns(pl.lit(sd_m1).alias("sd80_basis"), (pl.col("pts_M1") - 1.2815515655446004 * sd_m1).alias("pts_M1_lo80"), (pl.col("pts_M1") + 1.2815515655446004 * sd_m1).alias("pts_M1_hi80"))
    t = t.with_columns(pl.col("pts_M1").rank("ordinal", descending=True).alias("rank_M1")).sort("rank_M1")
    t.select("rank_M1", "team", "pts_M1", "pts_M1_lo80", "pts_M1_hi80", "pts_M3", "pts_B2", "pts82_last", "S", "G", "gd_hat").write_csv(outdir / f"standings_{season}_forecast.csv")
    meta = {"season": season, "roster_snapshot": stamp, "goalies_per_team": 2, "sd_points_M1": sd_m1, "map_a": float(a), "map_b": float(b)}
    (cfg.path(cfg.data.metrics_dir) / f"standings_{season}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [f"# {season}-{str(season + 1)[2:]} standings projection (first pass)", "",
             f"Roster snapshot {stamp} (published rosters before opening night, {snap.height} players; top 13 forwards and 7 defensemen by projected minutes, two goalies per team). "
             "Points per 82 games. **Status:** in the backtest this model matched but did not beat last season's goal differential (docs/standings, ST-3 failed, calibration slope 1.22), "
             f"so treat M1 as a roster-based view with about ±{1.2816 * sd_m1:.0f} points of 80% uncertainty per team, not a validated improvement on last season's results. "
             "M1 is the pre-registered primary; M3 (blend with last season's goal differential) and B2 (last season's goal differential alone) are shown for reference.", "",
             "| rank | team | M1 | 80% range | M3 blend | B2 history | last season |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['rank_M1']} | {r['team']} | {r['pts_M1']:.0f} | {r['pts_M1_lo80']:.0f}–{r['pts_M1_hi80']:.0f} | {r['pts_M3']:.0f} | {r['pts_B2']:.0f} | {r['pts82_last']:.0f} |" for r in t.to_dicts()]
    (cfg.path("reports/standings") / f"forecast_{season}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("forecast written for %d from snapshot %s", season, stamp)
    return t


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    forecast_next(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)


if __name__ == "__main__":
    main()
