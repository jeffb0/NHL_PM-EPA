"""First-season-ahead forecast from a published roster snapshot:  python -m src.standings.next 2026
(docs/standings/methodology.md §10)."""

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
from src.prospects import features as pf
from src.standings import candidates as cd
from src.standings import data as sd
from src.standings import forecast as fc
from src.standings import run as sr

log = logging.getLogger("standings")
PRESEL = {"F": 14, "D": 8}


def load_snapshot(raw, season: int, stamp: str | None = None) -> tuple[pl.DataFrame, str]:
    base = raw / "rosters_next" / str(season)
    stamp = stamp or sorted(p.name for p in base.iterdir() if p.is_dir())[-1]
    rows = []
    for f in sorted((base / stamp).glob("*.json.gz")):
        team = f.name.split(".")[0]
        d = json.load(gzip.open(f))
        for grp, pg in (("forwards", "F"), ("defensemen", "D"), ("goalies", "G")):
            for p in d.get(grp, []):
                rows.append(
                    {
                        "team": team,
                        "player_id": int(p["id"]),
                        "name": f"{p['firstName']['default']} {p['lastName']['default']}",
                        "pos_group": pg,
                        "birth_date": p.get("birthDate"),
                        "height_in": p.get("heightInInches"),
                        "weight_lb": p.get("weightInPounds"),
                    }
                )
    snap = pl.DataFrame(rows).with_columns(
        pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("height_in").cast(pl.Float64),
        pl.col("weight_lb").cast(pl.Float64),
    )
    return snap, stamp


def next_goalie_rates(proc, cfg, season: int, returning: bool = False) -> pl.DataFrame:
    """Rolling persistent save rates for ``season``: the goaltending model fitted on shots through season - 1 (same
    settings as src/goaltending forward step).

    With ``returning`` (the live game forecasts, docs/gamemodel §10 item 2) goalies with a fitted career effect who
    faced no shots in season - 1 are included, at that effect, with n_prev = 0; without it they are absent and fall to
    the replacement rate."""
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
    out = g.select("goalie", (pl.col("saved") / pl.col("n_prev")).alias("rate_v"), "n_prev")
    if returning:
        back = gm.returning_rates(tr.filter(pl.col("season") == T), lay, mode.beta)
        out = pl.concat([out, back.with_columns(pl.lit(0, out.schema["n_prev"]).alias("n_prev"))])
    return out.select("goalie", pl.lit(season, pl.Int64).alias("season"), "rate_v", "n_prev").with_columns(
        pl.lit(rep).alias("rep_v")
    )


def forecast_next(season: int = 2026, stamp: str | None = None) -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    raw = cfg.path(cfg.data.raw_dir)
    outdir = cfg.path(cfg.data.output_dir)
    snap, stamp = load_snapshot(raw, season, stamp)
    # manual roster knowledge: players who will not play for their listed team
    ov = cfg.path("data/external/roster_overrides.json")
    if ov.exists():
        gone = [int(u["player_id"]) for u in json.loads(ov.read_text(encoding="utf-8")).get("unavailable", [])]
        snap = snap.filter(~pl.col("player_id").is_in(gone))
    last = season - 1
    res = sd.results(proc)
    ids = res.filter(pl.col("season") == last).select("team", "team_id")
    snap = snap.join(ids, on="team", how="left")
    assert snap["team_id"].null_count() == 0, "unmapped team"
    roll = pl.read_parquet(proc / "wm_rolling_values.parquet")
    p = pd_.panel(proc, roll)
    bio0 = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    new_bio = snap.select("player_id", "birth_date", "height_in", "weight_lb").unique("player_id")
    # the snapshot's bio wins where both exist
    bio = pl.concat([new_bio, bio0.filter(~pl.col("player_id").is_in(new_bio["player_id"]))])
    # pre-NHL priors (docs/prospects): used when that test's rule said to implement them; the M1 reference keeps the
    # original rate model
    pj_path = cfg.path(cfg.data.metrics_dir) / "prospects.json"
    pj = json.loads(pj_path.read_text()) if pj_path.exists() else {}
    yp = pl.read_parquet(proc / "prospect_features.parquet") if pj.get("implement") else None
    ej_path = cfg.path(cfg.data.metrics_dir) / "elite_prospects.json"
    # docs/prospects/elite.md supersedes the first pre-NHL columns
    if ej_path.exists() and json.loads(ej_path.read_text()).get("implement_A"):
        yp = pf.elite_table(raw, list(range(pr.FIRST_TARGET, season + 1)))
    M = pf.spec(yp)
    fr_plain = pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR)
    fr_val = pf.with_yp(fr_plain, yp)
    fr_min = pd_.design_frame(p, bio, 1)
    ros = snap.filter(pl.col("pos_group") != "G").select("team_id", "player_id", "pos_group")
    d = pf.with_yp(fc.roster_frame(p, bio, ros, season), yp)
    models = pm.fit_models(pr.train_rows(fr_val, 1, season), M)
    mm = pr._minutes_fit(pr.train_rows(fr_min, 1, season))
    alloc = fc.allocate_minutes(d, pr._minutes_pred(mm, d), fc.team_minutes(p, res, last))
    models0 = pm.fit_models(pr.train_rows(fr_plain, 1, season), Spec())
    S = fc._skater_sum(alloc, pm.predict_models(models0, alloc, Spec()), "S")
    # implemented improvements (docs/standings/improvements.md): C3 deployment-aware minutes, applied to a dressed-size
    # pre-selection (14 F, 8 D) so the
    # roster-relative feature is computed over a group like the backtest's opening lineups rather than a whole camp
    # roster; C1 portable team residual
    pre = (
        d.with_columns(pl.Series("_m", pr._minutes_pred(mm, d)))
        .with_columns(pl.col("_m").rank("ordinal", descending=True).over("team_id", "pos_group").alias("_rk"))
        .filter(pl.col("_rk") <= pl.col("pos_group").replace_strict(PRESEL, return_dtype=pl.Int64))
        .drop("_m", "_rk")
    )
    mm3 = cd.minutes_fit_c3(pr.train_rows(fr_min, 1, season))
    alloc3 = fc.allocate_minutes(pre, cd.minutes_pred_c3(mm3, pre), fc.team_minutes(p, res, last))
    rate3 = pm.predict_models(models, alloc3, M)
    S3 = fc._skater_sum(alloc3, rate3, "S_c3")
    tres = cd.team_residual(proc, p, res)
    Rc3 = cd.carried_residual(alloc3, p, tres, season).rename({"Rc": "Rc3"})
    per, lg = sd.goalie_usage(proc)
    rates = next_goalie_rates(proc, cfg, season)
    gk = snap.filter(pl.col("pos_group") == "G").select(
        "team_id", pl.col("player_id").alias("goalie"), pl.lit(season, pl.Int64).alias("season")
    )
    G = fc.goalie_value(gk, season, rates, per, lg, max_goalies=2)

    # points: the primary is M1 with C1 and C3 (improvements.md §4); M1, M3 (blend) and B2 (prior goal differential) are
    # shown for reference, all fitted on earlier seasons
    tab = pl.read_parquet(proc / "standings_table.parquet")
    act = tab.filter(pl.col("gp") > 0)
    a, b = sr.ols(act["gd82"].to_numpy()[:, None], act["pts82"].to_numpy())
    ct = pl.read_parquet(
        proc / ("standings_candidates_table_yp.parquet" if yp is not None else "standings_candidates_table.parquet")
    ).filter((pl.col("gp") > 0) & pl.col("SG3_dm").is_not_null())
    yc = (ct["pts82"].to_numpy() - (a + b * ct["SG3_dm"].to_numpy())) / b
    c = float((ct["Rc3_dm"].to_numpy() * yc).sum() / (ct["Rc3_dm"].to_numpy() ** 2).sum())
    cj = json.loads((cfg.path(cfg.data.metrics_dir) / "standings_candidates.json").read_text())
    # out-of-sample error of the implemented model, 2021-2025
    sd_new = float(pj["standings"]["rmse_C1_C3_MY"] if yp is not None else cj["rmse"]["C1_C3"])
    # player-level table for auditing: projected rate, minutes, value and the carried share of last season's team
    # residual
    prev_rho = (
        p.filter(pl.col("season") == last)
        .select("player_id", "team")
        .join(tres.filter(pl.col("season") == last).select("team", "rho"), on="team", how="left")
        .select("player_id", "rho")
    )
    players = (
        alloc3.with_columns(
            pl.Series("proj_rate60", rate3),
            pl.Series("proj_rate60_no_prenhl", pm.predict_models(models0, alloc3, Spec())),
        )
        .with_columns((pl.col("proj_rate60") * pl.col("mins") / 60).alias("proj_gar"))
        .join(prev_rho, on="player_id", how="left")
        .with_columns((pl.col("rho").fill_null(0.0) * pl.col("mins") * c).alias("carried_residual"))
        .join(snap.select("player_id", "name", "team"), on="player_id", how="left")
    )
    players.select(
        "team",
        "name",
        "pos_group",
        "age",
        "no_history",
        "mins",
        "proj_rate60",
        "proj_rate60_no_prenhl",
        "proj_gar",
        "carried_residual",
    ).sort("team", "proj_gar", descending=[False, True]).write_csv(outdir / f"standings_{season}_players.csv")
    t = (
        ids.join(S, on="team_id", how="left")
        .join(S3, on="team_id", how="left")
        .join(Rc3, on="team_id", how="left")
        .join(G, on="team_id", how="left")
    )
    t = t.with_columns(
        (pl.col("S") + pl.col("G")).alias("SG"), (pl.col("S_c3") + pl.col("G")).alias("SG3")
    ).with_columns(*[(pl.col(x) - pl.col(x).mean()).alias(f"{x}_dm") for x in ("SG", "SG3", "Rc3")])
    t = t.with_columns(
        (pl.col("SG3_dm") + c * pl.col("Rc3_dm")).alias("gd_hat"), (c * pl.col("Rc3_dm")).alias("carried_residual")
    )
    t = t.with_columns((a + b * pl.col("gd_hat")).alias("pts"), (a + b * pl.col("SG_dm")).alias("pts_M1"))
    prev = res.filter(pl.col("season") == last).select(
        "team_id", pl.col("pts82").alias("pts82_last"), pl.col("gd82").alias("gd82_last")
    )
    t = t.join(prev, on="team_id")
    trf = act.filter(pl.col("SG_dm").is_not_null() & pl.col("gd82_prev").is_not_null())
    X3 = lambda d_: np.column_stack([d_["SG_dm"].to_numpy(), d_["gd82_prev"].to_numpy()])
    b3 = sr.ols(X3(trf), trf["pts82"].to_numpy())
    t = t.with_columns(
        pl.Series("pts_M3", sr._pred(b3, np.column_stack([t["SG_dm"].to_numpy(), t["gd82_last"].to_numpy()])))
    )
    tp = act.filter(pl.col("gd82_prev").is_not_null())
    b2 = sr.ols(tp["gd82_prev"].to_numpy()[:, None], tp["pts82"].to_numpy())
    t = t.with_columns(pl.Series("pts_B2", sr._pred(b2, t["gd82_last"].to_numpy()[:, None])))
    z80 = 1.2815515655446004
    t = t.with_columns(
        pl.lit(sd_new).alias("sd80_basis"),
        (pl.col("pts") - z80 * sd_new).alias("pts_lo80"),
        (pl.col("pts") + z80 * sd_new).alias("pts_hi80"),
    )
    t = t.with_columns(pl.col("pts").rank("ordinal", descending=True).alias("rank")).sort("rank")
    t.select(
        "rank",
        "team",
        "pts",
        "pts_lo80",
        "pts_hi80",
        "pts_M1",
        "pts_M3",
        "pts_B2",
        "pts82_last",
        "S_c3",
        "G",
        "carried_residual",
        "gd_hat",
    ).write_csv(outdir / f"standings_{season}_forecast.csv")
    meta = {
        "season": season,
        "roster_snapshot": stamp,
        "goalies_per_team": 2,
        "model": "M1 + C1 (portable team residual) + C3 (deployment-aware minutes)"
        + (" + pre-NHL priors" if yp is not None else ""),
        "c_residual": c,
        "sd_points_out_of_sample": sd_new,
        "map_a": float(a),
        "map_b": float(b),
    }
    (cfg.path(cfg.data.metrics_dir) / f"standings_{season}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    lines = [
        f"# {season}-{str(season + 1)[2:]} standings projection",
        "",
        f"Roster snapshot {stamp} (published rosters before opening night, {snap.height} players; two goalies per team). Points per 82 games. "
        "Model: projected player rates × deployment-aware minutes, plus goalies, plus each player's carried share of last season's unexplained team result (docs/standings/improvements.md). "
        f"**Status:** in the backtest (2021–2025) this is 14.0 points RMSE, better than the original roster model (14.6) but still not better than last season's goal differential alone (13.7); "
        f"80% ranges are ±{z80 * sd_new:.0f} points. M1 (the original roster model), M3 (blend) and B2 (last season's goal differential) are shown for reference.",
        "",
        "| rank | team | points | 80% range | carried residual (goals) | M1 | M3 blend | B2 history | last season |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['rank']} | {r['team']} | {r['pts']:.0f} | {r['pts_lo80']:.0f}–{r['pts_hi80']:.0f} | {r['carried_residual']:+.1f} | {r['pts_M1']:.0f} | {r['pts_M3']:.0f} | {r['pts_B2']:.0f} | {r['pts82_last']:.0f} |"
        for r in t.to_dicts()
    ]
    (cfg.path("reports/standings") / f"forecast_{season}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("forecast written for %d from snapshot %s", season, stamp)
    return t


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    forecast_next(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)


if __name__ == "__main__":
    main()
