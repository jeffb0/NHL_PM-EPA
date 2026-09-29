"""Penalties taken and drawn at 5v5 (PEN), methodology §5.

Cost of a penalty class = mean net xG for the non-penalized team from the whistle until strength returns to even (or the
penalty length, capped), from the shared cross-fitted shot xG. Player value = minor-equivalent counts drawn minus taken,
as gamma-Poisson empirical-Bayes rates per 5v5 minute, times the cost of a minor, above the pooled fringe replacement rates.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.components import common
from src.config import load_config

log = logging.getLogger(__name__)
CLASS_OF = {2: 2, 4: 4, 5: 5}
TYPES = ["MIN", "MAJ", "BEN"]


def qualifying(proc, cap_s: dict[int, int]) -> pl.DataFrame:
    """5v5 penalties (minor / double minor / major) with net xG for the non-penalized team over the resulting man advantage."""
    ev = pl.read_parquet(proc / "events.parquet", columns=["game_id", "event_idx", "season", "period", "t_period", "type", "team_id", "home_skaters", "away_skaters",
                                                           "home_goalie", "away_goalie", "strength", "player1", "player2", "penalty_min", "penalty_type"]).sort("game_id", "event_idx")
    pen_all = ev.filter(pl.col("type") == "penalty")
    # coincident / multiple penalties at the same instant: no clean advantage, excluded
    multi = pen_all.group_by("game_id", "period", "t_period").agg(pl.len().alias("k")).filter(pl.col("k") >= 2).select("game_id", "period", "t_period")
    pen = (pen_all.filter((pl.col("strength") == "5v5") & pl.col("penalty_min").is_in(list(CLASS_OF)) & pl.col("penalty_type").is_in(TYPES))
           .join(multi.with_columns(pl.lit(True).alias("_m")), on=["game_id", "period", "t_period"], how="left").filter(pl.col("_m").is_null()).drop("_m"))
    shots = pl.read_parquet(proc / "shots_xg.parquet", columns=["game_id", "event_idx", "team_id", "xg"]).filter(pl.col("xg") > 0).join(
        ev.select("game_id", "event_idx", "period", "t_period"), on=["game_id", "event_idx"])
    even_t = ev.filter((pl.col("home_skaters") == pl.col("away_skaters")) & (pl.col("home_goalie") == 1) & (pl.col("away_goalie") == 1)).select("game_id", "period", "t_period")
    last_t = ev.group_by("game_id", "period").agg(pl.col("t_period").max().alias("t_last"))
    sg = {k[0]: g for k, g in shots.group_by(["game_id"])}
    eg = {(k[0], k[1]): g["t_period"].to_numpy() for k, g in even_t.sort("t_period").group_by(["game_id", "period"])}
    lt = {(r["game_id"], r["period"]): r["t_last"] for r in last_t.to_dicts()}
    rows = []
    for r in pen.iter_rows(named=True):
        g, p, t = r["game_id"], r["period"], r["t_period"]
        cls = CLASS_OF[r["penalty_min"]]
        cap = t + cap_s[cls]
        ets = eg.get((g, p), np.array([]))
        k = np.searchsorted(ets, t, side="right")          # first event strictly after t with even strength
        end = min(cap, float(ets[k]) if k < len(ets) else cap, float(lt.get((g, p), cap)))
        s = sg.get(g)
        net = 0.0
        if s is not None and end > t:
            w = s.filter((pl.col("period") == p) & (pl.col("t_period") > t) & (pl.col("t_period") <= end))
            if w.height:
                pen_team = w.filter(pl.col("team_id") == r["team_id"])["xg"].sum()
                net = float(w["xg"].sum() - 2 * pen_team)         # (other team's xG) - (penalized team's xG)
        rows.append((g, r["event_idx"], r["season"], cls, r["penalty_type"], r["player1"], r["player2"], float(end - t), net))
    return pl.DataFrame(rows, schema=["game_id", "event_idx", "season", "cls", "ptype", "committer", "drawer", "adv_s", "net_xg"], orient="row")


def class_costs(pen: pl.DataFrame) -> dict:
    out = {}
    for cls, g in sorted(pen.group_by(["cls"]).__iter__(), key=lambda kv: kv[0]):
        x = g["net_xg"].to_numpy()
        out[int(cls[0])] = {"n": int(len(x)), "cost": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))), "mean_adv_s": float(g["adv_s"].mean())}
    return out


def gamma_poisson(d: pl.DataFrame, n: str, min_toi: float) -> pl.DataFrame:
    """Posterior mean and variance of the per-minute rate of weighted count ``n``; prior mean = pooled rate of the status group,
    shape by method of moments from qualified regulars (by position)."""
    d = d.with_columns(pl.col(n).fill_null(0.0), pl.col("toi_ev").fill_null(0.0))
    grp = d.group_by("season", "pos_group", "fringe").agg((pl.col(n).sum() / pl.col("toi_ev").sum()).alias("m"))
    d = d.join(grp, on=["season", "pos_group", "fringe"], how="left")
    q = d.filter((~pl.col("fringe")) & (pl.col("toi_ev") >= min_toi))
    alpha = {}
    for (g,), x in q.group_by(["pos_group"]):
        m, e, c = x["m"].to_numpy(), x["toi_ev"].to_numpy(), x[n].to_numpy()
        inv = float(((c - m * e) ** 2 - m * e).sum() / ((m * e) ** 2).sum())
        alpha[g] = float(min(1.0 / max(inv, 1e-4), 1e4))
    a = d["pos_group"].replace_strict(alpha, default=50.0)
    return d.with_columns(a.alias("alpha")).with_columns(
        ((pl.col("alpha") + pl.col(n)) / (pl.col("alpha") / pl.col("m") + pl.col("toi_ev"))).alias(f"{n}_post"),
        ((pl.col("alpha") + pl.col(n)) / (pl.col("alpha") / pl.col("m") + pl.col("toi_ev")) ** 2).alias(f"{n}_postvar"),
        pl.when(pl.col("toi_ev") > 0).then(pl.col(n) / pl.col("toi_ev")).alias(f"{n}_raw"))


def deviance(n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.maximum(mu, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(n > 0, n * np.log(n / mu), 0.0)
    return 2 * (t - (n - mu))


def run() -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    cc = cfg.components
    pen = qualifying(proc, {int(k): v for k, v in cc.penalty_cap_s.items()})
    costs = class_costs(pen)
    by_type = {str(k[0]): {"n": g.height, "cost": float(g["net_xg"].mean())} for k, g in pen.group_by(["ptype"])}
    c2 = costs[2]["cost"]
    pen = pen.with_columns(pl.col("cls").replace_strict({k: v["cost"] / c2 for k, v in costs.items()}, return_dtype=pl.Float64).alias("w"))
    ok = common.good_game_ids(proc)
    pen = pen.join(ok, on="game_id")
    taken = pen.filter(pl.col("committer").is_not_null()).group_by(pl.col("committer").alias("player_id"), "season").agg(pl.col("w").sum().alias("taken_w"), pl.len().alias("taken_n"))
    drawn = pen.filter(pl.col("drawer").is_not_null()).group_by(pl.col("drawer").alias("player_id"), "season").agg(pl.col("w").sum().alias("drawn_w"), pl.len().alias("drawn_n"))
    b = common.base(proc)
    d = b.join(taken, on=["player_id", "season"], how="left").join(drawn, on=["player_id", "season"], how="left").with_columns(
        [pl.col(c).fill_null(0) for c in ("taken_w", "taken_n", "drawn_w", "drawn_n")])
    dt = gamma_poisson(d, "taken_w", cc.min_toi).select("player_id", "season", "taken_w_post", "taken_w_postvar", "taken_w_raw", "alpha")
    dd = gamma_poisson(d, "drawn_w", cc.min_toi).select("player_id", "season", "drawn_w_post", "drawn_w_postvar", "drawn_w_raw", pl.col("alpha").alias("alpha_d"))
    x = d.join(dt, on=["player_id", "season"]).join(dd, on=["player_id", "season"])
    rep = x.filter(pl.col("fringe")).group_by("season", "pos_group").agg((pl.col("drawn_w").sum() / pl.col("toi_ev").sum()).alias("rep_drawn"), (pl.col("taken_w").sum() / pl.col("toi_ev").sum()).alias("rep_taken"))
    x = x.join(rep, on=["season", "pos_group"], how="left").with_columns(
        (c2 * ((pl.col("drawn_w_post") - pl.col("taken_w_post")) - (pl.col("rep_drawn") - pl.col("rep_taken"))) * pl.col("toi_ev")).alias("pen_gar"),
        (c2 * pl.col("toi_ev") * (pl.col("drawn_w_postvar") + pl.col("taken_w_postvar")).sqrt()).alias("pen_gar_sd"),
        (pl.col("drawn_w_post") * 60).alias("drawn_per60_post"), (pl.col("taken_w_post") * 60).alias("taken_per60_post"))
    x.write_parquet(proc / "pen_player_season.parquet")

    diag: dict = {"class_costs": costs, "cost_by_type": by_type, "n_penalties": pen.height, "cost_minor": c2}
    diag["gate_PEN1_pass"] = bool(costs[2]["cost"] - 1.96 * costs[2]["se"] > 0)
    q = x.filter(pl.col("toi_ev") >= cc.min_toi)
    diag["sd_pen_gar_qualified"] = float(q["pen_gar"].std())
    diag["league_rates_per60"] = {"drawn_minor_equiv": float(x["drawn_w"].sum() / x["toi_ev"].sum() * 60), "taken_minor_equiv": float(x["taken_w"].sum() / x["toi_ev"].sum() * 60)}
    diag["alpha_by_pos"] = {"taken": {g: float(v) for g, v in dt.join(b.select("player_id", "season", "pos_group"), on=["player_id", "season"]).group_by("pos_group").agg(pl.col("alpha").first()).iter_rows()},
                            "drawn": {g: float(v) for g, v in dd.join(b.select("player_id", "season", "pos_group"), on=["player_id", "season"]).group_by("pos_group").agg(pl.col("alpha_d").first()).iter_rows()}}
    for kind in ("drawn", "taken"):
        t = x.with_columns((pl.col(f"{kind}_w_raw") * 60).alias("raw"), (pl.col(f"{kind}_w_post") * 60).alias("shr"))
        diag[f"yoy_{kind}"] = {"shrunk": common.season_centred_yoy(t, "shr", cc.min_toi), "raw": common.season_centred_yoy(t, "raw", cc.min_toi)}
        qq = x.filter(pl.col("toi_ev") >= cc.min_toi)
        nxt = qq.select("player_id", (pl.col("season") - 1).alias("season"), pl.col(f"{kind}_w").alias("n_next"), pl.col("toi_ev").alias("e_next"))
        j = qq.join(nxt, on=["player_id", "season"])
        n_next, e_next = j["n_next"].to_numpy(), j["e_next"].to_numpy()
        dev_raw = deviance(n_next, j[f"{kind}_w_raw"].to_numpy() * e_next)
        dev_shr = deviance(n_next, j[f"{kind}_w_post"].to_numpy() * e_next)
        diag[f"{kind}_forward"] = {**common.paired_z(dev_raw - dev_shr), "dev_raw": float(dev_raw.mean()), "dev_shrunk": float(dev_shr.mean())}
        diag[f"gate_PEN2_{kind}_pass"] = bool(diag[f"{kind}_forward"]["z"] > cfg.finishing.se_threshold)
    diag["gate_PEN3_pass"] = bool(diag["yoy_drawn"]["shrunk"]["all"]["r"] >= 0.15 and diag["yoy_taken"]["shrunk"]["all"]["r"] >= 0.15)
    (met / "pen_component.json").write_text(json.dumps(diag, indent=2, default=float))
    log.info("PEN: minor cost %.3f xG (%.3f-%.3f); n=%d; YOY drawn %.2f taken %.2f; SD gar %.2f", c2, c2 - 1.96 * costs[2]["se"], c2 + 1.96 * costs[2]["se"], pen.height,
             diag["yoy_drawn"]["shrunk"]["all"]["r"], diag["yoy_taken"]["shrunk"]["all"]["r"], diag["sd_pen_gar_qualified"])
    return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
