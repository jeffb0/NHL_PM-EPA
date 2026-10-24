"""Turnover cost (TO), methodology §3.

TO_i = 5v5 sum over player i's credit rows of (value at lambda = 1) - (value at lambda = 0) for the components
giveaway (direct), unobserved_loss and takeaway_against (pooled over the losing team's on-ice skaters): the opponent's
possession-start value after our turnovers, charged to the committers. Shrunk with the same empirical-Bayes scheme as
PM-EPA and measured above the pooled fringe rate.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.components import common
from src.config import load_config
from src.defense.stints import TURNOVER_STARTS
from src.metrics.aggregate import bootstrap_ci

log = logging.getLogger(__name__)
COMPS = ["giveaway", "unobserved_loss", "takeaway_against"]


def game_terms(proc) -> pl.DataFrame:
    """Per player-game 5v5 lambda-term (lam1 - lam0) by component."""
    ok = common.good_game_ids(proc).lazy()

    def load(f):
        return (
            pl.scan_parquet(proc / f)
            .filter(
                (pl.col("strength") == "5v5") & pl.col("player_id").is_not_null() & pl.col("component").is_in(COMPS)
            )
            .join(ok, on="game_id")
            .group_by("game_id", "season", "player_id", "component")
            .agg(pl.col("value").sum())
            .collect()
        )

    a, b = load("credits.parquet"), load("credits_lam1.parquet")
    j = b.join(a, on=["game_id", "season", "player_id", "component"], how="full", coalesce=True, suffix="_l0")
    j = j.with_columns((pl.col("value").fill_null(0.0) - pl.col("value_l0").fill_null(0.0)).alias("d"))
    w = j.pivot(on="component", index=["game_id", "season", "player_id"], values="d")
    for c in COMPS:
        if c not in w.columns:
            w = w.with_columns(pl.lit(0.0).alias(c))
    w = w.with_columns(pl.col(c).fill_null(0.0) for c in COMPS)
    return w.with_columns(
        (pl.col("giveaway") + pl.col("unobserved_loss") + pl.col("takeaway_against")).alias("to_total"),
        pl.col("giveaway").alias("to_direct"),
        (pl.col("unobserved_loss") + pl.col("takeaway_against")).alias("to_pooled"),
    ).select("game_id", "season", "player_id", "to_total", "to_direct", "to_pooled")


def accounting_check(proc, gt: pl.DataFrame) -> dict:
    """Gate TO-1: -sum(T2) over all players vs the ledger's turnover-born plus unobserved start values that DPM
    excludes."""
    poss = pl.read_parquet(proc / "poss_valued_epa.parquet", columns=["poss_id", "start_type"])
    ok = common.good_game_ids(proc)
    cr = (
        pl.scan_parquet(proc / "credits.parquet")
        .filter((pl.col("strength") == "5v5") & (pl.col("phase") == "start"))
        .collect()
        .join(ok, on="game_id")
        .join(poss, on="poss_id", how="left")
    )
    tb = cr.filter((pl.col("component") == "takeaway") | pl.col("start_type").is_in(list(TURNOVER_STARTS)))[
        "value"
    ].sum()
    un = cr.filter(pl.col("start_type") == "unobserved")["value"].sum()
    t2 = float(gt["to_total"].sum())
    return {
        "sum_T2": t2,
        "ledger_turnover_born_starts": float(tb),
        "ledger_unobserved_starts": float(un),
        "ratio_to_turnover_born_only": float(-t2 / tb),
        "ratio_to_turnover_born_plus_unobserved": float(-t2 / (tb + un)),
    }


def rink_diagnostic(proc, gt: pl.DataFrame, toi: pl.DataFrame) -> dict:
    """Scorer / rink bias: spread of arena means of the per-60 turnover charge vs what sampling noise alone gives."""
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "season", "home_id")
    g = (
        gt.drop("season")
        .join(toi, on=["game_id", "player_id"], how="right")
        .with_columns(pl.col("to_total").fill_null(0.0))
        .group_by("game_id")
        .agg(pl.col("to_total").sum().alias("x"), pl.col("toi_5v5").sum().alias("t"))
        .join(games, on="game_id")
        .filter(pl.col("t") > 0)
        .with_columns((pl.col("x") / pl.col("t") * 3600).alias("r"))
    )
    a = g.group_by("season", "home_id").agg(
        pl.col("r").mean().alias("m"), pl.col("r").var().alias("v"), pl.len().alias("n")
    )
    res = {}
    for (s,), x in a.group_by(["season"]):
        between = float(x["m"].var())
        noise = float((x["v"] / x["n"]).mean())
        res[int(s)] = {
            "arena_sd": float(np.sqrt(max(between - noise, 0.0))),
            "total_sd": float(np.sqrt(between)),
            "noise_sd": float(np.sqrt(noise)),
        }
    return {
        "per_season": dict(sorted(res.items())),
        "mean_arena_sd_excess": float(np.mean([v["arena_sd"] for v in res.values()])),
        "mean_game_rate_sd": float(g["r"].std()),
    }


def arena_effects(proc, gt: pl.DataFrame, col: str = "to_total") -> tuple[pl.DataFrame, dict]:
    """Arena x season scorer effect on the per-60 turnover charge, from *visiting* players only.

    Each visitor's game rate is compared with the same player's away rate in that season (leaving the game out), and the
    residuals are averaged by arena. Visitors are used because half the skaters in a home game are the home team's own
    players, whose style would otherwise leak into the arena effect. Effects are empirical-Bayes shrunk (the
    between-arena
    variance is estimated per season) after centring the arena means (so the shrunk effects average ~0).
    Units: goals per 60 player-minutes. ``col`` picks the charge part (total, direct or pooled).
    """
    toi = (
        pl.read_parquet(proc / "toi_game.parquet")
        .select("game_id", "player_id", "team_id", "toi_5v5")
        .join(common.good_game_ids(proc), on="game_id")
    )
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "season", "home_id")
    d = (
        toi.join(gt.drop("season"), on=["game_id", "player_id"], how="left")
        .join(games, on="game_id")
        .with_columns(pl.col(col).fill_null(0.0), (pl.col("team_id") != pl.col("home_id")).alias("away"))
    )
    v = d.filter(pl.col("away") & (pl.col("toi_5v5") > 120)).with_columns(
        pl.col(col).sum().over("player_id", "season").alias("sx"),
        pl.col("toi_5v5").sum().over("player_id", "season").alias("st"),
    )
    v = v.filter(pl.col("st") - pl.col("toi_5v5") > 1800).with_columns(
        (
            (pl.col(col) / pl.col("toi_5v5") * 3600)
            - ((pl.col("sx") - pl.col(col)) / (pl.col("st") - pl.col("toi_5v5")) * 3600)
        ).alias("resid")
    )
    a = v.group_by("season", "home_id").agg(
        ((pl.col("resid") * pl.col("toi_5v5")).sum() / pl.col("toi_5v5").sum()).alias("m"),
        (pl.col("resid").var() / pl.len()).alias("noise"),
    )
    rows, summary = [], {}
    for (s,), x in a.group_by(["season"]):
        between = float(x["m"].var())
        tau2 = max(between - float(x["noise"].mean()), 0.0)
        mu = float(x["m"].mean())
        w = tau2 / (tau2 + x["noise"].to_numpy())
        eff = w * (x["m"].to_numpy() - mu)
        rows.append(
            pl.DataFrame(
                {"season": int(s), "home_id": x["home_id"], "arena_effect": eff},
                schema={"season": pl.Int64, "home_id": pl.Int64, "arena_effect": pl.Float64},
            )
        )
        summary[int(s)] = {
            "arena_sd_excess": float(np.sqrt(tau2)),
            "noise_sd": float(np.sqrt(float(x["noise"].mean()))),
            "shrunk_sd": float(np.std(eff)),
        }
    return pl.concat(rows), dict(sorted(summary.items()))


def _changer_yoy(t: pl.DataFrame, col: str, min_toi: float) -> dict:
    """Season-centred YOY r for players who kept vs changed their primary team (arena / system bias shows up as a
    gap)."""
    q = t.filter(pl.col("toi_ev") >= min_toi).with_columns(
        (pl.col(col) - pl.col(col).mean().over("season", "pos_group")).alias("c")
    )
    j = q.join(
        q.select(
            "player_id",
            (pl.col("season") - 1).alias("season"),
            pl.col("c").alias("c_n"),
            pl.col("primary_team").alias("team_n"),
        ),
        on=["player_id", "season"],
    )
    out = {}
    for tag, x in (
        ("stayers", j.filter(pl.col("primary_team") == pl.col("team_n"))),
        ("changers", j.filter(pl.col("primary_team") != pl.col("team_n"))),
    ):
        out[tag] = {"r": float(np.corrcoef(x["c"], x["c_n"])[0, 1]), "n": x.height}
    return out


def run() -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    cc = cfg.components
    gt = game_terms(proc)
    arena, arena_summary = arena_effects(proc, gt)
    toi = (
        pl.read_parquet(proc / "toi_game.parquet")
        .select("game_id", "player_id", "toi_5v5")
        .join(common.good_game_ids(proc), on="game_id")
    )
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "season", "home_id")
    pg = (
        toi.join(games, on="game_id")
        .join(gt.drop("season"), on=["game_id", "player_id"], how="left")
        .join(arena, on=["season", "home_id"], how="left")
        .with_columns(
            pl.col("to_total").fill_null(0.0),
            pl.col("to_direct").fill_null(0.0),
            pl.col("to_pooled").fill_null(0.0),
            pl.col("arena_effect").fill_null(0.0),
        )
        .with_columns(
            pl.col("to_total").alias("to_unadj"),
            (pl.col("to_total") - pl.col("arena_effect") * pl.col("toi_5v5") / 3600).alias("to_total"),
        )
    )
    ci = bootstrap_ci(pg, "to_total", "toi_5v5", cc.bootstrap_reps, 0.95, cfg.seed)
    agg = pg.group_by("player_id", "season").agg(
        pl.col("to_total").sum(), pl.col("to_unadj").sum(), pl.col("to_direct").sum(), pl.col("to_pooled").sum()
    )
    b = common.base(proc)
    d = (
        b.join(agg, on=["player_id", "season"], how="left")
        .join(
            ci.select("player_id", "season", pl.col("boot_se_per60").alias("se60")),
            on=["player_id", "season"],
            how="left",
        )
        .with_columns(
            pl.col("to_total").fill_null(0.0),
            pl.col("to_unadj").fill_null(0.0),
            pl.col("to_direct").fill_null(0.0),
            pl.col("to_pooled").fill_null(0.0),
        )
        .with_columns(pl.col("to_total").alias("total"))
    )
    val, rep = common.eb_rate_component(d, cc.min_toi)
    val_u, _ = common.eb_rate_component(d.with_columns(pl.col("to_unadj").alias("total")), cc.min_toi)
    out = (
        d.select(
            "player_id",
            "season",
            "pos_group",
            "fringe",
            "primary_team",
            "toi_ev",
            "to_total",
            "to_unadj",
            "to_direct",
            "to_pooled",
            "se60",
        )
        .join(val, on=["player_id", "season"])
        .rename(
            {
                "gar": "to_gar",
                "gar_sd": "to_gar_sd",
                "rate": "to_rate_per60",
                "post_per60": "to_post_per60",
                "rep_per60": "to_rep_per60",
            }
        )
        .join(
            val_u.select(
                "player_id",
                "season",
                pl.col("gar").alias("to_gar_unadj"),
                pl.col("rate").alias("to_rate_unadj_per60"),
                pl.col("post_per60").alias("to_post_unadj_per60"),
            ),
            on=["player_id", "season"],
        )
    )
    out.write_parquet(proc / "to_player_season.parquet")

    # ---- diagnostics and gates
    diag: dict = {"accounting": accounting_check(proc, gt), "arena_effects": arena_summary}
    diag["team_changer_yoy"] = {
        k: _changer_yoy(out, col, cc.min_toi)
        for k, col in (
            ("adjusted_raw", "to_rate_per60"),
            ("unadjusted_raw", "to_rate_unadj_per60"),
            ("adjusted_shrunk", "to_post_per60"),
            ("unadjusted_shrunk", "to_post_unadj_per60"),
        )
    }
    diag["gate_TO1_pass"] = bool(0.75 <= diag["accounting"]["ratio_to_turnover_born_plus_unobserved"] <= 1.25)
    tab = out.with_columns(pl.col("to_post_per60").alias("shr"), pl.col("to_rate_per60").alias("raw"))
    diag["yoy_shrunk"] = common.season_centred_yoy(tab, "shr", cc.min_toi)
    diag["yoy_raw"] = common.season_centred_yoy(tab, "raw", cc.min_toi)
    diag["yoy_direct_only"] = common.season_centred_yoy(
        tab.with_columns((pl.col("to_direct") / pl.col("toi_ev") * 60).alias("dir")), "dir", cc.min_toi
    )
    diag["gate_TO3_pass"] = bool(diag["yoy_shrunk"]["F"]["r"] >= 0.30 and diag["yoy_shrunk"]["D"]["r"] >= 0.30)
    q = tab.filter(pl.col("toi_ev") >= cc.min_toi).with_columns(
        (pl.col("shr") - pl.col("shr").mean().over("season", "pos_group")).alias("shr_c"),
        (pl.col("raw") - pl.col("raw").mean().over("season", "pos_group")).alias("raw_c"),
    )
    nxt = q.select(
        "player_id", (pl.col("season") - 1).alias("season"), pl.col("raw_c").alias("y"), pl.col("toi_ev").alias("w")
    )
    j = q.join(nxt, on=["player_id", "season"])
    dsq = (j["w"] / j["w"].mean()).to_numpy() * (
        (j["raw_c"] - j["y"]).to_numpy() ** 2 - (j["shr_c"] - j["y"]).to_numpy() ** 2
    )
    diag["TO4_shrunk_vs_raw_next_season_mse"] = {
        **common.paired_z(dsq),
        "mse_raw": float(((j["raw_c"] - j["y"]) ** 2).mean()),
        "mse_shrunk": float(((j["shr_c"] - j["y"]) ** 2).mean()),
    }
    diag["gate_TO4_pass"] = bool(diag["TO4_shrunk_vs_raw_next_season_mse"]["z"] > cfg.finishing.se_threshold)
    diag["rink"] = rink_diagnostic(
        proc, gt, pl.read_parquet(proc / "toi_game.parquet").select("game_id", "player_id", "toi_5v5")
    )
    diag["replacement_rates_per60"] = rep.sort("season", "pos_group").to_dicts()
    diag["sd_gar_qualified"] = float(out.filter(pl.col("toi_ev") >= cc.min_toi)["to_gar"].std())
    diag["direct_share_of_total_abs"] = float(
        out["to_direct"].abs().sum() / (out["to_direct"].abs().sum() + out["to_pooled"].abs().sum())
    )
    (met / "to_component.json").write_text(json.dumps(diag, indent=2, default=float))
    log.info(
        "TO: accounting ratio %.2f; YOY shrunk F %.2f D %.2f; TO-4 z=%.1f; SD gar %.2f",
        diag["accounting"]["ratio_to_turnover_born_plus_unobserved"],
        diag["yoy_shrunk"]["F"]["r"],
        diag["yoy_shrunk"]["D"]["r"],
        diag["TO4_shrunk_vs_raw_next_season_mse"]["z"],
        diag["sd_gar_qualified"],
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
