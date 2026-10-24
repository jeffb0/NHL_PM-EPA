"""Role-adjusted turnover variant (TO-RA), methodology §8b.

Compares each player's arena-adjusted turnover charge with players who have a similar amount of *observed successful*
puck involvement (carry, move_start, move_end, shot_attempt credit rows per 60), within season and position, then
shrinks
and measures the residual above the pooled fringe residual. A variant beside TO, not a replacement.

    python -m src.components.turnover_ra     (needs `python -m src.components.turnover` first)
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.components import common
from src.config import load_config

log = logging.getLogger(__name__)
INV_COMPONENTS = ["carry", "move_start", "move_end", "shot_attempt"]
MIN_FIT_TOI = 200.0


def involvement(proc) -> pl.DataFrame:
    """Per player-season count of successful observed puck events at 5v5 (usable games)."""
    ok = common.good_game_ids(proc).lazy()
    return (
        pl.scan_parquet(proc / "credits.parquet")
        .filter(
            (pl.col("strength") == "5v5")
            & pl.col("player_id").is_not_null()
            & pl.col("component").is_in(INV_COMPONENTS)
        )
        .join(ok, on="game_id")
        .group_by("player_id", "season")
        .agg(pl.len().alias("inv_n"))
        .collect()
    )


def expected_rate(d: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Weighted quadratic fit of the per-60 charge on involvement per (season, position); returns d with `expected` and
    the R^2s."""
    parts, r2 = [], {}
    for (s, g), x in d.group_by(["season", "pos_group"]):
        fit = x.filter((pl.col("toi_ev") >= MIN_FIT_TOI) & pl.col("rate").is_not_null())
        w = fit["toi_ev"].to_numpy()
        iv, y = fit["inv60"].to_numpy(), fit["rate"].to_numpy()
        coef = np.polynomial.polynomial.polyfit(iv, y, 2, w=np.sqrt(w))
        pred_fit = np.polynomial.polynomial.polyval(iv, coef)
        ybar = np.average(y, weights=w)
        r2[f"{int(s)}_{g}"] = float(1 - np.sum(w * (y - pred_fit) ** 2) / np.sum(w * (y - ybar) ** 2))
        parts.append(
            x.with_columns(pl.Series("expected", np.polynomial.polynomial.polyval(x["inv60"].to_numpy(), coef)))
        )
    return pl.concat(parts), r2


def run() -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    cc = cfg.components
    to = pl.read_parquet(proc / "to_player_season.parquet")
    d = (
        to.select(
            "player_id",
            "season",
            "pos_group",
            "fringe",
            "primary_team",
            "toi_ev",
            "se60",
            "to_rate_per60",
            "to_gar",
            "to_post_per60",
        )
        .join(involvement(proc), on=["player_id", "season"], how="left")
        .with_columns(pl.col("inv_n").fill_null(0))
    )
    d = d.with_columns(
        pl.col("to_rate_per60").alias("rate"),
        pl.when(pl.col("toi_ev") > 0).then(pl.col("inv_n") / pl.col("toi_ev") * 60).alias("inv60"),
    ).filter(pl.col("inv60").is_not_null())
    d, r2 = expected_rate(d)
    d = d.with_columns((pl.col("rate") - pl.col("expected")).alias("resid60")).with_columns(
        (pl.col("resid60") * pl.col("toi_ev") / 60).alias("total")
    )
    val, rep = common.eb_rate_component(d, cc.min_toi)
    out = (
        d.select(
            "player_id",
            "season",
            "pos_group",
            "fringe",
            "primary_team",
            "toi_ev",
            "inv60",
            "rate",
            "expected",
            "resid60",
            "to_gar",
        )
        .join(val, on=["player_id", "season"])
        .rename(
            {
                "gar": "to_ra_gar",
                "gar_sd": "to_ra_gar_sd",
                "post_per60": "to_ra_post_per60",
                "rep_per60": "to_ra_rep_per60",
            }
        )
    )
    out.write_parquet(proc / "to_ra_player_season.parquet")

    q = out.filter(pl.col("toi_ev") >= cc.min_toi)
    diag: dict = {
        "r2_by_season_position": r2,
        "r2_mean_by_position": {
            g: float(np.mean([v for k, v in r2.items() if k.endswith("_" + g)])) for g in ("F", "D")
        },
    }
    diag["corr_rate_vs_inv"] = {
        g: float(np.corrcoef(x["rate"], x["inv60"])[0, 1]) for (g,), x in q.group_by(["pos_group"])
    }
    diag["corr_resid_vs_inv"] = {
        g: float(np.corrcoef(x["resid60"], x["inv60"])[0, 1]) for (g,), x in q.group_by(["pos_group"])
    }
    diag["gate_RA1_pass"] = bool(all(abs(v) < 0.15 for v in diag["corr_resid_vs_inv"].values()))
    tab = out.with_columns(
        pl.col("rate").alias("raw_rate"), pl.col("to_ra_post_per60").alias("shr"), pl.col("resid60").alias("raw")
    )
    diag["yoy_shrunk"] = common.season_centred_yoy(tab, "shr", cc.min_toi)
    diag["yoy_raw_residual"] = common.season_centred_yoy(tab, "raw", cc.min_toi)
    diag["yoy_original_TO_shrunk"] = common.season_centred_yoy(
        to.with_columns(pl.col("to_post_per60").alias("shr")), "shr", cc.min_toi
    )
    diag["gate_RA2_pass"] = bool(diag["yoy_shrunk"]["F"]["r"] >= 0.30 and diag["yoy_shrunk"]["D"]["r"] >= 0.30)
    qq = tab.filter(pl.col("toi_ev") >= cc.min_toi).with_columns(
        (pl.col("shr") - pl.col("shr").mean().over("season", "pos_group")).alias("shr_c"),
        (pl.col("raw") - pl.col("raw").mean().over("season", "pos_group")).alias("raw_c"),
    )
    j = qq.join(
        qq.select(
            "player_id", (pl.col("season") - 1).alias("season"), pl.col("raw_c").alias("y"), pl.col("toi_ev").alias("w")
        ),
        on=["player_id", "season"],
    )
    dsq = (j["w"] / j["w"].mean()).to_numpy() * (
        (j["raw_c"] - j["y"]).to_numpy() ** 2 - (j["shr_c"] - j["y"]).to_numpy() ** 2
    )
    diag["RA3_shrunk_vs_raw_next_season"] = common.paired_z(dsq)
    diag["gate_RA3_pass"] = bool(diag["RA3_shrunk_vs_raw_next_season"]["z"] > cfg.finishing.se_threshold)
    # stayers vs changers (raw residual and original)
    from src.components.turnover import _changer_yoy

    diag["team_changer_yoy"] = {
        "role_adjusted_shrunk": _changer_yoy(tab.with_columns(pl.col("shr").alias("c")), "shr", cc.min_toi),
        "original_shrunk": _changer_yoy(to.rename({"to_post_per60": "shr"}), "shr", cc.min_toi),
    }
    # what changed
    diag["spearman_TO_vs_TORA_qualified"] = {
        g: float(x.select(pl.corr(pl.col("to_gar").rank(), pl.col("to_ra_gar").rank()))[0, 0])
        for (g,), x in q.group_by(["pos_group"])
    }
    q5 = q.with_columns(
        pl.col("inv60")
        .qcut(5, labels=[f"Q{i}" for i in range(1, 6)], allow_duplicates=True)
        .over("pos_group", "season")
        .alias("inv_q")
    )
    diag["mean_gar_by_involvement_quintile"] = {
        r["inv_q"]: {"to_gar": r["to"], "to_ra_gar": r["ra"], "n": r["n"]}
        for r in q5.group_by("inv_q")
        .agg(pl.col("to_gar").mean().alias("to"), pl.col("to_ra_gar").mean().alias("ra"), pl.len().alias("n"))
        .sort("inv_q")
        .to_dicts()
    }
    diag["sd_gar_qualified"] = {"TO": float(q["to_gar"].std()), "TO_RA": float(q["to_ra_gar"].std())}
    (met / "to_ra_component.json").write_text(json.dumps(diag, indent=2, default=float))
    log.info(
        "TO-RA: R2 F %.2f D %.2f; corr(resid, inv) %s; YOY shrunk F %.2f D %.2f; RA-3 z=%.1f",
        diag["r2_mean_by_position"]["F"],
        diag["r2_mean_by_position"]["D"],
        {k: round(v, 2) for k, v in diag["corr_resid_vs_inv"].items()},
        diag["yoy_shrunk"]["F"]["r"],
        diag["yoy_shrunk"]["D"]["r"],
        diag["RA3_shrunk_vs_raw_next_season"]["z"],
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
