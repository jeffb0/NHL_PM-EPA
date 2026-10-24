"""Does the unclaimed turnover value explain PM-EPA's under-scaling on team goal differential?

Two candidate definitions of turnover cost, both built from the lambda = 1 minus lambda = 0 credit difference:

  T2 (charge only)  = diff of {giveaway, unobserved_loss, takeaway_against}: the opponent's possession-start value
                      after our turnovers, charged to the committers. It is exactly the slice DPM removes as
                      `start_turnover` (identity check below).
  T1 (symmetric)    = T2 + diff of `takeaway`: additionally credits the taker with the value denied to the opponent.

    python -m src.valuation.turnover_test
"""

from __future__ import annotations

import json
import logging

import polars as pl

from src.config import load_config
from src.defense.stints import RECOVERY_STARTS, TURNOVER_STARTS  # noqa: F401
from src.valuation import recon

log = logging.getLogger(__name__)
COMPS = ["takeaway", "takeaway_against", "giveaway", "unobserved_loss"]


def lam_terms(proc) -> pl.DataFrame:
    """Per player-season 5v5 sums of the lambda-term (lam1 - lam0) for each turnover component."""

    def load(f):
        return (
            pl.scan_parquet(proc / f)
            .filter(
                (pl.col("strength") == "5v5") & pl.col("player_id").is_not_null() & pl.col("component").is_in(COMPS)
            )
            .group_by("season", "player_id", "component")
            .agg(pl.col("value").sum())
            .collect()
        )

    a, b = load("credits.parquet"), load("credits_lam1.parquet")
    j = b.join(a, on=["season", "player_id", "component"], how="full", coalesce=True, suffix="_l0").with_columns(
        pl.col("value").fill_null(0.0), pl.col("value_l0").fill_null(0.0)
    )
    j = j.with_columns((pl.col("value") - pl.col("value_l0")).alias("d"))
    w = j.pivot(on="component", index=["season", "player_id"], values="d")
    for c in COMPS:
        if c not in w.columns:
            w = w.with_columns(pl.lit(0.0).alias(c))
    # only the value columns: player_id must stay an integer key
    w = w.with_columns(pl.col(c).fill_null(0.0) for c in COMPS)
    return w.with_columns(
        (pl.col("giveaway") + pl.col("unobserved_loss") + pl.col("takeaway_against")).alias("T2"),
        (pl.col("giveaway") + pl.col("unobserved_loss") + pl.col("takeaway_against") + pl.col("takeaway")).alias("T1"),
        pl.col("takeaway").alias("taker_credit"),
    ).select("season", "player_id", "T2", "T1", "taker_credit")


def identity_check(proc) -> dict:
    """Sum of T2 over all players vs minus the ledger's turnover-born start values (what DPM excludes), 5v5."""
    poss = pl.read_parquet(proc / "poss_valued_epa.parquet", columns=["poss_id", "start_type"])
    cr = (
        pl.scan_parquet(proc / "credits.parquet")
        .filter((pl.col("strength") == "5v5") & (pl.col("phase") == "start"))
        .collect()
        .join(poss, on="poss_id", how="left")
    )
    led = cr.filter((pl.col("component") == "takeaway") | pl.col("start_type").is_in(list(TURNOVER_STARTS)))[
        "value"
    ].sum()
    t2 = lam_terms(proc)["T2"].sum()
    return {
        "ledger_start_turnover_5v5": float(led),
        "sum_T2_all_players": float(t2),
        "ratio_minus_T2_over_ledger": float(-t2 / led),
    }


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    t = pl.read_parquet(proc / "player_value_season.parquet")
    dpm = pl.read_parquet(proc / "dpm_player_season.parquet").select(
        "player_id", "season", "toi_5v5", pl.col("dpm_start_turnover_per60"), pl.col("dpm_start_faceoff_per60")
    )
    ps = pl.read_parquet(proc / "player_season.parquet").select(
        "player_id", "season", "toi_ev", "pm_epa_ev", "pm_epa_per60_ev_lam1"
    )
    d = (
        t.join(lam_terms(proc), on=["player_id", "season"], how="left")
        .join(dpm.drop("toi_5v5"), on=["player_id", "season"], how="left")
        .join(ps.select("player_id", "season", "pm_epa_ev"), on=["player_id", "season"], how="left")
        .with_columns(
            pl.col("T2").fill_null(0.0),
            pl.col("T1").fill_null(0.0),
            pl.col("taker_credit").fill_null(0.0),
            (pl.col("dpm_start_turnover_per60").fill_null(0.0) * pl.col("toi_5v5") / 60).alias("dpm_excl_turnover"),
            (pl.col("dpm_start_faceoff_per60").fill_null(0.0) * pl.col("toi_5v5") / 60).alias("dpm_excl_faceoff"),
        )
    )
    cols = ["pm_gar", "def_gar", "fin_gar", "T2", "T1", "taker_credit", "dpm_excl_turnover", "dpm_excl_faceoff"]
    j = recon.team_sums(d, cols).join(recon.team_gd5(proc), on=["season", "team"])
    base = ["pm_gar", "def_gar", "fin_gar"]
    specs = {
        "R0 current (pm, def, fin)": base,
        "R1 + T2 charge-only": base + ["T2"],
        "R2 + T1 symmetric": base + ["T1"],
        "R3 + T2 + taker credit": base + ["T2", "taker_credit"],
        "R4 + DPM-excluded turnover (on-ice)": base + ["dpm_excl_turnover"],
        "R5 + T2 + DPM-excluded turnover": base + ["T2", "dpm_excl_turnover"],
    }
    res = {k: recon.regress(j, v) for k, v in specs.items()}
    res["T2_alone"] = recon.regress(j, ["T2"])
    res["T1_alone"] = recon.regress(j, ["T1"])
    res["identity"] = identity_check(proc)
    res["sd_team_totals"] = {c: float(j[c].std()) for c in cols}
    (met / "val_turnover_test.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = run()
    for k, v in out.items():
        if isinstance(v, dict) and "slopes" in v:
            print(
                f"{k:44s} R2={v['r2']:.3f}  "
                + "  ".join(f"{c}={s['slope']:.2f}±{s['se']:.2f}" for c, s in v["slopes"].items())
            )
    print("identity", out["identity"])
    print("team SDs", {k: round(v, 2) for k, v in out["sd_team_totals"].items()})
