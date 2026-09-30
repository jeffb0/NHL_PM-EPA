"""reports/transition/validation_report.md from tr_results.json (docs/transition/methodology.md §5)."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.calibration.report import _ok, _table
from src.calibration.report_pos import _corrs
from src.config import ROOT, load_config


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    r = json.loads((met / "tr_results.json").read_text(encoding="utf-8"))
    g, pl_, tl = r["gates"], r["player_level"], r["team_level"]
    out = ["# On-ice offense / transition component — validation report", "",
           "Spec and pre-registered gates: [methodology](../../docs/transition/methodology.md). OPM is the independent-season (ρ = 0) on-ice offense RAPM; TR is the part of OPM not already in PM-EPA (within position). Motivation: the Morrissey/Appleton review.", "",
           f"**{'TR is adopted as a published variant (`total_gar_all_tr`); the default total is unchanged.' if r['adopt'] else 'Not adopted: at least one gate failed. The total is unchanged.'}**", "",
           "## Gates", "", "| Gate | Result | Pass |", "|---|---|---|",
           f"| TR-1 year-over-year correlation of TR for defensemen ≥ 0.20 | defensemen {g['TR1']['yoy_D']:.2f}, forwards {g['TR1']['yoy_F']:.2f} | {_ok(g['TR1']['pass'])} |",
           f"| TR-2 adding TR improves next-season prediction (z > 2), all skaters and defensemen | all z = {g['TR2']['z_all']:.2f}; defensemen z = {g['TR2']['z_D']:.2f} | {_ok(g['TR2']['pass'])} |",
           f"| TR-3 adding TR improves held-out team-level error over the position-specific identity (z > 2) | {tl['cv_mse_with']:.1f} vs {tl['cv_mse_without']:.1f}, z = {g['TR3']['z']:.2f} | {_ok(g['TR3']['pass'])} |",
           f"| TR-4 the total with TR does no harm at the player level (deployment-adjusted goals z > −2) | goals z = {r['TR4_detail']['goals_z']:.2f}, xG z = {r['TR4_detail']['xg_z']:.2f} | {_ok(g['TR4']['pass'])} |", "",
           "## The measure", "",
           _table([{"position": k, "SD OPM per 60": v["opm_gar"], "SD PM-EPA per 60": v["pm_gar"], "SD TR per 60": v["tr_gar"], "β (OPM on PM-EPA)": r["beta_pos"][k],
                    "YOY OPM": r["yoy"]["opm"][k]["r"], "YOY PM-EPA": r["yoy"]["pm"][k]["r"], "YOY TR": r["yoy"]["tr"][k]["r"]} for k, v in r["sd_per60"].items()], 3), "",
           "## Incremental validity at the player level (next-season deployment-adjusted goal differential per 60)", "",
           _table([{"skaters": k, "rows": v["n"], "TR coefficient ± SE": f"{v['tr_coef']:.2f} ± {v['tr_se']:.2f}", "paired z, with TR better": v["paired_z_with_tr_better"]} for k, v in pl_.items()], 2), "",
           "## Team level", "",
           f"Ridge weights on the team sums of TR: forwards {tl['weight_tr_F']:.2f} ({tl['ci_tr_F'][0]:.2f} to {tl['ci_tr_F'][1]:.2f}), defensemen {tl['weight_tr_D']:.2f} ({tl['ci_tr_D'][0]:.2f} to {tl['ci_tr_D'][1]:.2f}); "
           f"unadjusted slopes {tl['ols_slope_tr_F']['slope']:.2f} ± {tl['ols_slope_tr_F']['se']:.2f} and {tl['ols_slope_tr_D']['slope']:.2f} ± {tl['ols_slope_tr_D']['se']:.2f}. Applied weights (clipped to [0, 3]): forwards {r['weights_applied']['F']:.2f}, defensemen {r['weights_applied']['D']:.2f}.", ""]
    d = pl.read_parquet(proc / "player_value_all_tr.parquet")
    q = d.filter(pl.col("season") == 2022)
    rows = []
    for n in ("Josh Morrissey", "Mason Appleton", "Adam Fox", "Jaccob Slavin", "Cale Makar", "Quinn Hughes", "Roman Josi"):
        x = q.filter(pl.col("player_name") == n)
        if x.height:
            rows.append({"player": n, "position": x["position"][0], "GAR position-weighted": x["total_gar_all_ridge"][0], "OPM": x["opm_gar"][0], "TR": x["tr_gar"][0], "GAR with TR": x["total_gar_all_tr"][0], "± (GAR sd)": x["total_gar_all_sd"][0] if "total_gar_all_sd" in x.columns else None})
    out += ["## Named players, 2022", "", _table(rows, 2), ""]
    qq = d.filter(pl.col("toi_5v5") >= 500)
    par = qq.group_by("pos_group").agg(pl.len().alias("player-seasons"), pl.col("total_gar_all_ridge").mean().alias("mean GAR position-weighted"), pl.col("total_gar_all_tr").mean().alias("mean GAR with TR"),
                                       pl.col("total_gar_all_ridge").std().alias("SD position-weighted"), pl.col("total_gar_all_tr").std().alias("SD with TR")).sort("pos_group")
    out += ["## Defensemen against forwards", "", _table(par.to_dicts(), 2), ""]
    cb, ca = _corrs(d, proc, "total_gar_all_ridge"), _corrs(d, proc, "total_gar_all_tr")
    out += ["External-validity look (not a criterion): correlation of total GAR per 60 with ice time per game and points per 60.", "",
            _table([{"position": k, "TOI/game before": cb[k]["toi"], "TOI/game with TR": ca[k]["toi"], "points/60 before": cb[k]["points"], "points/60 with TR": ca[k]["points"]} for k in ("D", "F")], 3), ""]
    top = q.filter((pl.col("pos_group") == "D") & (pl.col("toi_5v5") >= 500)).with_columns(pl.col("total_gar_all_ridge").rank("ordinal", descending=True).alias("rank_before"), pl.col("total_gar_all_tr").rank("ordinal", descending=True).alias("rank_after"))
    n_d = top.height
    out += [f"Defensemen, 2022 (n = {n_d}): largest rank gains and losses from adding TR.", "",
            _table(top.with_columns((pl.col("rank_before") - pl.col("rank_after")).alias("rank_gain")).sort("rank_gain", descending=True).head(8).select("player_name", "team", "total_gar_all_ridge", "total_gar_all_tr", "rank_before", "rank_after").to_dicts(), 1), "",
            _table(top.with_columns((pl.col("rank_before") - pl.col("rank_after")).alias("rank_gain")).sort("rank_gain").head(6).select("player_name", "team", "total_gar_all_ridge", "total_gar_all_tr", "rank_before", "rank_after").to_dicts(), 1), ""]
    mo = top.filter(pl.col("player_name") == "Josh Morrissey")
    if mo.height:
        out += [f"Morrissey among {n_d} defensemen, 2022: rank {mo['rank_before'][0]} before, {mo['rank_after'][0]} with TR.", ""]
    _ = np
    nar = ROOT / "reports" / "transition" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "transition" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
