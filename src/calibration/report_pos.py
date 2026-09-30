"""reports/calibration/validation_report_pos.md: the position-specific weights (methodology §5d)."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.calibration.report import _ok, _table
from src.config import ROOT, load_config

NAMES = {"pm": "possession", "def": "defense", "fin": "finishing", "to": "turnover", "pen": "penalties", "pp": "power play", "pk": "penalty kill"}


def _corrs(t: pl.DataFrame, proc, col: str) -> dict:
    ps = pl.read_parquet(proc / "player_season.parquet").select(pl.col("player_id").cast(t["player_id"].dtype), pl.col("season").cast(t["season"].dtype), "gp", "toi_ev", "goals", "a1", "a2")
    j = t.join(ps, on=["player_id", "season"]).filter((pl.col("toi_5v5") >= 500) & (pl.col("gp") >= 40)).with_columns(
        (pl.col("toi_ev") / pl.col("gp")).alias("toi_pg"), ((pl.col("goals") + pl.col("a1") + pl.col("a2")) / pl.col("toi_ev") * 60).alias("p60"), (pl.col(col) / pl.col("toi_5v5") * 60).alias("g60"))
    return {g: {"toi": float(np.corrcoef(s["g60"], s["toi_pg"])[0, 1]), "points": float(np.corrcoef(s["g60"], s["p60"])[0, 1])} for g in ("D", "F") for s in [j.filter(pl.col("pos_group") == g)]}


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    if not (met / "cal_team_all_pos.json").exists():
        return
    t = json.loads((met / "cal_team_all_pos.json").read_text(encoding="utf-8"))
    p = json.loads((met / "cal_player_all_pos.json").read_text(encoding="utf-8"))
    d = pl.read_parquet(proc / "player_value_calibrated_all_pos.parquet")
    adj = p["deployment_adjusted"]["y_gd_adj"]
    out = ["# Position-specific weights — validation report", "",
           "Spec and pre-registered rules: [methodology §5d](../../docs/calibration/methodology.md). The all-strength identity with the seven main components entering separately for forwards and defensemen "
           "(18 weights with faceoffs, 4v4, 3v3 and the goalie pooled). **Exploratory:** the specification was chosen after four were tried on the same data (§5d).", "",
           f"**{'The position-specific weights were adopted' if t['adopt'] else 'Not adopted'}**: held-out team-level error {t['cv_mse_position']:.1f}, against {t['cv_mse_pooled_ridge']:.1f} for the pooled ridge and {t['cv_mse_ones']:.1f} for all ones "
           f"(paired z = {t['paired_z_vs_pooled_ridge']:.2f} against pooled, {t['paired_z_vs_ones']:.2f} against ones; both must exceed 2).", "",
           "## Gates", "", "| Gate | Result | Pass |", "|---|---|---|",
           f"| D-cal-pos adoption: z > 2 against all ones and against the pooled ridge | {t['paired_z_vs_ones']:.2f} and {t['paired_z_vs_pooled_ridge']:.2f} | {_ok(t['adopt'])} |",
           f"| CAL-3 no harm at player level, deployment-adjusted (position-weighted vs uncalibrated total) | goals z = {adj['paired_z_calibrated_better']:.2f}, xG z = {p['deployment_adjusted']['y_xgd_adj']['paired_z_calibrated_better']:.2f} (needs > −2) | {_ok(p['gate_CAL3_pass_adjusted'])} |",
           f"| CAL-5 identical to the uncalibrated total when weights are 1 | {'weights adopted, non-zero by design' if t['adopt'] else 'max change 0'} | {_ok(t['gate_CAL5_pass'])} |", "",
           "## Weights (ridge toward 1, λ = {:.3g}; applied weights are clipped to [0, 3])".format(t["lambda"]), "",
           _table([{"component": NAMES[c], "forwards weight (95% CI)": f"{t['weights_ridge'][c + '_F']:.2f} ({t['weights_ci95'][c + '_F'][0]:.2f} to {t['weights_ci95'][c + '_F'][1]:.2f})",
                    "defensemen weight (95% CI)": f"{t['weights_ridge'][c + '_D']:.2f} ({t['weights_ci95'][c + '_D'][0]:.2f} to {t['weights_ci95'][c + '_D'][1]:.2f})",
                    "F unshrunk slope": f"{t['unshrunk_ols']['slopes'][c + '_F']['slope']:.2f} ± {t['unshrunk_ols']['slopes'][c + '_F']['se']:.2f}", "D unshrunk slope": f"{t['unshrunk_ols']['slopes'][c + '_D']['slope']:.2f} ± {t['unshrunk_ols']['slopes'][c + '_D']['se']:.2f}"} for c in NAMES], 2), "",
           "Pooled: " + ", ".join(f"{c} {t['weights_ridge'][c]:.2f}" for c in ("fo", "4v4", "3v3", "gk")) + ". DPM and the special-teams RAPM parts have no unshrunk analogue, so their slopes also carry shrinkage compression.", ""]
    # named players and position parity
    q = d.filter(pl.col("season") == 2022)
    rows = []
    for n in ("Josh Morrissey", "Mason Appleton", "Adam Fox", "Jaccob Slavin", "Mark Scheifele", "Kyle Connor"):
        r = q.filter(pl.col("player_name") == n)
        if r.height:
            rows.append({"player": n, "position": r["position"][0], "GAR before": r["total_gar_all"][0], "GAR position-weighted": r["total_gar_all_ridge"][0], "change": r["total_gar_all_ridge"][0] - r["total_gar_all"][0], "± (GAR sd)": r["total_gar_all_sd"][0]})
    out += ["## Named players, 2022", "", _table(rows, 2), ""]
    qq = d.filter(pl.col("toi_5v5") >= 500)
    par = qq.group_by("pos_group").agg(pl.len().alias("player-seasons"), pl.col("total_gar_all").mean().alias("mean GAR before"), pl.col("total_gar_all_ridge").mean().alias("mean GAR position-weighted"),
                                       pl.col("total_gar_all").std().alias("SD before"), pl.col("total_gar_all_ridge").std().alias("SD position-weighted")).sort("pos_group")
    out += ["## Defensemen against forwards (qualified skaters, 500+ 5v5 minutes)", "", _table(par.to_dicts(), 2), ""]
    cb, ca = _corrs(d, proc, "total_gar_all"), _corrs(d, proc, "total_gar_all_ridge")
    out += ["External-validity look (not a criterion): correlation of total GAR per 60 with ice time per game and with points per 60.", "",
            _table([{"position": g, "TOI/game before": cb[g]["toi"], "TOI/game after": ca[g]["toi"], "points/60 before": cb[g]["points"], "points/60 after": ca[g]["points"]} for g in ("D", "F")], 3), ""]
    top = q.filter((pl.col("pos_group") == "D") & (pl.col("toi_5v5") >= 500)).with_columns(pl.col("total_gar_all").rank("ordinal", descending=True).alias("rank_before"), pl.col("total_gar_all_ridge").rank("ordinal", descending=True).alias("rank_after"))
    out += ["Defensemen, 2022: largest gains and losses in rank among the 210 qualified.", "",
            _table(top.with_columns((pl.col("rank_before") - pl.col("rank_after")).alias("rank_gain")).sort("rank_gain", descending=True).head(8).select("player_name", "team", "total_gar_all", "total_gar_all_ridge", "rank_before", "rank_after").to_dicts(), 1), "",
            _table(top.with_columns((pl.col("rank_before") - pl.col("rank_after")).alias("rank_gain")).sort("rank_gain").head(6).select("player_name", "team", "total_gar_all", "total_gar_all_ridge", "rank_before", "rank_after").to_dicts(), 1), ""]
    nar = ROOT / "reports" / "calibration" / "narrative_pos.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "calibration" / "validation_report_pos.md"
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
