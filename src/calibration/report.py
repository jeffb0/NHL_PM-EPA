"""reports/calibration/validation_report.md from cal_team.json and cal_player.json (methodology §4)."""

from __future__ import annotations

import json

import polars as pl

from src.config import ROOT, load_config

NAMES = {"pm": "possession (PM-EPA)", "def": "defense (DPM)", "fin": "finishing (FIN)", "to": "turnover cost (TO)", "fo": "faceoffs (FO)"}


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _table(rows: list[dict], nd: int = 3) -> str:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_f(r.get(c), nd) for c in cols) + " |")
    return "\n".join(lines)


def _ok(b):
    return "✅" if b else "❌"


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    t = json.loads((met / "cal_team.json").read_text(encoding="utf-8"))
    p = json.loads((met / "cal_player.json").read_text(encoding="utf-8")) if (met / "cal_player.json").exists() else None
    comps = t["comps"]
    out = ["# Joint calibration of component scales — validation report", "",
           "Spec and pre-registered rules: [methodology](../../docs/calibration/methodology.md). The calibration is estimated on the *unshrunk* components at team level "
           "(where shrinkage cannot masquerade as scale error) with a ridge prior toward 1, and applied to the shrunk player values.", ""]
    verdict = ("**The calibrated weights were adopted**" if t["adopt"] else "**No calibration was justified: all weights are set to 1**")
    out += [f"{verdict} (leave-one-season-out team-level error of the calibrated sum vs the all-ones sum: MSE {t['cv_mse_calibrated']:.1f} vs {t['cv_mse_ones']:.1f}, paired z = {t['paired_z_calibrated_vs_ones']:.2f}; adoption needs z > 2).", ""]
    out += ["## Gates", "", "| Gate | Result | Pass |", "|---|---|---|",
            "| CAL-1 unshrunk joint regression reported | see the table below | ✅ |",
            f"| CAL-2 decision rule | z = {t['paired_z_calibrated_vs_ones']:.2f} → {'adopt' if t['adopt'] else 'weights = 1'} | ✅ |"]
    if p:
        g = p["y_gd"]
        out.append(f"| CAL-3 no harm at player level (next-season within-team error, ridge-weighted variant vs uncalibrated; moot while weights stay 1) | paired z = {g['paired_z_calibrated_better']:.2f} (needs > −2) | {_ok(p['gate_CAL3_pass'])} |")
    out += [f"| CAL-4 adopted weights within [0.5, 2] | {', '.join(f'{k} {v:.2f}' for k, v in t['weights_applied'].items())} | {_ok(t['gate_CAL4_pass'])} |",
            f"| CAL-5 consistency: applied columns equal the uncalibrated ones exactly when weights are 1 | max absolute change {'0' if not t['adopt'] else 'non-zero by design'} | {_ok(t['gate_CAL5_pass'])} |", ""]

    out += ["## Team-level scale (284 team-seasons, 5v5 goal differential)", "",
            _table([{"component": NAMES[c], "shrunk slope ± SE": f"{t['shrunk_ols']['slopes'][c]['slope']:.2f} ± {t['shrunk_ols']['slopes'][c]['se']:.2f}",
                     "unshrunk slope ± SE": f"{t['unshrunk_ols']['slopes'][c]['slope']:.2f} ± {t['unshrunk_ols']['slopes'][c]['se']:.2f}",
                     "ridge weight (95% CI)": f"{t['weights_ridge'][c]:.2f} ({t['weights_ci95'][c][0]:.2f}–{t['weights_ci95'][c][1]:.2f})", "applied weight": t["weights_applied"][c]} for c in comps], 2), "",
            f"R²: shrunk {t['shrunk_ols']['r2']:.3f}, unshrunk {t['unshrunk_ols']['r2']:.3f}. Ridge λ = {t['lambda']:.3g} (chosen by leave-one-season-out CV"
            + ("; **at the edge of the grid**" if t["lambda_grid_edge"] else "") + "). "
            "DPM has no unshrunk analogue and appears with its published values in both columns. "
            f"Team goal differential on team xG differential (context for the xG-scale components): slope {t['xg_to_goals_factor']['slope']:.2f} ± {t['xg_to_goals_factor']['se']:.2f}, R² {t['xg_to_goals_factor']['r2']:.2f}.", "",
            "Cross-validated team-level squared error by λ (the last entry, ∞, is all weights = 1):", "",
            _table([{"λ": k, "MSE": v} for k, v in t["cv_mse_by_lambda"].items()][::3] + [{"λ": "inf", "MSE": t["cv_mse_by_lambda"]["inf"]}], 1), ""]

    if p:
        out += ["## Player level, next season (team-season fixed effects, teammate-context control)", "",
                f"{p['n_player_seasons']:,} player-seasons (≥ 200 5v5 minutes in t+1), {p['n_team_seasons']} team-seasons. Outcome: on-ice 5v5 differential per 60 in t+1; predictors from season t.", "",
                _table([{"outcome": "goals" if y == "y_gd" else "xG", "total": lab, "own-rate slope ± SE": f"{p[y][k]['own_slope']:.2f} ± {p[y][k]['own_se']:.2f}", "teammate slope": p[y][k]["teammate_slope"], "within-team R²": p[y][k]["r2_within"]}
                        for y in ("y_gd", "y_xgd") for k, lab in (("uncalibrated", "uncalibrated"), ("calibrated", "calibrated"))], 3), "",
                "Slopes below 1 here combine shrinkage calibration and genuine season-to-season change in talent, so they are not used to set the scale. "
                f"Calibrated vs uncalibrated total, paired by team-season: goals z = {p['y_gd']['paired_z_calibrated_better']:.2f}, xG z = {p['y_xgd']['paired_z_calibrated_better']:.2f} (positive favors the calibrated total).", "",
                "Forward slope of next-season on-ice goal differential on each season-t component (own rate; teammate context controlled):", "",
                _table([{"component": NAMES.get(c, c), "goals: slope ± SE": f"{v['slope']:.2f} ± {v['se']:.2f}", "xG: slope ± SE": (f"{p['y_xgd']['forward_component_slopes'][c]['slope']:.2f} ± {p['y_xgd']['forward_component_slopes'][c]['se']:.2f}" if c in p["y_xgd"]["forward_component_slopes"] else "—")}
                        for c, v in p["y_gd"]["forward_component_slopes"].items()], 2), ""]
    cp = proc / "player_value_calibrated.parquet"
    if cp.exists():
        c = pl.read_parquet(cp)
        q = c.filter(pl.col("toi_5v5") >= cfg.valuation.min_toi)
        sp = q.select(pl.corr(pl.col("total_gar").rank(), pl.col("total_gar_ridge").rank())).item()
        out += ["## Effect on the table", "",
                f"Applied weights: {'the ridge weights' if t['adopt'] else 'all 1 (no calibration justified), so `total_gar_cal` equals `total_gar`'}. "
                f"The ridge-weighted variant (`total_gar_ridge`, not adopted unless stated above) changes total goals above replacement by a mean absolute {t['mean_abs_change_gar_ridge_qualified']:.3f} "
                f"among qualified skaters, with a Spearman rank correlation of {sp:.4f} against `total_gar`.", ""]
        top = q.filter(pl.col("season") == q["season"].max()).with_columns((pl.col("total_gar_ridge") - pl.col("total_gar")).alias("d"))
        cols = ["player_name", "position", "pm_gar", "fin_gar", "total_gar", "total_gar_ridge", "d"]
        out += ["Largest changes under the ridge weights, latest season (qualified):", "", _table(top.sort("d", descending=True).head(6).select(cols).to_dicts(), 2), "",
                _table(top.sort("d").head(6).select(cols).to_dicts(), 2), ""]
    nar = ROOT / "reports" / "calibration" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "calibration" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
