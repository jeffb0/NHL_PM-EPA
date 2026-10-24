"""reports/calibration/validation_report_all.md: the all-strength extension (methodology §5b) from cal_team_all.json and
cal_player_all.json."""

from __future__ import annotations

import json

import polars as pl

from src.calibration.report import _ok, _table
from src.config import ROOT, load_config

NAMES = {
    "pm": "possession (PM-EPA)",
    "def": "defense (DPM)",
    "fin": "finishing, 5v5 (FIN)",
    "to": "turnover cost (TO-RA)",
    "fo": "faceoffs (FO)",
    "pen": "penalties (PEN)",
    "pp": "power play",
    "pk": "penalty kill",
    "4v4": "4v4",
    "3v3": "3v3",
    "gk": "goaltending (GK)",
}


def run(gk: bool = False) -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    sfx = "_gk" if gk else ""
    if not (met / f"cal_team_all{sfx}.json").exists():
        return
    t = json.loads((met / f"cal_team_all{sfx}.json").read_text(encoding="utf-8"))
    p = (
        json.loads((met / f"cal_player_all{sfx}.json").read_text(encoding="utf-8"))
        if (met / f"cal_player_all{sfx}.json").exists()
        else None
    )
    v2 = json.loads((met / "cal_team_all.json").read_text(encoding="utf-8")) if gk else None
    comps = t["comps"]
    title = "Joint calibration with the goalie (v3)" if gk else "Joint calibration, all-strength extension"
    spec = (
        (
            "[methodology §5c](../../docs/calibration/methodology.md). Eleven components (the ten of the all-strength extension plus the team's goalie value, summed over shots faced by the defending team) against all-strength team goal differential. "
            "The v2 extension ([report](validation_report_all.md)) and the 5v5 calibration ([report](validation_report.md)) are unchanged."
        )
        if gk
        else (
            "[methodology §5b](../../docs/calibration/methodology.md). Ten components (the five 5v5 components, penalties and the four special-teams components) "
            "against all-strength team goal differential, estimated on the unshrunk versions with a ridge prior toward 1. The 5v5 calibration ([report](validation_report.md)) is unchanged."
        )
    )
    out = [f"# {title} — validation report", "", "Spec and pre-registered rules: " + spec, ""]
    verdict = (
        "**The calibrated weights were adopted** (rule of methodology §3)"
        if t["adopt"]
        else "**No calibration was justified: all weights are set to 1**"
    )
    out += [
        f"{verdict}: leave-one-season-out team-level error of the calibrated sum vs the all-ones sum, MSE {t['cv_mse_calibrated']:.1f} vs {t['cv_mse_ones']:.1f}, paired z = {t['paired_z_calibrated_vs_ones']:.2f} (adoption needs z > 2).",
        "",
    ]
    out += [
        "## Gates",
        "",
        "| Gate | Result | Pass |",
        "|---|---|---|",
        "| CAL-1 unshrunk joint regression reported | see the table below | ✅ |",
        f"| CAL-2 decision rule | z = {t['paired_z_calibrated_vs_ones']:.2f} → {'adopt' if t['adopt'] else 'weights = 1'} | ✅ |",
    ]
    if p:
        raw, adj = p["y_gd"], p["deployment_adjusted"]["y_gd_adj"]
        out += [
            f"| CAL-3 no harm at player level, as pre-registered (raw all-strength on-ice goal differential, next season) | paired z = {raw['paired_z_calibrated_better']:.2f} (needs > −2) | {_ok(p['gate_CAL3_pass'])} |",
            f"| CAL-3 no harm at player level, deployment-adjusted (methodology §7, deviation 4) | paired z = {adj['paired_z_calibrated_better']:.2f} (needs > −2) | {_ok(p['gate_CAL3_pass_adjusted'])} |",
        ]
    out += [
        f"| CAL-4 adopted weights within [0.5, 2] | {', '.join(f'{k} {v:.2f}' for k, v in t['weights_applied'].items())} | {_ok(t['gate_CAL4_pass'])} |",
        f"| CAL-5 consistency: applied columns equal the uncalibrated ones exactly when weights are 1 | {'weights adopted, so non-zero by design' if t['adopt'] else 'max absolute change 0'} | {_ok(t['gate_CAL5_pass'])} |",
        "",
        f"## Team level ({t['n_team_seasons']} team-seasons, all-strength goal differential, SD {t['sd_team_gd_all']:.1f})",
        "",
        _table(
            [
                {
                    "component": NAMES[c],
                    "shrunk slope ± SE": f"{t['shrunk_ols']['slopes'][c]['slope']:.2f} ± {t['shrunk_ols']['slopes'][c]['se']:.2f}",
                    "unshrunk slope ± SE": f"{t['unshrunk_ols']['slopes'][c]['slope']:.2f} ± {t['unshrunk_ols']['slopes'][c]['se']:.2f}",
                    "ridge weight (95% CI)": f"{t['weights_ridge'][c]:.2f} ({t['weights_ci95'][c][0]:.2f}–{t['weights_ci95'][c][1]:.2f})",
                    "applied weight": t["weights_applied"][c],
                }
                for c in comps
            ],
            2,
        ),
        "",
        f"R²: shrunk {t['shrunk_ols']['r2']:.3f}, unshrunk {t['unshrunk_ols']['r2']:.3f}. Ridge λ = {t['lambda']:.3g} (leave-one-season-out CV). DPM and the special-teams RAPM parts appear with their published values in both columns "
        "(no unshrunk analogue), so their slopes are subject to shrinkage compression.",
        "",
        "Cross-validated team-level squared error by λ (the last entry, ∞, is all weights = 1):",
        "",
        _table(
            [{"λ": k, "MSE": v} for k, v in t["cv_mse_by_lambda"].items()][::3]
            + [{"λ": "inf", "MSE": t["cv_mse_by_lambda"]["inf"]}],
            1,
        ),
        "",
    ]
    if v2:
        out += [
            "## Change from the all-strength extension without the goalie (v2)",
            "",
            f"Cross-validated team-level error with all weights at 1: {v2['cv_mse_ones']:.1f} (v2) → {t['cv_mse_ones']:.1f} (with the goalie); calibrated: {v2['cv_mse_calibrated']:.1f} → {t['cv_mse_calibrated']:.1f}. "
            f"Unshrunk R²: {v2['unshrunk_ols']['r2']:.3f} → {t['unshrunk_ols']['r2']:.3f}. Paired z for adopting weights: {v2['paired_z_calibrated_vs_ones']:.2f} → {t['paired_z_calibrated_vs_ones']:.2f}.",
            "",
            _table(
                [
                    {
                        "component": NAMES[c],
                        "v2 ridge weight": v2["weights_ridge"].get(c),
                        "v3 ridge weight (95% CI)": f"{t['weights_ridge'][c]:.2f} ({t['weights_ci95'][c][0]:.2f}–{t['weights_ci95'][c][1]:.2f})",
                    }
                    for c in comps
                ],
                2,
            ),
            "",
        ]
    if p:
        out += [
            "## Player level, next season (team-season fixed effects, teammate context)",
            "",
            f"{p['n_player_seasons']:,} player-seasons (≥ 250 modeled minutes in t+1), {p['n_team_seasons']} team-seasons. Predictors per modeled minute in season t; the outcome is on-ice goal (or xG) differential per 60 in t+1 over the 5v5 and special-teams stints. "
            "\"Calibrated\" is the ridge-weighted total (the weights adopted above). The **raw** outcome rewards power-play time and penalizes penalty-kill time whatever the player's skill; the **deployment-adjusted** outcome subtracts each stint's league mean for the side's state and adds the player's minute shares by state as controls.",
            "",
            _table(
                [
                    {
                        "outcome": lab,
                        "total": k,
                        "own-rate slope ± SE": f"{blk[y][k]['own_slope']:.2f} ± {blk[y][k]['own_se']:.2f}",
                        "teammate slope": blk[y][k]["teammate_slope"],
                        "within-team R²": blk[y][k]["r2_within"],
                    }
                    for lab, blk, ys in (
                        ("raw goals", p, ("y_gd",)),
                        ("raw xG", p, ("y_xgd",)),
                        ("adjusted goals", p["deployment_adjusted"], ("y_gd_adj",)),
                        ("adjusted xG", p["deployment_adjusted"], ("y_xgd_adj",)),
                    )
                    for y in ys
                    for k in ("uncalibrated", "calibrated")
                ],
                3,
            ),
            "",
            f"Calibrated vs uncalibrated total, paired by team-season (positive favors calibrated): raw goals z = {p['y_gd']['paired_z_calibrated_better']:.2f}, raw xG z = {p['y_xgd']['paired_z_calibrated_better']:.2f}; "
            f"adjusted goals z = {p['deployment_adjusted']['y_gd_adj']['paired_z_calibrated_better']:.2f}, adjusted xG z = {p['deployment_adjusted']['y_xgd_adj']['paired_z_calibrated_better']:.2f}.",
            "",
            "Forward slope of next-season on-ice differential on each season-t component (own rate; teammates and state shares controlled in the adjusted columns):",
            "",
            _table(
                [
                    {
                        "component": NAMES.get(c, c),
                        "raw goals": f"{v['slope']:.2f} ± {v['se']:.2f}",
                        "adjusted goals": (lambda d: f"{d['slope']:.2f} ± {d['se']:.2f}" if d else "—")(
                            p["deployment_adjusted"]["y_gd_adj"]["forward_component_slopes"].get(c)
                        ),
                        "adjusted xG": (lambda d: f"{d['slope']:.2f} ± {d['se']:.2f}" if d else "—")(
                            p["deployment_adjusted"]["y_xgd_adj"]["forward_component_slopes"].get(c)
                        ),
                    }
                    for c, v in p["y_gd"]["forward_component_slopes"].items()
                ],
                2,
            ),
            "",
        ]
    cp = proc / f"player_value_calibrated_all{sfx}.parameter".replace(".parameter", ".parquet")
    if cp.exists():
        c = pl.read_parquet(cp)
        q = c.filter(pl.col("toi_5v5") >= cfg.valuation.min_toi)
        sp = q.select(pl.corr(pl.col("total_gar_all").rank(), pl.col("total_gar_all_cal").rank())).item()
        out += [
            "## Effect on the table",
            "",
            f"Applied weights: {'the ridge weights' if t['adopt'] else 'all 1'}. `total_gar_all_cal` differs from `total_gar_all` by a mean absolute {t['mean_abs_change_gar_ridge_qualified']:.3f} goals among qualified skaters; "
            f"Spearman rank correlation {sp:.4f}. Both columns are published in `player_value_calibrated_all`.",
            "",
        ]
        top = q.filter(pl.col("season") == q["season"].max()).with_columns(
            (pl.col("total_gar_all_cal") - pl.col("total_gar_all")).alias("d")
        )
        cols = ["player_name", "position", "pm_gar", "st_pp_gar", "total_gar_all", "total_gar_all_cal", "d"]
        out += [
            "Largest changes, latest season (qualified):",
            "",
            _table(top.sort("d", descending=True).head(6).select(cols).to_dicts(), 2),
            "",
            _table(top.sort("d").head(6).select(cols).to_dicts(), 2),
            "",
        ]
    nar = ROOT / "reports" / "calibration" / f"narrative_all{sfx}.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "calibration" / f"validation_report_all{sfx}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
