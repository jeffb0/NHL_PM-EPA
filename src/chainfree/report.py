"""reports/chainfree/validation_report.md from cf_values.json, wm_validation.json, proj_validation(_chainfree).json,
cf_diagnostics.json."""

from __future__ import annotations

import json

from src.calibration.report import _ok, _table
from src.config import ROOT, load_config


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)

    def load(n):
        return json.loads((met / n).read_text(encoding="utf-8"))

    v, wm, pj, pc, dg = (
        load("cf_values.json"),
        load("wm_validation.json")["chainfree"],
        load("proj_validation.json"),
        load("proj_validation_chainfree.json"),
        load("cf_diagnostics.json"),
    )
    pr_c, pr_k = pc["revised"], pj["revised"]
    out = [
        "# Chain-free defense and special teams — validation report",
        "",
        "Spec and pre-registered gates: [methodology](../../docs/chainfree/methodology.md). The defensive component and the special-teams RAPM parts re-estimated with ρ = 0 (each season fitted independently) in place of the chained prior, "
        "then evaluated against real results. The published defaults are unchanged.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Pass |",
        "|---|---|---|",
        f"| CF-1 chain-free rolling total not significantly worse than the chained one at the team level (z > −2) | z = {wm['CF1']['z_rho0_better']:.2f} (RMSE {wm['rmse']['M_rho0']:.1f} vs {wm['rmse']['M_rr']:.1f}) | {_ok(wm['CF1']['pass'])} |",
        f"| CF-2 with chain-free components PR-6 passes (projected beat last season's rates, z > 2) | z = {pr_c['gates']['PR6']['z']:.2f} (RMSE {pr_c['team']['rmse']['projected']:.1f} vs {pr_c['team']['rmse']['rolling_last_season']:.1f}) | {_ok(pr_c['gates']['PR6']['pass'])} |",
        f"| CF-3 projected rates still beat Marcel at the player level (z > 2) | z = {pr_c['gates']['PR2']['z']:.2f} | {_ok(pr_c['gates']['PR2']['pass'])} |",
        "",
        "## Persistence by construction (year-over-year correlation, season-centred)",
        "",
        _table(
            [
                {
                    "component": k,
                    "forwards, chained ρ": v["yoy"][f"{k}_chained"]["F"]["r"],
                    "forwards, ρ = 0": v["yoy"][f"{k}_rho0"]["F"]["r"],
                    "defensemen, chained ρ": v["yoy"][f"{k}_chained"]["D"]["r"],
                    "defensemen, ρ = 0": v["yoy"][f"{k}_rho0"]["D"]["r"],
                }
                for k in ("defense", "pp", "pk")
            ],
            2,
        ),
        "",
        f"Correlation between the chained and chain-free values among qualified skaters: defense {v['corr_chained_vs_rho0']['defense']:.2f}, power play {v['corr_chained_vs_rho0']['pp']:.2f}, penalty kill {v['corr_chained_vs_rho0']['pk']:.2f}. "
        f"Standard deviation of goals above replacement: defense {v['sd_qualified']['def_chained']:.2f} → {v['sd_qualified']['def_rho0']:.2f}, power play {v['sd_qualified']['pp_chained']:.2f} → {v['sd_qualified']['pp_rho0']:.2f}, penalty kill {v['sd_qualified']['pk_chained']:.2f} → {v['sd_qualified']['pk_rho0']:.2f}.",
        "",
        "## Forward slopes (next-season deployment-adjusted on-ice goal differential per 60 on the component's rate; state shares controlled)",
        "",
        _table([{"component": k, "slope": s["slope"], "SE": s["se"]} for k, s in v["forward_slopes"].items()], 2),
        "",
        "## Whole model (M protocol, test seasons 2020–2025)",
        "",
        _table(
            [
                {"total": "chain-free rolling-refit", "RMSE (goals)": wm["rmse"]["M_rho0"]},
                {"total": "chained rolling-refit", "RMSE (goals)": wm["rmse"]["M_rr"]},
                {"total": "chain-free rolling-refit + prior team GD", "RMSE (goals)": wm["rmse"]["M+B1_rho0"]},
                {"total": "chained rolling-refit + prior team GD", "RMSE (goals)": wm["rmse"]["M+B1_rr"]},
            ],
            2,
        ),
        "",
        _table(
            [
                {"comparison": k.replace("_", " "), "mean difference": x["mean_diff"], "z": x["z"]}
                for k, x in wm.items()
                if isinstance(x, dict) and "z" in x
            ],
            2,
        ),
        "",
        "## Projection layer on chain-free component rates",
        "",
        _table(
            [
                {
                    "gate": g,
                    "chained": (
                        f"z = {pr_k['gates'][g]['z']:.2f}"
                        if "z" in pr_k["gates"][g]
                        else (
                            f"{pr_k['gates'][g]['slope']:.2f}"
                            if "slope" in pr_k["gates"][g]
                            else f"{pr_k['gates'][g]['coverage_80']:.1%}"
                        )
                    ),
                    "chain-free": (
                        f"z = {pr_c['gates'][g]['z']:.2f}"
                        if "z" in pr_c["gates"][g]
                        else (
                            f"{pr_c['gates'][g]['slope']:.2f}"
                            if "slope" in pr_c["gates"][g]
                            else f"{pr_c['gates'][g]['coverage_80']:.1%}"
                        )
                    ),
                }
                for g in pr_c["gates"]
            ]
        ),
        "",
        f"Results-target variant (exploratory): team RMSE {pc['results_target']['team']['rmse']['results_target']:.1f} against {pc['results_target']['team']['rmse']['rolling_last_season']:.1f} for last season's chain-free rates (z = {pc['results_target']['team']['results_vs_rolling']['z']:.2f}).",
        "",
        "## Why projections lose to last season's values at the team level (exploratory diagnostics)",
        "",
        "Team-level M protocol, test seasons 2021–2025; each row replaces the roster-weighted sum.",
        "",
        _table(
            [
                {"sum of": k, "team RMSE": x["rmse"], "z vs last season": x["z_vs_rolling"]}
                for k, x in dg["history_length"].items()
            ],
            2,
        ),
        "",
        _table(
            [
                {
                    "extra ballast M0 (minutes) in the projection features": k,
                    "team RMSE": x["team_rmse"],
                    "z vs last season": x["z_vs_rolling"],
                }
                for k, x in dg["ballast_sweep"].items()
            ],
            2,
        ),
        "",
        f"Within-season correlation of the summed values with team goal differential: projected {dg['within_season']['corr_projected']:.2f}, last season {dg['within_season']['corr_last_season']:.2f}; "
        f"SD of the sums {dg['within_season']['sd_projected']:.1f} against {dg['within_season']['sd_last_season']:.1f} goals.",
        "",
    ]
    nar = ROOT / "reports" / "chainfree" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "chainfree" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
