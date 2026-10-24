"""reports/teamcontext/validation_report.md from tc_results.json (docs/teamcontext/methodology.md §4)."""

from __future__ import annotations

import json

from src.calibration.report import _ok, _table
from src.config import ROOT, load_config


def run() -> None:
    cfg = load_config()
    r = json.loads((cfg.path(cfg.data.metrics_dir) / "tc_results.json").read_text(encoding="utf-8"))
    g, a, b, c = r["gates"], r["A"], r["B"], r["C"]
    out = [
        '# Team context ("system / chemistry") — report',
        "",
        "Spec and pre-registered gates: [methodology](../../docs/teamcontext/methodology.md). No coach data exists in the repository, so context is the team-season: coaching, system, linemates and arena are bundled and not separable here.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Pass |",
        "|---|---|---|",
        f"| TC-1 value travels less than it stays (stayer minus mover slope, total rate; 95% interval excludes 0) | {g['TC1']['stayer_minus_mover']:.2f} ({g['TC1']['ci95'][0]:.2f} to {g['TC1']['ci95'][1]:.2f}) | {_ok(g['TC1']['pass'])} |",
        f"| TC-2 movers: destination-context coefficient positive, t > 2 | {g['TC2']['coef']:.2f}, t = {g['TC2']['t']:.2f} | {_ok(g['TC2']['pass'])} |",
        f"| TC-3 context features improve the projection for movers (z > 2) and do not hurt overall (z > −2) | movers z = {g['TC3']['movers_z']:.2f}; all z = {g['TC3']['all_z']:.2f} | {_ok(g['TC3']['pass'])} |",
        f"| TC-4 team level: projections with context beat those without (z > 2) | z = {g['TC4']['z']:.2f} | {_ok(g['TC4']['pass'])} |",
        "",
        f"## A. How much of a player's value travels ({a['n_transitions']} consecutive-season pairs with ≥ 300 minutes in both; {a['n_movers']} movers)",
        "",
        _table(
            [
                {
                    "component": k,
                    "stayer slope": v["stayer_slope"],
                    "mover slope": v["mover_slope"],
                    "stayer r": v["stayer_r"],
                    "mover r": v["mover_r"],
                    "share that travels (mover / stayer slope)": v["share_that_travels"],
                    "95% interval": f"{v['share_that_travels_ci95'][0]:.2f} to {v['share_that_travels_ci95'][1]:.2f}",
                }
                for k, v in a.items()
                if isinstance(v, dict) and "stayer_slope" in v
            ],
            2,
        ),
        "",
        "### A, chain-free components (defense and special-teams RAPM re-estimated with ρ = 0; the chained prior carries a player's own coefficient across teams, which inflates how much of the published defensive value 'travels')",
        "",
        _table(
            [
                {
                    "component": k,
                    "stayer slope": v["stayer_slope"],
                    "mover slope": v["mover_slope"],
                    "share that travels": v["share_that_travels"],
                    "95% interval": f"{v['share_that_travels_ci95'][0]:.2f} to {v['share_that_travels_ci95'][1]:.2f}",
                }
                for k, v in r["A_chainfree"].items()
                if isinstance(v, dict)
                and "stayer_slope" in v
                and k in ("total", "defense", "power play", "penalty kill")
            ],
            2,
        ),
        "",
        "## B. Context regression (next-season total rate on the shrunk history, age terms and context; player-cluster SEs)",
        "",
        _table(
            [
                {
                    "group": k,
                    "rows": v["n"],
                    **{f"{c_} coefficient (t)": f"{v[c_]['coef']:.2f} ({v[c_]['t']:.1f})" for c_ in v if c_ != "n"},
                }
                for k, v in b.items()
            ],
            2,
        ),
        "",
        "## C. Do context features improve the projections? (test seasons 2020–2025, players with ≥ 300 target minutes)",
        "",
        _table(
            [
                {
                    "rows": k,
                    "MSE base": c[k]["mse"]["base"],
                    "MSE + context": c[k]["mse"]["ctx"],
                    "MSE + context + prior team GD": c[k]["mse"]["ctx_gd"],
                    "context z": c[k]["ctx_vs_base_z"],
                    "context + GD z": c[k]["ctx_gd_vs_base_z"],
                }
                for k in ("all", "movers", "stayers")
            ],
            4,
        ),
        "",
        f"Team level (M protocol, test seasons {', '.join(map(str, c['team']['test_seasons']))}):",
        "",
        _table(
            [
                {"projection": k.replace("S_", "").replace("_", " "), "team RMSE (goals)": v}
                for k, v in c["team"]["rmse"].items()
            ],
            2,
        ),
        "",
        _table(
            [
                {"comparison": k.replace("_", " "), "mean difference": v["mean_diff"], "z": v["z"]}
                for k, v in c["team"].items()
                if isinstance(v, dict) and "z" in v
            ],
            2,
        ),
        "",
    ]
    nar = ROOT / "reports" / "teamcontext" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "teamcontext" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
