"""reports/projection/validation_report.md from proj_validation.json (docs/projection/methodology.md §6)."""

from __future__ import annotations

import json

import polars as pl

from src.calibration.report import _ok, _table
from src.config import ROOT, load_config


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    R = json.loads((met / "proj_validation.json").read_text(encoding="utf-8"))
    pre, rev = R["as_preregistered"], R["revised"]
    gp, gr = pre["gates"], rev["gates"]
    h = rev["horizon"]

    fz = lambda g: f"z = {g['z']:.2f}"
    fs = lambda g: f"{g['slope']:.2f} ± {g['se']:.2f}"
    fc = lambda g: f"{g['coverage_80']:.1%}"

    def both(k, fmt):
        return fmt(gp[k]) + " → " + fmt(gr[k])

    def mark(k):
        return f"{_ok(gp[k]['pass'])} → {_ok(gr[k]['pass'])}"

    out = [
        "# Projection and aging layer — validation report",
        "",
        "Spec and pre-registered gates: [methodology](../../docs/projection/methodology.md). Weighted ridge projections of each skater's all-strength rate (goals above replacement per 60 modeled minutes) from shrunk component histories, "
        "age and body size; finishing and faceoffs enter as **rolling-refit** values so no forecast uses later information. Expanding-window evaluation on target seasons 2020–2025 (horizon 1), players with at least 300 target-season minutes, "
        "minute-weighted error, player-cluster bootstrap z.",
        "",
        f"Each cell shows the result **as pre-registered** (training rows need ≥ {pre['floor']:.0f} target-season minutes) → **revised** (≥ {rev['floor']:.0f}; methodology §9, deviation 1). Everything below the gate table is from the revised run.",
        "",
        "## Gates",
        "",
        "| Gate | As pre-registered → revised | Pass |",
        "|---|---|---|",
        f"| PR-1 beats the position-mean baseline (z > 2) | {both('PR1', fz)} | {mark('PR1')} |",
        f"| PR-2 beats Marcel (weights 5/4/3, 1,000-minute ballast, no age) (z > 2) | {both('PR2', fz)} | {mark('PR2')} |",
        f"| PR-3 the age terms improve the projection (z > 2) | {both('PR3', fz)} | {mark('PR3')} |",
        f"| PR-4 calibration slope of actual on projected rate in [0.8, 1.2] | {both('PR4', fs)} | {mark('PR4')} |",
        f"| PR-5 80% interval coverage in [70%, 90%] | {both('PR5', fc)} | {mark('PR5')} |",
        f"| PR-6 team level: projected rates beat last season's rolling-refit rates (z > 2) | {both('PR6', fz)} (RMSE {pre['team']['rmse']['projected']:.1f} vs {pre['team']['rmse']['rolling_last_season']:.1f} → {rev['team']['rmse']['projected']:.1f} vs {rev['team']['rmse']['rolling_last_season']:.1f}) | {mark('PR6')} |",
        "",
        "## Player-level accuracy by horizon (revised run)",
        "",
        _table(
            [
                {
                    "horizon (seasons ahead)": k,
                    "test seasons": f"{v['seasons'][0]}–{v['seasons'][-1]}",
                    "rows": v["n_rows"],
                    "MSE projection": v["mse_proj"],
                    "MSE position mean": v["mse_b_mean"],
                    "MSE Marcel": v["mse_b_marcel"],
                    "MSE last season": v["mse_b_last"],
                    "z vs mean": v["z_vs_mean"],
                    "z vs Marcel": v["z_vs_marcel"],
                    "z vs last": v["z_vs_last"],
                }
                for k, v in h.items()
            ],
            4,
        ),
        "",
        "## Variants (horizon 1; `z` positive means the main model is better)",
        "",
        _table(
            [
                {"variant": k, "MSE": v["mse"], "main model better, z": v["M_vs_variant_z"]}
                for k, v in rev["variants"].items()
            ],
            4,
        ),
        "",
        "## Team level (goalie proxy and intercept, plus the row's term, fitted on earlier seasons; test seasons "
        + ", ".join(map(str, rev["team"]["test_seasons"]))
        + ")",
        "",
        _table([{"model": k.replace("_", " "), "RMSE (goals)": v} for k, v in rev["team"]["rmse"].items()], 2),
        "",
        _table(
            [
                {"comparison": k.replace("_", " "), "mean difference": v["mean_diff"], "z": v["z"]}
                for k, v in rev["team"].items()
                if isinstance(v, dict) and "z" in v
            ],
            2,
        ),
        "",
        "## What persists: fitted weights on the shrunk component histories (horizon 1, per unit of shrunk rate; revised run)",
        "",
        _table(
            [
                {"feature": f, "forwards": R["component_weights"]["F"][f], "defensemen": R["component_weights"]["D"][f]}
                for f in R["component_weights"]["F"]
                if f.startswith(("rt_", "no_hist"))
            ],
            3,
        ),
        "",
        f"Minutes model: mean absolute error {rev['minutes']['mae_model']:.0f} against {rev['minutes']['mae_last_season']:.0f} for last season's minutes ({rev['minutes']['n']} player-seasons).",
        "",
    ]
    ac = ROOT / "reports" / "projection" / "aging_curves.csv"
    if ac.exists():
        a = pl.read_csv(ac)
        show = a.filter(pl.col("age").is_in([20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0]))
        out += [
            "## Aging curves (effect on projected rate per 60 relative to age 27, 95% player-cluster bootstrap interval)",
            "",
            _table(
                [
                    {
                        "age": r_["age"],
                        "position": r_["pos_group"],
                        "effect": r_["effect_per60"],
                        "lo95": r_["lo95"],
                        "hi95": r_["hi95"],
                    }
                    for r_ in show.sort("pos_group", "age").to_dicts()
                ],
                3,
            ),
            "",
        ]
    out += [
        "## Limitations",
        "",
        "* About nine seasons and shortened 2020 and 2021 seasons: few transitions, especially at the young and old ends; the aging curves are conditional on staying in the league (survivorship).",
        "* Injuries, role changes and trades are not modeled; minutes are projected conditional on playing.",
        "* Skaters only. No EDGE tracking features are used (none are in the repository); `Spec.extra` accepts any extra feature columns.",
        "",
    ]
    nar = ROOT / "reports" / "projection" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "projection" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
