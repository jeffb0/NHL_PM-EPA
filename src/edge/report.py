"""reports/edge/validation_report.md from edge_results.json (docs/edge/methodology.md §5)."""

from __future__ import annotations

import json

from src.calibration.report import _ok, _table
from src.config import ROOT, load_config

NAMES = {"max_speed": "max skating speed (mph)", "bursts60": "bursts over 20 mph per 60", "dist60": "distance skated per 60 (miles)", "shot_speed": "top shot speed (mph)", "oz_share": "offensive-zone time share"}


def run() -> None:
    cfg = load_config()
    r = json.loads((cfg.path(cfg.data.metrics_dir) / "edge_results.json").read_text(encoding="utf-8"))
    g, t = r["gates"], r["test"]
    a = t["all_five"]
    out = ["# NHL EDGE tracking features — report", "",
           "Spec and pre-registered gates: [methodology](../../docs/edge/methodology.md). EDGE season summaries (max skating speed, bursts over 20 mph, distance, top shot speed, zone time) exist for 2021–2025 only, so the projection test has two test seasons (2024, 2025) and a training window of target seasons 2022 onward.", "",
           "## Gates", "", "| Gate | Result | Pass |", "|---|---|---|",
           f"| ED-1 EDGE coverage of skater-seasons with ≥ 300 minutes (2021–2025) ≥ 90% | {g['ED1']['share']:.1%} ({r['coverage']['with_edge']:,} of {r['coverage']['skater_seasons_300min']:,}) | {_ok(g['ED1']['pass'])} |",
           f"| ED-2 max-speed age curve declines after 27 (27→33 interval excludes 0) | forwards {g['ED2']['change_F']:.2f} mph ({g['ED2']['F'][0]:.2f} to {g['ED2']['F'][1]:.2f}); defensemen {g['ED2']['change_D']:.2f} ({g['ED2']['D'][0]:.2f} to {g['ED2']['D'][1]:.2f}) | {_ok(g['ED2']['pass'])} |",
           f"| ED-3 EDGE features improve the horizon-1 projection on 2024–2025 (z > 2) | z = {g['ED3']['z']:.2f} (MSE {a['mse']:.4f} vs {a['mse_base']:.4f}; {t['n_rows']:,} player-seasons) | {_ok(g['ED3']['pass'])} |",
           f"| ED-4 80% interval coverage with EDGE features in [70%, 90%] | {g['ED4']['coverage_80']:.1%} | {_ok(g['ED4']['pass'])} |", "",
           "## Feature sets", "",
           _table([{"features": k, "MSE": t[k]["mse"], "MSE without": t[k]["mse_base"], "z vs without": t[k]["z_vs_base"], f"players 30+ (n = {t[k]['n_age30']})": t[k]["z_vs_base_age30"], "MSE 30+": t[k]["mse_age30"], "MSE 30+ without": t[k]["mse_base_age30"], "80% coverage": t[k]["coverage_80"]}
                   for k in ("all_five", "speed_only", "speed_distance")], 4), "",
           "## Does speed explain aging? Age effect on the projected rate, 27 to 34 (goals per 60), EDGE-era data", "",
           _table([{"position": k, "without EDGE": v["without_edge"], "with EDGE features": v["with_edge"]} for k, v in r["age_effect_27_to_34"].items()], 3), "",
           "## Exploratory: is a change in speed a leading indicator of next season's value? (next-season total rate on this season's rate, EDGE level and EDGE change, age terms; player-cluster t)", "",
           _table([{"feature": NAMES[k.split('|')[0]], "position": k.split('|')[1], "rows": v["n"], "level coefficient (t)": f"{v['level_coef']:.4f} ({v['level_t']:.1f})", "change coefficient (t)": f"{v['change_coef']:.4f} ({v['change_t']:.1f})"}
                   for k, v in r["leading_indicator"].items() if "|" in k], 3), "",
           "## Physical aging curves (season-centred; change from age 27 to 33, minute-weighted, 95% player-cluster interval)", "",
           _table([{"feature": NAMES[k.split('|')[0]], "position": k.split('|')[1], "27 → 33": v["change_27_to_33"], "95% interval": f"{v['change_27_to_33_ci95'][0]:.3f} to {v['change_27_to_33_ci95'][1]:.3f}", "rows": v["n"]} for k, v in r["physical_aging"].items()], 3), "",
           "## Standardized EDGE coefficients in the projection ridge (per unit of the season-and-position z-score; goals per 60)", "",
           _table([{"feature": f, "forwards": r["edge_coefficients"]["F"][f], "defensemen": r["edge_coefficients"]["D"][f]} for f in r["edge_coefficients"]["F"] if f.startswith(("z_", "edge_"))], 4), ""]
    nar = ROOT / "reports" / "edge" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "edge" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
