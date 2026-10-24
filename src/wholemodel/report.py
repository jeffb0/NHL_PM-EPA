"""reports/wholemodel/validation_report.md from wm_validation.json (methodology §5)."""

from __future__ import annotations

import json

from src.calibration.report import _ok, _table
from src.config import ROOT, load_config

MODEL_NAMES = {
    "constant": "intercept only",
    "B0": "B0 goalie proxy only",
    "B1": "B1 + prior team goal differential",
    "B2": "B2 + prior team xG differential",
    "B3": "B3 + naive roster-weighted on-ice xG",
    "M": "M + total_gar_all (headline)",
    "M+B1": "M+B1 + total and prior team goal differential",
    "M_fixed": "M with slope fixed at 1",
    "M_no_oracle": "M, previous team and minutes (no oracle roster)",
}
VAR_NAMES = {
    "calibrated": "calibrated weights (total_gar_all_cal)",
    "five_v_five_total": "5v5 total only",
    "original_turnover": "original (not role-adjusted) turnover",
    "leakage_clean": "leakage-clean total (unshrunk finishing, faceoffs, ST finishing)",
    "no_persistent_effects": "no persistent-effect components (exploratory)",
}


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    r = json.loads((met / "wm_validation.json").read_text(encoding="utf-8"))
    g = r["gates"]
    out = [
        "# Whole-model validation — report",
        "",
        "Spec and pre-registered gates: [methodology](../../docs/wholemodel/methodology.md). The question: does the published total (`total_gar_all`, weights 1) predict next-season team goal differential, and does it add information beyond simple baselines? "
        f"{r['n_test']} out-of-sample team-seasons ({r['test_seasons'][0]}–{r['test_seasons'][-1]}), expanding-window OLS, every model including the same goalie proxy.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Pass |",
        "|---|---|---|",
        f"| WM-1 M beats B0 (paired z > 2) | z = {g['WM1']['z']:.2f}; out-of-sample R² vs B0 {r['models']['M']['oos_r2_vs_B0']:.2f} | {_ok(g['WM1']['pass'])} |",
        f"| WM-2 M beats the naive roster-weighted on-ice xG baseline B3 (z > 2) | z = {g['WM2']['z']:.2f} (RMSE {r['models']['M']['rmse']:.1f} vs {r['models']['B3']['rmse']:.1f}) | {_ok(g['WM2']['pass'])} |",
        f"| WM-3 M+B1 beats B1: the total adds information beyond the team's prior results (z > 2) | z = {g['WM3']['z']:.2f} (RMSE {r['models']['M+B1']['rmse']:.1f} vs {r['models']['B1']['rmse']:.1f}) | {_ok(g['WM3']['pass'])} |",
        f"| WM-4 calibration slope of actual on the fixed-slope prediction in [0.7, 1.3] | {g['WM4']['slope']:.2f} ± {g['WM4']['se']:.2f} | {_ok(g['WM4']['pass'])} |",
        f"| WM-5 leakage-clean total beats B0 (z > 2) and differs from the headline by |z| < 2 | clean vs B0 z = {g['WM5']['clean_vs_B0_z']:.2f}; headline minus clean z = {g['WM5']['headline_minus_clean_z']:.2f} | {_ok(g['WM5']['pass'])} |",
        f"| WM-6 M better than B0 in ≥ 4 of 6 test seasons | {g['WM6']['seasons_better']} of 6 | {_ok(g['WM6']['pass'])} |",
        "",
        "## Models (out of sample)",
        "",
        _table(
            [
                {"model": MODEL_NAMES[k], "RMSE (goals)": v["rmse"], "R² vs B0": v["oos_r2_vs_B0"]}
                for k, v in r["models"].items()
            ],
            2,
        ),
        "",
        "Paired comparisons (squared-error difference, positive favors the first-named model):",
        "",
        _table(
            [
                {"comparison": k.replace("_", " "), "mean difference": v["mean_diff"], "z": v["z"]}
                for k, v in r["paired"].items()
            ],
            2,
        ),
        "",
        "## By test season",
        "",
        _table(
            [
                {"season": k, "MSE M": v["mse_M"], "MSE B0": v["mse_B0"], "MSE B1": v["mse_B1"]}
                for k, v in r["by_season"].items()
            ],
            0,
        ),
        "",
        "## Variants and drop-one (each replaces the total in M; `z vs headline` positive means the headline is better)",
        "",
        _table(
            [
                {
                    "total": VAR_NAMES.get(k, k.replace("drop_", "without ")),
                    "RMSE": v["rmse"],
                    "z vs headline": v["vs_headline"]["z"],
                    "z vs B0": v["vs_B0"]["z"],
                }
                for k, v in r["variants"].items()
            ],
            2,
        ),
        "",
    ]
    if r.get("rolling_refit"):
        rr = r["rolling_refit"]
        out += [
            "## Rolling refit of finishing and faceoffs (methodology §5b)",
            "",
            "The finishing and faceoff models, and the finishing part of the special-teams components, are refit on seasons up to each season only; the rolling-refit total replaces those parts in the headline total.",
            "",
            _table(
                [
                    {"model": "M with the rolling-refit total", "RMSE (goals)": rr["rmse"]["M_rr"]},
                    {"model": "M with the headline total", "RMSE (goals)": r["models"]["M"]["rmse"]},
                    {"model": "M+B1 with the rolling-refit total", "RMSE (goals)": rr["rmse"]["M+B1_rr"]},
                    {"model": "M+B1 with the headline total", "RMSE (goals)": r["models"]["M+B1"]["rmse"]},
                    {"model": "B1 prior team goal differential", "RMSE (goals)": r["models"]["B1"]["rmse"]},
                    {"model": "B3 naive roster-weighted on-ice xG", "RMSE (goals)": r["models"]["B3"]["rmse"]},
                ],
                2,
            ),
            "",
            _table(
                [
                    {"comparison": k.replace("_", " "), "mean difference": v["mean_diff"], "z": v["z"]}
                    for k, v in rr.items()
                    if isinstance(v, dict) and "z" in v
                ],
                2,
            ),
            "",
            f"Pre-registered rule: headline better than the rolling-refit total with z > 2 means the headline is leakage-sensitive. Headline better by z = {rr['headline_better_z']:.2f}: **{rr['verdict']}**.",
            "",
        ]
    nar = ROOT / "reports" / "wholemodel" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "wholemodel" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
