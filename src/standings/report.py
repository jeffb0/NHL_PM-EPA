"""Validation report for the standings backtest (docs/standings/methodology.md)."""

from __future__ import annotations

import json

from src.config import load_config

LABEL = {
    "B0": "B0 league mean",
    "B1": "B1 prior-season points",
    "B2": "B2 prior-season goal differential",
    "B3": "B3 prior-season xG differential",
    "M1": "M1 roster model (no fitted slope)",
    "M2": "M2 roster model, fitted slopes",
    "M3": "M3 roster model + prior goal differential",
    "M1_R": "variant R: last season's rates (no projection)",
    "M3_R": "variant R + prior goal differential",
    "P_res": "variant P-res: results-target projection",
    "M1_oracle": "M1 with the actual season roster and minutes (diagnostic)",
}
GATE = {
    "ST1": "M1 beats B0 (z > 2)",
    "ST2": "M1 not worse than best history baseline (z > −2)",
    "ST3": "blend M3 beats best history baseline (z > 2)",
    "ST4": "calibration slope in [0.8, 1.2]",
    "ST5": "80% coverage in [70%, 90%]",
}


def run() -> str:
    cfg = load_config()
    s = json.loads((cfg.path(cfg.data.metrics_dir) / "standings_backtest.json").read_text())
    L = [
        "# Standings backtest — validation report",
        "",
        f"Test seasons {s['test_seasons'][0]}–{s['test_seasons'][-1]} ({s['n_team_seasons']} team-seasons); every model and baseline is fitted on earlier seasons only. "
        f"Error is in points per 82 games. Best history baseline: **{LABEL[s['best_history']]}**.",
        "",
        "## Gates",
        "",
        "| gate | rule | statistic | result |",
        "|---|---|---|---|",
    ]
    for k, v in s["gates"].items():
        stat = (
            f"z = {v['z']:.2f}"
            if "z" in v and "slope" not in v
            else (f"slope {v['slope']:.2f} ± {v['se']:.2f}" if "slope" in v else f"{v['coverage_80']:.0%}")
        )
        L.append(f"| {k} | {GATE[k]} | {stat} | {'PASS' if v['pass'] else 'FAIL'} |")
    L += [
        "",
        "## Error by model",
        "",
        "| model | RMSE | MAE | Spearman (mean) | top-16 overlap |",
        "|---|---|---|---|---|",
    ]
    for k in ("B0", "B1", "B2", "B3", "M1", "M2", "M3", "M1_R", "M3_R", "P_res", "M1_oracle"):
        rk = s["rank"].get(k)
        L.append(
            f"| {LABEL[k]} | {s['rmse'][k]:.2f} | {s['mae'][k]:.2f} | {rk['spearman_mean']:.2f} | {rk['top16_overlap']:.0%} |"
            if rk
            else f"| {LABEL[k]} | {s['rmse'][k]:.2f} | {s['mae'][k]:.2f} | | |"
        )
    L += ["", "## Paired comparisons (z > 0 favors the first model)", "", "| comparison | z |", "|---|---|"]
    L += [f"| {k.replace('_', ' ')} | {v:.2f} |" for k, v in s["z"].items()]
    L += ["", "## RMSE by season", "", "| season | M1 | M3 | B2 |", "|---|---|---|---|"]
    L += [f"| {t} | {v['M1']:.1f} | {v['M3']:.1f} | {v['B2']:.1f} |" for t, v in s["rmse_by_season"].items()]
    text = "\n".join(L) + "\n"
    out = cfg.path("reports/standings")
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    run()
