"""Validation report for the game-level model (docs/gamemodel/methodology.md)."""

from __future__ import annotations

import json

from src.config import load_config

LABEL = {
    "G0": "G0 home-ice constant",
    "G1": "G1 team rating on goals",
    "G2": "G2 team rating on blended margin",
    "P": "P player ratings, updated in season",
    "P_pre": "P-pre preseason player prior only",
    "P_team": "P-team preseason prior + team-level update",
    "P_prev": "P-prev previous game's lineup",
}
GATE = {
    "GM1": "P beats G0 (z > 2)",
    "GM2": "P beats the better team rating (z > 2)",
    "GM3": "in-season updating helps: P beats P-pre (z > 2)",
    "GM4": "player attribution helps: P beats P-team (z > 2)",
    "GM5": "calibration slope in [0.8, 1.2]",
}


def run() -> str:
    cfg = load_config()
    s = json.loads((cfg.path(cfg.data.metrics_dir) / "game_model.json").read_text())
    L = [
        "# Game-level model — validation report",
        "",
        f"Test seasons 2021–2025, {s['n_test_games']} games; hyperparameters tuned on 2019–2020 only and frozen; the logistic step is fitted on earlier seasons. "
        f"Best team-rating benchmark: **{LABEL[s['best_team_benchmark']]}**.",
        "",
        "## Gates",
        "",
        "| gate | rule | statistic | result |",
        "|---|---|---|---|",
    ]
    for k, v in s["gates"].items():
        stat = f"z = {v['z']:.2f}" if "z" in v else f"slope {v['slope']:.2f}"
        L.append(f"| {k} | {GATE[k]} | {stat} | {'PASS' if v['pass'] else 'FAIL'} |")
    L += [
        "",
        "## Scores (lower log loss and Brier are better)",
        "",
        "| model | log loss | Brier | accuracy | calibration slope |",
        "|---|---|---|---|---|",
    ]
    for k in ("G0", "G1", "G2", "P_pre", "P_team", "P_prev", "P"):
        cs = s["calibration_slope"].get(k)
        L.append(
            f"| {LABEL[k]} | {s['logloss'][k]:.4f} | {s['brier'][k]:.4f} | {s['accuracy'][k]:.1%} | {'' if cs is None else f'{cs:.2f}'} |"
        )
    L += [
        "",
        "## Paired comparisons (z > 0 favors the first model; second z clusters by date)",
        "",
        "| comparison | log-loss difference | z | z (date clusters) |",
        "|---|---|---|---|",
    ]
    L += [
        f"| {k.replace('_vs_', ' vs ').replace('_', '-')} | {v['mean_diff']:.4f} | {v['z']:.2f} | {v['z_cluster_date']:.2f} |"
        for k, v in s["paired"].items()
    ]
    L += ["", "## Tuned hyperparameters (2019–2020)", "", "| model | values |", "|---|---|"]
    L += [
        f"| {k} | {', '.join(f'{a} = {b:g}' for a, b in v.items() if a != 'logloss_tune')} |"
        for k, v in s["tuned"].items()
    ]
    L += [
        "",
        "## Log loss by season",
        "",
        "| season | G0 | G1 | G2 | P-pre | P-team | P |",
        "|---|---|---|---|---|---|---|",
    ]
    L += [
        f"| {t} | {v['G0']:.4f} | {v['G1']:.4f} | {v['G2']:.4f} | {v['P_pre']:.4f} | {v['P_team']:.4f} | {v['P']:.4f} |"
        for t, v in s["logloss_by_season"].items()
    ]
    m = s.get("market")
    L += [
        "",
        "## Market comparison",
        "",
        (
            "No odds file supplied (`data/external/odds.csv`); no claim about the betting market is made."
            if not m
            else f"{m['n_games']} games with closing odds: market log loss {m['logloss_market']:.4f}, P {m['logloss_P']:.4f} (P vs market z = {m['P_vs_market']['z']:.2f})."
        ),
    ]
    text = "\n".join(L) + "\n"
    out = cfg.path("reports/gamemodel")
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    run()
