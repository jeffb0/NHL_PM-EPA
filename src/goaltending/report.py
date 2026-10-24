"""reports/goaltending/validation_report.md from the step outputs (docs/goaltending/methodology.md §6)."""

from __future__ import annotations

import json

import polars as pl

from src.calibration.report import _ok, _table
from src.config import ROOT, load_config


def run() -> None:
    cfg = load_config()
    gc = cfg.goaltending
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    v = json.loads((met / "gk_validation.json").read_text(encoding="utf-8"))
    tune = json.loads((met / "gk_tune.json").read_text(encoding="utf-8"))
    fw = json.loads((met / "gk_forward.json").read_text(encoding="utf-8"))
    s = v["summary"]
    gk5 = (
        json.loads((met / "gk_wholemodel.json").read_text(encoding="utf-8"))
        if (met / "gk_wholemodel.json").exists()
        else None
    )
    out = [
        "# Goaltending component — validation report",
        "",
        "Spec and pre-registered rules: [methodology](../../docs/goaltending/methodology.md). Goals above replacement for goalies, all strengths: a logistic mixed model on unblocked shots with the xG offset, the shooter's fitted shift, "
        f"a season × strength effect, a persistent and a season-deviation goalie effect{' and a defending team-season effect' if v['variant'] == 'G2' else ''}, plus an unpenalized replacement effect for goalie-seasons with fewer than {gc.fringe_shots} shots.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Pass |",
        "|---|---|---|",
        f"| GK-1 goals on non-empty-net shots attributed to a goalie ≥ 99.5% | {v['GK1']['coverage']:.2%} | {_ok(v['GK1']['pass'])} |",
        f"| GK-2 year-over-year r of single-season shrunk effects (≥ {gc.min_shots_yoy} shots both seasons) ≥ {gc.yoy_min} | {v['GK2']['yoy_shrunk']['r']:.2f} (n = {v['GK2']['yoy_shrunk']['n']}); raw goals minus xG: {v['GK2']['yoy_raw_goals_minus_xg']['r']:.2f} | {_ok(v['GK2']['pass'])} |",
        f"| GK-3 beats the goalie-free baseline on test folds {gc.test_folds} (paired z > {gc.z_threshold}) | z = {fw['GK3']['z']:.2f}; with prior fringe status added (tried): z = {fw['GK3']['with_prior_fringe_status']['z']:.2f} | {_ok(fw['GK3']['pass'])} |",
        f"| GK-4 forward slope of realized goals saved per shot on the prior estimate in [{gc.slope_band[0]}, {gc.slope_band[1]}] | {fw['GK4']['slope']:.2f} ± {fw['GK4']['se']:.2f} (n = {fw['GK4']['n_goalie_seasons']}); with prior fringe status added (tried): {fw['GK4']['with_prior_fringe_status']['slope']:.2f} ± {fw['GK4']['with_prior_fringe_status']['se']:.2f} | {_ok(fw['GK4']['pass'])} |",
    ]
    if gk5:
        out.append(
            f"| GK-5 rolling goalie term beats the crude proxy at predicting next-season team goal differential (paired z > 2) | z = {gk5['B0_roll_vs_B0']['z']:.2f} (RMSE {gk5['rmse']['B0_roll']:.1f} vs {gk5['rmse']['B0']:.1f}) | {_ok(gk5['B0_roll_vs_B0']['z'] > 2)} |"
        )
    out += [
        "",
        "## Model choice and variance components",
        "",
        f"Variant chosen by the D-type rule on the {gc.validation_fold} fold: **{tune['variant']}** (G2 vs G1 held-out log-loss z = {tune['validation']['G2_vs_G1']['z']:.2f}, threshold {gc.z_threshold}). "
        f"Standard deviations (logit units): {', '.join(f'{k} {x:.3f}' for k, x in s['sd'].items())} (u persistent, d season deviation{', t defending team-season' if v['variant'] == 'G2' else ''}). "
        f"The replacement goalie saves {s['final']['replacement_saved_per_100_shots']:.2f} goals per 100 shots relative to an average one. Correlation of goalie-season values between G1 and G2: {s['final']['corr_G1_G2']:.2f}.",
        "",
        "## Forward folds (each refit on data through the training season)",
        "",
        _table(
            [
                {
                    "trained through": f["train_end"],
                    "test season": f["test_season"],
                    "log-loss baseline": f["logloss_baseline"],
                    "log-loss model": f["logloss_model"],
                    "paired z": f["z"],
                    "z, with fringe status": f["z_with_status"],
                }
                for f in fw["folds"]
            ],
            4,
        ),
        "",
        f"## Scale (goalies with at least {gc.starter_shots} shots)",
        "",
        _table(
            [
                {
                    "goalie-seasons": s["starters"],
                    "mean goals above replacement": s["mean_gar_starters"],
                    "SD goals": s["sd_gar_starters"],
                    "SD wins": s["sd_gar_starters_wins"],
                    "mean posterior SD (goals)": s["mean_sd_starters"],
                }
            ],
            2,
        ),
        "",
    ]
    gp = proc / "goalie_player_season.parquet"
    if gp.exists():
        t = pl.read_parquet(gp)
        last = t["season"].max()
        cols = ["player_name", "team", "shots", "goals", "expected_goals", "gsaa", "goalie_gar", "gsaa_sd"]
        x = t.filter((pl.col("season") == last) & (pl.col("shots") >= gc.starter_shots)).sort(
            "goalie_gar", descending=True
        )
        out += [
            f"## Leaders, {last} (≥ {gc.starter_shots} shots)",
            "",
            _table(x.head(8).select(cols).to_dicts(), 1),
            "",
            "Lowest:",
            "",
            _table(x.tail(5).select(cols).to_dicts(), 1),
            "",
        ]
    out += [
        "## Limitations",
        "",
        "* Expected goals omit screens, traffic and pre-shot movement; the team-season effect absorbs part of it.",
        "* Goalie and defense effects are weakly separated: starters face most of their team's shots.",
        "* Save quality is rare-event data: values are strongly shrunk and intervals wide; the delta-method interval does not include the replacement level's own uncertainty.",
        "* No rest, back-to-back or workload adjustment; the value is per shot faced.",
        "",
    ]
    nar = ROOT / "reports" / "goaltending" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "goaltending" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
