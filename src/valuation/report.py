"""reports/valuation/integration_report.md from the val_* metrics and the combined table (methodology §8-§9)."""

from __future__ import annotations

import json

import polars as pl

from src.config import ROOT, load_config


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _table(rows: list[dict], nd: int = 3) -> str:
    cols = list(dict.fromkeys(k for r in rows for k in r))  # union of keys: rows may carry different columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_f(r.get(c), nd) for c in cols) + " |")
    return "\n".join(lines)


def run() -> None:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    gpw = json.loads((met / "val_gpw.json").read_text(encoding="utf-8"))
    com = json.loads((met / "val_combine.json").read_text(encoding="utf-8"))
    t = pl.read_parquet(proc / "player_value_season.parquet")
    last = t["season"].max()
    vc = cfg.valuation
    out = [
        "# Integrated player value — goals and wins above replacement (5v5)",
        "",
        "Components: possession (PM-EPA, shrunk), defense (DPM valuation rate, refit with a replacement group), finishing (FIN), turnover cost (TO), faceoffs (FO) and penalties (PEN). "
        "Spec: [finishing methodology §8-§9](../../docs/finishing/methodology.md) and [components methodology](../../docs/components/methodology.md). All values are 5v5, in goals unless labeled wins; "
        "intervals combine the component standard deviations assuming independence.",
        "",
    ]
    out += [
        "## Goals per win (derived from the data)",
        "",
        f"**{gpw['goals_per_win']:.2f} goals per win** (95% CI {gpw['ci95'][0]:.2f}–{gpw['ci95'][1]:.2f}; R² {gpw['r2']:.2f}; {gpw['n_team_seasons']} team-seasons, 2017-18 to 2025-26). "
        "OLS through the origin of (wins − GP/2) on goal differential (shootouts excluded).",
        "",
        _table([{"season": s, "goals per win": v} for s, v in gpw["per_season_goals_per_win"].items()], 2),
        "",
    ]
    out += [
        f"## Replacement level (top {vc.regular_forwards} F / {vc.regular_defense} D per team are regulars; the rest are the replacement pool)",
        "",
        "Per-60 replacement rates by season and position (positive = better than the regulars' mean for defense and possession):",
        "",
        _table(
            [
                {
                    "season": r["season"],
                    "PM-EPA F": r["pm_rep_F"],
                    "PM-EPA D": r["pm_rep_D"],
                    "DPM F": r["dpm_rep_F"],
                    "DPM D": r["dpm_rep_D"],
                }
                for r in com["replacement_rates"]
            ],
            4,
        ),
        "",
        "The DPM rates come from the refit with unpenalized fringe-count covariates, so they are not biased toward zero by ridge shrinkage. "
        "Finishing replacement is the fringe pool's mean shooter shift from the fitted model.",
        "",
        f"**The DPM replacement level is pooled over all seasons (the default).** The per-season estimate is noisy: across seasons it has an SD of {com['dpm_pooled_replacement']['sd_across_seasons_F']:.3f} (F) and "
        f"{com['dpm_pooled_replacement']['sd_across_seasons_D']:.3f} (D) xGA/60, i.e. about {com['dpm_pooled_replacement']['goals_per_1000min_per_sd_F']:.1f} / "
        f"{com['dpm_pooled_replacement']['goals_per_1000min_per_sd_D']:.1f} goals per 1,000 minutes of common shift for every player of that position, and it flips sign for forwards in some seasons. "
        f"The default (`def_gar`, `total_gar`) therefore uses the pooled effect (F {com['dpm_pooled_replacement']['F']:.3f}, D {com['dpm_pooled_replacement']['D']:.3f} per 60); "
        "the pre-registered per-season version is kept as `def_gar_perseason` / `total_gar_perseason`. The table above shows the per-season rates for reference only.",
        "",
    ]
    out += [
        "## Scale checks",
        "",
        f"SD of goals above replacement among qualified skaters (≥ {vc.min_toi:.0f} 5v5 min): "
        + ", ".join(f"{k} {v:.2f}" for k, v in com["component_sd_qualified"].items())
        + ".",
        "",
        "Correlation between components (qualified): "
        + ", ".join(f"{k} {v:.2f}" for k, v in com["component_corr_qualified"].items())
        + ".",
        "",
        "### Team reconciliation",
        "",
        f"{com['reconciliation']['n_team_seasons']} team-seasons: 5v5 goal differential regressed on the sum of player goals above replacement (by primary team). "
        "A slope near 1 means the components are on the goal scale. **Slopes are reported, not used to rescale.**",
        "",
        _table(
            [
                {"regressor": k, "slope": com["reconciliation"][k]["slope"], "r": com["reconciliation"][k]["r"]}
                for k in ("total_gar", "total_gar_perseason", "pm_gar", "def_gar", "fin_gar", "to_gar", "fo_gar")
                if k in com["reconciliation"]
            ],
            3,
        ),
        "",
        "Joint regressions (slope ± SE), 5v5 components only (penalties are realized on special teams and are checked against special-teams goal differential below):",
        "",
        _table(
            [
                {
                    "regression": name,
                    "R²": r["r2"],
                    **{
                        c.replace("_raw_gar", "")
                        .replace("to_gar_orig", "to (original)")
                        .replace("_gar", ""): f"{s['slope']:.2f} ± {s['se']:.2f}"
                        for c, s in r["slopes"].items()
                    },
                }
                for name, r in (
                    ("three components (PM, DPM, FIN)", com["reconciliation"]["joint_core"]),
                    ("five 5v5 components, shrunk", com["reconciliation"]["joint"]),
                    (
                        "five 5v5 components, possession and turnover unshrunk",
                        com["reconciliation"].get("joint_unshrunk"),
                    ),
                    (
                        "five 5v5 components, original (not role-adjusted) turnover instead of the default TO-RA",
                        com["reconciliation"].get("joint_to_orig"),
                    ),
                )
                if r is not None
            ],
            3,
        ),
        "",
        f"SD of team 5v5 goal differential {com['reconciliation']['sd_team_5v5_gd']:.1f}; SD of summed player value {com['reconciliation']['sd_sum_total_gar']:.1f}.",
        "",
    ]
    if "penalty_vs_special_teams_gd" in com["reconciliation"]:
        pst = com["reconciliation"]["penalty_vs_special_teams_gd"]
        out += [
            f"Penalty check against special-teams goal differential (non-5v5 goals): slope {pst['slope']:.2f} ± {pst['se']:.2f}, R² {pst['r2']:.3f} (special-teams GD SD {pst['sd_st_gd']:.1f}).",
            "",
        ]
    out += [
        "## Sensitivity to the regular/fringe cutoff",
        "",
        _table(
            [
                {
                    "cutoff": k,
                    "corr with baseline total": v["corr_total_gar"],
                    "rank corr": v["rank_corr_spearman"],
                    "mean shift (goals per 1000 min)": v["mean_shift_goals_per_1000min"],
                }
                for k, v in com["sensitivity"].items()
            ],
            3,
        ),
        "",
    ]
    cols = [
        "player_name",
        "team",
        "position",
        "toi_5v5",
        "pm_gar",
        "def_gar",
        "fin_gar",
        "to_gar",
        "fo_gar",
        "pen_gar",
        "total_gar",
        "total_gar_lo",
        "total_gar_hi",
        "total_war",
    ]
    q = t.filter((pl.col("season") == last) & (pl.col("toi_5v5") >= vc.min_toi)).sort("total_gar", descending=True)
    out += [
        f"## Total value, {last}-{str(last + 1)[2:]} (≥ {vc.min_toi:.0f} 5v5 min)",
        "",
        "**Top 15**",
        "",
        _table(q.head(15).select(cols).to_dicts(), 2),
        "",
        "**Bottom 10**",
        "",
        _table(q.tail(10).select(cols).to_dicts(), 2),
        "",
    ]
    w = (
        t.filter(pl.col("season") >= last - 2)
        .group_by("player_id", "player_name", "position")
        .agg(
            pl.col("toi_5v5").sum(),
            pl.col("fin_attempts_5v5").sum().alias("attempts"),
            pl.col("raw_g_minus_xg_5v5").sum().alias("raw_g_minus_xg"),
            pl.col("fin_gar").sum().alias("fin_gar"),
            pl.col("fin_gaa_5v5").sum().alias("fin_gaa"),
        )
        .filter(pl.col("attempts") >= 300)
        .with_columns((pl.col("fin_gaa") - pl.col("raw_g_minus_xg")).alias("model_minus_raw"))
    )
    c3 = ["player_name", "position", "attempts", "raw_g_minus_xg", "fin_gaa", "fin_gar", "model_minus_raw"]
    out += [
        f"## Finishing vs raw G−xG, last three seasons pooled ({last - 2}-{str(last - 1)[2:]} to {last}-{str(last + 1)[2:]}; ≥ 300 5v5 attempts)",
        "",
        "`raw_g_minus_xg` = goals − xG_f; `fin_gaa` = model goals above average (shrunk, goalie-adjusted). "
        "`model_minus_raw` is what shrinkage and the goalie adjustment removed.",
        "",
        "**Best finishers (model)**",
        "",
        _table(w.sort("fin_gaa", descending=True).head(12).select(c3).to_dicts(), 2),
        "",
        "**Worst finishers (model)**",
        "",
        _table(w.sort("fin_gaa").head(12).select(c3).to_dicts(), 2),
        "",
        "**Largest raw over-performance that the model discounts most**",
        "",
        _table(w.sort("model_minus_raw").head(10).select(c3).to_dicts(), 2),
        "",
        "**Largest raw under-performance that the model discounts most**",
        "",
        _table(w.sort("model_minus_raw", descending=True).head(10).select(c3).to_dicts(), 2),
        "",
    ]
    nar = ROOT / "reports" / "valuation" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    p = ROOT / "reports" / "valuation" / "integration_report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
