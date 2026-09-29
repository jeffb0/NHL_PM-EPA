"""reports/specialteams/validation_report.md from the step outputs (methodology §6)."""

from __future__ import annotations

import json

import polars as pl

from src.config import ROOT, load_config

NAMES = {"pp": "power play", "pk": "penalty kill", "4v4": "4v4", "3v3": "3v3"}


def _ok(b):
    return "✅" if b else "❌"


def _table(rows: list[dict], nd: int = 2) -> str:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    f = lambda v: "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))  # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f(r.get(c)) for c in cols) + " |" for r in rows])


def run() -> None:
    cfg = load_config()
    sc = cfg.specialteams
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    v = json.loads((met / "st_validation.json").read_text(encoding="utf-8"))
    sel = json.loads((met / "st_selection.json").read_text(encoding="utf-8"))
    stj = json.loads((met / "st_stints.json").read_text(encoding="utf-8"))
    cls = ["pp", "pk", "4v4", "3v3"]
    grp = {"pp": "adv", "pk": "adv", "4v4": "4v4", "3v3": "3v3"}
    gated = {"pp", "pk"}
    out = ["# Special-teams components — validation report", "",
           "Spec and pre-registered rules: [methodology](../../docs/specialteams/methodology.md). Four components (power play, penalty kill, 4v4, 3v3) on the goals-above-replacement scale: "
           "strength-specific ridge RAPM on shot xG plus finishing above the fringe pool. PP and PK are gated; 4v4 and 3v3 are reported (few minutes; see the limitations).", "",
           "## Gates", "", "| Gate | Result | Pass |", "|---|---|---|"]
    c1 = v["ST1"]
    out.append(f"| ST-1 coverage: shot xG assigned to a valid stint (≥ 90%) and non-5v5 skater-minutes in valid stints (≥ 85%) | xG: PP/PK {c1['xg_coverage']['adv']:.1%}, 4v4 {c1['xg_coverage']['4v4']:.1%}, 3v3 {c1['xg_coverage']['3v3']:.1%}; minutes {c1['minutes_coverage']:.1%} | "
               f"PP/PK {_ok(c1['pass_by_group']['adv'])} · 4v4 {_ok(c1['pass_by_group']['4v4'])} · 3v3 {_ok(c1['pass_by_group']['3v3'])} · minutes {_ok(c1['minutes_pass'])} |")
    out.append("| ST-2 beats the covariates-only baseline on test pairs (paired game-bootstrap z > 2) | " + ", ".join(f"{NAMES[c]} z = {v['ST2'][grp[c]]['z']:.2f}" for c in cls) + " | " + " · ".join(f"{NAMES[c]} {_ok(v['ST2'][grp[c]]['pass'])}" for c in cls) + " |")
    out.append(f"| ST-3 year-over-year r of the shrunk net rate ≥ {sc.yoy_min:.2f} for forwards and defensemen | " + "; ".join(f"{NAMES[c]} F {v['ST3'][c]['yoy']['F']['r']:.2f} / D {v['ST3'][c]['yoy']['D']['r']:.2f}" for c in cls) + " | " + " · ".join(f"{NAMES[c]} {_ok(v['ST3'][c]['pass'])}" for c in cls) + " |")
    out.append(f"| ST-4 team reconciliation slope in [{sc.slope_band[0]}, {sc.slope_band[1]}] | " + "; ".join(f"{NAMES[c]} {v['ST4'][c]['slope']:.2f} ± {v['ST4'][c]['se']:.2f}" for c in cls) + " | " + " · ".join(f"{NAMES[c]} {_ok(v['ST4'][c]['pass'])}" for c in cls) + " |")
    out.append(f"| ST-6 state finishing values sum to the finishing model's non-5v5 total | max abs diff {v['ST6']['max_abs_diff_vs_fin_gar_other']:.1e} | {_ok(v['ST6']['pass'])} |")
    fails = [f"{NAMES[c]} ST-3" for c in gated if not v["ST3"][c]["pass"]] + [f"{NAMES[c]} ST-4" for c in gated if not v["ST4"][c]["pass"]] + [f"{NAMES[c]} ST-2" for c in gated if not v["ST2"][grp[c]]["pass"]] + (["PP/PK ST-1"] if not c1["pass_by_group"]["adv"] else [])
    out += ["", ("**Gated failures (reported, not tuned away): " + ", ".join(fails) + ".**") if fails else "**All gated checks passed.**", "",
            "## Fit choices (pre-registered rules)", "",
            _table([{"model": {"adv": "PP and PK (mismatched states)", "4v4": "4v4", "3v3": "3v3"}[g], "ridge α (CV)": sel[g]["alpha"], "prior ρ": sel[g]["rho"],
                     "test MSE model": sel[g]["test"]["mse_model"], "test MSE baseline": sel[g]["test"]["mse_baseline"], "test games": sel[g]["test"]["n_games"]} for g in ("adv", "4v4", "3v3")], 3), "",
            "## Coverage", "", _table([{"group": r["grp"], "stints": r["stints"], "hours": r["hours"], "shot xG": r["xg"], "goals": r["goals"]} for r in stj["by_group"]], 1), "",
            "## Year-over-year stability (independent single-season fits)", "",
            _table([{"component": NAMES[c], "F r": v["ST3"][c]["yoy"]["F"]["r"], "F n": v["ST3"][c]["yoy"]["F"]["n"], "D r": v["ST3"][c]["yoy"]["D"]["r"], "D n": v["ST3"][c]["yoy"]["D"]["n"]} for c in cls]), "",
            "## Team reconciliation", "",
            "Team-season goal differential in the state regressed on the team's summed component value (primary team). Slopes compress toward 1 for shrunk sums, and special-teams goals are rare, so the intervals are wide.", "",
            _table([{"component": NAMES[c], "slope": v["ST4"][c]["slope"], "SE": v["ST4"][c]["se"], "r": v["ST4"][c]["r"], "team-seasons": v["ST4"][c]["n"], "SD team GD (goals)": v["ST4"][c]["sd_team_gd"], "SD summed value": v["ST4"][c]["sd_team_gar"]} for c in cls]), "",
            "## Relation to the penalty component (reported)", "",
            "Correlation of a player's component value per minute with the penalty component's drawn and taken rates (players with ≥ 100 minutes in the state). The penalty component values the extra special-teams time; these values are per minute, so a positive correlation with *drawn* reflects that the players who draw penalties are also strong in the state, not that time is counted twice.", "",
            _table([{"component": NAMES[c], "drawn/60": v["ST5"][c]["drawn"], "taken/60": v["ST5"][c]["taken"], "players": v["ST5"][c]["n"]} for c in ("pp", "pk")]), "",
            "## Scale", "",
            _table([{"component": NAMES[c], "qualified player-seasons": s["players_qualified"], "median minutes": s["median_minutes"], "SD goals": s["sd_gar"], "SD RAPM part": s["sd_rapm"], "SD finishing part": s["sd_fin"]} for c, s in v["summary"].items()]), ""]
    tab = pl.read_parquet(proc / "player_value_season.parquet") if (proc / "player_value_season.parquet").exists() else None
    if tab is not None and "st_pp_gar" in tab.columns:
        last = tab["season"].max()
        for c in ("pp", "pk"):
            x = tab.filter((pl.col("season") == last) & (pl.col(f"st_{c}_min") >= 100)).sort(f"st_{c}_gar", descending=True)
            cols = ["player_name", "team", "position", f"st_{c}_min", f"st_{c}_rapm_gar", f"st_{c}_fin_gar", f"st_{c}_gar"]
            out += [f"## Leaders, {NAMES[c]}, {last} (≥ 100 minutes)", "", _table(x.head(8).select(cols).to_dicts()), "", "Lowest:", "", _table(x.tail(5).select(cols).to_dicts()), ""]
    out += ["## Limitations", "",
            "* Shot xG only; no possession-value ledger in special-teams states.",
            "* PP units are collinear (the same five skaters share ice time), so RAPM is a regularized estimate; PP1 defensemen are the weakest-identified group (stability r below the gate).",
            "* 4v4 and 3v3 have about 7 median minutes per player-season; their values are dominated by shrinkage and finishing, their gates are not met, and they are published as requested with a low-information flag.",
            "* The finishing part carries no interval (non-5v5 posterior draws were not saved); the component's interval is the ridge bootstrap only.",
            "* Deployment is not valued: the component is per special-teams minute above replacement.", ""]
    nar = ROOT / "reports" / "specialteams" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    path = ROOT / "reports" / "specialteams" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
