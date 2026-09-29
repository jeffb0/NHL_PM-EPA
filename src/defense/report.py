"""reports/defense/validation_report.md from the dpm_* metrics (methodology §6 gates)."""

from __future__ import annotations

import json

import polars as pl

from src.config import ROOT, load_config
from src.defense.stints import COMPONENTS


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        lines.append("| " + " | ".join(_f(v, 3) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    j = {k: json.loads((met / f"dpm_{k}.json").read_text()) for k in ("stints", "alphas", "selection", "validation")}
    st, al, sel, val = j["stints"], j["alphas"], j["selection"], j["validation"]
    stab, rep = val["stability"], val["reported"]
    model, rho = sel["D1_model"], sel["D2"]["rho"]
    tab = pl.read_parquet(proc / "dpm_player_season.parquet")
    last = max(cfg.data.seasons)

    out = ["# Defensive PM-EPA (DPM) — Validation Report", "",
           f"Headline model **{model}**, prior ρ = **{rho}**. Spec and pre-registered rules: "
           "[docs/defense/methodology.md](../../docs/defense/methodology.md). Units: xG against per 60 5v5 minutes; "
           "positive = fewer xG against than an average skater.", ""]

    out += ["## Gates", "", "| Gate | Result | Pass |", "|---|---|---|"]
    g1 = sel["G1"]
    out.append(f"| G1 headline beats B0 on test pairs | ΔMSE {g1['diff_vs_B0']:.4f} (SE {g1['se']:.4f}) | {'✅' if g1['pass'] else '❌'} |")
    y = stab["yoy_dpm"]
    out.append(f"| G2 YOY ≥ 0.25 (season-centred, ≥{cfg.defense.min_toi:.0f} min) | F {y['F']['r']:.2f} · D {y['D']['r']:.2f} | {'✅' if stab['G2']['pass'] else '❌'} |")
    x = stab["yoy_xga_rapm"]
    if model == "M0":  # the headline at rho = 0 *is* plain xGA RAPM, so the comparison is an identity
        out.append(f"| G3 YOY ≥ plain xGA RAPM | trivially met: with M0 the ρ = 0 headline is xGA RAPM (F {x['F']['r']:.2f} · D {x['D']['r']:.2f}) | ➖ |")
    else:
        out.append(f"| G3 YOY ≥ plain xGA RAPM | xGA RAPM F {x['F']['r']:.2f} · D {x['D']['r']:.2f} | {'✅' if stab['G3']['pass'] else '❌'} |")
    out.append(f"| G4 component coefficients sum to xGA RAPM | max abs diff {al['G4_max_abs_diff']:.1e} | {'✅' if al['G4_max_abs_diff'] < 1e-8 else '❌'} |")
    out.append(f"| G5 ledger → stint assignment | {st['G5_assignment_rate']:.4f} of 5v5 ledger value | {'✅' if st['G5_assignment_rate'] >= 0.98 else '❌'} |")
    out.append("")

    out += ["## Data", "", _table(pl.DataFrame(st["by_season"]).select("season", "stints", "hours", "xg", "goals")), ""]

    out += ["## Ridge penalties (game-grouped CV, train seasons)", "",
            _table(pl.DataFrame({"target": list(al["chosen"]), "alpha": [float(v) for v in al["chosen"].values()]})), ""]

    out += ["## Model selection (D1, D2) and held-out test (G1)", "",
            "Minute-weighted MSE of next-season stint xGA/60. Lower is better. B0 = league average defense.", ""]
    rows = []
    for m in ["B0", "M0", "M1", "M2", "M3"]:
        rows.append({"model": m, "validation": sel["validation"]["mse"].get(m), "test": sel["test"]["mse"].get(m),
                     "test Δ vs B0": sel["test"]["diff"].get(f"{m}-B0") if m != "B0" else None,
                     "SE": sel["test"]["diff_se"].get(f"{m}-B0") if m != "B0" else None,
                     "test goals": sel["test_goals"]["mse"].get(m)})
    out += [_table(pl.DataFrame(rows)), ""]
    out += ["M3 stacking weights: " + ", ".join(f"{c} {w:.2f}" for c, w in sel["M3_weights"].items()), "",
            "D2 (prior ρ, validation MSE): " + ", ".join(f"ρ={k}: {v:.5f}" for k, v in sel["D2"]["mse"].items()), "",
            "Test pairs separately: " + "; ".join(f"{p}: " + ", ".join(f"{m} {v:.4f}" for m, v in d.items()) for p, d in sel["test_by_pair"].items()), ""]

    out += ["## Stability (independent ρ = 0 fits)", ""]
    srows = [{"metric": "DPM (headline)", "F": stab["yoy_dpm"]["F"]["r"], "D": stab["yoy_dpm"]["D"]["r"]},
             {"metric": "xGA RAPM", "F": stab["yoy_xga_rapm"]["F"]["r"], "D": stab["yoy_xga_rapm"]["D"]["r"]},
             {"metric": "DPM, team changers", "F": stab["yoy_dpm_team_changers"]["F"]["r"], "D": stab["yoy_dpm_team_changers"]["D"]["r"]},
             {"metric": "xGA RAPM, team changers", "F": stab["yoy_xga_rapm_team_changers"]["F"]["r"], "D": stab["yoy_xga_rapm_team_changers"]["D"]["r"]},
             {"metric": "xGF RAPM (offense, same fit)", "F": stab["yoy_xgf_rapm"]["F"]["r"], "D": stab["yoy_xgf_rapm"]["D"]["r"]}]
    srows += [{"metric": f"component: {c}", "F": stab["yoy_components"][c]["F"]["r"], "D": stab["yoy_components"][c]["D"]["r"]} for c in COMPONENTS]
    out += ["Season-centred YOY r:", "", _table(pl.DataFrame(srows)), ""]
    sh = stab["split_half"]
    out += ["Split-half r (odd vs even games):", "",
            _table(pl.DataFrame([{"season": int(s), "F DPM": v["F"]["dpm"], "F xGA RAPM": v["F"]["xga_rapm"],
                                  "D DPM": v["D"]["dpm"], "D xGA RAPM": v["D"]["xga_rapm"]} for s, v in sh.items()])), ""]

    out += ["## Reported (not gated)", "",
            "| check | F | D |", "|---|---|---|"]
    for k, lab in [("r_with_pm_epa", "r with PM-EPA/60 (overlap)"), ("r_with_xgf_rapm", "r with xGF RAPM"),
                   ("r_with_onice_xga60", "r with −(raw on-ice xGA/60)")]:
        if k in rep:
            out.append(f"| {lab} | {_f(rep[k].get('F'), 2)} | {_f(rep[k].get('D'), 2)} |")
    out += ["", f"Team level (season-centred): r(Σ TOI-weighted skater DPM, −team 5v5 xGA/60) = {rep['team_r_dpm_vs_xga60']:.2f}", ""]

    out += [f"## Face validity, {last}-{str(last + 1)[2:]} (≥{cfg.defense.min_toi:.0f} 5v5 min; inspected, never tuned)", ""]
    cols = ["player_name", "team", "toi_5v5", "dpm_per60", "dpm_lo", "dpm_hi", "xga_rapm_per60"]
    for g in ("D", "F"):
        q = tab.filter((pl.col("season") == last) & pl.col("qualified") & (pl.col("pos_group") == g)).sort("dpm_per60", descending=True)
        out += [f"**{'Defense' if g == 'D' else 'Forwards'}: top 15**", "", _table(q.head(15).select(cols)), "",
                f"**{'Defense' if g == 'D' else 'Forwards'}: bottom 10**", "", _table(q.tail(10).select(cols)), ""]
    path = ROOT / "reports" / "defense" / "validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
