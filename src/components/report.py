"""reports/components/validation_report.md from the to_/fo_/pen_ component metrics (methodology §7 gates)."""

from __future__ import annotations

import json

import polars as pl

from src.config import ROOT, load_config


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _table(rows: list[dict], nd: int = 3) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_f(r[c], nd) for c in cols) + " |")
    return "\n".join(lines)


def _ok(b):
    return "✅" if b else "❌"


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    load = lambda n: json.loads((met / n).read_text(encoding="utf-8"))  # noqa: E731
    to, fo, pen = load("to_component.json"), load("fo_component.json"), load("pen_component.json")
    thr = cfg.finishing.se_threshold
    rec = json.loads((met / "val_combine.json").read_text(encoding="utf-8"))["reconciliation"] if (met / "val_combine.json").exists() else None
    out = ["# Turnover cost, faceoffs and penalties (5v5) — validation report", "",
           "Spec and pre-registered gates: [methodology](../../docs/components/methodology.md). Values are goals (xG scale) above the fringe replacement pool, "
           "empirical-Bayes shrunk where the data are counts or rates.", ""]

    acc = to["accounting"]
    fw = fo["forward_test"]
    out += ["## Gates", "", "| Gate | Result | Pass |", "|---|---|---|",
            f"| TO-1 turnover accounting: −ΣT2 ÷ (DPM-excluded turnover-born + unobserved starts) | {acc['ratio_to_turnover_born_plus_unobserved']:.2f} (against turnover-born only: {acc['ratio_to_turnover_born_only']:.2f}) | {_ok(to['gate_TO1_pass'])} |"]
    if rec:
        s = rec["joint"]["slopes"].get("to_gar")
        if s:
            out.append(f"| TO-2 team-level slope of the turnover component (joint regression, shrunk) | {s['slope']:.2f} ± {s['se']:.2f} (unshrunk: {rec['joint_unshrunk']['slopes']['to_raw_gar']['slope']:.2f}) | {_ok(0.5 <= s['slope'] <= 1.5)} |")
    ra = json.loads((met / "to_ra_component.json").read_text(encoding="utf-8")) if (met / "to_ra_component.json").exists() else None
    out += [f"| TO-3 season-centred YOY of shrunk TO per 60 (≥ {cfg.components.min_toi:.0f} min) | F {to['yoy_shrunk']['F']['r']:.2f} · D {to['yoy_shrunk']['D']['r']:.2f} | {_ok(to['gate_TO3_pass'])} |",
            f"| TO-4 shrunk beats raw at predicting next-season TO per 60 | z = {to['TO4_shrunk_vs_raw_next_season_mse']['z']:.1f} | {_ok(to['gate_TO4_pass'])} |",
            f"| FO-1 shrunk taker skill beats no-taker on next-season draws | z = {fw['paired']['Mshr_vs_M0']['z']:.1f} | {_ok(fo['gate_FO1_pass'])} |",
            f"| FO-2 shrunk beats raw win-rate log-odds | z = {fw['paired']['Mshr_vs_Mraw']['z']:.1f} | {_ok(fo['gate_FO2_pass'])} |",
            f"| FO-3 calibration slope of realized next-season excess wins | {fw['slope']['Mshr']['slope']:.2f} ({fw['slope']['Mshr']['lo']:.2f}–{fw['slope']['Mshr']['hi']:.2f}) | {_ok(fo['gate_FO3_pass'])} |",
            f"| FO-4 draw swing positive at every dot | min dot value {min(fo['dot_values'].values()):.4f} | {_ok(fo['gate_FO4_pass'])} |",
            f"| PEN-1 cost per minor above 0 | {pen['cost_minor']:.3f} xG ({pen['class_costs']['2']['cost'] - 1.96 * pen['class_costs']['2']['se']:.3f}–{pen['class_costs']['2']['cost'] + 1.96 * pen['class_costs']['2']['se']:.3f}) | {_ok(pen['gate_PEN1_pass'])} |",
            f"| PEN-2 shrunk beats raw at predicting next-season counts (Poisson deviance) | drawn z = {pen['drawn_forward']['z']:.1f} · taken z = {pen['taken_forward']['z']:.1f} | {_ok(pen['gate_PEN2_drawn_pass'] and pen['gate_PEN2_taken_pass'])} |",
            f"| PEN-3 season-centred YOY (shrunk), drawn and taken per 60 | drawn {pen['yoy_drawn']['shrunk']['all']['r']:.2f} · taken {pen['yoy_taken']['shrunk']['all']['r']:.2f} | {_ok(pen['gate_PEN3_pass'])} |",
            ]
    if ra:
        out += [f"| RA-1 role adjustment removes the involvement gradient: absolute r (residual, involvement) | F {ra['corr_resid_vs_inv']['F']:.2f} · D {ra['corr_resid_vs_inv']['D']:.2f} | {_ok(ra['gate_RA1_pass'])} |",
                f"| RA-2 season-centred YOY of shrunk TO-RA per 60 | F {ra['yoy_shrunk']['F']['r']:.2f} · D {ra['yoy_shrunk']['D']['r']:.2f} | {_ok(ra['gate_RA2_pass'])} |",
                f"| RA-3 shrunk TO-RA beats raw at predicting next-season residual | z = {ra['RA3_shrunk_vs_raw_next_season']['z']:.1f} | {_ok(ra['gate_RA3_pass'])} |"]
    out += ["", f"Gate thresholds use z > {thr}. TO-2 is evaluated after integration (`python -m src.valuation.run`).", ""]

    out += ["## Turnover cost", "",
            f"Accounting by slice (5v5, usable games): T2 = {acc['sum_T2']:,.0f} xG charged; DPM-excluded turnover-born starts {acc['ledger_turnover_born_starts']:,.0f}; unobserved starts {acc['ledger_unobserved_starts']:,.0f}. "
            "T2 also charges giveaways and losses whose opponent start was not observed after the turnover, so the valuation defense column excludes `start_unobserved` as well.", "",
            "### Arena (scorer) effect on the per-60 turnover charge, visitors only", "",
            "League mean charge is about −0.20 per 60. Excess arena SD = spread of arena means beyond sampling noise:", "",
            _table([{"season": int(s), "excess arena SD": v["arena_sd_excess"], "noise SD": v["noise_sd"]} for s, v in to["arena_effects"].items()], 4), "",
            "Arena effects of 15–20% of the mean charge in 2017–2023 disappear in 2024-25, when the NHL's recording changed. They are subtracted (shrunk) from every game at that arena. "
            "Stayer/changer stability, season-centred YOY r:", "",
            _table([{"version": k.replace("_", " "), "stayers": v["stayers"]["r"], "changers": v["changers"]["r"], "gap": v["stayers"]["r"] - v["changers"]["r"]} for k, v in to["team_changer_yoy"].items()]), "",
            f"Direct (giveaway) share of the absolute charge: {to['direct_share_of_total_abs']:.2f}. SD of goals above replacement among qualified skaters: {to['sd_gar_qualified']:.2f}. "
            "The remaining stayer/changer gap is consistent with real style and system persistence; it is not fully arena bias.", ""]

    if ra:
        out += ["### Role-adjusted turnover variant (TO-RA)", "",
                "Compares each player's arena-adjusted charge with players of similar *observed successful* puck involvement (carry, move_start, move_end, shot_attempt credit rows per 60) in the same season and position, "
                f"then shrinks and measures above the fringe residual. A variant beside TO (`to_ra_gar`), not a replacement. Involvement explains only {ra['r2_mean_by_position']['F']:.0%} (forwards) and {ra['r2_mean_by_position']['D']:.0%} (defensemen) of the variance in charge, "
                f"so rankings move little (Spearman with TO: F {ra['spearman_TO_vs_TORA_qualified']['F']:.2f}, D {ra['spearman_TO_vs_TORA_qualified']['D']:.2f}; SD of goals {ra['sd_gar_qualified']['TO']:.2f} → {ra['sd_gar_qualified']['TO_RA']:.2f}).", "",
                f"Correlation of the raw per-60 charge with involvement is F {ra['corr_rate_vs_inv']['F']:.2f}, D {ra['corr_rate_vs_inv']['D']:.2f} (more involvement, larger charge: the puck-dominance bias) and of the residual F {ra['corr_resid_vs_inv']['F']:.2f}, D {ra['corr_resid_vs_inv']['D']:.2f}. "
                f"YOY shrunk: TO-RA F {ra['yoy_shrunk']['F']['r']:.2f} · D {ra['yoy_shrunk']['D']['r']:.2f} vs TO F {ra['yoy_original_TO_shrunk']['F']['r']:.2f} · D {ra['yoy_original_TO_shrunk']['D']['r']:.2f}.", "",
                "Mean goals above replacement by involvement quintile (Q5 = most involved):", "",
                _table([{"involvement quintile": k, "TO": v["to_gar"], "TO-RA": v["to_ra_gar"], "player-seasons": v["n"]} for k, v in ra["mean_gar_by_involvement_quintile"].items()], 3), ""]
        rp = proc / "to_ra_player_season.parquet"
        if rp.exists():
            nm = pl.read_parquet(proc / "players.parquet").select("player_id", (pl.col("first_name") + " " + pl.col("last_name")).alias("player"))
            w = (pl.read_parquet(rp).filter(pl.col("toi_ev") >= cfg.components.min_toi).group_by("player_id").agg(pl.col("to_gar").sum(), pl.col("to_ra_gar").sum(), pl.col("inv60").mean(), pl.col("toi_ev").sum())
                 .filter(pl.col("toi_ev") >= 3000).join(nm, on="player_id"))
            cols = ["player", "inv60", "to_gar", "to_ra_gar"]
            out += ["Most charged skaters, pooled over qualifying seasons (≥ 3,000 5v5 minutes; goals above replacement):", "", _table(w.sort("to_gar").head(10).select(cols).to_dicts(), 1), "",
                    "Most charged under TO-RA:", "", _table(w.sort("to_ra_gar").head(8).select(cols).to_dicts(), 1), "",
                    "Least charged under TO-RA:", "", _table(w.sort("to_ra_gar", descending=True).head(6).select(cols).to_dicts(), 1), ""]
    out += ["## Faceoffs", "",
            f"{fo['n_draws']:,} five-on-five draws with both takers known; home taker wins {fo['home_win_rate']:.3f}. Mean swing S (value of winning vs losing a draw) = {fo['mean_swing']:.4f} xG. "
            "Start value V by dot (winner's frame): " + ", ".join(f"({k}) {v:.4f}" for k, v in fo["dot_values"].items()) + ".", "",
            f"Taker skill SD (logit): centers {fo['sd_production']['u_C']:.2f} persistent + {fo['sd_production']['d_C']:.2f} season; others {fo['sd_production']['u_O']:.2f} + {fo['sd_production']['d_O']:.2f}. "
            f"Side (handedness × dot side) coefficient {fo['side_coefficient']:.3f}; fringe-taker effect {fo['fringe_effect_logit']:.3f} logit.", "",
            "Forward prediction (fit ≤ Y, predict Y+1; variance components from 2017-21):", "",
            _table([{"model": m, **{f"test {y}": fw["per_fold_logloss"][str(y)][m] for y in cfg.finishing.test_folds}, "pooled": fw["pooled_logloss"][m]} for m in ("M0", "Mraw", "Mshr")], 5), "",
            _table([{"comparison": k.replace("_vs_", " vs "), "Δ log-loss/draw": v["mean"], "z": v["z"]} for k, v in fw["paired"].items()], 5), "",
            "Calibration slopes of realized next-season excess wins on predicted (taker-seasons with ≥ " f"{cfg.components.min_draws_next} draws): "
            + "; ".join(f"{m} {v['slope']:.2f} ({v['lo']:.2f}–{v['hi']:.2f}, n = {v['taker_seasons']})" for m, v in fw["slope"].items()) + ".", ""]
    fp = proc / "fo_player_season.parquet"
    if fp.exists():
        names = pl.read_parquet(proc / "players.parquet").select("player_id", (pl.col("first_name") + " " + pl.col("last_name")).alias("player_name"))
        e = pl.read_parquet(fp).join(names, on="player_id", how="left").filter(pl.col("fo_draws") >= 500).with_columns((pl.col("fo_wins") / pl.col("fo_draws")).alias("win_pct"))
        cols = ["player_name", "season", "fo_draws", "win_pct", "fo_gaa", "fo_gar", "fo_gar_sd"]
        out += ["Best and worst taker-seasons (≥ 500 draws; goals above replacement):", "", _table(e.sort("fo_gar", descending=True).head(8).select(cols).to_dicts(), 3), "", _table(e.sort("fo_gar").head(6).select(cols).to_dicts(), 3), ""]

    out += ["## Penalties", "",
            _table([{"class (min)": k, "penalties": v["n"], "net xG for the advantaged team": v["cost"], "SE": v["se"], "mean advantage (s)": v["mean_adv_s"]} for k, v in pen["class_costs"].items()], 3), "",
            "Majors are almost all coincident (fighting) and are excluded, so the 5-minute class rests on very few penalties and has almost no effect. "
            f"League rates per 60 (minor-equivalents): drawn {pen['league_rates_per60']['drawn_minor_equiv']:.3f}, taken {pen['league_rates_per60']['taken_minor_equiv']:.3f}. "
            f"SD of goals above replacement among qualified skaters: {pen['sd_pen_gar_qualified']:.2f}.", "",
            _table([{"rate": k, "YOY raw": pen[f"yoy_{k}"]["raw"]["all"]["r"], "YOY shrunk": pen[f"yoy_{k}"]["shrunk"]["all"]["r"], "next-season deviance raw": pen[f"{k}_forward"]["dev_raw"],
                     "deviance shrunk": pen[f"{k}_forward"]["dev_shrunk"]} for k in ("drawn", "taken")], 3), ""]
    if rec and "penalty_vs_special_teams_gd" in rec:
        p = rec["penalty_vs_special_teams_gd"]
        out += [f"Team-level check: penalty value is realized on special teams, not at 5v5. Regressing team special-teams goal differential (all non-5v5 goals) on the summed penalty value gives a slope "
                f"of {p['slope']:.2f} ± {p['se']:.2f} (R² {p['r2']:.3f}; special-teams GD SD {p['sd_st_gd']:.1f}, dominated by power-play and penalty-kill skill and luck). "
                "Against 5v5 goal differential the penalty component has no relationship, as expected.", ""]
    nar = ROOT / "reports" / "components" / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    p = ROOT / "reports" / "components" / "validation_report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
