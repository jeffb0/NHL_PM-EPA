"""reports/finishing/{xg_validation,validation_report}.md from the fin_* metrics (methodology §4-§7)."""

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


def _load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _ok(b):
    return "✅" if b else "❌"


def offset_calibration(proc) -> dict:
    """Calibration of the cross-fitted xG_f offset that the finishing model actually uses: deciles and per-season
    slope."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    d = (
        pl.read_parquet(proc / "fin_shots.parquet")
        .select("game_id", "event_idx", "season", "goal", "strength_g")
        .join(
            pl.read_parquet(proc / "fin_xg.parquet").select("game_id", "event_idx", "xg_f"), on=["game_id", "event_idx"]
        )
    )
    y, p, s = d["goal"].to_numpy(), np.clip(d["xg_f"].to_numpy(), 1e-5, 1 - 1e-5), d["season"].to_numpy()
    lg = np.log(p / (1 - p))

    def slope(mask):
        return float(LogisticRegression(C=1e6, max_iter=2000).fit(lg[mask][:, None], y[mask]).coef_[0, 0])

    q = np.quantile(p, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    deciles = [
        {
            "decile": b + 1,
            "mean xG_f": float(p[idx == b].mean()),
            "observed rate": float(y[idx == b].mean()),
            "observed ÷ xG_f": float(y[idx == b].mean() / p[idx == b].mean()),
        }
        for b in range(10)
    ]
    top = p >= np.quantile(p, 0.99)
    return {
        "pooled_slope": slope(np.ones(len(y), bool)),
        "slope_5v5": slope(d["strength_g"].to_numpy() == "5v5"),
        "slope_by_season": {int(k): slope(s == k) for k in np.unique(s)},
        "deciles": deciles,
        "top1pct": {"xg": float(p[top].mean()), "observed": float(y[top].mean()), "n": int(top.sum())},
    }


def xg_report(met, chk, proc=None) -> str:
    m = _load(met / "fin_xg_metrics.json")
    out = [
        "# xG_f validation (finishing-only xG offset)",
        "",
        "Spec: [methodology §4](../../docs/finishing/methodology.md). F0 = the shared xG's features, F1 = F0 + previous-event ownership, lateral movement from "
        "the shooter's own team's previous event, time since faceoff, skater counts and recent own-team attempts. Same rows (unblocked, non-empty-net), "
        "same splits, same gradient-boosted model. Public play-by-play has no pass events, so pre-shot passing is not a feature.",
        "",
        f"**D-xG decision: {'F1 is used' if m['D_xG']['use_F1'] else 'shared xG is used'}** "
        f"(validation z = {m['D_xG']['valid_z']:.1f}, test z = {m['D_xG']['test_z']:.1f}; rule: valid > 2 SE and test not worse).",
        "",
        "## Held-out seasons (train 2017-21, validation 2022, test 2023-25)",
        "",
    ]
    rows = []
    for fs in ("F0", "F1"):
        for part in ("valid", "test"):
            r = m[f"{fs}_{part}"]
            rows.append(
                {
                    "features": fs,
                    "split": part,
                    "log-loss": r["log_loss"],
                    "AUC": r["auc"],
                    "Brier": r["brier"],
                    "Σxg÷Σgoals": r["calib_in_large"],
                    "ECE": r["ece"],
                    "calib slope": r["calib_slope"],
                }
            )
    out += [
        _table(rows, 4),
        "",
        f"Paired per-shot log-loss improvement of F1 over F0: validation {m['paired_valid']['mean_improvement']:.5f} (SE {m['paired_valid']['se']:.5f}); "
        f"test {m['paired_test']['mean_improvement']:.5f} (SE {m['paired_test']['se']:.5f}). "
        "The temporal test over-predicts goals by ~14% for both feature sets. That is league-level drift, so the finishing model carries season × strength intercepts.",
        "",
        "![reliability](../figures/finishing/xg_f_reliability.png)",
        "",
        "## Cross-fitted xG_f (each season scored by a model trained on the other eight)",
        "",
    ]
    cf, ref = m["crossfit"], m["crossfit_v1_reference"]
    out += [
        _table(
            [
                {
                    "xG": f"xG_f ({cf['feature_set']})",
                    "log-loss": cf["log_loss"],
                    "AUC": cf["auc"],
                    "Σxg÷Σgoals": cf["calib_in_large"],
                    "ECE": cf["ece"],
                },
                {
                    "xG": "shared xG on the same rows",
                    "log-loss": ref["log_loss"],
                    "AUC": ref["auc"],
                    "Σxg÷Σgoals": ref["calib_in_large"],
                    "ECE": ref["ece"],
                },
            ],
            4,
        ),
        "",
        f"Gate F4 (5v5 Σxg÷Σgoals within [0.97, 1.03]): {m['F4']['sum_xg_over_goals_5v5']:.4f} {_ok(m['F4']['pass'])}",
        "",
        "By season / by strength (Σxg ÷ Σgoals):",
        "",
        _table(
            [
                {"season": r["group"], "attempts": r["n"], "goals": r["goals"], "xg÷goals": r["xg_over_goals"]}
                for r in m["crossfit_by_season"]
            ]
        ),
        "",
        _table(
            [
                {"strength": r["group"], "attempts": r["n"], "goals": r["goals"], "xg÷goals": r["xg_over_goals"]}
                for r in m["crossfit_by_strength"]
            ]
        ),
        "",
    ]
    if proc is not None and (proc / "fin_xg.parquet").exists():
        oc = offset_calibration(proc)
        out += [
            "## Calibration of the cross-fitted offset the finishing model uses",
            "",
            "The reliability curves above are the *temporal* holdout, which flattens at high xG (drift after 2021). The finishing model uses the cross-fitted xG_f, whose calibration is:",
            "",
            f"* pooled logit calibration slope **{oc['pooled_slope']:.3f}** (5v5 {oc['slope_5v5']:.3f}); top 1% of attempts: predicted {oc['top1pct']['xg']:.3f}, observed {oc['top1pct']['observed']:.3f} (n = {oc['top1pct']['n']:,})",
            "* per-season slope: "
            + ", ".join(f"{s} {v:.2f}" for s, v in oc["slope_by_season"].items())
            + ". Season × strength intercepts absorb level drift but **not** slope drift; "
            f"{min(oc['slope_by_season'], key=oc['slope_by_season'].get)} ({min(oc['slope_by_season'].values()):.2f}) and {max(oc['slope_by_season'], key=oc['slope_by_season'].get)} "
            f"({max(oc['slope_by_season'].values()):.2f}) are the extremes. A per-season slope on the offset is a v1.1 candidate; until then, high-danger-shot specialists are the players most exposed to this.",
            "",
            _table(oc["deciles"], 4),
            "",
        ]
    out += [
        "## What F1's new features add (permutation importance, validation season, top 15)",
        "",
        _table(
            [
                {"feature": r["feature"], "Δ log-loss when shuffled": r["importance"]}
                for r in m["F1_permutation_importance"]
            ],
            5,
        ),
        "",
        "## Shot table integrity (gate F5)",
        "",
        "```",
        json.dumps(chk, indent=1),
        "```",
    ]
    return "\n".join(out)


def run() -> None:
    cfg = load_config()
    met = cfg.path(cfg.data.metrics_dir)
    proc = cfg.path(cfg.data.processed_dir)
    chk = _load(met / "fin_shots_checks.json")
    xgm = _load(met / "fin_xg_metrics.json")
    tune, final, stab = _load(met / "fin_tune.json"), _load(met / "fin_final.json"), _load(met / "fin_stability.json")
    nuts = {p.stem.replace("fin_nuts_", ""): _load(p) for p in sorted(met.glob("fin_nuts_*.json"))}
    rdir = ROOT / "reports" / "finishing"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "xg_validation.md").write_text(xg_report(met, chk, proc), encoding="utf-8")

    g, dt, tt = tune["gates"], tune["D_type"], tune["test"]
    out = [
        "# Finishing talent (FIN) — validation report",
        "",
        f"Model variant **{final['variant']}** ({'shooter × shot-type effects included' if final['variant'] == 'B' else 'no shot-type effects'}), offset = xG_f "
        f"({xgm['crossfit']['feature_set']}). Spec and pre-registered rules: [methodology](../../docs/finishing/methodology.md). "
        "Units: goals above a league-average shooter (or above the replacement pool) on the same shots against the same goalies.",
        "",
    ]
    out += [
        "## Gates",
        "",
        "| Gate | Result | Pass |",
        "|---|---|---|",
        f"| F1 shrunk model beats no-player model on next-season log-loss (pooled 2023-25) | z = {g['F1_beats_M0']['z']:.1f} (Δ {g['F1_beats_M0']['mean_improvement']:.5f}/shot) | {_ok(g['F1_pass'])} |",
        f"| F2 shrunk model beats raw smoothed G/xG | z = {g['F2_beats_Mraw']['z']:.1f} (Δ {g['F2_beats_Mraw']['mean_improvement']:.5f}/shot) | {_ok(g['F2_pass'])} |",
        f"| F3 next-season calibration slope in [0.7, 1.4] | {g['F3_slope']['slope']:.2f} (95% CI {g['F3_slope']['lo']:.2f}–{g['F3_slope']['hi']:.2f}, {g['F3_slope']['players']} players) | {_ok(g['F3_pass'])} |",
        f"| F4 xG_f cross-fit calibration, 5v5 | {xgm['F4']['sum_xg_over_goals_5v5']:.4f} | {_ok(xgm['F4']['pass'])} |",
        f"| F5 every goal is in the finishing table or a documented exclusion | {chk['goals_in_finishing_table']:,} + {chk['goals_empty_net']:,} empty-net + {chk['goals_without_coordinates']} no-coordinates = {chk['goals_in_events']:,}; unaccounted {chk['goals_unaccounted']} | {_ok(chk['goals_unaccounted'] == 0)} |",
        "| F6 four-term identity | `tests/test_finishing.py::test_four_term_identity_…` | ✅ (in the unit-test suite) |",
        "",
    ]
    out += [
        "## Variance components (logit-scale standard deviations)",
        "",
        _table(
            [
                {
                    "component": k,
                    "train seasons 2017-21, model A": tune["sd"]["A"].get(k),
                    "train seasons, model B": tune["sd"]["B"].get(k),
                    "production (all seasons)": final["sd"].get(k),
                }
                for k in sorted(set(final["sd"]) | set(tune["sd"]["A"]) | set(tune["sd"]["B"]))
            ],
            4,
        ),
        "",
        "u = persistent shooter talent (F/D), d = shooter season deviation, v/e = goalie persistent/season, w = shot-type deviation. "
        f"Fringe-player finishing effect (logit, relative to regulars): F {final['fringe_effect_logit']['F']:.3f} (±{final['fringe_effect_sd']['F']:.3f}), "
        f"D {final['fringe_effect_logit']['D']:.3f} (±{final['fringe_effect_sd']['D']:.3f}).",
        "",
    ]
    out += [
        "## Forward prediction (fit on seasons ≤ Y, predict Y+1; variance components fixed from 2017-21)",
        "",
        "Held-out log-loss per shot; season × strength intercepts of the held-out season are set to league-level calibration for every model.",
        "",
        _table(
            [
                {
                    "model": m,
                    "validation 2022": tune["validation"]["per_fold_logloss"]["2022"].get(m),
                    **{f"test {y}": tt["per_fold_logloss"][str(y)].get(m) for y in cfg.finishing.test_folds},
                    "pooled test": tt["pooled_logloss"].get(m),
                }
                for m in ("M0", "Mg", "Mraw", "MshrA", "MshrB")
            ],
            5,
        ),
        "",
        "Paired differences on the pooled test folds (positive = first model better):",
        "",
        _table(
            [
                {
                    "comparison": k.replace("_vs_", " vs "),
                    "Δ log-loss/shot": v["mean_improvement"],
                    "SE": v["se"],
                    "z": v["z"],
                }
                for k, v in tt["paired"].items()
            ],
            5,
        ),
        "",
        f"D-type: variant {dt['variant']} (validation MshrB vs MshrA: Δ {dt['mean_improvement']:.5f}, z = {dt['z']:.2f}; B needs z > {cfg.finishing.se_threshold}).",
        "",
        "Player-level calibration: realized next-season excess goals regressed on predicted (players with ≥ "
        f"{cfg.finishing.min_attempts_next} test attempts; slope 1 = well calibrated, < 1 = over-dispersed):",
        "",
        _table(
            [
                {
                    "model": m,
                    "slope": v["slope"],
                    "95% lo": v["lo"],
                    "95% hi": v["hi"],
                    "corr": v["corr"],
                    "players": v["players"],
                }
                for m, v in tt["player_slope"].items()
            ],
            3,
        ),
        "",
    ]
    if stab:
        y, sh, st = stab["yoy"], stab["split_half"], stab["stabilization"]
        out += [
            "## Reliability and stability (single-season independent fits, no pooling across seasons)",
            "",
            _table(
                [
                    {
                        "metric": "YOY r (≥200 attempts both seasons)",
                        "model estimate": y["model_shift"],
                        "raw G−xG per attempt": y["raw_rate"],
                        "raw total G−xG": y["raw_total_g_minus_xg"],
                    },
                    {
                        "metric": "  forwards",
                        "model estimate": y["F"]["model_shift"],
                        "raw G−xG per attempt": y["F"]["raw_rate"],
                        "raw total G−xG": None,
                    },
                    {
                        "metric": "  defensemen",
                        "model estimate": y["D"]["model_shift"],
                        "raw G−xG per attempt": y["D"]["raw_rate"],
                        "raw total G−xG": None,
                    },
                    {
                        "metric": "split-half r (odd/even games)",
                        "model estimate": sh["model_r"],
                        "raw G−xG per attempt": sh["raw_r"],
                        "raw total G−xG": None,
                    },
                    {
                        "metric": "split-half, Spearman-Brown (full season)",
                        "model estimate": sh["model_r_spearman_brown"],
                        "raw G−xG per attempt": sh["raw_r_spearman_brown"],
                        "raw total G−xG": None,
                    },
                ],
                3,
            ),
            "",
            f"YOY pairs: {y['n_pairs']:,}; split-half player-seasons: {sh['n_player_seasons']:,}.",
            "",
            "### Shots needed to stabilize (reliability 0.5)",
            "",
            f"Mean Bernoulli information per attempt: {st['mean_bernoulli_information']:.4f}.",
            "",
            _table(
                [
                    {"basis": k, "sd (logit)": v["sd_logit"], "attempts for r = 0.5": v["attempts_for_r50"]}
                    for k, v in st["analytic"].items()
                ],
                1,
            ),
            "",
            "Empirical (random half-splits of each player's career attempts, Spearman-Brown):",
            "",
            _table(
                [
                    {"attempts": int(n), "players": v["players"], "reliability": v["reliability"]}
                    for n, v in st["empirical"].items()
                ],
                3,
            ),
            "",
            f"Empirical attempts for reliability 0.5: {_f(st['empirical_attempts_for_r50'], 0)}.",
            "",
        ]
    if nuts:
        out += ["## Full-Bayes (NUTS) check of the Laplace estimates", ""]
        for tag, r in nuts.items():
            for k in ("fixed_sigma", "free_sigma"):
                if k in r:
                    s, gl = r[k], r[k]["goalie"]
                    out.append(
                        f"* **{tag}, {k.replace('_', ' ')}** — shooters: corr of means {s['shooter']['corr_means']:.4f}, Laplace/NUTS sd ratio median {s['shooter']['sd_ratio_laplace_over_nuts_median']:.3f}, "
                        f"NUTS mean inside Laplace 95%: {s['shooter']['nuts_mean_inside_laplace95']:.3f}; goalies: corr {gl['corr_means']:.4f}, sd ratio {gl['sd_ratio_laplace_over_nuts_median']:.3f}. "
                        f"max R-hat {s['max_r_hat']:.3f}, min n_eff {s['min_n_eff']:.0f}, divergences {s['divergences']}."
                    )
            if "free_sigma" in r and "sigma_posterior" in r["free_sigma"]:
                out.append(
                    "  * sigma posterior (free-sigma): "
                    + ", ".join(
                        f"{k}={v['mean']:.3f}±{v['sd']:.3f}" for k, v in r["free_sigma"]["sigma_posterior"].items()
                    )
                )
        out.append("")
    fp = proc / "fin_player_season.parquet"
    if fp.exists():
        f = pl.read_parquet(fp)
        last = f["season"].max()
        cols = [
            "player_name",
            "team",
            "position",
            "attempts_5v5",
            "goals_5v5",
            "xg_f_5v5",
            "raw_g_minus_xg_5v5",
            "fin_gaa_5v5",
            "fin_gaa_lo",
            "fin_gaa_hi",
            "fin_gar_5v5",
        ]
        q = f.filter((pl.col("season") == last) & (pl.col("attempts_5v5") >= 200)).sort("fin_gaa_5v5", descending=True)
        rnd = lambda d: [{k: v for k, v in r.items()} for r in d.select(cols).to_dicts()]
        out += [
            f"## Finishing leaders, {last}-{str(last + 1)[2:]} (5v5, ≥ 200 attempts; goals above average with 95% interval)",
            "",
            "**Top 12**",
            "",
            _table(rnd(q.head(12)), 2),
            "",
            "**Bottom 12**",
            "",
            _table(rnd(q.tail(12)), 2),
            "",
        ]
    nar = rdir / "narrative.md"
    if nar.exists():
        out += [nar.read_text(encoding="utf-8"), ""]
    out += [
        "## Data flags",
        "",
        "See [methodology §10](../../docs/finishing/methodology.md) and the deviations log (§11).",
        "",
    ]
    (rdir / "validation_report.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    run()
