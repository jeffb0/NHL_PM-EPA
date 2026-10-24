"""xG_f: finishing-only xG offset with richer pre-shot context (methodology §4, decision D-xG).

Same model family and hyperparameters as the shared xG (src/xg/model.py); trained on unblocked,
non-empty-net attempts. Feature set F0 = the shared xG's features; F1 = F0 + previous-event ownership,
lateral movement from the shooter's own team's previous event, time since faceoff, skater counts and
recent own-team attempts. Both are fitted on the same rows and splits; F1 replaces F0 only if the
pre-registered rule passes. If it does not, xG_f is the shared cross-fitted xG.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance

from src.config import load_config
from src.finishing.shots import V2_EXTRA
from src.viz.style import SERIES, plt, save
from src.xg.features import design_matrix
from src.xg.model import evaluate, make_model

log = logging.getLogger(__name__)


def design(shots: pl.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
    X, names = design_matrix(shots)
    if feature_set == "F0":
        return X, names
    extra = np.column_stack([shots[c].cast(pl.Float64).fill_null(0).to_numpy() for c in V2_EXTRA])
    return np.column_stack([X, extra]), names + V2_EXTRA


def _ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def paired(l0: np.ndarray, l1: np.ndarray) -> dict:
    """Mean per-shot log-loss improvement of model 1 over model 0 (positive = model 1 better) and its SE."""
    d = l0 - l1
    return {"mean_improvement": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n": int(len(d))}


def calibration_by(shots: pl.DataFrame, col: str, xg: np.ndarray) -> list[dict]:
    df = shots.select(col, "goal").with_columns(pl.Series("xg", xg))
    return [
        {
            "group": str(r[col]),
            "n": r["n"],
            "goals": r["goals"],
            "xg": r["xg"],
            "xg_over_goals": r["xg"] / max(r["goals"], 1),
        }
        for r in df.group_by(col)
        .agg(pl.len().alias("n"), pl.col("goal").sum().alias("goals"), pl.col("xg").sum().alias("xg"))
        .sort(col)
        .to_dicts()
    ]


def _reliability_figure(y, p0, p1, strength, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharex=False, sharey=False)
    for ax, (title, m) in zip(
        axes,
        [
            ("All non-empty-net attempts", np.ones(len(y), bool)),
            ("5v5", strength == "5v5"),
            ("Power play", strength == "PP"),
        ],
    ):
        for p, lab, c in ((p0, "F0 shared features", SERIES[1]), (p1, "F1 + context", SERIES[0])):
            q = np.quantile(p[m], np.linspace(0, 1, 16))
            idx = np.clip(np.searchsorted(q, p[m], side="right") - 1, 0, 14)
            xs = [p[m][idx == b].mean() for b in range(15)]
            ys = [y[m][idx == b].mean() for b in range(15)]
            ax.plot(xs, ys, marker="o", ms=4, color=c, label=lab)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color="#8f8e89", lw=1, ls="--")
        ax.set_title(title)
        ax.set_xlabel("predicted goal probability")
        ax.set_ylabel("observed goal rate")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "xG_f reliability on held-out seasons 2023-25 (decile-of-prediction bins)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    save(fig, path, "xg_f_reliability")


def run(force: bool = False) -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    out_path = proc / "fin_xg.parquet"
    if out_path.exists() and (met / "fin_xg_metrics.json").exists() and not force:
        log.info("fin_xg cache present; skipping (use force=True to refit)")
        return json.loads((met / "fin_xg_metrics.json").read_text())
    shots = pl.read_parquet(proc / "fin_shots.parquet")
    y = shots["goal"].to_numpy()
    season = shots["season"].to_numpy()
    strength = shots["strength_g"].to_numpy()
    tr, va, te = (np.isin(season, s) for s in (cfg.splits.train, cfg.splits.validation, cfg.splits.test))
    Xs, names = {}, {}
    for fs in ("F0", "F1"):
        Xs[fs], names[fs] = design(shots, fs)
    res: dict = {
        "n_train": int(tr.sum()),
        "n_valid": int(va.sum()),
        "n_test": int(te.sum()),
        "features": {fs: names[fs] for fs in Xs},
    }
    models, pred = {}, {}
    for fs in ("F0", "F1"):
        m = make_model("gbm", names[fs], cfg.seed).fit(Xs[fs][tr], y[tr])
        models[fs] = m
        pred[fs] = {"valid": m.predict_proba(Xs[fs][va])[:, 1], "test": m.predict_proba(Xs[fs][te])[:, 1]}
        res[f"{fs}_valid"] = evaluate(y[va], pred[fs]["valid"])
        res[f"{fs}_test"] = evaluate(y[te], pred[fs]["test"])
        log.info(
            "%s valid ll %.5f auc %.4f | test ll %.5f auc %.4f",
            fs,
            res[f"{fs}_valid"]["log_loss"],
            res[f"{fs}_valid"]["auc"],
            res[f"{fs}_test"]["log_loss"],
            res[f"{fs}_test"]["auc"],
        )
    res["paired_valid"] = paired(_ll(y[va], pred["F0"]["valid"]), _ll(y[va], pred["F1"]["valid"]))
    res["paired_test"] = paired(_ll(y[te], pred["F0"]["test"]), _ll(y[te], pred["F1"]["test"]))
    pv, pt = res["paired_valid"], res["paired_test"]
    use_v2 = bool(pv["mean_improvement"] > cfg.finishing.se_threshold * pv["se"] and pt["mean_improvement"] >= 0)
    res["D_xG"] = {
        "use_F1": use_v2,
        "valid_z": pv["mean_improvement"] / pv["se"],
        "test_z": pt["mean_improvement"] / pt["se"],
    }
    log.info(
        "D-xG: %s (valid z=%.1f, test z=%.1f)",
        "F1" if use_v2 else "shared xG",
        res["D_xG"]["valid_z"],
        res["D_xG"]["test_z"],
    )
    sub = np.where(va)[0][:40000]
    pi = permutation_importance(
        models["F1"], Xs["F1"][sub], y[sub], scoring="neg_log_loss", n_repeats=3, random_state=cfg.seed
    )
    res["F1_permutation_importance"] = sorted(
        ({"feature": n, "importance": float(v)} for n, v in zip(names["F1"], pi.importances_mean)),
        key=lambda r: -r["importance"],
    )[:15]
    res["test_calibration_by_strength"] = {
        fs: calibration_by(shots.filter(pl.Series(te)), "strength_g", pred[fs]["test"]) for fs in pred
    }
    res["test_calibration_by_type"] = {
        fs: calibration_by(shots.filter(pl.Series(te)), "shot_type_c", pred[fs]["test"]) for fs in pred
    }
    _reliability_figure(
        y[te], pred["F0"]["test"], pred["F1"]["test"], strength[te], cfg.path(cfg.data.figures_dir) / "finishing"
    )

    # ---- cross-fitted xG_f for every finishing-table shot
    if use_v2:
        xg_f = np.zeros(len(shots))
        for s in cfg.data.seasons:
            m = make_model("gbm", names["F1"], cfg.seed).fit(Xs["F1"][season != s], y[season != s])
            sel = season == s
            xg_f[sel] = m.predict_proba(Xs["F1"][sel])[:, 1]
            log.info("cross-fit xG_f season %s", s)
        fset = "F1"
    else:
        xg_f = shots["xg_v1"].to_numpy()
        fset = "shared"
    out = shots.select("game_id", "event_idx", "season", "xg_v1").with_columns(
        pl.Series("xg_f", xg_f), pl.lit(fset).alias("feature_set")
    )
    out.write_parquet(out_path)
    res["crossfit"] = {"feature_set": fset, **evaluate(y, xg_f)}
    res["crossfit"].pop("reliability", None)
    res["crossfit_v1_reference"] = {
        k: v for k, v in evaluate(y, shots["xg_v1"].to_numpy()).items() if k != "reliability"
    }
    res["crossfit_by_season"] = calibration_by(shots, "season", xg_f)
    res["crossfit_by_strength"] = calibration_by(shots, "strength_g", xg_f)
    m5 = strength == "5v5"
    ratio5 = float(xg_f[m5].sum() / y[m5].sum())
    res["F4"] = {"sum_xg_over_goals_5v5": ratio5, "pass": bool(0.97 <= ratio5 <= 1.03)}
    (met / "fin_xg_metrics.json").write_text(json.dumps(res, indent=2))
    log.info(
        "xG_f (%s) cross-fit log-loss %.5f auc %.4f; 5v5 sum xG / goals = %.4f",
        fset,
        res["crossfit"]["log_loss"],
        res["crossfit"]["auc"],
        ratio5,
    )
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
