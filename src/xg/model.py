"""Shot xG model: logistic baseline vs gradient boosting, temporal holdouts, cross-fitting.

Protocol (brief §6.3, §9.1)
  1. Fit logit and GBM on train seasons; compare on the validation season -> choose.
  2. Isotonic calibrator fit on validation-season predictions; kept only if it improves
     *test*-season log-loss (reported either way).
  3. Report test-season log-loss, Brier, AUC, calibration-in-the-large, ECE, slope.
  4. Final per-shot xG is cross-fitted: season Y scored by a model trained on seasons != Y.
Blocked shots receive xG = 0 by definition (a blocked attempt cannot be a goal).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.config import load_config
from src.xg.features import UNBLOCKED, build_shot_features, design_matrix

log = logging.getLogger(__name__)


def make_model(kind: str, names: list[str], seed: int):
    if kind == "logit":
        spl = [names.index(c) for c in ("dist", "angle", "dt_prev", "speed_prev")]
        rest = [i for i in range(len(names)) if i not in spl]
        ct = ColumnTransformer([
            ("spl", SplineTransformer(n_knots=7, degree=3, extrapolation="constant"), spl),
            ("lin", StandardScaler(), rest),
        ])
        return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000))
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=seed,
    )


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "mean_obs": float(y[m].mean())})
    return out


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    tab = calibration_table(y, p)
    ece = sum(r["n"] * abs(r["mean_pred"] - r["mean_obs"]) for r in tab) / len(y)
    logit_p = np.log(p / (1 - p))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(logit_p[:, None], y).coef_[0, 0]
    return {
        "n": int(len(y)), "goals": int(y.sum()), "sum_xg": float(p.sum()),
        "log_loss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)), "calib_in_large": float(p.sum() / y.sum()),
        "ece": float(ece), "calib_slope": float(slope), "reliability": tab,
    }


def run() -> None:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    metrics_dir = cfg.path(cfg.data.metrics_dir)
    model_dir = cfg.path(f"{cfg.data.output_dir}/models")
    events = pl.read_parquet(proc / "events.parquet")
    players = pl.read_parquet(proc / "players.parquet")
    shots = build_shot_features(events, players, cfg.xg.rush_window_s, cfg.possession.rebound_window_s)
    unb = shots.filter(pl.col("type").is_in(UNBLOCKED))
    X, names = design_matrix(unb)
    y = unb["goal"].to_numpy()
    season = unb["season"].to_numpy()
    tr = np.isin(season, cfg.splits.train)
    va = np.isin(season, cfg.splits.validation)
    te = np.isin(season, cfg.splits.test)

    results: dict = {"features": names, "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum())}
    fitted = {}
    for kind in ("logit", "gbm"):
        m = make_model(kind, names, cfg.seed).fit(X[tr], y[tr])
        fitted[kind] = m
        results[f"{kind}_valid"] = evaluate(y[va], m.predict_proba(X[va])[:, 1])
        log.info("%s valid log-loss %.5f auc %.4f", kind, results[f"{kind}_valid"]["log_loss"], results[f"{kind}_valid"]["auc"])
    chosen = min(("logit", "gbm"), key=lambda k: results[f"{k}_valid"]["log_loss"])
    results["chosen"] = chosen
    for kind, m in fitted.items():
        results[f"{kind}_test"] = evaluate(y[te], m.predict_proba(X[te])[:, 1])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(fitted[chosen].predict_proba(X[va])[:, 1], y[va])
    p_te_raw = fitted[chosen].predict_proba(X[te])[:, 1]
    results[f"{chosen}_test_isotonic"] = evaluate(y[te], iso.predict(p_te_raw))
    use_iso = cfg.xg.calibrate == "isotonic" and (
        results[f"{chosen}_test_isotonic"]["log_loss"] < results[f"{chosen}_test"]["log_loss"] * 0.999
    )
    results["isotonic_used"] = bool(use_iso)
    if chosen == "logit":
        coefs = fitted["logit"][-1].coef_[0]
        results["logit_n_coef"] = int(len(coefs))
    else:
        from sklearn.inspection import permutation_importance
        sub = np.where(va)[0][:40000]
        pi = permutation_importance(fitted["gbm"], X[sub], y[sub], scoring="neg_log_loss", n_repeats=3, random_state=cfg.seed)
        results["gbm_permutation_importance"] = sorted(
            ({"feature": n, "importance": float(v)} for n, v in zip(names, pi.importances_mean)), key=lambda r: -r["importance"]
        )[:20]
    joblib.dump({"model": fitted[chosen], "names": names, "isotonic": iso if use_iso else None}, model_dir / "xg_temporal.joblib")

    # ---- cross-fitted xG for every shot attempt
    X_all, _ = design_matrix(shots)
    s_all = shots["season"].to_numpy()
    xg = np.zeros(len(shots))
    unb_mask = shots["type"].is_in(UNBLOCKED).to_numpy()
    for s in cfg.data.seasons:
        tr_s = season != s
        m = make_model(chosen, names, cfg.seed).fit(X[tr_s], y[tr_s])
        sel = (s_all == s) & unb_mask
        xg[sel] = m.predict_proba(X_all[sel])[:, 1]
        log.info("cross-fit season %s: %d shots", s, sel.sum())
    out = shots.select(
        "game_id", "event_idx", "season", "type", "team_id", "player1", "x", "y", "dist", "angle",
        "shot_type_c", "rebound", "rush", "strength", "empty_net", "goal",
    ).with_columns(pl.Series("xg", xg))
    out.write_parquet(proc / "shots_xg.parquet")
    cf = out.filter(pl.col("type").is_in(UNBLOCKED))
    results["crossfit_all"] = evaluate(cf["goal"].to_numpy(), cf["xg"].to_numpy())
    results["crossfit_by_season"] = {
        str(s): {k: v for k, v in evaluate(g["goal"].to_numpy(), g["xg"].to_numpy()).items() if k != "reliability"}
        for (s,), g in cf.group_by(["season"])
    }
    Path(metrics_dir / "xg_metrics.json").write_text(json.dumps(results, indent=2))
    log.info("xG chosen=%s test log-loss=%.5f; cross-fit calib-in-large=%.4f", chosen, results[f"{chosen}_test"]["log_loss"], results["crossfit_all"]["calib_in_large"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
