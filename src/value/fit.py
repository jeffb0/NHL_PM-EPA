"""Value-function bake-off (Phases 4-5) and final cross-fitted V(s) artifacts.

Bake-off protocol (decided before looking at results)
  * Train on ``splits.train`` seasons, evaluate on ``splits.validation``; alive observations,
    EN excluded. Primary metric: MSE of V vs realized possession xG (G).
  * Candidates are compared by the *paired* per-observation squared-error difference vs the
    best candidate (mean +- SE). Candidates within 2 SE of the best are "statistically tied";
    among tied candidates we prefer higher V stability (season-to-season correlation of
    predictions) and then fewer parameters (parsimony, guardrail §14).
  * The chosen spec is then scored once on the untouched ``splits.test`` seasons.
Final artifacts
  * V models cross-fitted by season (season Y predicted by a model trained on seasons != Y)
  * v_state_{5v5,pp,pk,4v4_3v3}.parquet from an all-season fit + metadata json (N, metrics, sha256)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

import joblib
import numpy as np
import polars as pl

from src.config import load_config
from src.states.mapping import STRENGTH_GROUP
from src.value.estimators import GBMValue, HazardValue, TabularValue

log = logging.getLogger(__name__)
SCORING_END = {"shot_sog", "shot_miss", "goal"}
CENSORED_END = {"period_end", "unrecorded_stoppage"}


def terminal_values(poss: pl.DataFrame, shots_xg: pl.DataFrame) -> pl.DataFrame:
    """Attach terminal xG (G), goal flag and end class to each possession."""
    sx = shots_xg.select("game_id", pl.col("event_idx").alias("terminal_event_idx"), "xg")
    p = poss.join(sx, on=["game_id", "terminal_event_idx"], how="left")
    return p.with_columns(
        pl.when(pl.col("end_type").is_in(list(SCORING_END))).then(pl.col("xg").fill_null(0)).otherwise(0.0).alias("G"),
        (pl.col("end_type") == "goal").cast(pl.Int8).alias("goal"),
        pl.when(pl.col("end_type").is_in(list(SCORING_END))).then(1)
        .when(pl.col("end_type").is_in(list(CENSORED_END))).then(0).otherwise(2).alias("end_class"),
    ).drop("xg")


def alive_frame(obs: pl.DataFrame, poss: pl.DataFrame, exclude_en: bool = True) -> pl.DataFrame:
    """Alive observations with return G, time-to-end tau and Markov next-state linkage."""
    d = obs.filter(pl.col("role").is_in(["start", "mid"])).join(
        poss.select("poss_id", "G", "goal", "end_class", "end_t", pl.col("G").alias("terminal_xg")), on="poss_id"
    )
    d = d.with_columns((pl.col("end_t") - pl.col("t")).clip(0, None).alias("tau"))
    d = d.with_columns(pl.col("strength").replace_strict(STRENGTH_GROUP, default="other").alias("_sg"))
    if exclude_en:
        d = d.filter(~pl.col("_sg").is_in(["EN", "other"]))
    d = d.sort("poss_id", "seq").with_row_index("_row")
    nxt = pl.col("_row").shift(-1).over("poss_id")
    d = d.with_columns(nxt.alias("next_idx"), pl.col("G").alias("R"))
    return d.drop("_sg")


DECISION_KINDS = ["shot", "giveaway", "lost_takeaway"]


def decision_frame(obs: pl.DataFrame, poss: pl.DataFrame, exclude_en: bool = True) -> pl.DataFrame:
    """Terminal (decision-point) observations with a location; target = the possession's G.

    V_dec(s) = E[G_p | p's terminal decision happens at s]. Each row is valued as a `control`
    state at its location, so the model learns location x strength (x score) only.
    """
    d = obs.filter(pl.col("obs_kind").is_in(DECISION_KINDS) & pl.col("x").is_not_null() & pl.col("y").is_not_null()).join(
        poss.select("poss_id", "G", "goal"), on="poss_id")
    d = d.with_columns(pl.col("strength").replace_strict(STRENGTH_GROUP, default="other").alias("_sg"))
    if exclude_en:
        d = d.filter(~pl.col("_sg").is_in(["EN", "other"]))
    return d.drop("_sg")


def decision_bakeoff(dec: pl.DataFrame, cfg) -> dict:
    """Same protocol as V: train seasons -> validation MSE; chosen spec scored once on test."""
    tr = dec.filter(pl.col("season").is_in(cfg.splits.train))
    va = dec.filter(pl.col("season").is_in(cfg.splits.validation))
    te = dec.filter(pl.col("season").is_in(cfg.splits.test))
    tv = dec.filter(pl.col("season").is_in(cfg.splits.train + cfg.splits.validation))
    specs = ["gbm/cont/score", "gbm/cont", "mc/zones/k=auto/score", "mc/grid10/k=auto/score", "mc/grid10/k=auto"]
    rows, preds = [], {}
    g = va["G"].to_numpy()
    for n in specs:
        p = make(parse_spec(n), cfg.seed).fit(tr).predict(va)
        preds[n] = p
        rows.append({"spec": n, **{k: v for k, v in score(p, g, va["goal"].to_numpy()).items() if k != "reliability"}})
    best = min(rows, key=lambda r: r["mse"])["spec"]
    for r in rows:
        diff = (preds[r["spec"]] - g) ** 2 - (preds[best] - g) ** 2
        r["d_mse_vs_best"], r["se_d_mse"] = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
        r["tied_with_best"] = bool(r["d_mse_vs_best"] <= 2 * r["se_d_mse"] + 1e-15)
    # parsimony among ties: prefer tabular (fewer assumptions) only if tied; else best
    tied = [r for r in rows if r["tied_with_best"]]
    chosen = sorted(tied, key=lambda r: (not r["spec"].startswith("mc"), r["mse"]))[0]["spec"] if tied else best
    test = score(make(parse_spec(chosen), cfg.seed).fit(tv).predict(te), te["G"].to_numpy(), te["goal"].to_numpy())
    return {"rows": rows, "best_by_mse": best, "chosen": chosen, "test": test, "n_train": tr.height, "n_valid": va.height, "n_test": te.height}


def score(v: np.ndarray, g: np.ndarray, goal: np.ndarray) -> dict:
    v = np.clip(v, 1e-6, 1 - 1e-6)
    mse = float(np.mean((v - g) ** 2))
    mse0 = float(np.mean((g.mean() - g) ** 2))
    soft_ll = float(-np.mean(g * np.log(v) + (1 - g) * np.log(1 - v)))
    X = np.column_stack([np.ones_like(v), v])
    beta = np.linalg.lstsq(X, g, rcond=None)[0]
    q = np.quantile(v, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(q, v, side="right") - 1, 0, 9)
    rel = [{"bin": b, "n": int((idx == b).sum()), "mean_v": float(v[idx == b].mean()), "mean_g": float(g[idx == b].mean()),
            "goal_rate": float(goal[idx == b].mean())} for b in range(10) if (idx == b).any()]
    ece = sum(r["n"] * abs(r["mean_v"] - r["mean_g"]) for r in rel) / len(v)
    return {
        "n": int(len(v)), "mse": mse, "r2_vs_const": 1 - mse / mse0, "soft_logloss": soft_ll,
        "calib_intercept": float(beta[0]), "calib_slope": float(beta[1]), "ece": float(ece),
        "brier_goal": float(np.mean((v - goal) ** 2)), "mean_v": float(v.mean()), "mean_g": float(g.mean()),
        "reliability": rel,
    }


def make(spec: dict, seed: int):
    if spec["method"] in ("mc", "markov"):
        return TabularValue(spec["scheme"], spec.get("k", "auto"), spec["method"], spec.get("score", False))
    if spec["method"] == "gbm":
        return GBMValue(spec.get("score", False), seed)
    return HazardValue(spec["scheme"], seed=seed)


def spec_name(s: dict) -> str:
    parts = [s["method"], s.get("scheme", "cont")]
    if s["method"] in ("mc", "markov"):
        parts.append(f"k={s.get('k', 'auto')}")
    if s.get("score"):
        parts.append("score")
    if s.get("hits"):
        parts.append("hits")
    return "/".join(parts)


def bakeoff(alive: pl.DataFrame, alive_hits: pl.DataFrame | None, cfg) -> dict:
    tr_s, va_s = cfg.splits.train, cfg.splits.validation
    specs = []
    for scheme in ["zones", "grid10", "grid20"]:
        for k in ["auto", 0, 10, 50, 200]:
            specs.append({"method": "mc", "scheme": scheme, "k": k})
    specs += [
        {"method": "mc", "scheme": "zones", "k": "auto", "score": True},
        {"method": "mc", "scheme": "grid10", "k": "auto", "score": True},
        {"method": "gbm"}, {"method": "gbm", "score": True},
        {"method": "hazard", "scheme": "zones"},
    ]
    if alive_hits is not None:
        specs += [{"method": "markov", "scheme": "zones", "k": "auto", "hits": True},
                  {"method": "mc", "scheme": "zones", "k": "auto", "hits": True}]
    tr = alive.filter(pl.col("season").is_in(tr_s))
    va = alive.filter(pl.col("season").is_in(va_s))
    g, goal = va["G"].to_numpy(), va["goal"].to_numpy()
    va5 = (va["strength"] == "5v5").to_numpy()
    preds, rows = {}, []
    for s in specs:
        t0 = time.time()
        if s.get("hits"):
            trh = alive_hits.filter(pl.col("season").is_in(tr_s))
            # fit with hit mid-observations; score on the same validation rows as everyone else
            m = make(s, cfg.seed).fit(trh)
            p = m.predict(va)
        else:
            m = make(s, cfg.seed).fit(tr)
            p = m.predict(va)
        name = spec_name(s)
        preds[name] = p
        r = score(p, g, goal)
        r5 = score(p[va5], g[va5], goal[va5])
        n_params = len(m.table) if hasattr(m, "table") else None
        rows.append({"spec": name, **{k: v for k, v in r.items() if k != "reliability"}, "mse_5v5": r5["mse"],
                     "n_states": n_params, "k_used": getattr(m, "k_used", None), "fit_s": round(time.time() - t0, 1)})
        log.info("bakeoff %-28s mse=%.6f r2=%.4f slope=%.3f (%.0fs)", name, r["mse"], r["r2_vs_const"], r["calib_slope"], time.time() - t0)
    best = min(rows, key=lambda r: r["mse"])["spec"]
    se_best = (preds[best] - g) ** 2
    for r in rows:
        diff = (preds[r["spec"]] - g) ** 2 - se_best
        r["d_mse_vs_best"] = float(diff.mean())
        r["se_d_mse"] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        r["tied_with_best"] = bool(r["d_mse_vs_best"] <= 2 * r["se_d_mse"] + 1e-15)
    return {"rows": rows, "best_by_mse": best, "reliability": {k: score(v, g, goal)["reliability"] for k, v in preds.items() if k in (best,)}}


def stability(alive: pl.DataFrame, specs: list[dict], cfg) -> dict:
    """Season-to-season stability of V: fit on Y and on Y+1 separately, correlate predictions
    over a fixed evaluation set (validation-season alive observations). All consecutive pairs,
    so a scoring-regime break (e.g. 2023 -> 2024) shows up as a low-r pair."""
    va = alive.filter(pl.col("season").is_in(cfg.splits.validation))
    seasons = cfg.data.seasons
    out = {}
    for s in specs:
        cors = []
        for a, b in zip(seasons[:-1], seasons[1:]):
            pa = make(s, cfg.seed).fit(alive.filter(pl.col("season") == a)).predict(va)
            pb = make(s, cfg.seed).fit(alive.filter(pl.col("season") == b)).predict(va)
            cors.append(float(np.corrcoef(pa, pb)[0, 1]))
        pairs = [f"{a}-{b}" for a, b in zip(seasons[:-1], seasons[1:])]
        within = [c for (a, b), c in zip(zip(seasons[:-1], seasons[1:]), cors) if regime_of(a, cfg) == regime_of(b, cfg)]
        out[spec_name(s)] = {"pairs": dict(zip(pairs, cors)), "mean_r": float(np.mean(within)), "mean_r_all": float(np.mean(cors))}
        log.info("stability %-28s mean r=%.4f", spec_name(s), np.mean(cors))
    return out


def regime_of(season: int, cfg) -> int:
    for i, r in enumerate(cfg.data.scoring_regimes):
        if season in r:
            return i
    return -1


def crossfit_train_seasons(season: int, cfg) -> list[int]:
    """Seasons used to fit the V model applied to ``season`` (never ``season`` itself)."""
    others = [s for s in cfg.data.seasons if s != season]
    if cfg.value.crossfit_within_regime:
        same = [s for s in others if regime_of(s, cfg) == regime_of(season, cfg)]
        return same or others
    return others


def parse_spec(name: str) -> dict:
    parts = name.split("/")
    s: dict = {"method": parts[0]}
    if parts[0] in ("mc", "markov", "hazard"):
        s["scheme"] = parts[1]
    for p in parts[2:]:
        if p.startswith("k="):
            v = p[2:]
            s["k"] = v if v == "auto" else float(v)
        elif p == "score":
            s["score"] = True
        elif p == "hits":
            s["hits"] = True
    return s


def choose(bo: dict, stab: dict) -> str:
    tied = [r for r in bo["rows"] if r["tied_with_best"] and r["spec"] in stab]
    if not tied:
        return bo["best_by_mse"]
    # prefer stability, then parsimony (fewer states; continuous models count as large)
    tied.sort(key=lambda r: (-round(stab[r["spec"]]["mean_r"], 3), r["n_states"] or 10**6))
    return tied[0]["spec"]


def run(skip_bakeoff: bool = False) -> None:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    vdir = cfg.path(cfg.data.value_dir)
    mdir = cfg.path(cfg.data.metrics_dir)
    poss = terminal_values(pl.read_parquet(proc / "possessions.parquet"), pl.read_parquet(proc / "shots_xg.parquet"))
    poss.write_parquet(proc / "possessions_valued.parquet")
    obs = pl.read_parquet(proc / "observations.parquet")
    alive = alive_frame(obs, poss, cfg.epa.exclude_empty_net)
    results_path = mdir / "value_bakeoff.json"
    if not skip_bakeoff or not results_path.exists():
        alive_hits = None
        hits_path = proc / "observations_hits.parquet"
        if hits_path.exists():
            ph = terminal_values(pl.read_parquet(proc / "possessions_hits.parquet"), pl.read_parquet(proc / "shots_xg.parquet"))
            alive_hits = alive_frame(pl.read_parquet(hits_path), ph, cfg.epa.exclude_empty_net)
        ck = mdir / "value_bakeoff_rows.checkpoint.json"
        if ck.exists():
            bo = json.loads(ck.read_text())
            log.info("loaded bake-off checkpoint (%d specs)", len(bo["rows"]))
        else:
            bo = bakeoff(alive, alive_hits, cfg)
            ck.write_text(json.dumps(bo))
        cand = sorted([r for r in bo["rows"] if r["tied_with_best"]], key=lambda r: r["mse"])[:8]
        extra = ["mc/zones/k=auto", "mc/grid10/k=auto", "mc/grid20/k=auto", "gbm/cont"]
        names = list(dict.fromkeys([r["spec"] for r in cand] + extra))
        stab = stability(alive, [parse_spec(n) for n in names if "hits" not in n], cfg)
        chosen = choose(bo, stab)
        # untouched test seasons, scored once for chosen + main challengers
        tr = alive.filter(pl.col("season").is_in(cfg.splits.train + cfg.splits.validation))
        te = alive.filter(pl.col("season").is_in(cfg.splits.test))
        test = {}
        for n in dict.fromkeys([chosen, "mc/zones/k=auto", "mc/grid10/k=auto", "gbm/cont", "hazard/zones"]):
            m = make(parse_spec(n), cfg.seed).fit(tr)
            p = m.predict(te)
            test[n] = score(p, te["G"].to_numpy(), te["goal"].to_numpy())
            te5 = (te["strength"] == "5v5").to_numpy()
            test[n]["by_strength"] = {
                sg: {k: v for k, v in score(p[msk], te["G"].to_numpy()[msk], te["goal"].to_numpy()[msk]).items() if k != "reliability"}
                for sg, msk in [("5v5", te5), ("PP", (te["strength"] == "PP").to_numpy()), ("PK", (te["strength"] == "PK").to_numpy()),
                                ("4v4_3v3", te["strength"].is_in(["4v4", "3v3"]).to_numpy())]
            }
        results = {"bakeoff": bo, "stability": stab, "chosen": chosen, "test": test}
        results_path.write_text(json.dumps(results, indent=2))
    results = json.loads(results_path.read_text())
    chosen = results["chosen"]
    spec = parse_spec(chosen)
    log.info("chosen V spec: %s", chosen)

    # ---- cross-fitted models by season + all-season model
    models = {}
    for s in cfg.data.seasons:
        models[s] = make(spec, cfg.seed).fit(alive.filter(pl.col("season").is_in(crossfit_train_seasons(s, cfg))))
    models["all"] = make(spec, cfg.seed).fit(alive)
    # ---- decision-point value V_dec (arrival states: shot / giveaway / lost-puck locations)
    dec = decision_frame(obs, poss, cfg.epa.exclude_empty_net)
    dres = decision_bakeoff(dec, cfg)
    (mdir / "value_decision.json").write_text(json.dumps(dres, indent=2))
    dspec = parse_spec(dres["chosen"])
    dmodels = {s: make(dspec, cfg.seed).fit(dec.filter(pl.col("season").is_in(crossfit_train_seasons(s, cfg)))) for s in cfg.data.seasons}
    dmodels["all"] = make(dspec, cfg.seed).fit(dec)
    log.info("V_dec chosen %s (valid R2 %.4f, test R2 %.4f)", dres["chosen"],
             next(r["r2_vs_const"] for r in dres["rows"] if r["spec"] == dres["chosen"]), dres["test"]["r2_vs_const"])
    # attempt value V_att(s) = E[xG | shot attempt at s] (blocked = 0), same spec, shot rows only
    att = dec.filter(pl.col("obs_kind") == "shot")
    amodels = {s: make(dspec, cfg.seed).fit(att.filter(pl.col("season").is_in(crossfit_train_seasons(s, cfg)))) for s in cfg.data.seasons}
    amodels["all"] = make(dspec, cfg.seed).fit(att)
    a_tr, a_te = att.filter(pl.col("season").is_in(cfg.splits.train + cfg.splits.validation)), att.filter(pl.col("season").is_in(cfg.splits.test))
    dres["attempt_test"] = {k: v for k, v in score(make(dspec, cfg.seed).fit(a_tr).predict(a_te), a_te["G"].to_numpy(), a_te["goal"].to_numpy()).items() if k != "reliability"}
    (mdir / "value_decision.json").write_text(json.dumps(dres, indent=2))
    joblib.dump({"spec": chosen, "models": models, "dec_spec": dres["chosen"], "dec_models": dmodels, "att_models": amodels}, vdir / "v_models.joblib")

    # ---- tabular artifacts per strength (always export the tabular zones+grid10 tables for transparency)
    table_spec = spec if spec["method"] in ("mc", "markov") else {"method": "mc", "scheme": "grid10", "k": "auto"}
    tab = make(table_spec, cfg.seed).fit(alive)
    frame = tab.to_frame()
    meta = {"chosen_spec": chosen, "table_spec": spec_name(table_spec), "k_used": tab.k_used,  # per level
            "n_alive_obs": alive.height, "seasons": cfg.data.seasons, "test_metrics": {k: {m: v for m, v in d.items() if m not in ("reliability", "by_strength")} for k, d in results["test"].items()},
            "files": {}}
    for sg, fname in [("5v5", "v_state_5v5"), ("PP", "v_state_pp"), ("PK", "v_state_pk"), ("4v4_3v3", "v_state_4v4_3v3")]:
        f = vdir / f"{fname}.parquet"
        frame.filter(pl.col("sgroup") == sg).sort("state_id").write_parquet(f)
        meta["files"][f.name] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "n_states": frame.filter(pl.col("sgroup") == sg).height,
                                 "n_obs": int(frame.filter(pl.col("sgroup") == sg)["n"].sum())}
    (vdir / "v_state_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote value surfaces: %s", list(meta["files"]))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    run(skip_bakeoff="--skip-bakeoff" in sys.argv)
