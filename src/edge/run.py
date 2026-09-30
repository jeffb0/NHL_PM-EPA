"""EDGE tracking features: physical aging curves and the projection test (docs/edge/methodology.md).

    python -m src.edge.run        (needs ``python -m src.ingestion.edge`` first)
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.projection.model import Spec, age_basis

log = logging.getLogger("edge")
FIRST_TARGET_EDGE = 2022        # first target season whose previous season has EDGE data
TEST = [2024, 2025]
FEATS = ["max_speed", "bursts60", "dist60", "shot_speed", "oz_share"]
Z = {f: f"z_{f}" for f in FEATS}
SETS = {"all_five": [*Z.values(), "edge_missing"], "speed_only": [Z["max_speed"], Z["bursts60"], "edge_missing"], "speed_distance": [Z["max_speed"], Z["bursts60"], Z["dist60"], "edge_missing"]}


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def edge_frame(p: pl.DataFrame, edge: pl.DataFrame, bio: pl.DataFrame) -> pl.DataFrame:
    """Skater-season rows with the derived EDGE features (per 60 modeled minutes), age, and season x position z-scores."""
    d = (p.join(edge, on=["player_id", "season"], how="inner").join(bio.select("player_id", "birth_date"), on="player_id", how="left")
         .with_columns(pd_.age_at(pl.col("season"), pl.col("birth_date")).alias("age"), pl.col("max_speed_mph").alias("max_speed"), (pl.col("bursts_over20") / pl.col("m") * 60).alias("bursts60"),
                       (pl.col("distance_mi") / pl.col("m") * 60).alias("dist60"), pl.col("shot_speed_mph").alias("shot_speed"), pl.col("oz_share")))
    med = d.group_by("pos_group").agg(pl.col("age").median().alias("_a"))
    d = d.join(med, on="pos_group").with_columns(pl.col("age").fill_null(pl.col("_a"))).drop("_a")
    return d.with_columns([((pl.col(f) - pl.col(f).mean().over("season", "pos_group")) / pl.col(f).std().over("season", "pos_group")).alias(Z[f]) for f in FEATS])


def _age_curve(X: np.ndarray, y: np.ndarray, w: np.ndarray, sel: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Weighted least squares on the rows ``sel``; the age effect at ``ages`` (intercept excluded)."""
    A = X[sel].T @ (w[sel, None] * X[sel])
    beta = np.linalg.solve(A, X[sel].T @ (w[sel] * y[sel]))
    return age_basis(ages) @ beta[1:]


def physical_aging(d: pl.DataFrame, reps: int = 200, seed: int = 0) -> dict:
    """Each EDGE feature (season x position centred) on the age basis, minute-weighted, player-cluster bootstrap; effect relative to age 27."""
    ages = np.array([22.0, 25.0, 27.0, 30.0, 33.0, 36.0])
    rng = np.random.default_rng(seed)
    out: dict = {}
    for f in FEATS:
        for g in ("F", "D"):
            s = d.filter((pl.col("pos_group") == g) & (pl.col("m") >= 300) & pl.col(f).is_not_null()).with_columns((pl.col(f) - pl.col(f).mean().over("season")).alias("_y"))
            X = np.column_stack([np.ones(s.height), age_basis(s["age"].to_numpy())])
            y, w = s["_y"].to_numpy(), s["m"].to_numpy()
            ids = s["player_id"].to_numpy()
            uniq = np.unique(ids)
            idx = {u: np.flatnonzero(ids == u) for u in uniq}

            base = _age_curve(X, y, w, np.arange(s.height), ages)
            boots = []
            for _ in range(reps):
                sel = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
                boots.append(_age_curve(X, y, w, sel, ages))
            bb = np.array(boots)
            d34 = bb[:, 4] - bb[:, 2]        # 33 vs 27
            out[f"{f}|{g}"] = {"n": s.height, "effect_by_age": dict(zip(map(str, ages), base.tolist())), "lo95": dict(zip(map(str, ages), np.quantile(bb, 0.025, 0).tolist())), "hi95": dict(zip(map(str, ages), np.quantile(bb, 0.975, 0).tolist())),
                               "change_27_to_33": float(base[4] - base[2]), "change_27_to_33_ci95": [float(np.quantile(d34, 0.025)), float(np.quantile(d34, 0.975))]}
    return out


def leading_indicator(p: pl.DataFrame, d: pl.DataFrame) -> dict:
    """Exploratory (methodology section 8, deviation 1): does a change in speed or bursts between t - 1 and t predict next-season value beyond this
    season's value and the level? Next-season total rate on this season's rate, the EDGE level at t, the EDGE change from t - 1 to t, and the age basis;
    minute-weighted, player-cluster SEs; forwards and defensemen separately, rows with >= 300 modeled minutes at t and t + 1."""
    from src.teamcontext.run import _wls_cluster
    cur = d.select("player_id", "season", "pos_group", "age", "m", "r_total", *Z.values())
    prv = d.select("player_id", (pl.col("season") + 1).alias("season"), *[pl.col(c_).alias(c_ + "_prev") for c_ in Z.values()])
    nxt = p.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("m").alias("m_next"), pl.col("r_total").alias("r_next"))
    j = cur.join(prv, on=["player_id", "season"]).join(nxt, on=["player_id", "season"]).filter((pl.col("m") >= 300) & (pl.col("m_next") >= 300))
    out: dict = {"n": j.height}
    for g in ("F", "D"):
        s = j.filter(pl.col("pos_group") == g)
        for f in ("max_speed", "bursts60", "dist60"):
            col = Z[f]
            dz = (s[col] - s[col + "_prev"]).to_numpy()
            X = np.column_stack([np.ones(s.height), s["r_total"].to_numpy(), s[col].to_numpy(), dz, age_basis(s["age"].to_numpy() + 1.0)])
            beta, se = _wls_cluster(X, s["r_next"].to_numpy(), s["m_next"].to_numpy(), s["player_id"].to_numpy())
            out[f"{f}|{g}"] = {"n": s.height, "level_coef": float(beta[2]), "level_t": float(beta[2] / se[2]), "change_coef": float(beta[3]), "change_t": float(beta[3] / se[3]), "own_rate_coef": float(beta[1])}
    return out


def with_edge_features(p: pl.DataFrame, d: pl.DataFrame) -> pl.DataFrame:
    """Add the prior season's z-scored EDGE features to every target row (missing -> 0 and ``edge_missing`` = 1)."""
    prev = d.select("player_id", (pl.col("season") + 1).alias("season"), *Z.values())
    x = p.join(prev, on=["player_id", "season"], how="left")
    miss = pl.col(Z["max_speed"]).is_null()
    return x.with_columns(miss.cast(pl.Float64).alias("edge_missing"), *[pl.col(c).fill_null(0.0) for c in Z.values()])


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    edge = pl.read_parquet(proc / "edge_player_season.parquet")
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    d = edge_frame(p, edge, bio)
    res: dict = {}
    cand = p.filter((pl.col("season") >= 2021) & (pl.col("m") >= 300))
    res["coverage"] = {"skater_seasons_300min": cand.height, "with_edge": int(cand.join(edge, on=["player_id", "season"], how="inner").height)}
    res["coverage"]["share"] = res["coverage"]["with_edge"] / res["coverage"]["skater_seasons_300min"]
    res["physical_aging"] = physical_aging(d)

    res["leading_indicator"] = leading_indicator(p, d)
    p2 = with_edge_features(p, d)
    ft_all, fa = pd_.design_frame(p2, bio, 1, floor=pr.REVISED_FLOOR), pd_.design_frame(p2, bio, 1, targets_only=False)
    ft = ft_all.filter(pl.col("season") >= FIRST_TARGET_EDGE)
    base = Spec()
    parts: dict[str, list] = {"base": []}
    for name in SETS:
        parts[name] = []
    for T in TEST:
        fb = pr.forecast(fa, ft, 1, T, base, p2).filter(pl.col("m") >= pr.EVAL_MIN)
        parts["base"].append(fb.select("player_id", "season", "m", "y", "age", pl.col("proj").alias("base"), pl.col("sig_a").alias("sa_base"), pl.col("sig_b").alias("sb_base")))
        for name, cols in SETS.items():
            fe = pr.forecast(fa, ft, 1, T, Spec(extra=tuple(cols)), p2).filter(pl.col("m") >= pr.EVAL_MIN)
            parts[name].append(fe.select("player_id", "season", pl.col("proj").alias(name), pl.col("sig_a").alias(f"sa_{name}"), pl.col("sig_b").alias(f"sb_{name}")))
    e = pl.concat(parts["base"])
    for name in SETS:
        e = e.join(pl.concat(parts[name]), on=["player_id", "season"])
    res["test"] = {"n_rows": e.height, "seasons": TEST}
    for name in SETS:
        sub30 = e.filter(pl.col("age") >= 30)
        sd = np.sqrt(e[f"sa_{name}"].to_numpy() + e[f"sb_{name}"].to_numpy() / e["m"].to_numpy())
        res["test"][name] = {"mse": float((e["m"] * (e["y"] - e[name]) ** 2).sum() / e["m"].sum()), "mse_base": float((e["m"] * (e["y"] - e["base"]) ** 2).sum() / e["m"].sum()),
                             "z_vs_base": pr.paired_cluster(e, name, "base")["z"], "n_age30": sub30.height, "z_vs_base_age30": pr.paired_cluster(sub30, name, "base")["z"],
                             "mse_age30": float((sub30["m"] * (sub30["y"] - sub30[name]) ** 2).sum() / sub30["m"].sum()), "mse_base_age30": float((sub30["m"] * (sub30["y"] - sub30["base"]) ** 2).sum() / sub30["m"].sum()),
                             "coverage_80": float((np.abs(e["y"].to_numpy() - e[name].to_numpy()) <= pr.Z80 * sd).mean())}
    # does speed explain aging? age effects with and without the EDGE features on the EDGE-era data
    ages = np.array([27.0, 30.0, 34.0])
    spec_e = Spec(extra=tuple(SETS["all_five"]))
    m0, m1 = pm.fit_models(ft, base), pm.fit_models(ft, spec_e)
    c0, c1 = pm.age_curve(m0, base, ages), pm.age_curve(m1, spec_e, ages)
    res["age_effect_27_to_34"] = {g: {"without_edge": float(c0[g][2] - c0[g][0]), "with_edge": float(c1[g][2] - c1[g][0])} for g in ("F", "D")}
    res["edge_coefficients"] = {g: dict(zip(pm.feature_names(spec_e), m1[g].raw_coef().tolist())) for g in ("F", "D")}
    ma = res["physical_aging"]
    res["gates"] = {"ED1": {"share": res["coverage"]["share"], "pass": bool(res["coverage"]["share"] >= 0.90)},
                    "ED2": {"F": ma["max_speed|F"]["change_27_to_33_ci95"], "D": ma["max_speed|D"]["change_27_to_33_ci95"], "change_F": ma["max_speed|F"]["change_27_to_33"], "change_D": ma["max_speed|D"]["change_27_to_33"],
                            "pass": bool(ma["max_speed|F"]["change_27_to_33_ci95"][1] < 0 and ma["max_speed|D"]["change_27_to_33_ci95"][1] < 0)},
                    "ED3": {"z": res["test"]["all_five"]["z_vs_base"], "pass": bool(res["test"]["all_five"]["z_vs_base"] > 2)},
                    "ED4": {"coverage_80": res["test"]["all_five"]["coverage_80"], "pass": bool(0.70 <= res["test"]["all_five"]["coverage_80"] <= 0.90)}}
    _save(met / "edge_results.json", res)
    log.info("gates %s", {k: v["pass"] for k, v in res["gates"].items()})
    from src.edge.report import run as report
    report()
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
