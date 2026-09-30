"""Team-level calibration on the accounting identity (methodology §2-§3).

    GD5_k = a + sum_c w_c S_ck + e_k   (S = team sum of the *unshrunk* component above replacement)

with a ridge penalty pulling every w_c toward 1 and lambda chosen by leave-one-season-out cross-validation.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.valuation import recon

COMPS = ["pm", "def", "fin", "to", "fo"]
RAW = {"pm": "pm_raw_gar", "def": "def_gar", "fin": "fin_raw_gar", "to": "to_raw_gar", "fo": "fo_raw_gar"}
SHR = {"pm": "pm_gar", "def": "def_gar", "fin": "fin_gar", "to": "to_gar", "fo": "fo_gar"}
# all-strength extension (methodology §5b): penalties and the four special-teams components
COMPS_ALL = COMPS + ["pen", "pp", "pk", "4v4", "3v3"]
RAW |= {"pen": "pen_raw_gar", **{c: f"st_{c}_raw_gar" for c in ("pp", "pk", "4v4", "3v3")}}
SHR |= {"pen": "pen_gar", **{c: f"st_{c}_gar" for c in ("pp", "pk", "4v4", "3v3")}}
# goalie extension (methodology §5c): the team's goalie value, summed over shots faced by the defending team
COMPS_ALL_GK = COMPS_ALL + ["gk"]
RAW |= {"gk": "gk_raw_gar"}
SHR |= {"gk": "gk_gar"}
# position-specific weights (methodology §5d): seven main components enter twice (forwards' and defensemen's team sums)
SPLIT = ["pm", "def", "fin", "to", "pen", "pp", "pk"]
COMPS_POS = [f"{c}_{g}" for c in SPLIT for g in ("F", "D")] + ["fo", "4v4", "3v3", "gk"]
for _c in SPLIT:
    for _g in ("F", "D"):
        RAW[f"{_c}_{_g}"] = f"{RAW[_c]}__{_g}"
        SHR[f"{_c}_{_g}"] = f"{SHR[_c]}__{_g}"
LAMBDAS = np.geomspace(1e1, 1e6, 26)


def team_table(t: pl.DataFrame, proc, comps: list[str] = COMPS, target: str = "gd5") -> pl.DataFrame:
    """Team-season sums of the raw and shrunk components joined to the goal-differential target (``gd5`` or ``gd_all``)."""
    virtual = [c for c in comps if c.rsplit("_", 1)[-1] in ("F", "D") and c.rsplit("_", 1)[0] in SPLIT]
    skater = [c for c in comps if c != "gk" and c not in virtual]
    cols = sorted(set(RAW[c] for c in skater) | set(SHR[c] for c in skater))
    s = recon.team_sums(t, cols)
    for g in ("F", "D"):                       # position-specific team sums (weights by position, methodology §5d)
        bases = sorted({c.rsplit("_", 1)[0] for c in virtual if c.endswith("_" + g)})
        if bases:
            cc = sorted({RAW[b] for b in bases} | {SHR[b] for b in bases})
            sg = t.filter(pl.col("pos_group") == g).group_by("season", "team").agg([pl.col(x).sum().alias(f"{x}__{g}") for x in cc])
            s = s.join(sg, on=["season", "team"], how="full", coalesce=True)
    s = s.with_columns([pl.col(c).fill_null(0.0) for c in s.columns if c not in ("season", "team")])
    if "gk" in comps:
        s = s.join(pl.read_parquet(proc / "goalie_team_season.parquet").select("season", "team", "gk_gar", "gk_raw_gar"), on=["season", "team"], how="left").with_columns(pl.col("gk_gar").fill_null(0.0), pl.col("gk_raw_gar").fill_null(0.0))
    gd = recon.team_gd5(proc) if target == "gd5" else recon.team_gd_all(proc)
    return s.join(gd, on=["season", "team"]).sort("season", "team")


def xg_to_goals_factor(proc) -> dict:
    """Slope of team 5v5 goal differential on team 5v5 xG differential: context for the xG-scale components."""
    sh = pl.read_parquet(proc / "fin_shots.parquet", columns=["season", "team_id", "opp_team_id", "strength_g", "xg_v1"]).filter(pl.col("strength_g") == "5v5")
    games = pl.read_parquet(proc / "games.parquet")
    abbr = pl.concat([games.select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")), games.select(pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team"))]).unique("team_id")
    f = sh.group_by("season", "team_id").agg(pl.col("xg_v1").sum().alias("xgf"))
    a = sh.group_by("season", pl.col("opp_team_id").alias("team_id")).agg(pl.col("xg_v1").sum().alias("xga"))
    x = f.join(a, on=["season", "team_id"]).with_columns((pl.col("xgf") - pl.col("xga")).alias("xgd")).join(abbr, on="team_id").select("season", "team", "xgd")
    j = x.join(recon.team_gd5(proc), on=["season", "team"])
    r = recon.regress(j, ["xgd"])
    return {"slope": r["slopes"]["xgd"]["slope"], "se": r["slopes"]["xgd"]["se"], "r2": r["r2"], "n": r["n"]}


def ridge_to_one(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """argmin sum (y - a - Xw)^2 + lam sum (w - 1)^2 with a unpenalized. lam = inf returns w = 1."""
    xm, ym = X.mean(0), y.mean()
    if np.isinf(lam):
        w = np.ones(X.shape[1])
    else:
        Xc = X - xm
        r = (y - ym) - Xc.sum(1)
        w = 1.0 + np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ r)
    return float(ym - xm @ w), w


def loso_cv(X: np.ndarray, y: np.ndarray, season: np.ndarray, lams: np.ndarray) -> dict:
    """Leave-one-season-out squared errors per team-season for every lambda and for w = 1."""
    grid = list(lams) + [np.inf]
    err = np.zeros((len(grid), len(y)))
    for s in np.unique(season):
        te = season == s
        for i, lam in enumerate(grid):
            a, w = ridge_to_one(X[~te], y[~te], lam)
            err[i, te] = (y[te] - (a + X[te] @ w)) ** 2
    return {"lams": grid, "sq_err": err, "total": err.sum(1)}


def calibrate(j: pl.DataFrame, comps: list[str], boot: int, seed: int, target: str = "gd5") -> dict:
    cols = [RAW[c] for c in comps]
    X, y, season = j.select(cols).to_numpy(), j[target].to_numpy().astype(float), j["season"].to_numpy()
    cv = loso_cv(X, y, season, LAMBDAS)
    best = int(np.argmin(cv["total"][:-1]))
    lam = float(LAMBDAS[best])
    diff = cv["sq_err"][-1] - cv["sq_err"][best]                         # positive = calibrated better than all-ones
    z = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    a, w = ridge_to_one(X, y, lam)
    rng = np.random.default_rng(seed)
    bw = np.array([ridge_to_one(X[k], y[k], lam)[1] for k in (rng.integers(0, len(y), len(y)) for _ in range(boot))])
    ols = recon.regress(j, cols, y=target)
    adopt = bool(z > 2.0)
    return {"comps": comps, "lambda": lam, "lambda_grid_edge": bool(best in (0, len(LAMBDAS) - 1)), "weights_ridge": dict(zip(comps, w.tolist())),
            "weights_ci95": {c: [float(np.quantile(bw[:, i], 0.025)), float(np.quantile(bw[:, i], 0.975))] for i, c in enumerate(comps)},
            "cv_mse_ones": float(cv["sq_err"][-1].mean()), "cv_mse_calibrated": float(cv["sq_err"][best].mean()), "cv_mse_by_lambda": {f"{lam_i:.3g}": float(cv["total"][i] / len(y)) for i, lam_i in enumerate(cv["lams"])},
            "paired_z_calibrated_vs_ones": z, "adopt": adopt, "weights_applied": dict(zip(comps, (w if adopt else np.ones(len(comps))).tolist())),
            "unshrunk_ols": {"r2": ols["r2"], "n": ols["n"], "slopes": {c: ols["slopes"][RAW[c]] for c in comps}}}


def shrunk_ols(j: pl.DataFrame, comps: list[str], target: str = "gd5") -> dict:
    r = recon.regress(j, [SHR[c] for c in comps], y=target)
    return {"r2": r["r2"], "slopes": {c: r["slopes"][SHR[c]] for c in comps}}


def apply_weights_all(t: pl.DataFrame, weights: dict[str, float], gpw: float, tag: str) -> pl.DataFrame:
    """All-strength version: every component in ``weights`` (penalties and special teams included) times its weight, summed to
    ``total_gar_all_<tag>``; the change is measured against ``total_gar_all``."""
    cols = {c: f"{SHR[c]}_{tag}" for c in weights}
    t = t.with_columns(*[(pl.col(SHR[c]) * weights[c]).alias(cols[c]) for c in weights],
                       *[(pl.col(f"{SHR[c]}_sd") * weights[c]).alias(f"{cols[c]}_sd") for c in weights if f"{SHR[c]}_sd" in t.columns])
    sds = [pl.col(f"{cols[c]}_sd") ** 2 for c in weights if f"{cols[c]}_sd" in t.columns]
    sd_total = pl.sum_horizontal(sds).sqrt() if sds else pl.lit(0.0)
    return t.with_columns(pl.sum_horizontal([pl.col(v) for v in cols.values()]).alias(f"total_gar_all_{tag}"), sd_total.alias(f"total_gar_all_{tag}_sd")).with_columns(
        (pl.col(f"total_gar_all_{tag}") / gpw).alias(f"total_war_all_{tag}"), (pl.col(f"total_gar_all_{tag}") - pl.col("total_gar_all")).alias(f"{tag}_change_gar"))


def apply_weights(t: pl.DataFrame, weights: dict[str, float], gpw: float, tag: str = "cal") -> pl.DataFrame:
    """Add calibrated components, total and wins (suffix ``tag``): each shrunk component times its weight (penalties keep weight 1)."""
    t = t.with_columns(*[(pl.col(SHR[c]) * weights[c]).alias(f"{c}_gar_{tag}") for c in weights],
                       *[(pl.col(f"{c}_gar_sd") * weights[c]).alias(f"{c}_gar_{tag}_sd") for c in weights])
    parts = [pl.col(f"{c}_gar_{tag}") for c in weights] + [pl.col("pen_gar")]
    sds = [pl.col(f"{c}_gar_{tag}_sd") ** 2 for c in weights] + [pl.col("pen_gar_sd") ** 2]
    return t.with_columns(pl.sum_horizontal(parts).alias(f"total_gar_{tag}"), pl.sum_horizontal(sds).sqrt().alias(f"total_gar_{tag}_sd")).with_columns(
        (pl.col(f"total_gar_{tag}") / gpw).alias(f"total_war_{tag}"), (pl.col(f"total_gar_{tag}") - pl.col("total_gar")).alias(f"{tag}_change_gar"))


def apply_weights_pos(t: pl.DataFrame, weights: dict[str, float], gpw: float, tag: str) -> pl.DataFrame:
    """Position-specific version of ``apply_weights_all``: each split component's shrunk player column times the weight for his position
    (``<c>_F`` / ``<c>_D``), pooled components times their single weight; summed to ``total_gar_all_<tag>``."""
    cols = {}
    exprs = []
    sd_exprs = []
    bases = list(dict.fromkeys(c.rsplit("_", 1)[0] if c.rsplit("_", 1)[-1] in ("F", "D") and c.rsplit("_", 1)[0] in SPLIT else c for c in weights if c != "gk"))
    for b in bases:
        if f"{b}_F" in weights:
            w = pl.when(pl.col("pos_group") == "F").then(weights[f"{b}_F"]).otherwise(weights[f"{b}_D"])
        else:
            w = pl.lit(weights[b])
        cols[b] = f"{SHR[b]}_{tag}"
        exprs.append((pl.col(SHR[b]) * w).alias(cols[b]))
        if f"{SHR[b]}_sd" in t.columns:
            sd_exprs.append((pl.col(f"{SHR[b]}_sd") * w).alias(f"{cols[b]}_sd"))
    t = t.with_columns(*exprs, *sd_exprs)
    sds = [pl.col(f"{v}_sd") ** 2 for v in cols.values() if f"{v}_sd" in t.columns]
    sd_total = pl.sum_horizontal(sds).sqrt() if sds else pl.lit(0.0)
    return t.with_columns(pl.sum_horizontal([pl.col(v) for v in cols.values()]).alias(f"total_gar_all_{tag}"), sd_total.alias(f"total_gar_all_{tag}_sd")).with_columns(
        (pl.col(f"total_gar_all_{tag}") / gpw).alias(f"total_war_all_{tag}"), (pl.col(f"total_gar_all_{tag}") - pl.col("total_gar_all")).alias(f"{tag}_change_gar"))
