"""Player-level, out-of-sample check of the calibration (methodology §2, gate CAL-3).

For a player-season in t+1, the outcome is his on-ice 5v5 goal (or xG) differential per 60. Predictors are his own season-t
component rates and the season-t rates of the four skaters he shared the ice with in t+1 (teammate context), in a weighted
regression with team-season fixed effects. Using the next season keeps the estimate independent of the data the
components were built from (a same-season regression would return 1/B, B = shrinkage).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.calibration.team import SHR
from src.defense.stints import COMPONENTS


def rate_table(t: pl.DataFrame, comps: list[str]) -> pl.DataFrame:
    """Season-t per-60 rates: each shrunk component, the uncalibrated total and the ridge-calibrated total (adopted or not)."""
    per = t.filter(pl.col("toi_5v5") > 0)
    cols = [(pl.col(SHR[c]) / pl.col("toi_5v5") * 60).alias(f"r_{c}") for c in comps]
    return per.select("player_id", "season", *cols, (pl.col("total_gar") / pl.col("toi_5v5") * 60).alias("r_uncal"),
                      (pl.col("total_gar_ridge") / pl.col("toi_5v5") * 60).alias("r_cal"))


def onice_forward(proc, rates: pl.DataFrame) -> pl.DataFrame:
    """Per player-season (t+1): on-ice minutes, goal and xG differentials, and the mean prior-season rate of his teammates."""
    hx = [f"h_{c}" for c in COMPONENTS]
    ax = [f"a_{c}" for c in COMPONENTS]
    st = (pl.read_parquet(proc / "def_stints.parquet", columns=["season", "minutes", "home_on", "away_on", "h_goals", "a_goals", *hx, *ax]).filter(pl.col("season") >= 2018)
          .with_row_index("sid").with_columns(pl.sum_horizontal(hx).alias("xh"), pl.sum_horizontal(ax).alias("xa")))
    home = st.select("sid", "season", "minutes", (pl.col("h_goals") - pl.col("a_goals")).alias("gd"), (pl.col("xh") - pl.col("xa")).alias("xgd"), pl.col("home_on").alias("p"), pl.lit(0).alias("side"))
    away = st.select("sid", "season", "minutes", (pl.col("a_goals") - pl.col("h_goals")).alias("gd"), (pl.col("xa") - pl.col("xh")).alias("xgd"), pl.col("away_on").alias("p"), pl.lit(1).alias("side"))
    long = pl.concat([home, away]).explode("p").rename({"p": "player_id"})
    prev = rates.select("player_id", (pl.col("season") + 1).alias("season"), "r_uncal", "r_cal")
    long = long.join(prev, on=["player_id", "season"], how="left").with_columns(pl.col("r_uncal").fill_null(0.0), pl.col("r_cal").fill_null(0.0))
    tot = long.group_by("sid", "side").agg(pl.col("r_uncal").sum().alias("su"), pl.col("r_cal").sum().alias("sc"))
    long = long.join(tot, on=["sid", "side"]).with_columns(((pl.col("su") - pl.col("r_uncal")) / 4).alias("tm_u"), ((pl.col("sc") - pl.col("r_cal")) / 4).alias("tm_c"))
    return long.group_by("player_id", "season").agg(pl.col("minutes").sum().alias("minutes"), pl.col("gd").sum().alias("gd"), pl.col("xgd").sum().alias("xgd"),
                                                    ((pl.col("minutes") * pl.col("tm_u")).sum() / pl.col("minutes").sum()).alias("tm_u"),
                                                    ((pl.col("minutes") * pl.col("tm_c")).sum() / pl.col("minutes").sum()).alias("tm_c"))


def within(X: np.ndarray, w: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Weighted within-group demeaning (team-season fixed effects)."""
    out = X.astype(float).copy()
    for k in np.unique(g):
        m = g == k
        out[m] -= (w[m, None] * X[m]).sum(0) / w[m].sum()
    return out


def wls_fe(y: np.ndarray, X: np.ndarray, w: np.ndarray, g: np.ndarray) -> dict:
    """Weighted least squares with group fixed effects; cluster-robust SEs (clusters = groups) and per-group weighted SSE."""
    yt, Xt = within(y[:, None], w, g)[:, 0], within(X, w, g)
    A = Xt.T @ (w[:, None] * Xt)
    beta = np.linalg.solve(A, Xt.T @ (w * yt))
    u = yt - Xt @ beta
    meat = np.zeros_like(A)
    sse = {}
    for k in np.unique(g):
        m = g == k
        s = Xt[m].T @ (w[m] * u[m])
        meat += np.outer(s, s)
        sse[k] = float((w[m] * u[m] ** 2).sum())
    Ainv = np.linalg.inv(A)
    cov = Ainv @ meat @ Ainv
    tot = float((w * yt**2).sum())
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "r2_within": 1 - float((w * u**2).sum()) / tot, "sse_by_group": sse}


def dataset(t: pl.DataFrame, oi: pl.DataFrame, rates: pl.DataFrame, comps: list[str], min_minutes: float) -> pl.DataFrame:
    """Season t+1 outcomes joined to season-t own rates and the t+1 team-season."""
    team = t.select("player_id", "season", "team")
    own = rates.select("player_id", (pl.col("season") + 1).alias("season"), "r_uncal", "r_cal", *[f"r_{c}" for c in comps])
    d = oi.join(own, on=["player_id", "season"], how="inner").join(team, on=["player_id", "season"], how="inner").filter(pl.col("minutes") >= min_minutes)
    return d.with_columns((pl.col("gd") / pl.col("minutes") * 60).alias("y_gd"), (pl.col("xgd") / pl.col("minutes") * 60).alias("y_xgd"),
                          (pl.col("season").cast(pl.Utf8) + "_" + pl.col("team")).alias("tsk"))


def evaluate(d: pl.DataFrame, comps: list[str]) -> dict:
    w, g = d["minutes"].to_numpy(), d["tsk"].to_numpy()
    out: dict = {"n_player_seasons": d.height, "n_team_seasons": int(len(np.unique(g)))}
    for y_name in ("y_gd", "y_xgd"):
        y = d[y_name].to_numpy()
        fu = wls_fe(y, d.select("r_uncal", "tm_u").to_numpy(), w, g)
        fc = wls_fe(y, d.select("r_cal", "tm_c").to_numpy(), w, g)
        ks = sorted(fu["sse_by_group"])
        diff = np.array([fu["sse_by_group"][k] - fc["sse_by_group"][k] for k in ks])       # positive = calibrated better
        z = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
        out[y_name] = {"uncalibrated": {"own_slope": float(fu["beta"][0]), "own_se": float(fu["se"][0]), "teammate_slope": float(fu["beta"][1]), "r2_within": fu["r2_within"]},
                       "calibrated": {"own_slope": float(fc["beta"][0]), "own_se": float(fc["se"][0]), "teammate_slope": float(fc["beta"][1]), "r2_within": fc["r2_within"]},
                       "paired_z_calibrated_better": z, "mean_sse_diff_per_team_season": float(diff.mean())}
        cc = [c for c in comps if not (y_name == "y_xgd" and c == "fin")]                     # xG outcomes cannot see finishing
        Xc = d.select(*[f"r_{c}" for c in cc], "tm_u").to_numpy()
        fp = wls_fe(y, Xc, w, g)
        out[y_name]["forward_component_slopes"] = {c: {"slope": float(fp["beta"][i]), "se": float(fp["se"][i])} for i, c in enumerate(cc)}
        out[y_name]["forward_component_slopes"]["teammate_context"] = {"slope": float(fp["beta"][-1]), "se": float(fp["se"][-1])}
    return out
