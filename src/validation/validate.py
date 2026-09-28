"""Metric validation suite (brief §9.3). Writes reports/metrics/validation.json.

Tests
  yoy            r(metric_Y, metric_Y+1), players >= min TOI both seasons, by position; within-regime
                 pairs vs the 2023->2024 scoring-regime pair; team-changer subset
  split_half     odd vs even games within season (+ Spearman-Brown), >= 500 / 800 EV minutes
  convergent     correlations with P1/60, points/60, ixG/60, on-ice xGF/60, O-RAPM
  discriminant   correlations with individual SH%, on-ice SH%, PDO, G - ixG
  predictive     r(metric_Y, on-ice xGF/60_Y+1) and r(metric_Y, on-ice GF/60_Y+1) vs baselines
  rapm           3-season 5v5 xG RAPM (offense): r and incremental R^2 beyond ixG/60 + P1/60
  team           team-season PM-EPA vs xGF, GF, goal differential
  transition     do higher-value movement segments precede better possession outcomes?
  toi_curve      spread of PM-EPA/60 by TOI band (noise flattening)
  rink           scorekeeper (home-rink) effects on turnover counts and PM-EPA
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config

log = logging.getLogger(__name__)

CORE = ["pm_epa_per60_ev", "pm_direct_per60_ev", "pm_pooled_per60_ev"]
BASELINES = ["ixg_per60_ev", "p1_per60_ev", "pts_per60_ev", "a1_per60_ev", "g_per60_ev", "oi_xgf_per60_ev", "oi_xgf_pct_ev",
             "gv_per60_ev", "tk_per60_ev", "shot_quality_per60_ev"]
FINISHING = ["ind_sh_pct", "oi_sh_pct_ev", "oi_pdo_ev", "g_minus_xg"]


def _r(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")


def _spearman(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    ra = pl.Series(a[m]).rank().to_numpy()
    rb = pl.Series(b[m]).rank().to_numpy()
    return _r(ra, rb)


def _ci(r: float, n: int) -> list[float]:
    if not np.isfinite(r) or n < 5:
        return [float("nan")] * 2
    z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def regime(season: int, cfg) -> int:
    return next((i for i, r in enumerate(cfg.data.scoring_regimes) if season in r), -1)


def yoy_pairs(ps: pl.DataFrame, min_toi: float) -> pl.DataFrame:
    q = ps.filter(pl.col("toi_ev") >= min_toi)
    nxt = q.with_columns((pl.col("season") - 1).alias("season"))
    return q.join(nxt, on=["player_id", "season"], suffix="_next")


def yoy(ps: pl.DataFrame, metrics: list[str], cfg, min_toi: float) -> dict:
    pairs = yoy_pairs(ps, min_toi)
    pairs = pairs.with_columns(
        pl.col("season").map_elements(lambda s: regime(s, cfg), return_dtype=pl.Int64).alias("_ra"),
        (pl.col("season") + 1).map_elements(lambda s: regime(s, cfg), return_dtype=pl.Int64).alias("_rb"),
        (pl.col("primary_team") != pl.col("primary_team_next")).alias("changer"),
    )
    out = {}
    subsets = {
        "within_regime": pl.col("_ra") == pl.col("_rb"),
        "regime_break": pl.col("_ra") != pl.col("_rb"),
        "within_regime_team_changers": (pl.col("_ra") == pl.col("_rb")) & pl.col("changer"),
    }
    for pos in ["F", "D"]:
        for name, cond in subsets.items():
            d = pairs.filter((pl.col("pos_group") == pos) & cond)
            row = {"n": d.height}
            for m in metrics:
                a, b = d[m].to_numpy().astype(float), d[f"{m}_next"].to_numpy().astype(float)
                r = _r(a, b)
                # season-centred: removes season-level shifts (xG drift, scoring regime) that inflate pooled r
                dc = d.select((pl.col(m) - pl.col(m).mean().over("season")).alias("a"),
                              (pl.col(f"{m}_next") - pl.col(f"{m}_next").mean().over("season")).alias("b"))
                rc = _r(dc["a"].to_numpy().astype(float), dc["b"].to_numpy().astype(float))
                row[m] = {"r": r, "ci95": _ci(r, d.height), "spearman": _spearman(a, b), "r_centered": rc}
            out[f"{pos}/{name}"] = row
    by_pair = {}
    for (s,), d in pairs.group_by(["season"]):
        by_pair[f"{s}-{s + 1}"] = {pos: {"n": d.filter(pl.col("pos_group") == pos).height,
                                         "r": _r(d.filter(pl.col("pos_group") == pos)["pm_epa_per60_ev"].to_numpy(),
                                                 d.filter(pl.col("pos_group") == pos)["pm_epa_per60_ev_next"].to_numpy())}
                                   for pos in ["F", "D"]}
    out["by_pair_pm_epa_per60_ev"] = dict(sorted(by_pair.items()))
    return out


def split_half(pg: pl.DataFrame, players: pl.DataFrame, pairs: dict[str, tuple[str, str]], min_toi_min: float) -> dict:
    d = pg.sort("game_id").with_columns((pl.col("game_id").rank("ordinal").over("player_id", "season") % 2).alias("half"))
    agg = d.group_by("player_id", "season", "half").agg([pl.col(c).sum() for c in {c for p in pairs.values() for c in p}] + [pl.col("toi_5v5").sum().alias("_toi")])
    tot = agg.group_by("player_id", "season").agg(pl.col("_toi").sum().alias("tot"))
    agg = agg.join(tot, on=["player_id", "season"]).filter(pl.col("tot") >= min_toi_min * 60)
    agg = agg.join(players.select("player_id", "position"), on="player_id").with_columns(
        pl.when(pl.col("position") == "D").then(pl.lit("D")).otherwise(pl.lit("F")).alias("pos_group"))
    out = {}
    for pos in ["F", "D"]:
        a = agg.filter((pl.col("pos_group") == pos) & (pl.col("half") == 0))
        b = agg.filter((pl.col("pos_group") == pos) & (pl.col("half") == 1))
        j = a.join(b, on=["player_id", "season"], suffix="_b")
        res = {"n": j.height}
        for name, (num, den) in pairs.items():
            ra = j[num].to_numpy() * 3600 / np.maximum(j[den].to_numpy(), 1)
            rb = j[f"{num}_b"].to_numpy() * 3600 / np.maximum(j[f"{den}_b"].to_numpy(), 1)
            r = _r(ra, rb)
            res[name] = {"r": r, "spearman_brown": float(2 * r / (1 + r)) if np.isfinite(r) else None}
        out[pos] = res
    return out


def correlations(ps: pl.DataFrame, min_toi: float, targets: list[str]) -> dict:
    q = ps.filter(pl.col("toi_ev") >= min_toi)
    out = {}
    for pos in ["F", "D"]:
        d = q.filter(pl.col("pos_group") == pos)
        out[pos] = {"n": d.height}
        for m in CORE:
            out[pos][m] = {t: _r(d[m].to_numpy().astype(float), d[t].to_numpy().astype(float)) for t in targets}
    return out


def predictive(ps: pl.DataFrame, min_toi: float, cfg) -> dict:
    pairs = yoy_pairs(ps, min_toi)
    pairs = pairs.with_columns((pl.col("oi_gf") * 60 / pl.col("toi_ev_next")).alias("_"))
    pairs = pairs.with_columns((pl.col("oi_gf_next") * 60 / pl.col("toi_ev_next")).alias("oi_gf_per60_next"))
    out = {}
    for pos in ["F", "D"]:
        d = pairs.filter(pl.col("pos_group") == pos)
        out[pos] = {"n": d.height}
        for m in CORE + BASELINES[:7]:
            a = d[m].to_numpy().astype(float)
            out[pos][m] = {"next_oi_xgf_per60": _r(a, d["oi_xgf_per60_ev_next"].to_numpy().astype(float)),
                           "next_oi_gf_per60": _r(a, d["oi_gf_per60_next"].to_numpy().astype(float)),
                           "next_p1_per60": _r(a, d["p1_per60_ev_next"].to_numpy().astype(float))}
    return out


def _r2(X: np.ndarray, y: np.ndarray) -> float:
    """In-sample R^2 of an OLS fit of y on X (X includes an intercept column)."""
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2))


def rapm_validation(cfg, ps: pl.DataFrame, pg: pl.DataFrame, players: pl.DataFrame) -> dict:
    from src.validation.rapm import build_stints, choose_alpha, design, fit_rapm

    proc = cfg.path(cfg.data.processed_dir)
    goalies = set(players.filter(pl.col("position") == "G")["player_id"].to_list())
    windows = [cfg.data.seasons[i:i + 3] for i in range(0, len(cfg.data.seasons), 3)]
    out, alpha = {"windows": {}}, None
    for w in windows:
        stints = build_stints(proc, w, goalies)
        X, y, wt, groups, pl_ids = design(stints)
        if alpha is None:
            alpha, cv = choose_alpha(X, y, wt, groups)
            out["alpha"], out["alpha_cv"] = alpha, {str(k): v for k, v in cv.items()}
        rp = fit_rapm(X, y, wt, pl_ids, alpha)
        agg = pg.filter(pl.col("season").is_in(w)).group_by("player_id").agg(
            pl.col("pm_epa_ev").sum(), pl.col("pm_direct_ev").sum(), pl.col("pm_pooled_ev").sum(), pl.col("toi_5v5").sum(),
            pl.col("ixg_ev").sum(), pl.col("goals_ev").sum(), pl.col("a1_ev").sum(), pl.col("oi_xgf").sum())
        agg = agg.with_columns(
            (pl.col("pm_epa_ev") * 3600 / pl.col("toi_5v5")).alias("pm_epa_per60_ev"),
            (pl.col("pm_direct_ev") * 3600 / pl.col("toi_5v5")).alias("pm_direct_per60_ev"),
            (pl.col("pm_pooled_ev") * 3600 / pl.col("toi_5v5")).alias("pm_pooled_per60_ev"),
            (pl.col("ixg_ev") * 3600 / pl.col("toi_5v5")).alias("ixg_per60_ev"),
            ((pl.col("goals_ev") + pl.col("a1_ev")) * 3600 / pl.col("toi_5v5")).alias("p1_per60_ev"),
            (pl.col("oi_xgf") * 3600 / pl.col("toi_5v5")).alias("oi_xgf_per60_ev"),
        ).filter(pl.col("toi_5v5") >= 1500 * 60)
        d = agg.join(rp, on="player_id").join(players.select("player_id", "position"), on="player_id")
        res = {"n_stints": stints.height, "n_players": d.height}
        for pos in ["F", "D"]:
            dd = d.filter((pl.col("position") == "D") if pos == "D" else (pl.col("position") != "D"))
            y_r = dd["rapm_off_xgf60"].to_numpy()
            base = np.column_stack([np.ones(dd.height), dd["ixg_per60_ev"].to_numpy(), dd["p1_per60_ev"].to_numpy()])
            full = np.column_stack([base, dd["pm_epa_per60_ev"].to_numpy()])
            res[pos] = {"n": dd.height, **{f"r_{m}": _r(dd[m].to_numpy(), y_r) for m in ["pm_epa_per60_ev", "pm_direct_per60_ev", "pm_pooled_per60_ev", "ixg_per60_ev", "p1_per60_ev", "oi_xgf_per60_ev"]},
                        "r2_base_ixg_p1": _r2(base, y_r), "r2_base_plus_pm": _r2(full, y_r)}
        out["windows"][f"{w[0]}-{w[-1]}"] = res
        log.info("RAPM window %s: %s", w, res)
    return out


def team_level(proc, cfg) -> dict:
    ts = pl.read_csv(cfg.path(cfg.data.output_dir) / "pm_epa_team_season.csv")
    ev = pl.read_parquet(proc / "events.parquet", columns=["game_id", "season", "type", "team_id", "strength", "home_id", "away_id"])
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "home_id", "home_abbrev", "away_abbrev")
    xg = pl.read_parquet(proc / "shots_xg.parquet").select("game_id", "event_idx", "xg")
    ev2 = pl.read_parquet(proc / "events.parquet", columns=["game_id", "event_idx", "season", "type", "team_id", "strength"]).join(xg, on=["game_id", "event_idx"]).join(games, on="game_id")
    ev2 = ev2.filter(pl.col("strength") != "EN").with_columns(
        pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.col("home_abbrev")).otherwise(pl.col("away_abbrev")).alias("team"),
        pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.col("away_abbrev")).otherwise(pl.col("home_abbrev")).alias("opp"))
    # counts cast to signed ints: boolean sums are UInt32 and gf - ga would wrap around
    f = ev2.group_by("season", "team").agg(pl.col("xg").sum().alias("xgf"), (pl.col("type") == "goal").sum().cast(pl.Int64).alias("gf"))
    a = ev2.group_by("season", pl.col("opp").alias("team")).agg(pl.col("xg").sum().alias("xga"), (pl.col("type") == "goal").sum().cast(pl.Int64).alias("ga"))
    d = ts.join(f, on=["season", "team"]).join(a, on=["season", "team"]).with_columns(
        (pl.col("xgf") - pl.col("xga")).alias("xgd"), (pl.col("gf") - pl.col("ga")).alias("gd"))
    _ = ev
    out = {"n_team_seasons": d.height}
    for m in ["pm_epa_attributed", "pm_epa_direct", "pm_epa_pooled"]:
        out[m] = {t: _r(d[m].to_numpy().astype(float), d[t].to_numpy().astype(float)) for t in ["xgf", "gf", "xgd", "gd", "xga"]}
    # next-season prediction of goal differential
    nxt = d.select("team", (pl.col("season") - 1).alias("season"), pl.col("gd").alias("gd_next"), pl.col("xgd").alias("xgd_next"))
    j = d.join(nxt, on=["team", "season"])
    out["predict_next_gd"] = {m: _r(j[m].to_numpy().astype(float), j["gd_next"].to_numpy().astype(float)) for m in ["pm_epa_attributed", "xgd", "gd"]}
    return out


def transition_check(proc) -> dict:
    """Do higher-Δ movement segments precede better outcomes? (partially circular: see report)"""
    tr = pl.read_parquet(proc / "transitions.parquet").filter(pl.col("seg_type").is_in(["carry", "split"]) & (pl.col("strength") == "5v5"))
    poss = pl.read_parquet(proc / "possessions_valued.parquet").select("poss_id", "G", "end_class", "goal")
    d = tr.join(poss, on="poss_id").with_columns(pl.col("delta").qcut(5, labels=[f"Q{i}" for i in range(1, 6)]).alias("q"))
    g = d.group_by("q").agg(pl.len().alias("n"), pl.col("delta").mean().alias("mean_delta"), pl.col("G").mean().alias("mean_G"),
                            (pl.col("end_class") == 1).mean().alias("p_unblocked_shot"), pl.col("goal").mean().alias("p_goal")).sort("q")
    # non-circular variant: from-state held fixed (start kind + coarse zone of start), compare outcome by Δ quintile within stratum
    d2 = d.with_columns(pl.when(pl.col("x") >= 25).then(pl.lit("OZ")).when(pl.col("x") <= -25).then(pl.lit("DZ")).otherwise(pl.lit("NZ")).alias("from_zone"))
    strat = d2.group_by("from_kind", "from_zone", "q").agg(pl.len().alias("n"), pl.col("goal").mean().alias("p_goal")).filter(pl.col("n") >= 2000)
    return {"by_quintile": g.to_dicts(), "goal_rate_by_from_state_and_quintile": strat.sort("from_kind", "from_zone", "q").to_dicts()}


def toi_curve(ps: pl.DataFrame) -> list[dict]:
    d = ps.filter(pl.col("toi_ev") > 50).with_columns(pl.col("toi_ev").cut([100, 200, 300, 500, 700, 900, 1100], labels=["50-100", "100-200", "200-300", "300-500", "500-700", "700-900", "900-1100", "1100+"]).alias("band"))
    return d.group_by("band", "pos_group").agg(pl.len().alias("n"), pl.col("pm_epa_per60_ev").std().alias("sd"), pl.col("pm_epa_per60_ev").mean().alias("mean"),
                                             pl.col("boot_se_per60").median().alias("median_boot_se")).sort("pos_group", "band").to_dicts()


def rink_effects(proc) -> dict:
    ev = pl.read_parquet(proc / "events.parquet", columns=["game_id", "season", "type", "is_home"]).filter(pl.col("type").is_in(["giveaway", "takeaway", "hit"]))
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "home_abbrev")
    per_game = ev.join(games, on="game_id").group_by("season", "home_abbrev", "game_id").agg(
        (pl.col("type") == "giveaway").sum().alias("gv"), (pl.col("type") == "takeaway").sum().alias("tk"), (pl.col("type") == "hit").sum().alias("hit"))
    rink = per_game.group_by("season", "home_abbrev").agg(pl.col("gv").mean(), pl.col("tk").mean(), pl.col("hit").mean())
    lg = per_game.group_by("season").agg(pl.col("gv").mean().alias("gv_lg"), pl.col("tk").mean().alias("tk_lg"), pl.col("hit").mean().alias("hit_lg"))
    r = rink.join(lg, on="season").with_columns((pl.col("gv") / pl.col("gv_lg")).alias("gv_idx"), (pl.col("tk") / pl.col("tk_lg")).alias("tk_idx"), (pl.col("hit") / pl.col("hit_lg")).alias("hit_idx"))
    # persistence of rink index across seasons (a scorer effect persists; noise does not)
    nx = r.select("home_abbrev", (pl.col("season") - 1).alias("season"), pl.col("gv_idx").alias("gv_idx_next"), pl.col("tk_idx").alias("tk_idx_next"))
    j = r.join(nx, on=["home_abbrev", "season"])
    ha = ev.group_by("is_home").agg((pl.col("type") == "giveaway").sum().alias("gv"), (pl.col("type") == "takeaway").sum().alias("tk"))
    return {
        "rink_index_sd": {k: float(r[k].std()) for k in ["gv_idx", "tk_idx", "hit_idx"]},
        "rink_index_range": {k: [float(r[k].min()), float(r[k].max())] for k in ["gv_idx", "tk_idx"]},
        "rink_index_yoy_r": {"gv": _r(j["gv_idx"].to_numpy(), j["gv_idx_next"].to_numpy()), "tk": _r(j["tk_idx"].to_numpy(), j["tk_idx_next"].to_numpy())},
        "home_vs_away_counts": ha.to_dicts(),
    }


def home_away_pm(pg: pl.DataFrame) -> dict:
    d = pg.with_columns((pl.col("team_id") == pl.col("home_id")).alias("home"))
    g = d.group_by("home").agg((pl.col("pm_epa_ev").sum() * 3600 / pl.col("toi_5v5").sum()).alias("pm_epa_per60_ev"),
                               (pl.col("pm_direct_ev").sum() * 3600 / pl.col("toi_5v5").sum()).alias("direct_per60"))
    return {str(r["home"]): {k: v for k, v in r.items() if k != "home"} for r in g.to_dicts()}


def run(include_rapm: bool = True) -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    ps = pl.read_parquet(proc / "player_season.parquet")
    pg = pl.read_parquet(proc / "player_game.parquet")
    players = pl.read_parquet(proc / "players.parquet")
    mt = cfg.metrics.min_toi_ev
    res: dict = {"min_toi_ev": mt}
    res["yoy"] = yoy(ps, CORE + BASELINES + FINISHING, cfg, mt)
    res["yoy_min300"] = {k: v for k, v in yoy(ps, CORE + ["ixg_per60_ev", "p1_per60_ev", "oi_xgf_per60_ev"], cfg, 300).items() if "within_regime" in k and "changers" not in k}
    halves = {"pm_epa_per60_ev": ("pm_epa_ev", "toi_5v5"), "pm_direct_per60_ev": ("pm_direct_ev", "toi_5v5"), "pm_pooled_per60_ev": ("pm_pooled_ev", "toi_5v5"),
              "ixg_per60_ev": ("ixg_ev", "toi_5v5"), "oi_xgf_per60_ev": ("oi_xgf", "toi_5v5"), "gv_per60_ev": ("gv_ev", "toi_5v5")}
    res["split_half"] = {f"min{m}": split_half(pg, players, halves, m) for m in (500, 800)}
    res["convergent_discriminant"] = correlations(ps, mt, BASELINES + FINISHING)
    res["predictive"] = predictive(ps, mt, cfg)
    res["team"] = team_level(proc, cfg)
    res["transition"] = transition_check(proc)
    res["toi_curve"] = toi_curve(ps)
    res["rink"] = rink_effects(proc)
    res["home_away_pm"] = home_away_pm(pg)
    if include_rapm:
        res["rapm"] = rapm_validation(cfg, ps, pg, players)
    path = cfg.path(cfg.data.metrics_dir) / "validation.json"
    path.write_text(json.dumps(res, indent=2, default=str))
    log.info("validation written to %s", path)
    return res


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    run(include_rapm="--no-rapm" not in sys.argv)
