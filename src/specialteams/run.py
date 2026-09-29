"""Special-teams components end to end:  python -m src.specialteams.run [--from STEP]

Steps: stints -> alphas -> select -> final -> finishing -> validate -> report. Decision rules are pre-registered in
docs/specialteams/methodology.md §6 and applied mechanically here.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from src.config import load_config
from src.defense import ridge
from src.specialteams import finish, model, stints
from src.valuation.replacement import status

log = logging.getLogger("specialteams")

STEPS = ["stints", "alphas", "select", "final", "finishing", "validate", "report"]
GROUPS = ["adv", "4v4", "3v3"]
JITTER = 1e-6   # diagonal jitter for empty covariate columns in thin states (ridge.JITTER)
CLS_OF = {("adv", 1): "pp", ("adv", 2): "pk", ("4v4", 0): "4v4", ("3v3", 0): "3v3"}


def _paths(cfg):
    return cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.output_dir), cfg.path(cfg.data.metrics_dir)


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup(cfg, proc):
    return model.fringe_lookup(status(pl.read_parquet(proc / "player_season.parquet"), cfg.valuation.regular_forwards, cfg.valuation.regular_defense))


def _designs(st: pl.DataFrame, group: str, seasons: list[int], lookup: dict) -> dict:
    g = st.filter(pl.col("grp") == group)
    return {s: model.make_design(g.filter(pl.col("season") == s), group, lookup) for s in seasons}


# ----------------------------------------------------------------------------- steps
def step_stints(cfg):
    proc, _, met = _paths(cfg)
    goalies = set(pl.read_parquet(proc / "players.parquet").filter(pl.col("position") == "G")["player_id"].to_list())
    st, cv = stints.build(proc, cfg.data.seasons, goalies, cfg.specialteams.min_match_rate, cfg.specialteams.workers)
    st.write_parquet(proc / "st_stints.parquet")
    cv.write_parquet(proc / "st_coverage.parquet")
    cov = {g: float(x["xg_assigned"].sum() / x["xg_state"].sum()) for (g,), x in cv.group_by(["grp"])}
    # skater-minutes: non-5v5 time in valid stints vs the shift-chart totals (toi_game, seconds)
    valid_games = st.select("game_id").unique()
    toi = (pl.read_parquet(proc / "toi_game.parquet").join(valid_games, on="game_id").filter(~pl.col("player_id").is_in(list(goalies)))
           .select((pl.col("toi_all") - pl.col("toi_5v5")).sum()).item())
    in_st = float((st["minutes"] * (st["nh"] + st["na"]) * 60).sum())
    by = st.group_by("grp").agg(pl.len().alias("stints"), (pl.col("minutes").sum() / 60).alias("hours"), (pl.col("h_xg") + pl.col("a_xg")).sum().alias("xg"),
                                (pl.col("h_goals") + pl.col("a_goals")).sum().alias("goals")).to_dicts()
    _save(met / "st_stints.json", {"xg_coverage": cov, "minutes_coverage_vs_toi_game": in_st / toi, "games": valid_games.height, "by_group": by,
                                    "xg_coverage_by_season": {f"{g}_{s}": float(x["xg_assigned"].sum() / x["xg_state"].sum()) for (g, s), x in cv.group_by(["grp", "season"])}})
    log.info("stints=%d  xG coverage %s  minutes coverage %.3f", st.height, {k: round(v, 3) for k, v in cov.items()}, in_st / toi)


def step_alphas(cfg):
    proc, _, met = _paths(cfg)
    sc = cfg.specialteams
    st = pl.read_parquet(proc / "st_stints.parquet")
    lookup = _lookup(cfg, proc)
    res = {}
    for g in GROUPS:
        curves = {a: [] for a in sc.alphas}
        for s, d in _designs(st, g, sc.cv_seasons, lookup).items():
            cv = ridge.cv_alpha(d, sc.alphas, sc.cv_folds, cfg.seed + s)["xg"]
            for a in sc.alphas:
                curves[a].append(cv[a])
        mean = {str(a): float(np.mean(v)) for a, v in curves.items()}
        res[g] = {"cv_mse": mean, "chosen": float(min(mean, key=mean.get))}
        log.info("alpha CV %s: %s -> %s", g, {k: round(v, 3) for k, v in mean.items()}, res[g]["chosen"])
    _save(met / "st_alphas.json", res)


def _pair_losses(fits, designs_next, st_g, group, pairs, lookup, cache) -> pl.DataFrame:
    parts = []
    for y, y1 in pairs:
        d, beta, _ = fits[y]
        key = (group, y, y1)
        if key not in cache:
            cache[key] = model.make_design(st_g.filter(pl.col("season") == y1), group, lookup, players=d.players)
        nd = cache[key]
        b = beta[:, 0]
        cov_only = b.copy()
        cov_only[1:1 + 2 * d.P] = 0.0
        yv = nd.y()[:, 0]
        parts.append(pl.DataFrame({"game_id": nd.groups, "w": nd.w, "e_m": nd.w * (yv - nd.X @ b) ** 2, "e_b": nd.w * (yv - nd.X @ cov_only) ** 2}).group_by("game_id").agg(pl.all().sum()))
    return pl.concat(parts).group_by("game_id").agg(pl.all().sum())


def _boot_diff(pg: pl.DataFrame, a: str, b: str, reps: int, seed: int) -> tuple[float, float]:
    """Weighted MSE(a) - MSE(b) and its game-bootstrap SE."""
    W, A, B = pg["w"].to_numpy(), pg[a].to_numpy(), pg[b].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(W)
    diffs = []
    for _ in range(reps):
        k = rng.integers(0, n, n)
        diffs.append(A[k].sum() / W[k].sum() - B[k].sum() / W[k].sum())
    return float(A.sum() / W.sum() - B.sum() / W.sum()), float(np.std(diffs))


def step_select(cfg):
    proc, _, met = _paths(cfg)
    sc = cfg.specialteams
    st = pl.read_parquet(proc / "st_stints.parquet")
    lookup = _lookup(cfg, proc)
    seasons = sorted(cfg.data.seasons)
    alphas = _load(met / "st_alphas.json")
    res = {}
    for g in GROUPS:
        a = alphas[g]["chosen"]
        des = _designs(st, g, seasons, lookup)
        st_g = st.filter(pl.col("grp") == g)
        cache: dict = {}
        val, fits_by_rho = {}, {}
        for rho in sc.prior_rhos:
            fits_by_rho[rho] = model.chain_fits(des, a, rho)
            val[rho] = _pair_losses(fits_by_rho[rho], None, st_g, g, sc.validation_pairs, lookup, cache)
        mse = {rho: float(val[rho]["e_m"].sum() / val[rho]["w"].sum()) for rho in sc.prior_rhos}
        best = min(mse, key=mse.get)
        rho = best
        for r in sc.prior_rhos:                      # ascending rho = simpler first: within one SE of the best wins
            d, se = _boot_diff(val[r].join(val[best].select("game_id", pl.col("e_m").alias("e_best")), on="game_id"), "e_m", "e_best", 300, cfg.seed)
            if r == best or d <= se:
                rho = r
                break
        test = _pair_losses(fits_by_rho[rho], None, st_g, g, sc.test_pairs, lookup, cache)
        d, se = _boot_diff(test, "e_b", "e_m", sc.bootstrap_reps * 3, cfg.seed)      # baseline minus model: positive = model better
        res[g] = {"alpha": a, "rho": rho, "validation_mse_by_rho": {str(k): v for k, v in mse.items()},
                  "test": {"mse_model": float(test["e_m"].sum() / test["w"].sum()), "mse_baseline": float(test["e_b"].sum() / test["w"].sum()),
                           "baseline_minus_model": d, "se": se, "z": d / se if se > 0 else float("nan"), "n_games": test.height}}
        log.info("%s: alpha=%s rho=%s  test z=%.2f", g, a, rho, res[g]["test"]["z"])
    _save(met / "st_selection.json", res)


def _boot_season(args):
    from threadpoolctl import threadpool_limits

    d, alpha, reps, seed, prev, rho, cache = args
    ridge.JITTER = JITTER
    if cache.exists():
        return np.load(cache)
    with threadpool_limits(1):
        draws = ridge.bootstrap(d, {"xg": alpha}, reps, seed, ridge.prior_vector(d, prev, rho))
    np.save(cache, draws)
    return draws


def step_final(cfg):
    proc, _, met = _paths(cfg)
    sc = cfg.specialteams
    st = pl.read_parquet(proc / "st_stints.parquet")
    lookup = _lookup(cfg, proc)
    seasons = sorted(cfg.data.seasons)
    sel = _load(met / "st_selection.json")
    cache_dir = proc / "st_boot"
    cache_dir.mkdir(exist_ok=True)
    out = []
    for g in GROUPS:
        a, rho = sel[g]["alpha"], sel[g]["rho"]
        des = _designs(st, g, seasons, lookup)
        fits = model.chain_fits(des, a, rho)
        tasks = [(des[s], a, sc.bootstrap_reps, cfg.seed + s, fits[seasons[i - 1]][2] if i > 0 and rho > 0 else None, rho,
                  cache_dir / f"{g}_{s}_B{sc.bootstrap_reps}_a{a}_rho{rho}_seed{cfg.seed}.npy") for i, s in enumerate(seasons)]
        with ProcessPoolExecutor(min(sc.workers, len(tasks))) as pool:
            draws = dict(zip(seasons, pool.map(_boot_season, tasks)))
        fx = model.fringe_effects(fits, g)
        vals = model.player_values(fits, g, fx, lookup, draws).with_columns(pl.col("role").map_elements(lambda r, g=g: CLS_OF[(g, r)], return_dtype=pl.Utf8).alias("cls"))
        out.append(vals.with_columns(pl.lit(g).alias("grp")))
        _save(met / f"st_fringe_{g}.json", {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in fx.items()})
        log.info("final %s: fringe effects %s", g, {f"{k[0]}{k[1]}{k[2]}": round(v, 3) for k, v in fx.items()})
        # independent (rho = 0) fits for the stability gate
        ind = model.chain_fits(des, a, 0.0)
        model.player_values(ind, g, model.fringe_effects(ind, g), lookup).with_columns(
            pl.col("role").map_elements(lambda r, g=g: CLS_OF[(g, r)], return_dtype=pl.Utf8).alias("cls")).write_parquet(proc / f"st_rapm_ind_{g}.parquet")
    pl.concat(out).write_parquet(proc / "st_rapm_player_season.parquet")


def step_finishing(cfg):
    proc, _, met = _paths(cfg)
    fin, chk = finish.by_class(proc, cfg.valuation.regular_forwards, cfg.valuation.regular_defense)
    fin.write_parquet(proc / "st_finishing.parquet")
    _save(met / "st_finishing_check.json", chk)
    log.info("finishing by class: %s", chk)


def build_table(cfg) -> pl.DataFrame:
    """Wide player-season table of the four components: st_<cls>_gar (RAPM + finishing), _sd, _min, _rapm_gar, _fin_gar."""
    proc, _, _ = _paths(cfg)
    rapm = pl.read_parquet(proc / "st_rapm_player_season.parquet").select("player_id", "season", "cls", "minutes", "gar_rapm", "gar_rapm_sd", "rate_rapm")
    fin = pl.read_parquet(proc / "st_finishing.parquet").select("player_id", "season", "cls", "fin_gar_st", "attempts_st")
    j = rapm.join(fin, on=["player_id", "season", "cls"], how="full", coalesce=True).with_columns(
        pl.col("minutes").fill_null(0.0), pl.col("gar_rapm").fill_null(0.0), pl.col("gar_rapm_sd").fill_null(0.0), pl.col("fin_gar_st").fill_null(0.0), pl.col("attempts_st").fill_null(0))
    j = j.with_columns((pl.col("gar_rapm") + pl.col("fin_gar_st")).alias("gar"))
    wide = None
    for c in ("pp", "pk", "4v4", "3v3"):
        x = j.filter(pl.col("cls") == c).select("player_id", "season", pl.col("gar").alias(f"st_{c}_gar"), pl.col("gar_rapm_sd").alias(f"st_{c}_gar_sd"), pl.col("minutes").alias(f"st_{c}_min"),
                                                  pl.col("gar_rapm").alias(f"st_{c}_rapm_gar"), pl.col("fin_gar_st").alias(f"st_{c}_fin_gar"), pl.col("rate_rapm").alias(f"st_{c}_rapm_per60"))
        wide = x if wide is None else wide.join(x, on=["player_id", "season"], how="full", coalesce=True)
    num = [c for c in wide.columns if c.startswith("st_")]
    return wide.with_columns([pl.col(c).fill_null(0.0) for c in num])


def _yoy(ind: pl.DataFrame, cls: str, min_min: float) -> dict:
    t = ind.filter(pl.col("cls") == cls).with_columns((pl.col("rate_rapm") - pl.col("rate_rapm").mean().over("season", "pos_group")).alias("_c"))
    nxt = t.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("_c").alias("_n"), pl.col("minutes").alias("m_n"))
    j = t.join(nxt, on=["player_id", "season"]).filter((pl.col("minutes") >= min_min) & (pl.col("m_n") >= min_min))
    return {g: {"r": float(np.corrcoef(x["_c"], x["_n"])[0, 1]), "n": x.height} for (g,), x in j.group_by(["pos_group"])}


def step_validate(cfg):
    proc, _, met = _paths(cfg)
    sc = cfg.specialteams
    sel = _load(met / "st_selection.json")
    stj = _load(met / "st_stints.json")
    res: dict = {"gates": {}}
    # ST-1 coverage
    cov_xg = stj["xg_coverage"]
    res["ST1"] = {"xg_coverage": cov_xg, "minutes_coverage": stj["minutes_coverage_vs_toi_game"],
                  "pass_by_group": {g: bool(v >= 0.90) for g, v in cov_xg.items()}, "minutes_pass": bool(stj["minutes_coverage_vs_toi_game"] >= 0.85)}
    # ST-2 out of sample (per group; adv covers PP and PK)
    res["ST2"] = {g: {"z": v["test"]["z"], "pass": bool(v["test"]["z"] > sc.z_threshold)} for g, v in sel.items()}
    # ST-3 stability from independent fits
    yoy = {}
    for g in GROUPS:
        ind = pl.read_parquet(proc / f"st_rapm_ind_{g}.parquet")
        for c in ind["cls"].unique().to_list():
            yoy[c] = _yoy(ind, c, sc.min_minutes if c in ("pp", "pk") else sc.min_minutes_thin)
    res["ST3"] = {c: {"yoy": v, "pass": bool(all(x["r"] >= sc.yoy_min for x in v.values()) and len(v) == 2)} for c, v in yoy.items()}
    # ST-4 team reconciliation: class goal differential vs summed class value by primary team
    st = pl.read_parquet(proc / "st_stints.parquet")
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "home_id", "away_id")
    t = st.join(games, on="game_id").with_columns((pl.col("nh") > pl.col("na")).alias("h_adv"), (pl.col("h_goals") - pl.col("a_goals")).alias("gd_h"))
    rows = []
    for g in GROUPS:
        x = t.filter(pl.col("grp") == g)
        if g == "adv":
            pp = pl.concat([x.filter(pl.col("h_adv")).select("season", pl.col("home_id").alias("team_id"), pl.col("gd_h").alias("gd")),
                            x.filter(~pl.col("h_adv")).select("season", pl.col("away_id").alias("team_id"), (-pl.col("gd_h")).alias("gd"))]).with_columns(pl.lit("pp").alias("cls"))
            pk = pl.concat([x.filter(pl.col("h_adv")).select("season", pl.col("away_id").alias("team_id"), (-pl.col("gd_h")).alias("gd")),
                            x.filter(~pl.col("h_adv")).select("season", pl.col("home_id").alias("team_id"), pl.col("gd_h").alias("gd"))]).with_columns(pl.lit("pk").alias("cls"))
            rows += [pp, pk]
        else:
            rows.append(pl.concat([x.select("season", pl.col("home_id").alias("team_id"), pl.col("gd_h").alias("gd")), x.select("season", pl.col("away_id").alias("team_id"), (-pl.col("gd_h")).alias("gd"))]).with_columns(pl.lit(g).alias("cls")))
    gd = pl.concat(rows).group_by("season", "team_id", "cls").agg(pl.col("gd").sum())
    tab = build_table(cfg)
    ps = pl.read_parquet(proc / "player_season.parquet").select("player_id", "season", "primary_team")
    pt =pl.read_parquet(proc / "games.parquet").select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("primary_team")).unique("team_id")
    recon = {}
    for c in ("pp", "pk", "4v4", "3v3"):
        v = tab.join(ps.with_columns(pl.col("season").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)), on=["player_id", "season"]).group_by("season", "primary_team").agg(pl.col(f"st_{c}_gar").sum().alias("gar"))
        j = gd.filter(pl.col("cls") == c).join(pt, on="team_id").join(v, on=["season", "primary_team"])
        x, y = j["gar"].to_numpy(), j["gd"].to_numpy().astype(float)
        A = np.column_stack([np.ones(len(x)), x])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - A @ coef
        se = float(np.sqrt(resid @ resid / (len(y) - 2) / np.sum((x - x.mean()) ** 2)))
        recon[c] = {"slope": float(coef[1]), "se": se, "r": float(np.corrcoef(x, y)[0, 1]), "n": int(len(x)), "sd_team_gd": float(y.std()), "sd_team_gar": float(x.std()),
                    "pass": bool(sc.slope_band[0] <= coef[1] <= sc.slope_band[1])}
    res["ST4"] = recon
    # ST-5 (reported): correlation with the penalty component's drawn / taken rates
    pen = pl.read_parquet(proc / "pen_player_season.parquet").select("player_id", "season", "drawn_per60_post", "taken_per60_post").with_columns(pl.col("season").cast(pl.Int64), pl.col("player_id").cast(pl.Int64))
    corr = {}
    for c in ("pp", "pk"):
        x = tab.filter(pl.col(f"st_{c}_min") >= 100).join(pen, on=["player_id", "season"]).with_columns((pl.col(f"st_{c}_gar") / pl.col(f"st_{c}_min") * 60).alias("rate"))
        corr[c] = {"drawn": float(np.corrcoef(x["rate"], x["drawn_per60_post"])[0, 1]), "taken": float(np.corrcoef(x["rate"], x["taken_per60_post"])[0, 1]), "n": x.height}
    res["ST5"] = corr
    # ST-6 finishing consistency
    chk = _load(met / "st_finishing_check.json")
    res["ST6"] = {**chk, "pass": bool(chk["max_abs_diff_vs_fin_gar_other"] < 1e-6)}
    # scale summaries (reported)
    q = {c: tab.filter(pl.col(f"st_{c}_min") >= (sc.min_minutes if c in ("pp", "pk") else sc.min_minutes_thin)) for c in ("pp", "pk", "4v4", "3v3")}
    res["summary"] = {c: {"players_qualified": x.height, "sd_gar": float(x[f"st_{c}_gar"].std()), "sd_rapm": float(x[f"st_{c}_rapm_gar"].std()), "sd_fin": float(x[f"st_{c}_fin_gar"].std()),
                          "median_minutes": float(x[f"st_{c}_min"].median())} for c, x in q.items()}
    _save(met / "st_validation.json", res)
    log.info("ST-1 %s | ST-2 %s | ST-3 %s | ST-4 %s | ST-6 %s", res["ST1"]["pass_by_group"], {g: round(v["z"], 2) for g, v in res["ST2"].items()},
             {c: {k: round(x["r"], 2) for k, x in v["yoy"].items()} for c, v in res["ST3"].items()}, {c: round(v["slope"], 2) for c, v in recon.items()}, res["ST6"]["pass"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STEPS, default="stints")
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args()
    run_steps(args.only or STEPS[STEPS.index(args.start):])


def run_steps(steps: list[str]) -> None:
    cfg = load_config()
    ridge.JITTER = JITTER
    for s in steps:
        log.info("=== specialteams:%s ===", s)
        if s == "report":
            from src.specialteams.report import run as report
            report()
        else:
            globals()[f"step_{s}"](cfg)


if __name__ == "__main__":
    main()
