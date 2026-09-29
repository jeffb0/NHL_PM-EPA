"""Defensive PM-EPA end to end:  python -m src.defense.run [--from STEP]

Steps: stints -> alphas -> select (D1, D2, G1 on held-out pairs) -> final (fits, bootstrap CIs,
stability gates, leaderboards) -> report. Decision rules are pre-registered in
docs/defense/methodology.md §6 and applied mechanically here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from src.config import load_config
from src.defense import evaluate as ev
from src.defense import ridge
from src.defense.stints import COMPONENTS, build

log = logging.getLogger("defense")

TARGETS = COMPONENTS + ["xg"]
STEPS = ["stints", "alphas", "select", "final", "report"]
POS_GROUP = {"D": "D"}  # everything else is a forward


# ----------------------------------------------------------------------------- helpers
def _paths(cfg):
    proc = cfg.path(cfg.data.processed_dir)
    return proc, cfg.path(cfg.data.output_dir), cfg.path(cfg.data.metrics_dir)


def _load_json(path):
    return json.loads(path.read_text())


def _save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float))


def _stints(proc, seasons=None) -> pl.DataFrame:
    st = pl.read_parquet(proc / "def_stints.parquet")
    return st if seasons is None else st.filter(pl.col("season").is_in(seasons))


def chain_fits(stints: pl.DataFrame, seasons: list[int], alphas: dict, rho: float) -> dict:
    """Season fits in order; with rho > 0 each season's ridge prior is rho x the previous fit."""
    out, prev = {}, None
    for s in seasons:
        d = ridge.build_design(stints.filter(pl.col("season") == s), TARGETS)
        beta = ridge.fit(d, alphas, ridge.prior_vector(d, prev, rho))
        coefs = ridge.coef_frame(d, beta)
        out[s] = (d, beta, coefs)
        prev = coefs
    return out


def _pair_feats(fits: dict, stints: pl.DataFrame, pairs: list[list[int]]) -> list:
    return [ev.features(fits[y][0], fits[y][1], stints.filter(pl.col("season") == y1)) for y, y1 in pairs]


def _losses(feats: list, core, w_stack, reps, seed, target="xg") -> dict:
    per_game = pl.concat([ev.pair_losses(nd, b, f, core, w_stack, target) for nd, b, f in feats])
    return ev.summarize_losses(per_game, reps, seed)


def _boot_season(args):
    """Bootstrap draws for one season, cached on disk (keyed by season, reps, alphas, rho)."""
    from threadpoolctl import threadpool_limits

    st, alphas, reps, seed, prior_coefs, rho, cache = args
    if cache.exists():
        return np.load(cache)
    with threadpool_limits(1):  # one BLAS thread per worker: the pool already fills the cores
        d = ridge.build_design(st, TARGETS)
        draws = ridge.bootstrap(d, alphas, reps, seed, ridge.prior_vector(d, prior_coefs, rho))
    np.save(cache, draws)
    return draws


# ----------------------------------------------------------------------------- steps
def step_stints(cfg):
    proc, _, met = _paths(cfg)
    players = pl.read_parquet(proc / "players.parquet")
    goalies = set(players.filter(pl.col("position") == "G")["player_id"].to_list())
    st, q = build(proc, cfg.data.seasons, goalies, cfg.defense.min_match_rate, cfg.defense.workers)
    st.write_parquet(proc / "def_stints.parquet")
    rate = float(q["ledger_abs_assigned"].sum() / q["ledger_abs_5v5"].sum())
    by_season = st.group_by("season").agg(
        pl.len().alias("stints"), (pl.col("minutes").sum() / 60).alias("hours"),
        pl.sum_horizontal([pl.col(f"{s}_{c}") for s in "ha" for c in COMPONENTS]).sum().alias("xg"),
        (pl.col("h_goals") + pl.col("a_goals")).sum().alias("goals"),
    ).sort("season")
    info = {"G5_assignment_rate": rate, "games": q.height, "by_season": by_season.to_dicts()}
    _save_json(met / "dpm_stints.json", info)
    log.info("stints=%d games=%d  G5 assignment rate=%.4f", st.height, q.height, rate)


def step_alphas(cfg):
    proc, _, met = _paths(cfg)
    st = _stints(proc)
    curves = {t: {a: [] for a in cfg.defense.alphas} for t in TARGETS}
    for s in cfg.defense.cv_seasons:
        d = ridge.build_design(st.filter(pl.col("season") == s), TARGETS)
        cv = ridge.cv_alpha(d, cfg.defense.alphas, cfg.defense.cv_folds, cfg.seed + s)
        for t in TARGETS:
            for a in cfg.defense.alphas:
                curves[t][a].append(cv[t][a])
        log.info("alpha CV season %d done", s)
    mean = {t: {str(a): float(np.mean(v)) for a, v in c.items()} for t, c in curves.items()}
    chosen = {t: float(min(m, key=m.get)) for t, m in mean.items()}
    # G4: at a common alpha the component coefficients sum exactly to the xGA RAPM coefficient
    d = ridge.build_design(st.filter(pl.col("season") == cfg.defense.cv_seasons[-1]), TARGETS)
    b = ridge.fit(d, {t: chosen["xg"] for t in TARGETS})
    g4 = float(np.abs(b[:, :len(COMPONENTS)].sum(1) - b[:, len(COMPONENTS)]).max())
    _save_json(met / "dpm_alphas.json", {"cv_mse": mean, "chosen": chosen, "G4_max_abs_diff": g4})
    log.info("alphas=%s  G4=%.2e", chosen, g4)


def step_select(cfg):
    proc, _, met = _paths(cfg)
    dc = cfg.defense
    alphas = _load_json(met / "dpm_alphas.json")["chosen"]
    st = _stints(proc)
    seasons = sorted(cfg.data.seasons)
    core = dc.core_components
    res = {"alphas": alphas}
    # D1 at rho = 0
    fits = chain_fits(st, seasons, alphas, 0.0)
    train = _pair_feats(fits, st, dc.train_pairs)
    w_stack = ev.stack_weights(train)
    res["M3_weights"] = w_stack
    val = _pair_feats(fits, st, dc.validation_pairs)
    res["validation"] = _losses(val, core, w_stack, dc.loss_bootstrap_reps, cfg.seed)
    model = ev.pick_model(res["validation"], ["M0", "M1", "M2", "M3"])
    res["D1_model"] = model
    log.info("validation MSE %s -> D1 picks %s", res["validation"]["mse"], model)
    # D2: prior strength for the chosen model, same rule (ties -> rho = 0)
    rho_rows = {}
    per_rho = {}
    for rho in dc.prior_rhos:
        f = fits if rho == 0 else chain_fits(st, seasons, alphas, rho)
        per_rho[rho] = f
        v = _pair_feats(f, st, dc.validation_pairs)
        pg = pl.concat([ev.pair_losses(nd, b, fe, core, w_stack) for nd, b, fe in v]).select("game_id", "w", pl.col(model).alias(str(rho)))
        rho_rows[rho] = pg
    joined = rho_rows[dc.prior_rhos[0]]
    for rho in dc.prior_rhos[1:]:
        joined = joined.join(rho_rows[rho].drop("w"), on="game_id")
    W = joined["w"].to_numpy()
    E = np.column_stack([joined[str(r)].to_numpy() for r in dc.prior_rhos])
    mse = E.sum(0) / W.sum()
    rng = np.random.default_rng(cfg.seed)
    boots = np.array([E[k].sum(0) / W[k].sum() for k in (rng.integers(0, len(W), len(W)) for _ in range(dc.loss_bootstrap_reps))])
    best = int(np.argmin(mse))
    rho = dc.prior_rhos[best]
    for i, r in enumerate(dc.prior_rhos):  # ascending rho = simpler first
        if i == best or mse[i] - mse[best] <= np.std(boots[:, i] - boots[:, best]):
            rho = r
            break
    res["D2"] = {"mse": dict(zip(map(str, dc.prior_rhos), mse.tolist())), "rho": rho}
    log.info("D2 rho MSE %s -> rho=%s", res["D2"]["mse"], rho)
    # G1 on test pairs, reported once (with the chosen rho)
    test = _pair_feats(per_rho[rho], st, dc.test_pairs)
    res["test"] = _losses(test, core, w_stack, dc.loss_bootstrap_reps, cfg.seed)
    res["test_goals"] = _losses(test, core, w_stack, dc.loss_bootstrap_reps, cfg.seed, target="goals")
    res["test_by_pair"] = {f"{y}-{y1}": _losses([t], core, w_stack, dc.loss_bootstrap_reps, cfg.seed)["mse"] for (y, y1), t in zip(dc.test_pairs, test)}
    d = res["test"]["diff"][f"{model}-B0"]
    se = res["test"]["diff_se"][f"{model}-B0"]
    res["G1"] = {"diff_vs_B0": d, "se": se, "pass": bool(d < 0 and -d > 2 * se)}
    _save_json(met / "dpm_selection.json", res)
    log.info("test MSE %s  G1=%s", res["test"]["mse"], res["G1"])


def _player_meta(proc) -> pl.DataFrame:
    players = pl.read_parquet(proc / "players.parquet").select(
        "player_id", (pl.col("first_name") + " " + pl.col("last_name")).alias("player_name"), "position")
    return players.with_columns(pl.col("position").replace_strict(POS_GROUP, default="F").alias("pos_group"))


def _team_season(proc) -> pl.DataFrame:
    toi = pl.read_parquet(proc / "toi_game.parquet").join(pl.read_parquet(proc / "games.parquet").select("game_id", "season"), on="game_id")
    teams = pl.read_parquet(proc / "games.parquet").select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")).unique("team_id")
    return (toi.group_by("player_id", "season", "team_id").agg(pl.col("toi_5v5").sum())
            .sort("toi_5v5", descending=True).group_by("player_id", "season").agg(pl.col("team_id").first(), pl.col("team_id").n_unique().alias("n_teams"))
            .join(teams, on="team_id", how="left"))


def _season_table(fits, model, core, w_stack, min_toi, meta) -> pl.DataFrame:
    rows = []
    for s, (d, _, coefs) in fits.items():
        t = coefs.with_columns(ev.headline(coefs, model, core, w_stack), (-pl.col("def_xg")).alias("xga_rapm_per60"),
                               pl.col("off_xg").alias("xgf_rapm_per60"), pl.lit(s, pl.Int64).alias("season"))
        t = t.join(ridge.player_minutes(d), on="player_id")
        rows.append(t)
    out = pl.concat(rows).join(meta, on="player_id", how="left").filter(pl.col("toi_5v5") >= min_toi)
    return out


def _yoy(tab: pl.DataFrame, col: str, teams: pl.DataFrame | None = None) -> dict:
    """Season-centred (within season x position) YOY r; optionally only players who changed teams."""
    t = tab.with_columns((pl.col(col) - pl.col(col).mean().over("season", "pos_group")).alias("_c"))
    nxt = t.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("_c").alias("_n"))
    j = t.join(nxt, on=["player_id", "season"])
    if teams is not None:
        tm = teams.select("player_id", "season", "team_id")
        j = (j.join(tm, on=["player_id", "season"])
             .join(tm.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("team_id").alias("team_next")), on=["player_id", "season"])
             .filter(pl.col("team_id") != pl.col("team_next")))
    return {g: {"r": float(np.corrcoef(x["_c"], x["_n"])[0, 1]), "n": x.height} for (g,), x in j.group_by(["pos_group"])}


def step_final(cfg):
    proc, out_dir, met = _paths(cfg)
    dc = cfg.defense
    sel = _load_json(met / "dpm_selection.json")
    alphas, model, rho, w_stack = sel["alphas"], sel["D1_model"], sel["D2"]["rho"], sel["M3_weights"]
    if model == "M0":  # common alpha: the component columns then sum exactly to the headline (G4; methodology §8.6)
        alphas = {t: alphas["xg"] for t in TARGETS}
    core = dc.core_components
    st = _stints(proc)
    seasons = sorted(cfg.data.seasons)
    meta, teams = _player_meta(proc), _team_season(proc)
    fits = chain_fits(st, seasons, alphas, rho)
    fits0 = fits if rho == 0 else chain_fits(st, seasons, alphas, 0.0)

    # ---- bootstrap CIs (per season, in parallel)
    tasks = []
    reps = int(os.environ.get("DPM_BOOT_REPS", dc.bootstrap_reps))  # override for quick end-to-end checks
    cache_dir = proc / "dpm_boot"
    cache_dir.mkdir(exist_ok=True)
    for i, s in enumerate(seasons):
        prev = fits[seasons[i - 1]][2] if i > 0 and rho > 0 else None
        key = f"{s}_B{reps}_rho{rho}_a{'-'.join(str(int(alphas[t])) for t in TARGETS)}_seed{cfg.seed}.npy"
        tasks.append((st.filter(pl.col("season") == s), alphas, reps, cfg.seed + s, prev, rho, cache_dir / key))
    with ProcessPoolExecutor(min(dc.workers, len(tasks))) as pool:
        draws = list(pool.map(_boot_season, tasks))
    q = (1 - cfg.metrics.ci_level) / 2
    ci_rows = []
    for s, dr in zip(seasons, draws):
        d = fits[s][0]
        P = d.P
        # headline draws: same linear combination of defense coefficients as ev.headline
        comb = {"M0": {"xg": 1.0}, "M1": {c: 1.0 for c in core}, "M2": {c: 1.0 for c in core + ["quality"]},
                "M3": w_stack}[model]
        h = -sum(w * dr[:, 1 + P:1 + 2 * P, TARGETS.index(t)] for t, w in comb.items())
        ci_rows.append(pl.DataFrame({"player_id": d.players, "season": s,
                                     "dpm_lo": np.quantile(h, q, 0), "dpm_hi": np.quantile(h, 1 - q, 0), "dpm_se": h.std(0)}))
    ci = pl.concat(ci_rows).with_columns(pl.col("season").cast(pl.Int64))

    # ---- leaderboards (xga_rapm_per60 = plain single-season xGA RAPM, rho = 0, for reference)
    plain = pl.concat([c.select("player_id", (-pl.col("def_xg")).alias("xga_rapm_per60"), pl.lit(s, pl.Int64).alias("season")) for s, (_, _, c) in fits0.items()])
    tab = (_season_table(fits, model, core, w_stack, 0, meta).drop("xga_rapm_per60")
           .join(plain, on=["player_id", "season"], how="left").join(ci, on=["player_id", "season"], how="left"))
    tab = tab.join(teams.select("player_id", "season", "team", "n_teams"), on=["player_id", "season"], how="left")
    tab = tab.with_columns(
        (pl.col("toi_5v5") >= dc.min_toi).alias("qualified"),
        (pl.col("dpm_per60") * pl.col("toi_5v5") / 60).alias("dpm_total"),
        *[(-pl.col(f"def_{c}")).alias(f"dpm_{c}_per60") for c in COMPONENTS],
        # valuation-ready: opponent value from own turnovers / lost faceoffs is left to those components (§5)
        (pl.col("dpm_per60") + pl.col("def_start_turnover") + pl.col("def_start_faceoff")).alias("dpm_val_per60"),
    )
    tab = tab.with_columns(
        pl.when(pl.col("qualified")).then(pl.col("dpm_per60").rank("average").over("season", "pos_group", "qualified")
                                          / pl.col("dpm_per60").count().over("season", "pos_group", "qualified") * 100).alias("dpm_pctile"))
    pm = proc / "player_season.parquet"
    if pm.exists():
        tab = tab.join(pl.read_parquet(pm).select("player_id", "season", "pm_epa_per60_ev"), on=["player_id", "season"], how="left")
    keep = (["player_id", "player_name", "team", "position", "pos_group", "season", "toi_5v5", "qualified", "dpm_per60", "dpm_lo", "dpm_hi",
             "dpm_se", "dpm_pctile", "dpm_total", "dpm_val_per60", "xga_rapm_per60", "xgf_rapm_per60"]
            + [f"dpm_{c}_per60" for c in COMPONENTS] + (["pm_epa_per60_ev"] if pm.exists() else []))
    tab = tab.select(keep).sort("season", "pos_group", "dpm_per60", descending=[False, False, True])
    tab.write_parquet(proc / "dpm_player_season.parquet")
    tab.write_csv(out_dir / "dpm_player_season.csv")

    # ---- stability gates on independent (rho = 0) fits: a prior would inflate YOY mechanically
    ind = _season_table(fits0, model, core, w_stack, dc.min_toi, meta)
    stab = {"headline_model": model, "rho": rho,
            "yoy_dpm": _yoy(ind, "dpm_per60"), "yoy_xga_rapm": _yoy(ind, "xga_rapm_per60"),
            "yoy_dpm_team_changers": _yoy(ind, "dpm_per60", teams), "yoy_xga_rapm_team_changers": _yoy(ind, "xga_rapm_per60", teams),
            "yoy_components": {c: _yoy(ind.with_columns((-pl.col(f"def_{c}")).alias("_v")), "_v") for c in COMPONENTS},
            "yoy_xgf_rapm": _yoy(ind, "xgf_rapm_per60")}
    # split-half: odd vs even games within each season, rho = 0
    sh = {}
    for s in seasons:
        ss = st.filter(pl.col("season") == s)
        halves = []
        for par in (0, 1):
            d = ridge.build_design(ss.filter(pl.col("game_id") % 2 == par), TARGETS)
            c = ridge.coef_frame(d, ridge.fit(d, alphas))
            c = c.with_columns(ev.headline(c, model, core, w_stack), (-pl.col("def_xg")).alias("xga")).join(ridge.player_minutes(d), on="player_id")
            halves.append(c.filter(pl.col("toi_5v5") >= dc.min_toi / 2).join(meta, on="player_id"))
        j = halves[0].join(halves[1], on="player_id", suffix="_b")
        sh[s] = {g: {"dpm": float(np.corrcoef(x["dpm_per60"], x["dpm_per60_b"])[0, 1]), "xga_rapm": float(np.corrcoef(x["xga"], x["xga_b"])[0, 1]),
                     "n": x.height} for (g,), x in j.group_by(["pos_group"])}
    stab["split_half"] = sh
    g2 = {g: stab["yoy_dpm"][g]["r"] >= 0.25 for g in ("F", "D")}
    g3 = {g: stab["yoy_dpm"][g]["r"] >= stab["yoy_xga_rapm"][g]["r"] for g in ("F", "D")}
    stab["G2"] = {"by_pos": g2, "pass": all(g2.values())}
    stab["G3"] = {"by_pos": g3, "pass": all(g3.values())}

    # ---- reported, not gated
    q_tab = tab.filter(pl.col("qualified"))
    rep = {}
    if "pm_epa_per60_ev" in q_tab.columns:
        rep["r_with_pm_epa"] = {g: float(np.corrcoef(x.drop_nulls("pm_epa_per60_ev")["dpm_per60"], x.drop_nulls("pm_epa_per60_ev")["pm_epa_per60_ev"])[0, 1])
                                for (g,), x in q_tab.group_by(["pos_group"])}
    rep["r_with_xgf_rapm"] = {g: float(np.corrcoef(x["dpm_per60"], x["xgf_rapm_per60"])[0, 1]) for (g,), x in q_tab.group_by(["pos_group"])}
    # raw on-ice xGA/60 per player-season (from the stints) for convergent validity
    onice = []
    for s, (d, _, _) in fits0.items():
        P = d.P
        Xd = d.X[:, 1 + P:1 + 2 * P]
        xga = np.asarray(Xd.T @ (d.S[:, TARGETS.index("xg")] / 60.0)).ravel()
        mins = np.asarray(Xd.T @ d.w).ravel()
        onice.append(pl.DataFrame({"player_id": d.players, "season": s, "onice_xga60": 60 * xga / np.maximum(mins, 1e-9)}))
    oi = q_tab.join(pl.concat(onice).with_columns(pl.col("season").cast(pl.Int64)), on=["player_id", "season"])
    rep["r_with_onice_xga60"] = {g: float(np.corrcoef(x["dpm_per60"], -x["onice_xga60"])[0, 1]) for (g,), x in oi.group_by(["pos_group"])}
    # team level: TOI-weighted sum of skater DPM (x5 skaters) vs team 5v5 xGA/60, season-centred
    team_tab = (tab.join(teams.select("player_id", "season", "team_id"), on=["player_id", "season"])
                .group_by("season", "team_id").agg(((pl.col("dpm_per60") * pl.col("toi_5v5")).sum() / pl.col("toi_5v5").sum() * 5).alias("team_dpm")))
    txga = []
    for s in fits0:
        sst = st.filter(pl.col("season") == s).join(pl.read_parquet(proc / "games.parquet").select("game_id", "home_id", "away_id"), on="game_id")
        xg_h = sst.select(pl.sum_horizontal([pl.col(f"h_{c}") for c in COMPONENTS])).to_series()
        xg_a = sst.select(pl.sum_horizontal([pl.col(f"a_{c}") for c in COMPONENTS])).to_series()
        # away team defends home attacks and vice versa
        f = pl.concat([pl.DataFrame({"team_id": sst["away_id"], "xga": xg_h, "min": sst["minutes"]}),
                       pl.DataFrame({"team_id": sst["home_id"], "xga": xg_a, "min": sst["minutes"]})])
        txga.append(f.group_by("team_id").agg((60 * pl.col("xga").sum() / pl.col("min").sum()).alias("team_xga60")).with_columns(pl.lit(s, pl.Int64).alias("season")))
    tt = team_tab.join(pl.concat(txga), on=["season", "team_id"]).with_columns(
        (pl.col("team_dpm") - pl.col("team_dpm").mean().over("season")).alias("a"), (pl.col("team_xga60") - pl.col("team_xga60").mean().over("season")).alias("b"))
    rep["team_r_dpm_vs_xga60"] = float(np.corrcoef(tt["a"], -tt["b"])[0, 1])

    # ---- multi-season windows (rho = 0; reported)
    win = []
    for i in range(dc.window - 1, len(seasons)):
        ws = seasons[i - dc.window + 1:i + 1]
        d = ridge.build_design(st.filter(pl.col("season").is_in(ws)), TARGETS)
        c = ridge.coef_frame(d, ridge.fit(d, alphas))
        c = c.with_columns(ev.headline(c, model, core, w_stack), (-pl.col("def_xg")).alias("xga_rapm_per60"), pl.lit(f"{ws[0]}-{ws[-1]}").alias("window"))
        win.append(c.join(ridge.player_minutes(d), on="player_id").join(meta, on="player_id", how="left"))
    w = pl.concat(win).select("player_id", "player_name", "position", "pos_group", "window", "toi_5v5", "dpm_per60", "xga_rapm_per60",
                              *[(-pl.col(f"def_{c}")).alias(f"dpm_{c}_per60") for c in COMPONENTS])
    w.write_csv(out_dir / "dpm_player_window.csv")

    _save_json(met / "dpm_validation.json", {"stability": stab, "reported": rep})
    log.info("G2=%s G3=%s  yoy=%s", stab["G2"], stab["G3"], stab["yoy_dpm"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STEPS, default="stints")
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args()
    run_steps(args.only or STEPS[STEPS.index(args.start):])


def run_steps(steps: list[str]) -> None:
    cfg = load_config()
    for s in steps:
        log.info("=== defense:%s ===", s)
        if s == "report":
            from src.defense.report import run as report
            report()
        else:
            globals()[f"step_{s}"](cfg)


if __name__ == "__main__":
    main()
