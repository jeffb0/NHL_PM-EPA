"""Goaltending component end to end:  python -m src.goaltending.run [--from STEP]

Steps: data -> tune (variance components, G1 vs G2) -> forward (rolling folds: GK-3, GK-4, forecast inputs) -> final
(values)
-> validate (GK-1, GK-2, summaries) -> report. Decision rules: docs/goaltending/methodology.md §6, applied mechanically.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.goaltending import model as gm

log = logging.getLogger("goaltending")
STEPS = ["data", "tune", "forward", "final", "validate", "report"]


def _paths(cfg):
    return cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.output_dir), cfg.path(cfg.data.metrics_dir)


def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_shots(proc) -> pl.DataFrame:
    return pl.read_parquet(proc / "gk_shots.parquet")


def _with_status(df: pl.DataFrame, T: int) -> pl.DataFrame:
    """Season T+1 shots with each goalie's season-T fringe status (known at forecast time); goalies with no season-T
    shots count as replacement."""
    status = df.filter(pl.col("season") == T).group_by("goalie").agg(pl.col("gfringe").first().alias("_fr"))
    return (
        df.filter(pl.col("season") == T + 1)
        .drop("gfringe")
        .join(status, on="goalie", how="left")
        .with_columns(pl.col("_fr").fill_null(True).alias("gfringe"))
        .drop("_fr")
    )


def paired_z(loss_a: pl.DataFrame, a: str, b: str) -> dict:
    """Per-game sums of two loss columns; positive mean_diff = b minus a > 0 means a is better."""
    d = (loss_a[b] - loss_a[a]).to_numpy()
    return {
        "mean_improvement_per_game": float(d.mean()),
        "z": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
        "n_games": int(len(d)),
    }


def _game_loss(df: pl.DataFrame, probs: dict[str, np.ndarray]) -> pl.DataFrame:
    y = df["goal"].to_numpy().astype(float)
    return (
        pl.DataFrame({"game_id": df["game_id"], **{k: gm.logloss(p, y) for k, p in probs.items()}})
        .group_by("game_id")
        .agg(pl.all().sum())
    )


# ----------------------------------------------------------------------------- steps
def step_data(cfg):
    proc, _, met = _paths(cfg)
    shots = pl.read_parquet(
        proc / "fin_shots.parquet",
        columns=[
            "game_id",
            "event_idx",
            "season",
            "shooter",
            "pos_group",
            "goalie",
            "opp_team_id",
            "goal",
            "strength_g",
        ],
    )
    sh = pl.read_parquet(proc / "fin_shot_shifts.parquet")
    assert shots.height == sh.height and all(
        (shots[c] == sh[c]).all() for c in ("shooter", "season", "pos_group", "strength_g")
    ), "fin_shots and fin_shot_shifts are misaligned"
    xg = pl.read_parquet(proc / "fin_xg.parquet").select("game_id", "event_idx", "xg_f")
    df = shots.hstack(sh.select("T")).join(xg, on=["game_id", "event_idx"], how="left")
    assert df["xg_f"].null_count() == 0
    p = np.clip(df["xg_f"].to_numpy(), 1e-5, 1 - 1e-5)
    df = df.with_columns(
        pl.Series("offset", np.log(p / (1 - p)) + df["T"].to_numpy()), pl.col("opp_team_id").alias("team_id")
    )
    all_goals = (
        pl.read_parquet(proc / "shots_xg.parquet", columns=["goal", "empty_net"])
        .filter(~pl.col("empty_net"))["goal"]
        .sum()
    )
    with_goalie = int(df.filter(pl.col("goalie").is_not_null())["goal"].sum())
    df = df.filter(pl.col("goalie").is_not_null())
    df = df.with_columns((pl.len().over("goalie", "season") < cfg.goaltending.fringe_shots).alias("gfringe")).select(
        "game_id", "event_idx", "season", "goalie", "team_id", "goal", "strength_g", "xg_f", "offset", "gfringe"
    )
    df.write_parquet(proc / "gk_shots.parquet")
    _save(
        met / "gk_data.json",
        {
            "shots": df.height,
            "goals_with_goalie": with_goalie,
            "goals_not_empty_net": int(all_goals),
            "coverage": with_goalie / int(all_goals),
            "goalies": int(df["goalie"].n_unique()),
        },
    )
    log.info("goalie shots %d, goal coverage %.4f", df.height, with_goalie / int(all_goals))


def step_tune(cfg):
    proc, _, met = _paths(cfg)
    gc = cfg.goaltending
    df = load_shots(proc)
    train = df.filter(pl.col("season").is_in(gc.sigma_seasons))
    res: dict = {"n_train_shots": train.height}
    s2 = {}
    for name, team in (("G1", False), ("G2", True)):
        lay = gm.build_layout(train, team, fringe=True)
        mode, s, trace = gm.fit(train, lay, em_iters=gc.em_iters, em_tol_sd=gc.em_tol_sd)
        s2[name] = s
        res.setdefault("sd", {})[name] = {k: float(np.sqrt(v)) for k, v in s.items()}
        res.setdefault("em_converged", {})[name] = bool(trace[-1].get("converged", False)) if trace else None
        log.info("EM %s: sd %s converged=%s", name, res["sd"][name], res["em_converged"][name])
    res["s2"] = s2
    # validation fold: fit on seasons before it, predict it with the persistent effect only (D-type rule)
    vf = gc.validation_fold
    tr, te = (
        df.filter(pl.col("season") < vf),
        df.filter(pl.col("season") == vf).with_columns(pl.lit(False).alias("gfringe")),
    )
    probs = {}
    for name, team in (("G1", False), ("G2", True)):
        lay = gm.build_layout(tr, team, fringe=True)
        mode, _, _ = gm.fit(tr, lay, s2=s2[name])
        p = gm.shot_probs(te, lay, mode.beta, cell_season=pl.lit(vf - 1))
        probs[name] = p["v"]
        probs["baseline"] = p["cell_only"]
    loss = _game_loss(te, probs)
    res["validation"] = {
        "G1_vs_baseline": paired_z(loss, "G1", "baseline"),
        "G2_vs_baseline": paired_z(loss, "G2", "baseline"),
        "G2_vs_G1": paired_z(loss, "G2", "G1"),
    }
    use_team = bool(res["validation"]["G2_vs_G1"]["z"] > gc.z_threshold)
    res["variant"] = "G2" if use_team else "G1"
    _save(met / "gk_tune.json", res)
    log.info("D-type: %s (G2 vs G1 z=%.2f)", res["variant"], res["validation"]["G2_vs_G1"]["z"])


def step_forward(cfg):
    proc, _, met = _paths(cfg)
    gc = cfg.goaltending
    tune = _load(met / "gk_tune.json")
    team = tune["variant"] == "G2"
    s2 = tune["s2"][tune["variant"]]
    df = load_shots(proc)
    seasons = sorted(df["season"].unique().to_list())
    rates, folds, real = [], [], []
    for T in seasons[:-1]:
        tr, te = df.filter(pl.col("season") <= T), _with_status(df, T)
        lay = gm.build_layout(tr, team, fringe=True)
        mode, _, _ = gm.fit(tr, lay, s2=s2)
        ptr = gm.shot_probs(tr.filter(pl.col("season") == T), lay, mode.beta)
        cur = tr.filter(pl.col("season") == T).with_columns(
            pl.Series("saved_v", ptr["base"] - ptr["v"]), pl.Series("saved_s", ptr["base"] - ptr["with_status"])
        )
        g = cur.group_by("goalie").agg(
            pl.col("saved_v").sum().alias("saved"), pl.col("saved_s").sum().alias("saved_s"), pl.len().alias("n_prev")
        )
        rate = g.with_columns(
            (pl.col("saved") / pl.col("n_prev")).alias("rate_v"), (pl.col("saved_s") / pl.col("n_prev")).alias("rate_s")
        )
        # goals saved per shot by a replacement goalie relative to an average one (the fringe effect)
        rep = float((ptr["base"] - ptr["rep"]).mean())
        rates.append(
            rate.select("goalie", pl.lit(T + 1, pl.Int64).alias("season"), "rate_v", "n_prev").with_columns(
                pl.lit(rep).alias("rep_v")
            )
        )
        pte = gm.shot_probs(te, lay, mode.beta, cell_season=pl.lit(T))
        y = te["goal"].to_numpy().astype(float)
        loss = _game_loss(te, {"baseline": pte["cell_only"], "model": pte["v"], "with_status": pte["with_status"]})
        z = paired_z(loss, "model", "baseline")
        folds.append(
            {
                "train_end": T,
                "test_season": T + 1,
                "logloss_baseline": float(gm.logloss(pte["cell_only"], y).mean()),
                "logloss_model": float(gm.logloss(pte["v"], y).mean()),
                "z_with_status": paired_z(loss, "with_status", "baseline")["z"],
                **z,
            }
        )
        realized = (
            te.with_columns(pl.Series("saved", pte["cell_only"] - y))
            .group_by("goalie")
            .agg(pl.col("saved").sum().alias("saved"), pl.len().alias("n"))
        )
        real.append(
            realized.join(rate.select("goalie", "rate_v", "rate_s"), on="goalie", how="left").with_columns(
                pl.lit(T + 1).alias("season"), pl.col("rate_v").fill_null(0.0), pl.col("rate_s").fill_null(rep)
            )
        )
        log.info("fold %d->%d: model vs baseline z=%.2f", T, T + 1, z["z"])
    pl.concat(rates).write_parquet(proc / "gk_rolling_rates.parquet")
    # GK-3: test folds pooled
    test = [f for f in folds if f["test_season"] in gc.test_folds]
    tl = []
    for f in test:
        T = f["train_end"]
        tr, te = df.filter(pl.col("season") <= T), _with_status(df, T)
        lay = gm.build_layout(tr, team, fringe=True)
        mode, _, _ = gm.fit(tr, lay, s2=s2)
        p = gm.shot_probs(te, lay, mode.beta, cell_season=pl.lit(T))
        tl.append(_game_loss(te, {"baseline": p["cell_only"], "model": p["v"], "with_status": p["with_status"]}))
    pooled = pl.concat(tl)
    gk3 = paired_z(pooled, "model", "baseline")
    gk3_u = paired_z(pooled, "with_status", "baseline")
    # GK-4: forward slope of realized saved per shot on the prior persistent estimate (all folds; goalies with enough
    # shots)
    r = pl.concat(real).filter(pl.col("n") >= gc.min_shots_next)

    def slope(col: str) -> dict:
        x, yv, w = r[col].to_numpy(), (r["saved"] / r["n"]).to_numpy(), r["n"].to_numpy().astype(float)
        A = np.column_stack([np.ones(len(x)), x])
        W = np.diag(w)
        beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ yv)
        res_ = yv - A @ beta
        cov = np.linalg.inv(A.T @ W @ A) * float((w * res_**2).sum() / (len(x) - 2))
        return {
            "slope": float(beta[1]),
            "se": float(np.sqrt(cov[1, 1])),
            "n_goalie_seasons": int(len(x)),
            "intercept": float(beta[0]),
        }

    gk4, gk4_u = slope("rate_v"), slope("rate_s")
    out = {
        "folds": folds,
        "GK3": {
            **gk3,
            "pass": bool(gk3["z"] > gc.z_threshold),
            "test_folds": gc.test_folds,
            "with_prior_fringe_status": gk3_u,
        },
        "GK4": {
            **gk4,
            "pass": bool(gc.slope_band[0] <= gk4["slope"] <= gc.slope_band[1]),
            "with_prior_fringe_status": gk4_u,
        },
    }
    _save(met / "gk_forward.json", out)
    log.info("GK-3 z=%.2f  GK-4 slope %.2f ± %.2f", gk3["z"], gk4["slope"], gk4["se"])


def step_final(cfg):
    proc, out_dir, met = _paths(cfg)
    gc = cfg.goaltending
    tune = _load(met / "gk_tune.json")
    s2_all = tune["s2"]
    df = load_shots(proc)
    games = pl.read_parquet(proc / "games.parquet")
    abbr = pl.concat(
        [
            games.select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")),
            games.select(pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team")),
        ]
    ).unique("team_id")
    gpw = _load(met / "val_gpw.json")["goals_per_win"]
    tabs = {}
    for name, team in (("G1", False), ("G2", True)):
        lay = gm.build_layout(df, team, fringe=True)
        mode, _, _ = gm.fit(df, lay, s2=s2_all[name])
        p = gm.shot_probs(df, lay, mode.beta)
        d = df.with_columns(
            pl.Series("p_base", p["base"]),
            pl.Series("p_full", p["full"]),
            pl.Series("p_rep", p["rep"]),
            pl.Series("y", df["goal"].to_numpy().astype(float)),
            pl.Series("var", p["full"] * (1 - p["full"])),
        )
        g = d.group_by("goalie", "season").agg(
            pl.len().alias("shots"),
            pl.col("y").sum().alias("goals"),
            pl.col("p_base").sum().alias("expected_goals"),
            (pl.col("p_base") - pl.col("p_full")).sum().alias("gsaa"),
            (pl.col("p_rep") - pl.col("p_full")).sum().alias("goalie_gar"),
            (pl.col("p_base") - pl.col("p_rep")).sum().alias("rep_saved"),
            pl.col("var").sum().alias("W"),
            pl.col("team_id").mode().first().alias("team_id"),
        )
        tt = d.group_by("team_id", "season").agg(
            pl.len().alias("shots"),
            (pl.col("p_rep") - pl.col("p_full")).sum().alias("gk_gar"),
            (pl.col("p_rep") - pl.col("y")).sum().alias("gk_raw_gar"),
            (pl.col("p_base") - pl.col("p_full")).sum().alias("gk_gsaa"),
        )
        tabs[name] = (g, mode, lay, tt)
    name = tune["variant"]
    g, mode, lay, team_tab = tabs[name]
    V = mode.posterior_cov()
    ucol = g["goalie"].replace_strict(lay.goalie, default=-1, return_dtype=pl.Int64).to_numpy()
    dcol = (
        (g["goalie"] * 10000 + g["season"])
        .replace_strict(lay.goalie_season, default=-1, return_dtype=pl.Int64)
        .to_numpy()
    )
    var_eff = V[ucol, ucol] + V[dcol, dcol] + 2 * V[ucol, dcol]
    g = g.with_columns(pl.Series("sd_eff", np.sqrt(np.maximum(var_eff, 0)))).with_columns(
        (pl.col("W") * pl.col("sd_eff")).alias("gsaa_sd")
    )
    # goals saved per shot by the replacement (fringe-effect) goalie relative to average
    rep = float(g["rep_saved"].sum() / g["shots"].sum())
    g = g.with_columns(
        (pl.col("goals") - pl.col("expected_goals")).alias("raw_goals_minus_expected"),
        (pl.col("shots") < gc.fringe_shots).alias("fringe"),
    )
    g = g.with_columns(
        (pl.col("goalie_gar") / gpw).alias("goalie_wins"), (pl.col("gsaa_sd") / gpw).alias("goalie_wins_sd")
    )
    players = pl.read_parquet(proc / "players.parquet").select(
        pl.col("player_id").alias("goalie"), (pl.col("first_name") + " " + pl.col("last_name")).alias("player_name")
    )
    out = (
        g.join(players, on="goalie", how="left")
        .join(abbr, on="team_id", how="left")
        .rename({"goalie": "player_id"})
        .with_columns(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64))
    )
    out = out.select(
        "player_id",
        "player_name",
        "team",
        "season",
        "shots",
        "goals",
        "expected_goals",
        "gsaa",
        "goalie_gar",
        "gsaa_sd",
        "goalie_wins",
        "goalie_wins_sd",
        "fringe",
        "raw_goals_minus_expected",
    ).sort("season", "goalie_gar", descending=[False, True])
    (
        team_tab.join(abbr, on="team_id")
        .drop("team_id")
        .with_columns(pl.col("season").cast(pl.Int64))
        # goalie value by defending team-season (shrunk gk_gar; unshrunk gk_raw_gar): the joint calibration's regressors
        .write_parquet(proc / "goalie_team_season.parquet")
    )
    out.write_parquet(proc / "goalie_player_season.parquet")
    out.write_csv(out_dir / "goalie_player_season.csv")
    # G1 vs G2 agreement (goalie-season saved per shot)
    other = "G1" if name == "G2" else "G2"
    a = tabs[name][0].select("goalie", "season", (pl.col("gsaa") / pl.col("shots")).alias("a"), "shots")
    b = tabs[other][0].select("goalie", "season", (pl.col("gsaa") / pl.col("shots")).alias("b"))
    j = a.join(b, on=["goalie", "season"]).filter(pl.col("shots") >= gc.min_shots_yoy)
    _save(
        met / "gk_final.json",
        {
            "variant": name,
            "replacement_saved_per_100_shots": rep * 100,
            "corr_G1_G2": float(np.corrcoef(j["a"], j["b"])[0, 1]),
            "n_goalie_seasons": out.height,
        },
    )
    log.info("goalie values %d rows; replacement %.3f goals per 100 shots", out.height, rep * 100)


def _yoy(eff: pl.DataFrame, min_shots: int) -> dict:
    t = eff.with_columns((pl.col("eff") - pl.col("eff").mean().over("season")).alias("_c"))
    nxt = t.select(
        "goalie", (pl.col("season") - 1).alias("season"), pl.col("_c").alias("_n"), pl.col("shots").alias("s_n")
    )
    j = t.join(nxt, on=["goalie", "season"]).filter((pl.col("shots") >= min_shots) & (pl.col("s_n") >= min_shots))
    return {"r": float(np.corrcoef(j["_c"], j["_n"])[0, 1]), "n": j.height}


def step_validate(cfg):
    proc, _, met = _paths(cfg)
    gc = cfg.goaltending
    tune = _load(met / "gk_tune.json")
    dat = _load(met / "gk_data.json")
    fw = _load(met / "gk_forward.json")
    df = load_shots(proc)
    res: dict = {
        "GK1": {"coverage": dat["coverage"], "pass": bool(dat["coverage"] >= 0.995)},
        "GK3": fw["GK3"],
        "GK4": fw["GK4"],
        "variant": tune["variant"],
    }
    team = tune["variant"] == "G2"
    s2 = tune["s2"][tune["variant"]]
    s2_single = {"u": s2["u"] + s2["d"], **({"t": s2["t"]} if team else {})}
    shrunk = []
    for s in sorted(df["season"].unique().to_list()):
        d = df.filter(pl.col("season") == s)
        lay = gm.build_layout(d, team, persistent=True, season_dev=False)
        mode, _, _ = gm.fit(d, lay, s2=s2_single)
        lo, _ = lay.blocks["u"]
        eff = {g: float(mode.beta[lo + i]) for i, g in enumerate(sorted(lay.goalie, key=lay.goalie.get))}
        n = d.group_by("goalie").agg(
            pl.len().alias("shots"), (pl.col("goal").sum() - (pl.col("xg_f").sum())).alias("rawx")
        )
        shrunk.append(
            n.with_columns(
                pl.col("goalie").replace_strict(eff, return_dtype=pl.Float64).alias("eff"),
                pl.lit(s, pl.Int64).alias("season"),
            )
        )
    tab = pl.concat(shrunk)
    res["GK2"] = {
        "yoy_shrunk": _yoy(tab.select("goalie", "season", "shots", "eff"), gc.min_shots_yoy),
        "yoy_raw_goals_minus_xg": _yoy(
            tab.select("goalie", "season", "shots", (-pl.col("rawx") / pl.col("shots")).alias("eff")), gc.min_shots_yoy
        ),
    }
    res["GK2"]["pass"] = bool(res["GK2"]["yoy_shrunk"]["r"] >= gc.yoy_min)
    out = pl.read_parquet(proc / "goalie_player_season.parquet")
    starters = out.filter(pl.col("shots") >= gc.starter_shots)
    res["summary"] = {
        "starters": starters.height,
        "mean_gar_starters": float(starters["goalie_gar"].mean()),
        "sd_gar_starters": float(starters["goalie_gar"].std()),
        "sd_gar_starters_wins": float(starters["goalie_wins"].std()),
        "mean_sd_starters": float(starters["gsaa_sd"].mean()),
        "final": _load(met / "gk_final.json"),
        "sd": tune["sd"][tune["variant"]],
        "validation": tune["validation"],
    }
    _save(met / "gk_validation.json", res)
    log.info(
        "GK-1 %s | GK-2 r=%.2f | GK-3 z=%.2f | GK-4 slope %.2f",
        res["GK1"]["pass"],
        res["GK2"]["yoy_shrunk"]["r"],
        fw["GK3"]["z"],
        fw["GK4"]["slope"],
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STEPS, default="data")
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args()
    run_steps(args.only or STEPS[STEPS.index(args.start) :])


def run_steps(steps: list[str]) -> None:
    cfg = load_config()
    for s in steps:
        log.info("=== goaltending:%s ===", s)
        if s == "report":
            from src.goaltending.report import run as report

            report()
        else:
            globals()[f"step_{s}"](cfg)


if __name__ == "__main__":
    main()
