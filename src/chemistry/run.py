"""Chemistry model end to end:  python -m src.chemistry.run   (docs/chemistry/methodology.md)."""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import polars as pl

from src.chemistry import model as cm
from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun

log = logging.getLogger("chemistry")
ALPHA_B = [500.0, 1500.0, 5000.0, 15000.0]
RANKS = [2, 4, 8]
ALPHA_V = [1e3, 1e4, 1e5]
TUNE_FOLDS = [2019, 2020]
TEST = [2021, 2022, 2023, 2024, 2025]


def wmse(D: cm.Design, p: cm.Params) -> float:
    e = D.y - cm.predict(D, p)
    return float((D.w * e * e).sum() / D.w.sum())


def game_z(D: cm.Design, pa: cm.Params, pf: cm.Params, mask: np.ndarray | None = None) -> dict:
    """Paired z that F beats A: per-game sums of the minute-weighted squared-error difference."""
    d = D.w * ((D.y - cm.predict(D, pa)) ** 2 - (D.y - cm.predict(D, pf)) ** 2)
    g, w = D.game, D.w
    if mask is not None:
        d, g, w = d[mask], g[mask], w[mask]
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=d)
    return {"mean_diff_per_minute": float(d.sum() / w.sum()), "z": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), "n_games": int(len(s)), "minutes": float(w.sum())}


def new_combo_mask(D: cm.Design, T: int, proc) -> np.ndarray:
    """Stints with at least one skater who was not on that team (primary team) the season before."""
    g = pl.read_parquet(proc / "games.parquet").select("game_id", "home_abbrev", "away_abbrev")
    prev = pl.read_parquet(proc / "player_value_calibrated_all.parquet").filter(pl.col("season") == T - 1).select(pl.col("player_id").cast(pl.Int64), "team")
    teams = sorted(set(g["home_abbrev"].to_list()) | set(g["away_abbrev"].to_list()))
    code = {t: i for i, t in enumerate(teams)}
    pid = prev["player_id"].to_numpy()
    order = np.argsort(pid)
    pid, pteam = pid[order], np.array([code.get(t, -1) for t in prev["team"].to_list()])[order]
    gi = pl.DataFrame({"game_id": D.game}).join(g, on="game_id", how="left")
    out = np.zeros(len(D.game), dtype=bool)
    for ids, col in ((D.home, "home_abbrev"), (D.away, "away_abbrev")):
        tc = np.array([code[t] for t in gi[col].to_list()])
        idx = np.clip(np.searchsorted(pid, ids), 0, len(pid) - 1)
        last = np.where(pid[idx] == ids, pteam[idx], -1)
        out |= (last != tc[:, None]).any(1)
    return out


def run() -> dict:
    cfg = load_config()
    proc, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.metrics_dir)
    st = cm.stint_frame(proc)

    def fold(T: int):
        tr = st.filter(pl.col("season").is_in([T - 2, T - 1]))
        D, players, centre = cm.design(tr)
        Dt, _, _ = cm.design(st.filter(pl.col("season") == T), players, centre)
        return D, players, centre, Dt

    # ---- tuning on two forward folds
    folds = {T: fold(T) for T in TUNE_FOLDS}
    tune: dict = {"additive": {}, "fm": {}}
    for ab in ALPHA_B:
        tune["additive"][str(ab)] = float(np.mean([wmse(Dt, cm.fit_additive(D, pl_, c, ab)) for D, pl_, c, Dt in folds.values()]))
    ab = float(min(tune["additive"], key=tune["additive"].get))
    adds = {T: cm.fit_additive(D, pl_, c, ab) for T, (D, pl_, c, _) in folds.items()}
    log.info("additive alpha_b = %g (forward error %s)", ab, {k: round(v, 5) for k, v in tune["additive"].items()})
    for k, av in itertools.product(RANKS, ALPHA_V):
        tune["fm"][f"{k}|{av:g}"] = float(np.mean([wmse(folds[T][3], cm.fit_fm(folds[T][0], adds[T], k, ab, av)) for T in TUNE_FOLDS]))
        log.info("fm k=%d alpha_v=%g forward error %.5f", k, av, tune["fm"][f"{k}|{av:g}"])
    best = min(tune["fm"], key=tune["fm"].get)
    k, av = int(best.split("|")[0]), float(best.split("|")[1])
    res: dict = {"tuned": {"alpha_b": ab, "rank": k, "alpha_v": av, "forward_error_additive": tune["additive"][str(ab)], "forward_error_fm": tune["fm"][best]}, "tuning_grid": tune}
    del folds

    # ---- CH-1 on the test seasons, and the traits used by the game model (seasons from 2019)
    V_by_season, per, fits = {}, {}, {}
    tot = {"d": [], "dn": []}
    for T in TUNE_FOLDS + TEST:
        D, players, centre, Dt = fold(T)
        add = cm.fit_additive(D, players, centre, ab)
        fm = cm.fit_fm(D, add, k, ab, av)
        V_by_season[T] = dict(zip(players.tolist(), fm.V))
        fits[T] = (players, fm.V, np.asarray((D.Ah + D.Aa).T @ D.w).ravel())
        if T in TEST:
            mask = new_combo_mask(Dt, T, proc)
            per[str(T)] = {"wmse_A": wmse(Dt, add), "wmse_F": wmse(Dt, fm), "all": game_z(Dt, add, fm), "new_combinations": game_z(Dt, add, fm, mask), "share_minutes_new": float(Dt.w[mask].sum() / Dt.w.sum()),
                           "sd_additive": float(np.std((Dt.Ah - Dt.Aa) @ fm.beta)), "sd_synergy": float(np.std(cm._syn(Dt.Ah, fm.V)[0] - cm._syn(Dt.Aa, fm.V)[0]))}
            d = Dt.w * ((Dt.y - cm.predict(Dt, add)) ** 2 - (Dt.y - cm.predict(Dt, fm)) ** 2)
            for key, m in (("d", np.ones(len(d), bool)), ("dn", mask)):
                _, inv = np.unique(Dt.game[m], return_inverse=True)
                tot[key].append(np.bincount(inv, weights=d[m]))
            log.info("season %d: A %.5f F %.5f z %.2f (new combinations z %.2f)", T, per[str(T)]["wmse_A"], per[str(T)]["wmse_F"], per[str(T)]["all"]["z"], per[str(T)]["new_combinations"]["z"])
        if T == TEST[-1]:
            fm2 = cm.fit_fm(D, add, k, ab, av, seed=1)
            iu = np.triu_indices(len(players), 1)
            res["seed_check_pair_synergy_corr"] = float(np.corrcoef((fm.V @ fm.V.T)[iu], (fm2.V @ fm2.V.T)[iu])[0, 1])
    z = lambda s: float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))            # noqa: E731
    res["stint"] = {"by_season": per, "pooled_z": z(np.concatenate(tot["d"])), "pooled_z_new_combinations": z(np.concatenate(tot["dn"]))}
    # stability of pair synergies between non-overlapping fits (2021-22 and 2023-24), players with >= 500 minutes in both
    (p1, V1, m1), (p2, V2, m2) = fits[2023], fits[2025]
    common = np.intersect1d(p1[m1 >= 500], p2[m2 >= 500])
    G1, G2 = V1[np.searchsorted(p1, common)], V2[np.searchsorted(p2, common)]
    iu = np.triu_indices(len(common), 1)
    res["pair_synergy_stability"] = {"n_players": int(len(common)), "corr": float(np.corrcoef((G1 @ G1.T)[iu], (G2 @ G2.T)[iu])[0, 1])}

    # ---- CH-2: roster-fit covariate in the game model (frozen hyperparameters, original rate model)
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    pri = gd_.build_priors(proc, cfg, seasons, games)
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    s_arr, y_all = t["season"].to_numpy(), t["home_win"].to_numpy()
    te = t["season"].is_in(grun.TEST).to_numpy()

    def feat(T: int):
        Vd = V_by_season[T]
        zero = np.zeros(k)
        return lambda ids, m: cm.roster_synergy(np.array([Vd.get(i, zero) for i in ids.tolist()]), m / 60.0)

    def oos(X: np.ndarray) -> np.ndarray:
        p = np.full(len(y_all), np.nan)
        for T in grun.TEST:
            tr = (s_arr >= grun.TUNE[0]) & (s_arr < T)
            p[s_arr == T] = gm.logit_pred(gm.logit_fit(X[tr], y_all[tr]), X[s_arr == T])
        return p[te]

    game: dict = {}
    for name, kw in (("P", {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]}), ("P_pre", {"lam": 0.0, "alpha": 0.0, "n0": hp["n0"], "update": False})):
        parts = [gm.run_player_season(games, pri, T, feature=feat(T), **kw) for T in seasons]
        mu, ch = np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])
        X0 = np.column_stack([mu, t["b2b_home"].to_numpy(), t["b2b_away"].to_numpy()])
        X1 = np.column_stack([X0, ch])
        l0, l1 = gm.logloss(oos(X0), y_all[te]), gm.logloss(oos(X1), y_all[te])
        game[name] = {"logloss_without": float(l0.mean()), "logloss_with": float(l1.mean()), **grun.paired(l1, l0, t["date"].to_numpy()[te]),
                      "coef_all_seasons": float(gm.logit_fit(X1[s_arr >= grun.TUNE[0]], y_all)[-1]), "sd_chem_diff": float(ch.std()), "corr_chem_mu": float(np.corrcoef(ch, mu)[0, 1])}
        log.info("game %s: %.5f -> %.5f z %.2f", name, game[name]["logloss_without"], game[name]["logloss_with"], game[name]["z"])
    res["game"] = {"n_games": int(te.sum()), **game}
    g = {"CH1": {"z": res["stint"]["pooled_z"], "z_new_combinations": res["stint"]["pooled_z_new_combinations"], "pass": bool(res["stint"]["pooled_z"] > 2)},
         "CH2": {"z": game["P"]["z"], "pass": bool(game["P"]["z"] > 2)}, "CH3": {"z_P": game["P"]["z"], "z_P_pre": game["P_pre"]["z"], "pass": bool(game["P"]["z"] > -2 and game["P_pre"]["z"] > -2)}}
    res["gates"] = g
    res["implement"] = bool(g["CH2"]["pass"] and g["CH3"]["pass"])
    (met / "chemistry.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("gates %s implement=%s", {a: b["pass"] for a, b in g.items()}, res["implement"])
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
