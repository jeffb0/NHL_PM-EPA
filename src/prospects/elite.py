"""Elite-prospect priors and experience-weighted in-season learning:  python -m src.prospects.elite
(docs/prospects/elite.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.gamemodel import data as gd_
from src.gamemodel import model as gm
from src.gamemodel import run as grun
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.projection.model import Spec
from src.prospects import features as pf

log = logging.getLogger("prospects")
PLAYER_TEST = [2020, 2021, 2022, 2023, 2024, 2025]
KAPPAS = [0.0, 0.5, 1.0, 2.0, 4.0]


def player_level(p: pl.DataFrame, bio: pl.DataFrame, yp: pl.DataFrame) -> tuple[dict, dict]:
    fr_t = pf.with_yp(pd_.design_frame(p, bio, 1, floor=pr.REVISED_FLOOR), yp)
    fr_a = pf.with_yp(pd_.design_frame(p, bio, 1, targets_only=False), yp)
    M, ME = Spec(), pf.spec(yp)
    parts = []
    for T in PLAYER_TEST:
        tr, te = pr.train_rows(fr_t, 1, T), fr_a.filter(pl.col("season") == T)
        m0, m1 = pm.fit_models(tr, M), pm.fit_models(tr, ME)
        parts.append(
            te.select("player_id", "season", "pos_group", "m", "m_eff", pl.col("r_total").alias("y")).with_columns(
                pl.Series("M", pm.predict_models(m0, te, M)), pl.Series("ME", pm.predict_models(m1, te, ME))
            )
        )
    e = pl.concat(parts).join(
        yp.select("player_id", "season", "nhl_gp_before", "overall"), on=["player_id", "season"], how="left"
    )

    def cmp(d: pl.DataFrame) -> dict:
        if d.height < 5:
            return {"n": d.height}
        mse = lambda c: float((d["m"] * (d["y"] - d[c]) ** 2).sum() / d["m"].sum())
        return {
            "n": d.height,
            "mse_M": mse("M"),
            "mse_new": mse("ME"),
            "z": pr.paired_cluster(d, "ME", "M")["z"],
            "mean_actual": float((d["m"] * d["y"]).sum() / d["m"].sum()),
            "mean_M": float((d["m"] * d["M"]).sum() / d["m"].sum()),
            "mean_new": float((d["m"] * d["ME"]).sum() / d["m"].sum()),
        }

    rook = e.filter((pl.col("nhl_gp_before") == 0) & (pl.col("m") >= 200))
    out = {
        "true_rookies": cmp(rook),
        "first_round_rookies": cmp(rook.filter(pl.col("overall") <= 32)),
        "top10_rookies": cmp(rook.filter(pl.col("overall") <= 10)),
        "top3_rookies": cmp(rook.filter(pl.col("overall") <= 3)),
        "thin": cmp(e.filter((pl.col("m_eff") < 1000) & (pl.col("m") >= 200))),
        "all": cmp(e.filter(pl.col("m") >= 300)),
    }
    # final model on every season: coefficients and the implied debut ratings (a no-history player of typical size at
    # the given age, with the bin's mean NHL-equivalent production)
    tr = fr_t.filter(pl.col("season") >= pr.FIRST_TARGET)
    models = pm.fit_models(tr, ME)
    names = pm.feature_names(ME)
    k = len(pf.ELITE)
    coefs = {g: dict(zip(names[-k:], [float(x) for x in models[g].raw_coef()[-k:]])) for g in ("F", "D")}
    deb = fr_a.join(
        yp.select("player_id", "season", "nhl_gp_before", "overall", "top3", "p4_10", "p11_32", "d0", "d1"),
        on=["player_id", "season"],
        how="left",
    ).filter((pl.col("nhl_gp_before") == 0) & (pl.col("m_eff") == 0) & (pl.col("m") >= 200))
    implied = {}
    rng = np.random.default_rng(0)
    pids = tr["player_id"].unique().to_list()
    boots = []
    for _ in range(100):
        pick = pl.DataFrame({"player_id": rng.choice(pids, len(pids)).astype(np.int64)}).group_by("player_id").len()
        bt = tr.join(pick, on="player_id").with_columns((pl.col("m") * pl.col("len")).alias("m_w"))
        boots.append(pm.fit_models(bt, ME, lam={g: models[g].lam for g in ("F", "D")}, weight="m_w"))
    cases = {
        "top-3 pick, debut at 18 (D0)": ("top3", "d0"),
        "top-3 pick, debut a year later (D1)": ("top3", "d1"),
        "picks 4-10, D0": ("p4_10", "d0"),
        "picks 4-10, D1": ("p4_10", "d1"),
        "picks 11-32, D1": ("p11_32", "d1"),
        "later pick or undrafted, later debut": (None, None),
    }
    for g in ("F", "D"):
        dg = deb.filter(pl.col("pos_group") == g)
        if dg.height == 0:
            continue
        for label, (slot, timing) in cases.items():
            sub = dg.filter(
                (pl.col(slot) == 1) if slot else ((pl.col("top3") + pl.col("p4_10") + pl.col("p11_32")) == 0)
            ).filter((pl.col(timing) == 1) if timing else ((pl.col("d0") + pl.col("d1")) == 0))
            ref = sub if sub.height >= 3 else dg.filter(pl.col(slot) == 1) if slot else dg
            row = dg.head(1).with_columns(
                pl.lit(0.0).alias("yp_top3"),
                pl.lit(0.0).alias("yp_p4_10"),
                pl.lit(0.0).alias("yp_p11_32"),
                pl.lit(0.0).alias("yp_d0"),
                pl.lit(0.0).alias("yp_d1"),
                pl.lit(float(ref["yp_nhle"].mean())).alias("yp_nhle"),
                pl.lit(float(ref["yp_has_nhle"].mean())).alias("yp_has_nhle"),
                pl.lit(float(ref["age"].mean())).alias("age"),
                pl.lit(float(dg["height_in"].mean())).alias("height_in"),
                pl.lit(float(dg["weight_lb"].mean())).alias("weight_lb"),
            )
            if slot:
                row = row.with_columns(pl.lit(1.0).alias(f"yp_{slot}"))
            if timing:
                row = row.with_columns(pl.lit(1.0).alias(f"yp_{timing}"))
            est = float(pm.predict_models(models, row, ME)[0])
            bs = [float(pm.predict_models(b, row, ME)[0]) for b in boots]
            implied[f"{g} | {label}"] = {
                "rating60": est,
                "lo90": float(np.quantile(bs, 0.05)),
                "hi90": float(np.quantile(bs, 0.95)),
                "n_debuts_in_data": sub.height,
                "actual_mean_rate": float((sub["m"] * sub["r_total"]).sum() / sub["m"].sum()) if sub.height else None,
                "generic_model": float(pm.predict_models(pm.fit_models(tr, M), row, M)[0]),
            }
    return out, {"coefficients": coefs, "implied_debut_ratings": implied}


def run() -> dict:
    cfg = load_config()
    proc, raw, met = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.raw_dir), cfg.path(cfg.data.metrics_dir)
    yp = pf.elite_table(raw, list(range(2018, 2027)))
    yp.write_parquet(proc / "elite_features.parquet")
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    bio = pd_.player_bio(proc, raw)
    res: dict = {}
    res["player"], res["model"] = player_level(p, bio, yp)
    log.info(
        "player level: true rookies z %.2f (n %d), first-round rookies z %.2f, all z %.2f",
        res["player"]["true_rookies"]["z"],
        res["player"]["true_rookies"]["n"],
        res["player"]["first_round_rookies"].get("z", float("nan")),
        res["player"]["all"]["z"],
    )

    # ---- game level: preseason ratings from the new rate model against the current ones (hyperparameters frozen)
    hp = json.loads((met / "game_model.json").read_text())["tuned"]["P"]
    games = gd_.load_games(proc)
    seasons = grun.TUNE + grun.TEST
    t = games.table.filter(pl.col("season") >= grun.TUNE[0])
    tt = t["season"].is_in(grun.TUNE).to_numpy()
    te = t["season"].is_in(grun.TEST).to_numpy()
    y, dates = t["home_win"].to_numpy()[te], t["date"].to_numpy()[te]
    pre, full = (
        {"lam": 0.0, "alpha": 0.0, "n0": hp["n0"], "update": False},
        {"lam": hp["lam"], "alpha": hp["alpha"], "n0": hp["n0"]},
    )
    pri0, pri1 = gd_.build_priors(proc, cfg, seasons, games), gd_.build_priors(proc, cfg, seasons, games, yp)
    mu = lambda pri, **kw: np.concatenate([gm.run_player_season(games, pri, T, **kw) for T in seasons])
    ll = {
        (tag, name): gm.logloss(grun.predict(t, mu(pri, **kw))[te], y)
        for tag, pri in (("M", pri0), ("new", pri1))
        for name, kw in (("P_pre", pre), ("P", full))
    }
    res["game"] = {
        "n_games": int(te.sum()),
        "logloss": {f"{a}|{b}": float(v.mean()) for (a, b), v in ll.items()},
        "P_pre": grun.paired(ll[("new", "P_pre")], ll[("M", "P_pre")], dates),
        "P": grun.paired(ll[("new", "P")], ll[("M", "P")], dates),
    }
    log.info("game level: P-pre z %.2f, P z %.2f", res["game"]["P_pre"]["z"], res["game"]["P"]["z"])
    zr, za = res["player"]["true_rookies"]["z"], res["player"]["all"]["z"]
    g = {
        "EP1": {"z": zr, "pass": bool(zr > 2)},
        "EP2": {
            "z_all_players": za,
            "z_P_pre": res["game"]["P_pre"]["z"],
            "z_P": res["game"]["P"]["z"],
            "pass": bool(za > -2 and res["game"]["P_pre"]["z"] > -2 and res["game"]["P"]["z"] > -2),
        },
    }
    res["implement_A"] = bool(g["EP2"]["pass"] and zr > -1)

    # ---- part B: experience-weighted update shares, tuned on 2019-2020 with the adopted preseason ratings
    pri = pri1 if res["implement_A"] else pri0
    tune_t = t.filter(pl.col("season").is_in(grun.TUNE))
    surf = {}
    for k in KAPPAS:
        m_ = np.concatenate([gm.run_player_season(games, pri, T, kappa=k, **full) for T in seasons])
        surf[str(k)] = {"tune": grun.tune_score(tune_t, m_[tt]), "ll": gm.logloss(grun.predict(t, m_)[te], y)}
    best = float(min(surf, key=lambda k: surf[k]["tune"]))
    base = surf["0.0"]["ll"]
    res["part_B"] = {
        "tuning_logloss": {k: v["tune"] for k, v in surf.items()},
        "kappa": best,
        "test_logloss": {k: float(v["ll"].mean()) for k, v in surf.items()},
        "tuned_vs_0": grun.paired(surf[str(best)]["ll"], base, dates) if best != 0.0 else {"z": 0.0, "mean_diff": 0.0},
    }
    zb = res["part_B"]["tuned_vs_0"]["z"]
    g["EP3"] = {"z": zb, "kappa": best, "pass": bool(best != 0.0 and zb > 2)}
    res["gates"] = g
    res["implement_B"] = bool(g["EP3"]["pass"])
    (met / "elite_prospects.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info(
        "part B: kappa %g, z %.2f; gates %s; implement A=%s B=%s",
        best,
        zb,
        {k: v["pass"] for k, v in g.items()},
        res["implement_A"],
        res["implement_B"],
    )
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
