"""Joint calibration end to end:  python -m src.calibration.run [--only STEP ...]

Steps: team -> player -> team_all -> player_all -> report (team_all and player_all are the all-strength extension,
methodology §5b; team_gk and player_gk add the goalie, §5c). Needs the integrated table (``python -m
src.valuation.run``).
Writes reports/metrics/cal_*.json and data/outputs/player_value_calibrated.csv (methodology in docs/calibration/).
"""

from __future__ import annotations

import argparse
import json
import logging

import polars as pl

from src.calibration import player as pl_
from src.calibration import team as tm
from src.calibration import unshrunk
from src.config import load_config

log = logging.getLogger("calibration")
STEPS = ["team", "player", "team_all", "player_all", "team_gk", "player_gk", "team_pos", "player_pos", "report"]


def _paths(cfg):
    return cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.output_dir), cfg.path(cfg.data.metrics_dir)


def step_team(cfg):
    proc, out_dir, met = _paths(cfg)
    t = unshrunk.build(proc)
    j = tm.team_table(t, proc)
    res = tm.calibrate(j, tm.COMPS, boot=1000, seed=cfg.seed)
    res["shrunk_ols"] = tm.shrunk_ols(j, tm.COMPS)
    res["xg_to_goals_factor"] = tm.xg_to_goals_factor(proc)
    res["n_team_seasons"] = j.height
    gpw = json.loads((met / "val_gpw.json").read_text())["goals_per_win"]
    w = res["weights_applied"]
    res["gate_CAL4_pass"] = bool(all(0.5 <= v <= 2.0 for v in w.values()))
    tc = tm.apply_weights(t, w, gpw)
    # the ridge weights, reported and validated whether or not adopted
    tc = tm.apply_weights(tc, res["weights_ridge"], gpw, tag="ridge")
    res["mean_abs_change_gar_ridge_qualified"] = float(
        tc.filter(pl.col("toi_5v5") >= cfg.valuation.min_toi)["ridge_change_gar"].abs().mean()
    )
    res["mean_abs_change_gar_qualified"] = float(
        tc.filter(pl.col("toi_5v5") >= cfg.valuation.min_toi)["cal_change_gar"].abs().mean()
    )
    res["gate_CAL5_pass"] = bool((not res["adopt"]) == (tc["cal_change_gar"].abs().max() < 1e-9))
    tc.write_parquet(proc / "player_value_calibrated.parquet")
    tc.write_csv(out_dir / "player_value_calibrated.csv")
    (met / "cal_team.json").write_text(json.dumps(res, indent=2, default=float))
    log.info(
        "calibration: lambda=%.3g weights %s; paired z vs ones %.2f -> %s",
        res["lambda"],
        {k: round(v, 2) for k, v in res["weights_ridge"].items()},
        res["paired_z_calibrated_vs_ones"],
        "ADOPTED" if res["adopt"] else "weights set to 1",
    )


def step_player(cfg):
    proc, _, met = _paths(cfg)
    t = pl.read_parquet(proc / "player_value_calibrated.parquet")
    rates = pl_.rate_table(t, tm.COMPS)
    oi = pl_.onice_forward(proc, rates)
    d = pl_.dataset(t, oi, rates, tm.COMPS, min_minutes=200.0)
    res = pl_.evaluate(d, tm.COMPS)
    res["gate_CAL3_pass"] = bool(res["y_gd"]["paired_z_calibrated_better"] > -2.0)
    (met / "cal_player.json").write_text(json.dumps(res, indent=2, default=float))
    log.info(
        "player-level forward check: %d player-seasons; own slope (goals) %.2f; calibrated-better z %.2f",
        res["n_player_seasons"],
        res["y_gd"]["calibrated"]["own_slope"],
        res["y_gd"]["paired_z_calibrated_better"],
    )


def _team_all(cfg, gk: bool = False):
    """All-strength calibration (methodology §5b); with ``gk`` the team's goalie value is an eleventh regressor (§5c)
    and outputs carry the suffix ``_gk``."""
    proc, out_dir, met = _paths(cfg)
    sfx = "_gk" if gk else ""
    comps = tm.COMPS_ALL_GK if gk else tm.COMPS_ALL
    if not (proc / "st_rapm_player_season.parquet").exists() or (
        gk and not (proc / "goalie_team_season.parquet").exists()
    ):
        log.warning(
            "required outputs missing: skipping the all-strength calibration%s", " with the goalie" if gk else ""
        )
        return
    t = unshrunk.build(proc, all_strength=True)
    j = tm.team_table(t, proc, comps, target="gd_all")
    res = tm.calibrate(j, comps, boot=1000, seed=cfg.seed, target="gd_all")
    res["shrunk_ols"] = tm.shrunk_ols(j, comps, target="gd_all")
    res["n_team_seasons"] = j.height
    res["sd_team_gd_all"] = float(j["gd_all"].std())
    gpw = json.loads((met / "val_gpw.json").read_text())["goals_per_win"]
    w = res["weights_applied"]
    res["gate_CAL4_pass"] = bool(all(0.5 <= v <= 2.0 for v in w.values()))
    sk = lambda d: {k: v for k, v in d.items() if k != "gk"}
    tc = tm.apply_weights_all(t, sk(w), gpw, "cal")
    tc = tm.apply_weights_all(tc, sk(res["weights_ridge"]), gpw, "ridge")
    q = tc.filter(pl.col("toi_5v5") >= cfg.valuation.min_toi)
    res["mean_abs_change_gar_ridge_qualified"] = float(q["ridge_change_gar"].abs().mean())
    res["gate_CAL5_pass"] = bool((not res["adopt"]) == (tc["cal_change_gar"].abs().max() < 1e-9))
    if gk:
        g = pl.read_parquet(proc / "goalie_player_season.parquet").with_columns(
            (pl.col("goalie_gar") * w["gk"]).alias("goalie_gar_cal"),
            (pl.col("goalie_gar") * res["weights_ridge"]["gk"]).alias("goalie_gar_ridge"),
        )
        g.write_parquet(proc / "goalie_value_calibrated.parquet")
        g.write_csv(out_dir / "goalie_value_calibrated.csv")
        # one table for every player: skaters (all-strength total) and goalies (goalie value), calibrated columns follow
        # the decision (weights stay 1 unless adopted)
        sk_rows = tc.select(
            "player_id",
            "player_name",
            "position",
            "season",
            "team",
            pl.lit("skater").alias("role"),
            (pl.col("toi_5v5") + sum((pl.col(f"st_{c}_min") for c in ("pp", "pk", "4v4", "3v3")), pl.lit(0.0))).alias(
                "usage"
            ),
            pl.lit("minutes").alias("usage_unit"),
            pl.col("total_gar_all").alias("gar_uncalibrated"),
            pl.col("total_gar_all_cal").alias("gar"),
            pl.col("total_gar_all_sd").alias("gar_sd"),
        )
        gk_rows = g.select(
            "player_id",
            "player_name",
            pl.lit("G").alias("position"),
            "season",
            "team",
            pl.lit("goalie").alias("role"),
            pl.col("shots").cast(pl.Float64).alias("usage"),
            pl.lit("shots faced").alias("usage_unit"),
            pl.col("goalie_gar").alias("gar_uncalibrated"),
            pl.col("goalie_gar_cal").alias("gar"),
            pl.col("gsaa_sd").alias("gar_sd"),
        )
        allin = (
            pl.concat(
                [
                    sk_rows.with_columns(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64)),
                    gk_rows.with_columns(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64)),
                ]
            )
            .with_columns((pl.col("gar") / gpw).alias("wins"))
            .sort("season", "gar", descending=[False, True])
        )
        allin.write_parquet(proc / "player_value_all_in.parquet")
        allin.write_csv(out_dir / "player_value_all_in.csv")
    tc.write_parquet(proc / f"player_value_calibrated_all{sfx}.parquet")
    tc.write_csv(out_dir / f"player_value_calibrated_all{sfx}.csv")
    (met / f"cal_team_all{sfx}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info(
        "all-strength calibration%s: lambda=%.3g weights %s; paired z vs ones %.2f -> %s",
        " with the goalie" if gk else "",
        res["lambda"],
        {k: round(v, 2) for k, v in res["weights_ridge"].items()},
        res["paired_z_calibrated_vs_ones"],
        "ADOPTED" if res["adopt"] else "weights set to 1",
    )


def _player_all(cfg, gk: bool = False, sfx_override: str | None = None):
    proc, _, met = _paths(cfg)
    sfx = sfx_override if sfx_override is not None else ("_gk" if gk else "")
    if not (proc / f"player_value_calibrated_all{sfx}.parquet").exists():
        return
    t = pl.read_parquet(proc / f"player_value_calibrated_all{sfx}.parquet")
    rates = pl_.rate_table_all(t, tm.COMPS_ALL)
    oi = pl_.onice_forward_all(proc, rates)
    d = pl_.dataset(t, oi, rates, tm.COMPS_ALL, min_minutes=250.0)
    # as pre-registered: raw all-strength on-ice differential per 60
    res = pl_.evaluate(d, tm.COMPS_ALL)
    res["gate_CAL3_pass"] = bool(res["y_gd"]["paired_z_calibrated_better"] > -2.0)
    # deployment-adjusted (methodology §7 deviation 4)
    adj = pl_.evaluate(d, tm.COMPS_ALL, ys=("y_gd_adj", "y_xgd_adj"), controls=("sh_pp", "sh_pk", "sh_o"))
    res["deployment_adjusted"] = adj
    res["gate_CAL3_pass_adjusted"] = bool(adj["y_gd_adj"]["paired_z_calibrated_better"] > -2.0)
    (met / f"cal_player_all{sfx}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info(
        "all-strength player check%s: %d player-seasons; own slope (goals) %.2f; ridge-better z %.2f",
        " with the goalie" if gk else "",
        res["n_player_seasons"],
        res["y_gd"]["calibrated"]["own_slope"],
        res["y_gd"]["paired_z_calibrated_better"],
    )


def _team_all_pos(cfg):
    """Position-specific weights (methodology §5d): 18 weights, adoption needs paired z > 2 against all ones and against
    the pooled ridge (§5c)."""
    import numpy as np

    proc, out_dir, met = _paths(cfg)
    if not (proc / "goalie_team_season.parquet").exists():
        return
    t = unshrunk.build(proc, all_strength=True)
    jp = tm.team_table(t, proc, tm.COMPS_ALL_GK, target="gd_all")
    js = tm.team_table(t, proc, tm.COMPS_POS, target="gd_all")
    assert (jp["season"].to_list() == js["season"].to_list()) and (jp["team"].to_list() == js["team"].to_list())
    y, season = js["gd_all"].to_numpy().astype(float), js["season"].to_numpy()
    Xs, Xp = (
        js.select([tm.RAW[c] for c in tm.COMPS_POS]).to_numpy(),
        jp.select([tm.RAW[c] for c in tm.COMPS_ALL_GK]).to_numpy(),
    )
    cvs, cvp = tm.loso_cv(Xs, y, season, tm.LAMBDAS), tm.loso_cv(Xp, y, season, tm.LAMBDAS)
    bs, bp = int(np.argmin(cvs["total"][:-1])), int(np.argmin(cvp["total"][:-1]))
    lam = float(tm.LAMBDAS[bs])

    def z(a, b):  # positive = a has the larger error = b is better
        d = a - b
        return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))

    z_ones, z_pool = z(cvs["sq_err"][-1], cvs["sq_err"][bs]), z(cvp["sq_err"][bp], cvs["sq_err"][bs])
    adopt = bool(z_ones > 2 and z_pool > 2)
    _, w = tm.ridge_to_one(Xs, y, lam)
    rng = np.random.default_rng(cfg.seed)
    bw = np.array(
        [tm.ridge_to_one(Xs[k], y[k], lam)[1] for k in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    )
    ols = tm.recon.regress(js, [tm.RAW[c] for c in tm.COMPS_POS], y="gd_all")
    shr = tm.recon.regress(js, [tm.SHR[c] for c in tm.COMPS_POS], y="gd_all")
    wr = dict(zip(tm.COMPS_POS, w.tolist()))
    wc = {c: float(np.clip(v, 0.0, 3.0)) for c, v in wr.items()}  # applied weights are clipped to [0, 3]
    res = {
        "comps": tm.COMPS_POS,
        "lambda": lam,
        "n_team_seasons": js.height,
        "cv_mse_ones": float(cvs["sq_err"][-1].mean()),
        "cv_mse_position": float(cvs["sq_err"][bs].mean()),
        "cv_mse_pooled_ridge": float(cvp["sq_err"][bp].mean()),
        "paired_z_vs_ones": z_ones,
        "paired_z_vs_pooled_ridge": z_pool,
        "adopt": adopt,
        "weights_ridge": wr,
        "weights_clipped": wc,
        "weights_ci95": {
            c: [float(np.quantile(bw[:, i], 0.025)), float(np.quantile(bw[:, i], 0.975))]
            for i, c in enumerate(tm.COMPS_POS)
        },
        "unshrunk_ols": {"r2": ols["r2"], "slopes": {c: ols["slopes"][tm.RAW[c]] for c in tm.COMPS_POS}},
        "shrunk_ols": {"r2": shr["r2"], "slopes": {c: shr["slopes"][tm.SHR[c]] for c in tm.COMPS_POS}},
    }
    gpw = json.loads((met / "val_gpw.json").read_text())["goals_per_win"]
    applied = wc if adopt else {c: 1.0 for c in tm.COMPS_POS}
    sk = lambda d: {k: v for k, v in d.items() if k != "gk"}
    tc = tm.apply_weights_pos(t, sk(applied), gpw, "cal")
    tc = tm.apply_weights_pos(tc, sk(wc), gpw, "ridge")
    res["weights_applied"] = applied
    res["gate_CAL5_pass"] = bool((not adopt) == (tc["cal_change_gar"].abs().max() < 1e-9))
    tc.write_parquet(proc / "player_value_calibrated_all_pos.parquet")
    tc.write_csv(out_dir / "player_value_calibrated_all_pos.csv")
    (met / "cal_team_all_pos.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info(
        "position-specific calibration: lambda=%.3g CV MSE ones %.1f pooled %.1f position %.1f; z vs ones %.2f vs pooled %.2f -> %s",
        lam,
        res["cv_mse_ones"],
        res["cv_mse_pooled_ridge"],
        res["cv_mse_position"],
        z_ones,
        z_pool,
        "ADOPTED" if adopt else "not adopted",
    )


def step_team_pos(cfg):
    _team_all_pos(cfg)


def step_player_pos(cfg):
    _player_all(cfg, False, "_pos")


def step_team_all(cfg):
    _team_all(cfg, False)


def step_player_all(cfg):
    _player_all(cfg, False)


def step_team_gk(cfg):
    _team_all(cfg, True)


def step_player_gk(cfg):
    _player_all(cfg, True)


def run_steps(steps: list[str]) -> None:
    cfg = load_config()
    for s in steps:
        log.info("=== calibration:%s ===", s)
        if s == "report":
            from src.calibration.report import run

            run()
            from src.calibration.report_all import run as run_all

            run_all()
            run_all(gk=True)
            from src.calibration.report_pos import run as run_pos

            run_pos()
        else:
            globals()[f"step_{s}"](cfg)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args()
    run_steps(args.only or STEPS)


if __name__ == "__main__":
    main()
