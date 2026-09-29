"""Whole-model validation:  python -m src.wholemodel.run   (docs/wholemodel/methodology.md)."""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.wholemodel import build as wb

log = logging.getLogger("wholemodel")
TEST_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]


def se_arr(p: pl.DataFrame) -> np.ndarray:
    return ((p["y"] - p["pred"]) ** 2).to_numpy()


def paired(a: pl.DataFrame, b: pl.DataFrame) -> dict:
    """Squared-error difference b - a (positive favors a), paired over team-seasons."""
    j = a.join(b, on=["season", "team"], suffix="_b")
    d = ((j["y_b"] - j["pred_b"]) ** 2 - (j["y"] - j["pred"]) ** 2).to_numpy()
    return {"mean_diff": float(d.mean()), "z": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), "n": int(len(d))}


def summarize(p: pl.DataFrame, base: pl.DataFrame) -> dict:
    mse, mse0 = float(se_arr(p).mean()), float(se_arr(base).mean())
    return {"mse": mse, "rmse": mse ** 0.5, "oos_r2_vs_B0": 1 - mse / mse0, "n": p.height}


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    df = wb.build(proc)
    df.write_parquet(proc / "wm_team_table.parquet")
    df = df.filter(pl.col("season") >= 2018)
    fit = wb.expanding_fit
    P = {"B0": fit(df, ["G"], TEST_SEASONS), "B1": fit(df, ["G", "B_gd"], TEST_SEASONS), "B2": fit(df, ["G", "B_xgd"], TEST_SEASONS), "B3": fit(df, ["G", "S_naive"], TEST_SEASONS),
         "M": fit(df, ["G", "S_headline"], TEST_SEASONS), "M+B1": fit(df, ["G", "S_headline", "B_gd"], TEST_SEASONS), "M_fixed": fit(df, ["G"], TEST_SEASONS, fixed_slope="S_headline"),
         "M_no_oracle": fit(df, ["G", "S_no_oracle"], TEST_SEASONS), "constant": fit(df, [], TEST_SEASONS)}
    if "G_roll" in df.columns:      # same models with the rolling goalie component in place of the proxy (docs/goaltending/methodology.md §7)
        PG = {"B0_roll": fit(df, ["G_roll"], TEST_SEASONS), "B1_roll": fit(df, ["G_roll", "B_gd"], TEST_SEASONS), "B3_roll": fit(df, ["G_roll", "S_naive"], TEST_SEASONS),
              "M_roll": fit(df, ["G_roll", "S_headline"], TEST_SEASONS), "M+B1_roll": fit(df, ["G_roll", "S_headline", "B_gd"], TEST_SEASONS)}
        gk = {"rmse": {k: summarize(v, P["B0"])["rmse"] for k, v in {**{k: P[k] for k in ("B0", "B1", "B3", "M", "M+B1")}, **PG}.items()},
              "B0_roll_vs_B0": paired(PG["B0_roll"], P["B0"]), "M_roll_vs_B0_roll": paired(PG["M_roll"], PG["B0_roll"]), "M_roll_vs_B3_roll": paired(PG["M_roll"], PG["B3_roll"]),
              "M_roll_vs_B1_roll": paired(PG["M_roll"], PG["B1_roll"]), "M+B1_roll_vs_B1_roll": paired(PG["M+B1_roll"], PG["B1_roll"]), "M_roll_vs_M": paired(PG["M_roll"], P["M"])}
        (met / "gk_wholemodel.json").write_text(json.dumps(gk, indent=2, default=float), encoding="utf-8")
    res: dict = {"n_test": P["M"].height, "test_seasons": TEST_SEASONS}
    res["models"] = {k: summarize(v, P["B0"]) for k, v in P.items()}
    res["paired"] = {"M_vs_B0": paired(P["M"], P["B0"]), "M_vs_B3": paired(P["M"], P["B3"]), "M+B1_vs_B1": paired(P["M+B1"], P["B1"]), "M_vs_B1": paired(P["M"], P["B1"]), "M_vs_B2": paired(P["M"], P["B2"]),
                     "B0_vs_constant": paired(P["B0"], P["constant"]), "M_fixed_vs_B0": paired(P["M_fixed"], P["B0"])}
    # WM-4: calibration slope of actual on the fixed-slope prediction
    f = P["M_fixed"]
    x, y = f["pred"].to_numpy(), f["y"].to_numpy()
    A = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ coef
    res["calibration_slope"] = {"slope": float(coef[1]), "se": float(np.sqrt(resid @ resid / (len(y) - 2) / np.sum((x - x.mean()) ** 2))), "n": len(y)}
    # WM-6: by season
    by = {}
    for T in TEST_SEASONS:
        m = lambda p, T=T: float(se_arr(p.filter(pl.col("season") == T)).mean())  # noqa: E731
        by[str(T)] = {"mse_M": m(P["M"]), "mse_B0": m(P["B0"]), "mse_B1": m(P["B1"]), "M_better_than_B0": bool(m(P["M"]) < m(P["B0"]))}
    res["by_season"] = by
    # variants and drop-one, each under the M protocol and compared with the headline
    var = {}
    for k in [c for c in df.columns if c.startswith("S_") and c not in ("S_headline", "S_naive", "S_no_oracle")]:
        name = k[2:]
        p = fit(df, ["G", k], TEST_SEASONS)
        var[name] = {**summarize(p, P["B0"]), "vs_headline": paired(P["M"], p), "vs_B0": paired(p, P["B0"])}     # vs_headline: positive favors the headline
    res["variants"] = var
    # gates
    g = {}
    g["WM1"] = {"z": res["paired"]["M_vs_B0"]["z"], "pass": bool(res["paired"]["M_vs_B0"]["z"] > 2)}
    g["WM2"] = {"z": res["paired"]["M_vs_B3"]["z"], "pass": bool(res["paired"]["M_vs_B3"]["z"] > 2)}
    g["WM3"] = {"z": res["paired"]["M+B1_vs_B1"]["z"], "pass": bool(res["paired"]["M+B1_vs_B1"]["z"] > 2)}
    cs = res["calibration_slope"]
    g["WM4"] = {"slope": cs["slope"], "se": cs["se"], "pass": bool(0.7 <= cs["slope"] <= 1.3)}
    cl = var["leakage_clean"]
    g["WM5"] = {"clean_vs_B0_z": cl["vs_B0"]["z"], "headline_minus_clean_z": cl["vs_headline"]["z"], "pass": bool(cl["vs_B0"]["z"] > 2 and abs(cl["vs_headline"]["z"]) < 2)}
    g["WM6"] = {"seasons_better": sum(v["M_better_than_B0"] for v in by.values()), "pass": bool(sum(v["M_better_than_B0"] for v in by.values()) >= 4)}
    res["gates"] = g
    (met / "wm_validation.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("gates %s", {k: (round(v.get("z", v.get("slope", v.get("seasons_better", 0))), 2), v["pass"]) for k, v in g.items()})
    log.info("rmse %s", {k: round(v["rmse"], 2) for k, v in res["models"].items()})
    from src.wholemodel.report import run as report
    report()
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
