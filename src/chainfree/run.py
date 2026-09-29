"""Chain-free defense and special-teams components (docs/chainfree/methodology.md).

    python -m src.chainfree.run

Refits the DPM valuation model and the special-teams RAPM with rho = 0 (each season independent), writes
``data/processed/chainfree_values.parquet`` and ``reports/metrics/cf_values.json`` (persistence and forward-slope diagnostics).
The whole-model and projection evaluations of these values live in ``src.wholemodel.run`` and ``src.projection.run --chainfree``.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.defense import ridge
from src.specialteams import model as sm
from src.specialteams import run as st_run
from src.valuation import combine as cb
from src.valuation.replacement import status

log = logging.getLogger("chainfree")
CLASSES = ("pp", "pk", "4v4", "3v3")


def _yoy(t: pl.DataFrame, col: str, min_col: str, min_val: float) -> dict:
    d = t.filter(pl.col(min_col) >= min_val).with_columns((pl.col(col) - pl.col(col).mean().over("season", "pos_group")).alias("_c"))
    nxt = d.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("_c").alias("_n"))
    j = d.join(nxt, on=["player_id", "season"])
    return {g: {"r": float(np.corrcoef(x["_c"], x["_n"])[0, 1]), "n": x.height} for (g,), x in j.group_by(["pos_group"])}


def refit_defense(cfg, proc) -> pl.DataFrame:
    ps = pl.read_parquet(proc / "player_season.parquet")
    dpm = pl.read_parquet(proc / "dpm_player_season.parquet")
    sel = json.loads((cfg.path(cfg.data.metrics_dir) / "dpm_selection.json").read_text())
    st = status(ps, cfg.valuation.regular_forwards, cfg.valuation.regular_defense)
    stints = pl.read_parquet(proc / "def_stints.parquet")
    refit, fr = cb.dpm_refit(stints, st, sorted(cfg.data.seasons), sel["alphas"]["xg"], 0.0)
    dp = cb.dpm_component(refit, fr, st, ps, dpm)
    return dp.select(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64), pl.col("def_gar").alias("def_gar_rho0"))


def refit_special_teams(cfg, proc) -> pl.DataFrame:
    ridge.JITTER = st_run.JITTER
    stints = pl.read_parquet(proc / "st_stints.parquet")
    lookup = st_run._lookup(cfg, proc)
    seasons = sorted(cfg.data.seasons)
    sel = json.loads((cfg.path(cfg.data.metrics_dir) / "st_selection.json").read_text())
    parts = []
    for g in st_run.GROUPS:
        des = st_run._designs(stints, g, seasons, lookup)
        fits = sm.chain_fits(des, sel[g]["alpha"], 0.0)
        vals = sm.player_values(fits, g, sm.fringe_effects(fits, g), lookup)
        parts.append(vals.with_columns(pl.col("role").map_elements(lambda r, g=g: st_run.CLS_OF[(g, r)], return_dtype=pl.Utf8).alias("cls")).select("player_id", "season", "cls", "gar_rapm"))
        log.info("special-teams group %s refit with rho = 0 (alpha %s; the chained fit used rho %s)", g, sel[g]["alpha"], sel[g]["rho"])
    long = pl.concat(parts)
    wide = None
    for c in CLASSES:
        x = long.filter(pl.col("cls") == c).select("player_id", "season", pl.col("gar_rapm").alias(f"st_{c}_rapm_gar_rho0"))
        wide = x if wide is None else wide.join(x, on=["player_id", "season"], how="full", coalesce=True)
    return wide.with_columns([pl.col(c).fill_null(0.0) for c in wide.columns if c.endswith("_rho0")])


def forward_slopes(proc, t: pl.DataFrame, cols: dict[str, str]) -> dict:
    """Next-season deployment-adjusted on-ice goal differential per 60 on each component's rate per modeled minute (minute-weighted, state shares
    as controls, player-cluster robust SE)."""
    from src.projection.run import results_outcome
    out = results_outcome(proc)
    nxt = out.select("player_id", (pl.col("season") - 1).alias("season"), pl.col("y_res").alias("y_next"), "sh_pp", "sh_pk", "sh_o")
    m = pl.col("toi_5v5") + sum((pl.col(f"st_{c}_min") for c in CLASSES), pl.lit(0.0))
    d = t.with_columns(m.alias("_m")).filter(pl.col("_m") >= 300).join(nxt.with_columns(pl.col("player_id").cast(t["player_id"].dtype), pl.col("season").cast(t["season"].dtype)), on=["player_id", "season"])
    res = {}
    for name, col in cols.items():
        x = (d[col] / d["_m"] * 60).to_numpy()
        X = np.column_stack([np.ones(d.height), x, d["sh_pp"].to_numpy(), d["sh_pk"].to_numpy(), d["sh_o"].to_numpy()])
        y, w = d["y_next"].to_numpy(), d["_m"].to_numpy()
        A = X.T @ (w[:, None] * X)
        beta = np.linalg.solve(A, X.T @ (w * y))
        u = y - X @ beta
        meat = np.zeros_like(A)
        ids = d["player_id"].to_numpy()
        for pid in np.unique(ids):
            k = ids == pid
            s = X[k].T @ (w[k] * u[k])
            meat += np.outer(s, s)
        cov = np.linalg.inv(A) @ meat @ np.linalg.inv(A)
        res[name] = {"slope": float(beta[1]), "se": float(np.sqrt(cov[1, 1])), "n": int(d.height)}
    return res


def run() -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    met = cfg.path(cfg.data.metrics_dir)
    dfn = refit_defense(cfg, proc)
    stw = refit_special_teams(cfg, proc)
    vals = dfn.join(stw, on=["player_id", "season"], how="full", coalesce=True).with_columns([pl.col(c).fill_null(0.0) for c in [*stw.columns[2:], "def_gar_rho0"]])
    vals.write_parquet(proc / "chainfree_values.parquet")
    t = pl.read_parquet(proc / "player_value_calibrated_all.parquet")
    t = t.join(vals.with_columns(pl.col("player_id").cast(t["player_id"].dtype), pl.col("season").cast(t["season"].dtype)), on=["player_id", "season"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in vals.columns if c.endswith("_rho0")])
    res: dict = {}
    # persistence by construction: YOY of the chained and chain-free values
    tt = t.with_columns((pl.col("def_gar") / pl.col("toi_5v5") * 60).alias("def_rate"), (pl.col("def_gar_rho0") / pl.col("toi_5v5") * 60).alias("def_rate_rho0"),
                        *[(pl.col(f"st_{c}_rapm_gar") / pl.col(f"st_{c}_min") * 60).alias(f"{c}_rate") for c in ("pp", "pk")],
                        *[(pl.col(f"st_{c}_rapm_gar_rho0") / pl.col(f"st_{c}_min") * 60).alias(f"{c}_rate_rho0") for c in ("pp", "pk")]).with_columns(
        [pl.col(c).fill_nan(None) for c in ("pp_rate", "pk_rate", "pp_rate_rho0", "pk_rate_rho0")])
    res["yoy"] = {"defense_chained": _yoy(tt, "def_rate", "toi_5v5", 500), "defense_rho0": _yoy(tt, "def_rate_rho0", "toi_5v5", 500)}
    for c in ("pp", "pk"):
        sub = tt.filter(pl.col(f"{c}_rate").is_not_null() & pl.col(f"{c}_rate_rho0").is_not_null())
        res["yoy"][f"{c}_chained"] = _yoy(sub, f"{c}_rate", f"st_{c}_min", 100)
        res["yoy"][f"{c}_rho0"] = _yoy(sub, f"{c}_rate_rho0", f"st_{c}_min", 100)
    res["corr_chained_vs_rho0"] = {"defense": float(np.corrcoef(t.filter(pl.col("toi_5v5") >= 500)["def_gar"], t.filter(pl.col("toi_5v5") >= 500)["def_gar_rho0"])[0, 1]),
                                   **{c: float(np.corrcoef(t.filter(pl.col(f"st_{c}_min") >= 100)[f"st_{c}_rapm_gar"], t.filter(pl.col(f"st_{c}_min") >= 100)[f"st_{c}_rapm_gar_rho0"])[0, 1]) for c in ("pp", "pk")}}
    res["sd_qualified"] = {"def_chained": float(t.filter(pl.col("toi_5v5") >= 500)["def_gar"].std()), "def_rho0": float(t.filter(pl.col("toi_5v5") >= 500)["def_gar_rho0"].std()),
                           **{f"{c}_chained": float(t.filter(pl.col(f"st_{c}_min") >= 100)[f"st_{c}_rapm_gar"].std()) for c in ("pp", "pk")},
                           **{f"{c}_rho0": float(t.filter(pl.col(f"st_{c}_min") >= 100)[f"st_{c}_rapm_gar_rho0"].std()) for c in ("pp", "pk")}}
    res["forward_slopes"] = forward_slopes(proc, t, {"defense_chained": "def_gar", "defense_rho0": "def_gar_rho0", "pp_chained": "st_pp_rapm_gar", "pp_rho0": "st_pp_rapm_gar_rho0",
                                                       "pk_chained": "st_pk_rapm_gar", "pk_rho0": "st_pk_rapm_gar_rho0"})
    (met / "cf_values.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log.info("YOY defense chained %s -> rho0 %s", {k: round(v["r"], 2) for k, v in res["yoy"]["defense_chained"].items()}, {k: round(v["r"], 2) for k, v in res["yoy"]["defense_rho0"].items()})
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
