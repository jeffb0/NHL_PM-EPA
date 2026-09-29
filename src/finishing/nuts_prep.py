"""Export data and the Laplace fit for the full-Bayes (NUTS) check (main environment).

    python -m src.finishing.nuts_prep [--season 2024 | --window 2023 2025]

Writes data/processed/fin_nuts_<tag>.npz with the shots, indices, the empirical-Bayes variance
components and the Laplace posterior mean / sd of every shooter and goalie effect. The sampler
(``nuts_check``) runs in the optional JAX environment and compares its posterior with these.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.finishing import model
from src.finishing.model import Spec
from src.finishing.run import load_table

log = logging.getLogger(__name__)


def prepare(cfg, seasons: list[int], tag: str) -> str:
    proc = cfg.path(cfg.data.processed_dir)
    df = load_table(cfg).filter(pl.col("season").is_in(seasons) & pl.col("goalie").is_not_null())
    multi = len(seasons) > 1
    spec = Spec(fringe=False, season_dev=multi, goalie_season=multi, types=False)
    fit = model.fit_finishing(df, spec, em_iters=cfg.finishing.em_iters, em_tol_sd=cfg.finishing.em_tol_sd)
    lay = fit.layout
    sd = np.sqrt(np.diag(fit.mode.posterior_cov()))
    d = model.add_keys(df)

    def index(keys, mapping):
        cols = sorted(mapping.values())
        pos = {c: i for i, c in enumerate(cols)}
        inv = {c: k for k, c in mapping.items()}
        col = model._map(keys, mapping)
        return np.array([pos[c] for c in col]), np.array(cols), np.array([inv[c] for c in cols])

    cell, cell_cols, _ = index(d["k_cell"], lay.cells)
    u_idx, u_cols, u_ids = index(d["shooter"], lay.shooter)
    v_idx, v_cols, v_ids = index(d["goalie"], lay.goalie)
    posmap = lay.shooter_pos
    out = {"y": df["goal"].to_numpy().astype(np.float32), "off": df["offset"].to_numpy(), "cell": cell, "u_idx": u_idx, "v_idx": v_idx,
           "u_pos": np.array([1 if posmap[p] == "D" else 0 for p in u_ids]), "u_ids": u_ids, "v_ids": v_ids,
           "lap_u_mean": fit.beta[u_cols], "lap_u_sd": sd[u_cols], "lap_v_mean": fit.beta[v_cols], "lap_v_sd": sd[v_cols],
           "lap_cell_mean": fit.beta[cell_cols], "n_cell": len(cell_cols),
           "s2_names": np.array(list(fit.s2)), "s2_vals": np.array(list(fit.s2.values())), "multi": np.array(multi)}
    if multi:
        ss_idx, ss_cols, ss_ids = index(d["k_ss"], lay.shooter_season)
        gs_idx, gs_cols, _ = index(d["k_gs"], lay.goalie_season)
        out.update({"ss_idx": ss_idx, "gs_idx": gs_idx, "ss_pos": np.array([1 if posmap[k // 10000] == "D" else 0 for k in ss_ids]),
                    "lap_d_mean": fit.beta[ss_cols], "lap_d_sd": sd[ss_cols], "lap_e_mean": fit.beta[gs_cols], "lap_e_sd": sd[gs_cols]})
    path = proc / f"fin_nuts_{tag}.npz"
    np.savez_compressed(path, **out)
    log.info("wrote %s: %d shots, %d shooters, %d goalies; sd %s", path.name, len(out["y"]), len(u_ids), len(v_ids), {k: round(float(np.sqrt(v)), 4) for k, v in fit.s2.items()})
    return str(path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--window", type=int, nargs=2)
    args = ap.parse_args()
    cfg = load_config()
    if args.window:
        prepare(cfg, list(range(args.window[0], args.window[1] + 1)), f"window_{args.window[0]}_{args.window[1]}")
    else:
        prepare(cfg, [args.season], f"season_{args.season}")


if __name__ == "__main__":
    main()
