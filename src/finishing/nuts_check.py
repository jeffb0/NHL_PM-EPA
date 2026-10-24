"""Full-Bayes check of the Laplace finishing model with NUTS (optional JAX environment).

    .venv-bayes/Scripts/python -m src.finishing.nuts_check --input data/processed/fin_nuts_season_2024.npz

Two models on the exported data, both non-centered:
  * fixed-sigma : variance components fixed at the empirical-Bayes values -> isolates the error of the
                  Laplace (Gaussian) posterior approximation for the shooter/goalie effects
  * free-sigma  : half-normal hyperpriors on every sigma -> compares the EM point estimates with the
                  posterior of the standard deviations
Reports agreement of posterior means and sds with the Laplace values, and Laplace 95% interval coverage of
the NUTS posterior mean. Pure numpy/JAX/numpyro: importable without polars.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)


def make_model(d: dict, learn_sigma: bool):
    multi = bool(d["multi"])
    names = [str(n) for n in d["s2_names"]]
    s2 = dict(zip(names, [float(v) for v in d["s2_vals"]]))
    y, off, cell = jnp.asarray(d["y"]), jnp.asarray(d["off"], dtype=jnp.float32), jnp.asarray(d["cell"])
    u_idx, v_idx = jnp.asarray(d["u_idx"]), jnp.asarray(d["v_idx"])
    n_u, n_v, n_cell = len(d["u_ids"]), len(d["v_ids"]), int(d["n_cell"])
    u_pos = jnp.asarray(d["u_pos"])
    if multi:
        ss_idx, gs_idx, ss_pos = jnp.asarray(d["ss_idx"]), jnp.asarray(d["gs_idx"]), jnp.asarray(d["ss_pos"])
        n_ss, n_gs = len(d["ss_pos"]), len(d["lap_e_mean"])

    def sigma(name: str, init_key: str):
        if learn_sigma:
            return numpyro.sample(name, dist.HalfNormal(0.5))
        return jnp.sqrt(s2[init_key])

    def model():
        a = numpyro.sample("cell", dist.Normal(0.0, 5.0).expand([n_cell]))
        su = jnp.stack([sigma("sd_u_F", "u_F"), sigma("sd_u_D", "u_D")])
        u = numpyro.sample("zu", dist.Normal(0.0, 1.0).expand([n_u])) * su[u_pos]
        sv = sigma("sd_v", "v")
        v = numpyro.sample("zv", dist.Normal(0.0, 1.0).expand([n_v])) * sv
        eta = off + a[cell] + u[u_idx] + v[v_idx]
        if multi:
            sd_ = jnp.stack([sigma("sd_d_F", "d_F"), sigma("sd_d_D", "d_D")])
            dd = numpyro.sample("zd", dist.Normal(0.0, 1.0).expand([n_ss])) * sd_[ss_pos]
            se = sigma("sd_e", "e")
            ee = numpyro.sample("ze", dist.Normal(0.0, 1.0).expand([n_gs])) * se
            eta = eta + dd[ss_idx] + ee[gs_idx]
        numpyro.sample("y", dist.Bernoulli(logits=eta), obs=y)

    return model


def summarize(samples: dict, d: dict, learn_sigma: bool) -> dict:
    out = {}

    def eff(z, s):
        return np.asarray(z) * s

    if learn_sigma:
        sd_uF, sd_uD = np.asarray(samples["sd_u_F"])[:, None], np.asarray(samples["sd_u_D"])[:, None]
        pos = d["u_pos"][None, :]
        u = np.asarray(samples["zu"]) * np.where(pos == 1, sd_uD, sd_uF)
        v = np.asarray(samples["zv"]) * np.asarray(samples["sd_v"])[:, None]
        out["sigma_posterior"] = {
            k: {"mean": float(np.mean(samples[k])), "sd": float(np.std(samples[k]))}
            for k in samples
            if k.startswith("sd_")
        }
    else:
        s2 = dict(zip([str(n) for n in d["s2_names"]], [float(x) for x in d["s2_vals"]]))
        s = np.where(d["u_pos"] == 1, np.sqrt(s2["u_D"]), np.sqrt(s2["u_F"]))
        u, v = eff(samples["zu"], s), eff(samples["zv"], np.sqrt(s2["v"]))
    for name, draws, mean, sd in (
        ("shooter", u, d["lap_u_mean"], d["lap_u_sd"]),
        ("goalie", v, d["lap_v_mean"], d["lap_v_sd"]),
    ):
        m, s_ = draws.mean(0), draws.std(0)
        out[name] = {
            "n": int(len(m)),
            "corr_means": float(np.corrcoef(m, mean)[0, 1]),
            "mean_abs_diff_means": float(np.abs(m - mean).mean()),
            "sd_ratio_laplace_over_nuts_median": float(np.median(sd / s_)),
            "sd_ratio_p05_p95": [float(np.quantile(sd / s_, 0.05)), float(np.quantile(sd / s_, 0.95))],
            "nuts_mean_inside_laplace95": float(np.mean(np.abs(m - mean) <= 1.96 * sd)),
            "sd_nuts_median": float(np.median(s_)),
        }
    return out


def run(path: str, warmup: int, draws: int, seed: int) -> dict:
    d = dict(np.load(path, allow_pickle=False))
    res = {"input": Path(path).name, "n_shots": int(len(d["y"])), "multi_season": bool(d["multi"])}
    for tag, learn in (("fixed_sigma", False), ("free_sigma", True)):
        t0 = time.time()
        mcmc = MCMC(
            NUTS(make_model(d, learn), target_accept_prob=0.85),
            num_warmup=warmup,
            num_samples=draws,
            num_chains=4,
            chain_method="parallel",
            progress_bar=False,
        )
        mcmc.run(random.PRNGKey(seed))
        s = mcmc.get_samples()
        extra = mcmc.get_extra_fields()
        r = summarize(s, d, learn)
        r["seconds"] = time.time() - t0
        r["divergences"] = int(np.sum(np.asarray(extra["diverging"])))
        from numpyro.diagnostics import summary

        sm = summary(mcmc.get_samples(group_by_chain=True), prob=0.9, group_by_chain=True)
        r["max_r_hat"] = float(max(float(np.max(v["r_hat"])) for v in sm.values()))
        r["min_n_eff"] = float(min(float(np.min(v["n_eff"])) for v in sm.values()))
        res[tag] = r
        print(
            tag,
            json.dumps({k: v for k, v in r.items() if k in ("seconds", "divergences", "max_r_hat", "min_n_eff")}),
            flush=True,
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.input, a.warmup, a.draws, a.seed)
    out = Path(
        a.out
        or Path(a.input).parents[2]
        / "reports"
        / "metrics"
        / f"fin_nuts_{Path(a.input).stem.replace('fin_nuts_', '')}.json"
    )
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
