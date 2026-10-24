"""Component integration on a common goals-above-replacement scale (methodology §8-§9).

Each component becomes goals (xG scale) at 5v5 = rate x TOI / 60 above the replacement pool:

* possession  PM-EPA: empirical-Bayes shrunk rate; replacement = pooled raw fringe rate
* defense     DPM valuation-ready rate, refit once with unpenalized fringe-count covariates; replacement = fringe effect
* finishing   shot-level goals above the fringe pool's mean shooter shift (from the fitted model)

The published PM-EPA and DPM columns are never modified.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import polars as pl
from scipy import sparse
from scipy.special import expit

from src.defense import ridge
from src.valuation import recon

log = logging.getLogger(__name__)
# valuation-ready defense drops the slices claimed by the turnover and faceoff components (and the unobserved starts
# that
# the turnover charge already covers; docs/components/methodology.md §2)
DPM_TARGETS = ["xg", "start_turnover", "start_faceoff", "start_unobserved"]


# ------------------------------------------------------------------ possession (PM-EPA)
def pm_component(ps: pl.DataFrame, status: pl.DataFrame, min_toi: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Shrunk PM-EPA 5v5 value above replacement. Returns (per player-season, replacement rates)."""
    d = (
        ps.select("player_id", "season", "pos_group", "toi_ev", "pm_epa_ev", "pm_epa_per60_ev", "boot_se_per60")
        .join(status.select("player_id", "season", "fringe"), on=["player_id", "season"], how="left")
        .with_columns(
            pl.col("fringe").fill_null(True), pl.col("toi_ev").fill_null(0.0), pl.col("pm_epa_ev").fill_null(0.0)
        )
    )
    grp = d.group_by("season", "pos_group", "fringe").agg(
        (pl.col("pm_epa_ev").sum() / pl.col("toi_ev").sum() * 60).alias("m_status"),
        pl.col("toi_ev").sum().alias("pool_toi"),
        pl.len().alias("pool_n"),
    )
    q = d.filter(
        (~pl.col("fringe"))
        & (pl.col("toi_ev") >= min_toi)
        & pl.col("pm_epa_per60_ev").is_not_null()
        & pl.col("boot_se_per60").is_not_null()
    )
    tau = (
        q.group_by("season", "pos_group")
        .agg(
            pl.col("pm_epa_per60_ev").var().alias("v_x"),
            (pl.col("boot_se_per60") ** 2).mean().alias("v_se"),
            pl.len().alias("n_q"),
        )
        .with_columns((pl.col("v_x") - pl.col("v_se")).clip(1e-6, None).alias("tau2"))
    )
    rep = grp.filter(pl.col("fringe")).select(
        "season", "pos_group", pl.col("m_status").alias("pm_rep_per60"), "pool_toi", "pool_n"
    )
    d = d.join(
        grp.select("season", "pos_group", "fringe", "m_status"), on=["season", "pos_group", "fringe"], how="left"
    ).join(tau.select("season", "pos_group", "tau2"), on=["season", "pos_group"], how="left")
    d = d.join(rep.select("season", "pos_group", "pm_rep_per60"), on=["season", "pos_group"], how="left")
    se = pl.col("boot_se_per60").fill_null(1.0).clip(1e-3, None)
    d = (
        d.with_columns(
            (pl.col("tau2") / (pl.col("tau2") + se**2)).alias("B"),
            pl.col("pm_epa_per60_ev").fill_null(pl.col("m_status")).alias("x"),
        )
        .with_columns(
            (pl.col("m_status") + pl.col("B") * (pl.col("x") - pl.col("m_status"))).alias("pm_post_per60"),
            (pl.col("B") * se**2).sqrt().alias("pm_post_sd_per60"),
        )
        .with_columns(
            ((pl.col("pm_post_per60") - pl.col("pm_rep_per60")) * pl.col("toi_ev") / 60).alias("pm_gar"),
            (pl.col("pm_post_sd_per60") * pl.col("toi_ev") / 60).alias("pm_gar_sd"),
        )
    )
    return d.select(
        "player_id", "season", "pm_post_per60", "pm_rep_per60", "pm_gar", "pm_gar_sd", pl.col("B").alias("pm_shrink_B")
    ), rep


# ------------------------------------------------------------------ defense (DPM refit with a replacement group)
def _fringe_columns(st: pl.DataFrame, stat: pl.DataFrame) -> np.ndarray:
    fF = {p: 1.0 for p in stat.filter(pl.col("fringe") & (pl.col("pos_group") == "F"))["player_id"].to_list()}
    fD = {p: 1.0 for p in stat.filter(pl.col("fringe") & (pl.col("pos_group") == "D"))["player_id"].to_list()}
    home, away = st["home_on"].to_list(), st["away_on"].to_list()
    cnt = lambda lists, m: np.array([sum(m.get(p, 0.0) for p in lst) for lst in lists])
    hF, hD, aF, aD = cnt(home, fF), cnt(home, fD), cnt(away, fF), cnt(away, fD)
    # rows: home attacking (defenders = away), then away attacking (defenders = home); columns [defF, defD, attF, attD]
    return np.column_stack([np.r_[aF, hF], np.r_[aD, hD], np.r_[hF, aF], np.r_[hD, aD]])


def dpm_refit(
    stints: pl.DataFrame, status: pl.DataFrame, seasons: list[int], alpha: float, rho: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Season-chained DPM refit with fringe-count covariates. Returns (per player-season valuation rates, fringe
    effects)."""
    prev, rows, fr = None, [], []
    for s in seasons:
        st = stints.filter(pl.col("season") == s)
        d = ridge.build_design(st, DPM_TARGETS)
        extra = _fringe_columns(st, status.filter(pl.col("season") == s))
        d2 = dataclasses.replace(
            d, X=sparse.hstack([d.X, sparse.csr_matrix(extra)]).tocsr(), n_cov=d.n_cov + extra.shape[1]
        )
        beta = ridge.fit(d2, {t: alpha for t in DPM_TARGETS}, ridge.prior_vector(d2, prev, rho))
        coefs = ridge.coef_frame(d2, beta)
        f = beta[-4:, :]  # rows [defF, defD, attF, attD] x targets
        # value-ready = xg - turnover - faceoff - unobserved starts (raw xGA units)
        w = np.array([1.0, -1.0, -1.0, -1.0])
        c_f = {"F": float(f[0] @ w), "D": float(f[1] @ w)}
        fr.append(
            {
                "season": s,
                "c_val_fringe_F": c_f["F"],
                "c_val_fringe_D": c_f["D"],
                "dpm_rep_per60_F": -c_f["F"],
                "dpm_rep_per60_D": -c_f["D"],
                "n_stints": st.height,
                "att_fringe_F": float(f[2] @ w),
                "att_fringe_D": float(f[3] @ w),
            }
        )
        c_beta = (
            coefs["def_xg"] - coefs["def_start_turnover"] - coefs["def_start_faceoff"] - coefs["def_start_unobserved"]
        )
        rows.append(
            coefs.select("player_id").with_columns(pl.Series("c_val_beta", c_beta), pl.lit(s, pl.Int64).alias("season"))
        )
        prev = coefs
        log.info(
            "DPM refit %d: fringe effect on defense (xGA/60 per fringe skater) F %.4f D %.4f", s, c_f["F"], c_f["D"]
        )
    return pl.concat(rows), pl.DataFrame(fr)


def dpm_component(
    refit: pl.DataFrame, fr: pl.DataFrame, status: pl.DataFrame, ps: pl.DataFrame, dpm: pl.DataFrame
) -> pl.DataFrame:
    d = (
        refit.join(status.select("player_id", "season", "pos_group", "fringe"), on=["player_id", "season"], how="inner")
        .join(
            fr.select("season", "c_val_fringe_F", "c_val_fringe_D", "dpm_rep_per60_F", "dpm_rep_per60_D"), on="season"
        )
        .join(ps.select("player_id", "season", "toi_ev"), on=["player_id", "season"])
        .join(dpm.select("player_id", "season", "dpm_se", "dpm_val_per60"), on=["player_id", "season"], how="left")
    )
    c_f = pl.when(pl.col("pos_group") == "D").then(pl.col("c_val_fringe_D")).otherwise(pl.col("c_val_fringe_F"))
    rep = pl.when(pl.col("pos_group") == "D").then(pl.col("dpm_rep_per60_D")).otherwise(pl.col("dpm_rep_per60_F"))
    med = float(dpm["dpm_se"].drop_nulls().median())
    # DEFAULT (methodology §11 deviation 5): the replacement effect is pooled over all seasons, because the
    # per-season fringe estimate is noisy. The per-season estimate is kept as the *_perseason alternative.
    pooled = {"F": float(fr["c_val_fringe_F"].mean()), "D": float(fr["c_val_fringe_D"].mean())}
    c_pool = pl.when(pl.col("pos_group") == "D").then(pl.lit(pooled["D"])).otherwise(pl.lit(pooled["F"]))
    reg = 1 - pl.col("fringe").cast(pl.Float64)
    return (
        d.with_columns(
            (-pl.col("c_val_beta") + reg * c_pool).alias("dpm_var_per60"),
            (-c_pool).alias("dpm_rep_per60"),
            (-pl.col("c_val_beta") + reg * c_f).alias("dpm_var_perseason_per60"),
            rep.alias("dpm_rep_perseason_per60"),
            (-pl.col("c_val_beta") - pl.col("fringe").cast(pl.Float64) * c_f).alias("dpm_val_refit_per60"),
        )
        .with_columns(
            (pl.col("dpm_var_per60") * pl.col("toi_ev") / 60).alias("def_gar"),
            (pl.col("dpm_var_perseason_per60") * pl.col("toi_ev") / 60).alias("def_gar_perseason"),
            (pl.col("dpm_se").fill_null(med) * pl.col("toi_ev") / 60).alias("def_gar_sd"),
        )
        .select(
            "player_id",
            "season",
            "dpm_var_per60",
            "dpm_val_refit_per60",
            "dpm_rep_per60",
            "def_gar",
            "def_gar_perseason",
            "def_gar_sd",
            "dpm_val_per60",
        )
    )


# ------------------------------------------------------------------ finishing (from saved shot shifts)
def fin_from_shifts(shifts: pl.DataFrame, status: pl.DataFrame) -> pl.DataFrame:
    """5v5 finishing goals above the fringe pool (under ``status``), shot by shot, from the fitted shifts."""
    s = shifts.join(
        status.select(pl.col("player_id").alias("shooter"), "season", pl.col("fringe").alias("fringe_alt")),
        on=["shooter", "season"],
        how="left",
    ).with_columns(pl.col("fringe_alt").fill_null(True))
    s5 = s.filter(pl.col("strength_g") == "5v5")
    ref = s5.filter(pl.col("fringe_alt")).group_by("pos_group", "season").agg(pl.col("T").mean().alias("ref_rep"))
    s5 = s5.join(ref, on=["pos_group", "season"], how="left").with_columns(pl.col("ref_rep").fill_null(0.0))
    g = expit(s5["l"].to_numpy() + s5["T"].to_numpy()) - expit(s5["l"].to_numpy() + s5["ref_rep"].to_numpy())
    return (
        s5.with_columns(pl.Series("g", g))
        .group_by("shooter", "season")
        .agg(pl.col("g").sum().alias("fin_gar"))
        .rename({"shooter": "player_id"})
    )


# ------------------------------------------------------------------ table
CORE = ["pm", "def", "fin"]


def combine(
    ps: pl.DataFrame,
    pm: pl.DataFrame,
    dp: pl.DataFrame,
    fin: pl.DataFrame,
    fin_sd: pl.DataFrame | None,
    fin_memo: pl.DataFrame | None,
    gpw: float,
    extras: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Join the components and total them. ``extras`` = additional 5v5 components, each with player_id, season,
    `<name>_gar`, `<name>_gar_sd`."""
    names = CORE + list((extras or {}).keys())
    spine = ps.select(
        "player_id",
        "season",
        "player_name",
        "position",
        "pos_group",
        pl.col("primary_team").alias("team"),
        pl.col("toi_ev").alias("toi_5v5"),
        "gp",
        "pm_epa_per60_ev",
        "qualified_ev",
    )
    t = (
        spine.join(pm, on=["player_id", "season"], how="left")
        .join(dp, on=["player_id", "season"], how="left")
        .join(fin, on=["player_id", "season"], how="left")
    )
    if fin_sd is not None:
        t = t.join(fin_sd, on=["player_id", "season"], how="left")
    else:
        t = t.with_columns(pl.lit(None, pl.Float64).alias("fin_gar_sd"))
    if fin_memo is not None:
        t = t.join(fin_memo, on=["player_id", "season"], how="left")
    for name, df in (extras or {}).items():
        assert {f"{name}_gar", f"{name}_gar_sd"} <= set(df.columns), name
        # all columns: value, sd and the component's memo columns
        t = t.join(df, on=["player_id", "season"], how="left")
    t = t.with_columns(
        *[pl.col(f"{n}_gar").fill_null(0.0) for n in names], *[pl.col(f"{n}_gar_sd").fill_null(0.0) for n in names]
    )
    t = t.with_columns(
        sum((pl.col(f"{n}_gar") for n in names), pl.lit(0.0)).alias("total_gar"),
        sum((pl.col(f"{n}_gar_sd") ** 2 for n in names), pl.lit(0.0)).sqrt().alias("total_gar_sd"),
    )
    if "def_gar_perseason" in t.columns:  # alternative total with the per-season DPM replacement level
        t = t.with_columns(
            (pl.col("total_gar") - pl.col("def_gar") + pl.col("def_gar_perseason").fill_null(0.0)).alias(
                "total_gar_perseason"
            )
        )
    t = t.with_columns(
        (pl.col("total_gar") / gpw).alias("total_war"),
        (pl.col("total_gar_sd") / gpw).alias("total_war_sd"),
        (pl.col("total_gar") - 1.96 * pl.col("total_gar_sd")).alias("total_gar_lo"),
        (pl.col("total_gar") + 1.96 * pl.col("total_gar_sd")).alias("total_gar_hi"),
    )
    t = t.with_columns(*[(pl.col(f"{n}_gar") / gpw).alias(f"{n}_wins") for n in names])
    return t.sort("season", "total_gar", descending=[False, True])


def reconcile(t: pl.DataFrame, proc) -> dict:
    """Team-season 5v5 goal differential vs the sum of player components (above replacement), by primary team."""
    games = pl.read_parquet(proc / "games.parquet")
    abbr = pl.concat(
        [
            games.select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")),
            games.select(pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team")),
        ]
    ).unique("team_id")
    goals = pl.read_parquet(
        proc / "events.parquet", columns=["season", "type", "team_id", "opp_team_id", "strength"]
    ).filter((pl.col("type") == "goal") & (pl.col("strength") == "5v5"))
    gf = goals.group_by("season", "team_id").agg(pl.len().cast(pl.Int64).alias("gf"))  # signed: gf - ga must not wrap
    ga = goals.group_by("season", pl.col("opp_team_id").alias("team_id")).agg(pl.len().cast(pl.Int64).alias("ga"))
    gd = (
        gf.join(ga, on=["season", "team_id"], how="full", coalesce=True)
        .fill_null(0)
        .with_columns((pl.col("gf") - pl.col("ga")).alias("gd5"))
        .join(abbr, on="team_id")
    )
    extra = ["total_gar_perseason"] if "total_gar_perseason" in t.columns else []
    all_comps = [f"{n}_gar" for n in CORE + [c[:-4] for c in t.columns if c in ("to_gar", "fo_gar", "pen_gar")]]
    # 5v5 target: penalties are realized on special teams (checked separately)
    comps = [c for c in all_comps if c != "pen_gar"]
    raw_cols = [c for c in ("pm_raw_gar", "def_gar", "fin_gar", "to_raw_gar", "fo_gar") if c in t.columns]
    has_orig = "to_gar_orig" in t.columns
    sums = t.group_by("season", "team").agg(
        *[
            pl.col(c).sum().alias(c)
            for c in dict.fromkeys([*all_comps, "total_gar", *extra, *raw_cols, *(["to_gar_orig"] if has_orig else [])])
        ]
    )
    j = (
        gd.join(sums, on=["season", "team"])
        .filter(pl.col("gd5").is_not_null())
        .join(recon.team_gd_st(proc), on=["season", "team"], how="left")
    )
    y = j["gd5"].to_numpy().astype(float)
    out = {"n_team_seasons": int(j.height), "components": comps}
    for c in ("total_gar", *all_comps, *extra):
        x = j[c].to_numpy()
        A = np.column_stack([np.ones(len(x)), x])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        out[c] = {"slope": float(coef[1]), "intercept": float(coef[0]), "r": float(np.corrcoef(x, y)[0, 1])}

    def joint(cols: list[str]) -> dict:
        r = recon.regress(j, cols)
        return {"r2": r["r2"], "slopes": r["slopes"], **{f"{c[:-4]}_slope": v["slope"] for c, v in r["slopes"].items()}}

    out["joint"] = joint(comps)
    # the same regression with the unshrunk possession and turnover values (shrinkage compresses team sums)
    if len(raw_cols) > 3:
        out["joint_unshrunk"] = joint(raw_cols)
    if "pen_gar" in j.columns:
        r = recon.regress(j, ["pen_gar"], y="st_gd")
        out["penalty_vs_special_teams_gd"] = {
            "slope": r["slopes"]["pen_gar"]["slope"],
            "se": r["slopes"]["pen_gar"]["se"],
            "r2": r["r2"],
            "sd_st_gd": float(j["st_gd"].std()),
        }
    if has_orig:  # the original (not role-adjusted) turnover charge in place of the default TO-RA
        out["joint_to_orig"] = joint(["to_gar_orig" if c == "to_gar" else c for c in comps])
    # the three earlier components only (continuity with the earlier report)
    out["joint_core"] = joint([f"{n}_gar" for n in CORE])
    out["sd_team_5v5_gd"] = float(y.std())
    out["sd_sum_total_gar"] = float(j["total_gar"].std())
    return out
