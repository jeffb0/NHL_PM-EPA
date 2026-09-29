"""Strength-specific ridge RAPM for the special-teams states (methodology §3).

The design reuses ``src.defense.ridge``: two rows per stint (one per attacking side), player blocks
[offense | defense] penalized, everything else fixed effects. Roles: in a mismatched state a player's columns are
keyed ``player_id * 10 + role`` (1 = advantaged side, 2 = disadvantaged side); in 4v4 and 3v3 the role is 0.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import polars as pl
from scipy import sparse

from src.defense import ridge
from src.defense.stints import COMPONENTS

log = logging.getLogger(__name__)

ADV_STATES = [(5, 3), (4, 3), (6, 5), (6, 4), (6, 3), (4, 5), (3, 5), (3, 4), (5, 6), (4, 6), (3, 6)]   # ordered (attackers, defenders); (5, 4) is the reference
CLASSES = {"pp": ("adv", 1), "pk": ("adv", 2), "4v4": ("4v4", 0), "3v3": ("3v3", 0)}
GROUP_ROLES = {"adv": (1, 2), "4v4": (0,), "3v3": (0,)}


def fringe_lookup(status: pl.DataFrame) -> dict[tuple[int, int], tuple[bool, bool]]:
    """(season, player_id) -> (is_fringe, is_defenseman)."""
    return {(s, p): (bool(f), g == "D") for p, s, f, g in status.select("player_id", "season", "fringe", "pos_group").iter_rows()}


def _fr_keys(group: str) -> list[tuple[int, str, str]]:
    return [(r, side, pos) for r in GROUP_ROLES[group] for side in ("att", "def") for pos in ("F", "D")]


def as_stints_frame(st: pl.DataFrame, group: str) -> pl.DataFrame:
    """Rewrite one group's stints in the layout of ``ridge.build_design`` (role-keyed ids, xG in one ledger column)."""
    adv = group == "adv"
    roles = [((1, 2) if h > a else (2, 1)) if adv else (0, 0) for h, a in zip(st["nh"].to_list(), st["na"].to_list())]
    home_on = [[p * 10 + r[0] for p in lst] for lst, r in zip(st["home_on"].to_list(), roles)]
    away_on = [[p * 10 + r[1] for p in lst] for lst, r in zip(st["away_on"].to_list(), roles)]
    d = st.with_columns(pl.Series("home_on", home_on, dtype=pl.List(pl.Int64)), pl.Series("away_on", away_on, dtype=pl.List(pl.Int64)))
    zero = pl.lit(0.0)
    for side in "ha":
        for c in COMPONENTS:
            d = d.with_columns((pl.col(f"{side}_xg") if c == "move" else zero).alias(f"{side}_{c}"))
    return d


def extras(st: pl.DataFrame, group: str, lookup: dict) -> np.ndarray:
    """Extra unpenalized covariates for the 2n rows: skater-count state dummies (adv) and fringe counts by role, block, position."""
    n = st.height
    keys = _fr_keys(group)
    kidx = {k: i for i, k in enumerate(keys)}
    fr = np.zeros((2 * n, len(keys)))
    seasons = st["season"].to_list()
    home_on, away_on, nh, na = st["home_on"].to_list(), st["away_on"].to_list(), st["nh"].to_list(), st["na"].to_list()
    adv = group == "adv"
    for r in range(n):
        s = seasons[r]
        rh, ra = ((1, 2) if nh[r] > na[r] else (2, 1)) if adv else (0, 0)
        for row, att, dfn, r_att, r_def in ((r, home_on[r], away_on[r], rh, ra), (n + r, away_on[r], home_on[r], ra, rh)):
            for p in att:
                f, is_d = lookup.get((s, p), (True, False))
                if f:
                    fr[row, kidx[(r_att, "att", "D" if is_d else "F")]] += 1
            for p in dfn:
                f, is_d = lookup.get((s, p), (True, False))
                if f:
                    fr[row, kidx[(r_def, "def", "D" if is_d else "F")]] += 1
    cols = [fr]
    if adv:
        state = np.r_[np.column_stack([nh, na]), np.column_stack([na, nh])]
        cols.insert(0, np.column_stack([(state[:, 0] == a) & (state[:, 1] == b) for a, b in ADV_STATES]).astype(float))
    return np.hstack(cols)


def make_design(st: pl.DataFrame, group: str, lookup: dict, players: list[int] | None = None) -> ridge.Design:
    """Design for one group's stints (any set of seasons). The last ``len(_fr_keys(group))`` columns are the fringe counts."""
    fr = as_stints_frame(st, group)
    d = ridge.build_design(fr, ["xg"], players)
    ex = extras(st, group, lookup)
    return dataclasses.replace(d, X=sparse.hstack([d.X, sparse.csr_matrix(ex)]).tocsr(), n_cov=d.n_cov + ex.shape[1])


def n_fringe_cols(group: str) -> int:
    return len(_fr_keys(group))


def chain_fits(designs: dict[int, ridge.Design], alpha: float, rho: float) -> dict:
    """Season fits in order; with rho > 0 each season's ridge prior mean is rho x the previous season's coefficients."""
    out, prev = {}, None
    for s in sorted(designs):
        d = designs[s]
        beta = ridge.fit(d, {"xg": alpha}, ridge.prior_vector(d, prev, rho))
        coefs = ridge.coef_frame(d, beta)
        out[s] = (d, beta, coefs)
        prev = coefs
    return out


def predict_next(fit_d: ridge.Design, beta: np.ndarray, nxt: pl.DataFrame, group: str, lookup: dict) -> tuple[ridge.Design, np.ndarray, np.ndarray]:
    """Season-Y coefficients applied to season Y+1 stints: (design, full prediction, covariates-only prediction)."""
    nd = make_design(nxt, group, lookup, players=fit_d.players)
    b = beta[:, 0]
    P = fit_d.P
    cov_only = b.copy()
    cov_only[1:1 + 2 * P] = 0.0
    return nd, nd.X @ b, nd.X @ cov_only


def fringe_effects(fits: dict, group: str) -> dict[tuple[int, str, str], float]:
    """Pooled (mean over seasons) fitted fringe-count coefficients per (role, block, position)."""
    keys = _fr_keys(group)
    k = len(keys)
    mat = np.array([beta[-k:, 0] for (_, beta, _) in fits.values()])
    return dict(zip(keys, mat.mean(0).tolist()))


def player_values(fits: dict, group: str, fringe_fx: dict, lookup: dict, draws: dict[int, np.ndarray] | None = None) -> pl.DataFrame:
    """Per player-season-role rate above replacement (xG per 60) and goals, from the fitted coefficients.

    rate = (offense - regular * f_att) - (defense - regular * f_def): a regular's coefficients are measured against the pooled
    fringe effect of the same role, block and position; a fringe player carries his own shrunk deviation.
    """
    rows = []
    for s, (d, _, coefs) in fits.items():
        mins = ridge.player_minutes(d).rename({"toi_5v5": "minutes"})
        t = coefs.join(mins, on="player_id")
        ids = t["player_id"].to_numpy()
        pid, role = ids // 10, ids % 10
        fr, is_d = zip(*[lookup.get((s, int(p)), (True, False)) for p in pid])
        fr, is_d = np.array(fr), np.array(is_d)
        pos = np.where(is_d, "D", "F")
        f_att = np.array([fringe_fx[(int(r), "att", p)] for r, p in zip(role, pos)])
        f_def = np.array([fringe_fx[(int(r), "def", p)] for r, p in zip(role, pos)])
        reg = (~fr).astype(float)
        rate = (t["off_xg"].to_numpy() - reg * f_att) - (t["def_xg"].to_numpy() - reg * f_def)
        sd_rate = np.full(len(ids), np.nan)
        if draws is not None:
            P = d.P
            net = draws[s][:, 1:1 + P, 0] - draws[s][:, 1 + P:1 + 2 * P, 0]
            sd_rate = net.std(0)
        m = t["minutes"].to_numpy()
        rows.append(pl.DataFrame({"player_id": pid, "role": role, "season": s, "minutes": m, "fringe": fr, "pos_group": pos, "rate_rapm": rate,
                                  "gar_rapm": rate * m / 60.0, "gar_rapm_sd": sd_rate * m / 60.0, "off_xg": t["off_xg"], "def_xg": t["def_xg"]}))
    return pl.concat(rows).with_columns(pl.col("season").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("role").cast(pl.Int64))
