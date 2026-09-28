"""State-space definitions: spatial schemes, observation kinds, strength groups, state ids.

Coordinates are in the possessing team's attacking frame (feet): attacking goal line at
x=+89, own goal line at x=-89, blue lines at x=+-25, end-zone faceoff dots at x=+-69,
y=+-22 (circle radius 15 ft, so circle tops at x=+-54). +y is the attacker's left.

Spatial schemes (brief §5.1)
  zones   26 interpretable sub-zones (10 OZ, 6 NZ, 10 DZ); see ZONE_DOC.
  grid10  10 ft x 8.5 ft rectangles  -> 20 x 10 = 200 cells
  grid20  20 ft x 17 ft rectangles   -> 10 x 5  =  50 cells
Every scheme maps into its parent (grid -> zone -> coarse DZ/NZ/OZ) for shrinkage.

State kinds
  control   puck under control at a known location (takeaway; hit_mid; and the *arrival*
            state at terminal locations: shot, giveaway, lost_takeaway)
  faceoff   won faceoff (state reset; brief §4.2)
  loose_*   inferred recovery after a loose-puck origin. Origins with a reliable location
            (giveaway, block) keep the fine cell; origins whose recorded location is not
            where the puck was recovered (miss, sog, goal, rebound) use the coarse zone;
            ``loose_unobserved`` has no location.
"""

from __future__ import annotations

import numpy as np
import polars as pl

GOAL_X = 89.0
BLUE_X = 25.0

ZONE_DOC = {
    "OZ_behind": "x > 89 (behind attacking net)",
    "OZ_low_slot": "69 <= x <= 89, |y| <= 11 (net-front / inner slot)",
    "OZ_high_slot": "54 <= x < 69, |y| <= 11",
    "OZ_circle_L/R": "54 <= x <= 89, 11 < |y| <= 30 (faceoff circles, outside slot)",
    "OZ_boards_L/R": "54 <= x <= 89, |y| > 30 (low boards / corners)",
    "OZ_point_C/L/R": "25 <= x < 54 (point / top of zone); C if |y| <= 11",
    "NZ_att_C/L/R": "0 <= x < 25, lane by y (C if |y| <= 14)",
    "NZ_own_C/L/R": "-25 < x < 0, lane by y",
    "DZ_*": "mirror of OZ_* about x=0 (DZ_behind = behind own net, DZ_low_slot = own net-front, ...)",
}

PRECISE_LOOSE = {"loose_opp_giveaway", "loose_own_giveaway", "loose_opp_block", "loose_own_block"}
CONTROL_OBS = {"takeaway", "hit_mid", "shot", "giveaway", "lost_takeaway"}
STRENGTH_GROUP = {"5v5": "5v5", "PP": "PP", "PK": "PK", "4v4": "4v4_3v3", "3v3": "4v4_3v3", "EN": "EN", "other": "other"}


def _side(y: np.ndarray, width: float) -> np.ndarray:
    return np.where(np.abs(y) <= width, "C", np.where(y > 0, "L", "R"))


def zone_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """26-zone scheme. Vectorized; NaN coordinates -> 'NA'."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape, "NA", dtype=object)
    ok = ~(np.isnan(x) | np.isnan(y))
    ax = np.abs(x)
    # end zones computed on |x| with the prefix chosen by sign; y mirrored for DZ so that
    # "L/R" stays the attacker's left/right in both ends
    lr = np.where(y > 0, "L", "R")
    ez = np.full(x.shape, "", dtype=object)
    ez[ax > GOAL_X] = "behind"
    slot = (np.abs(y) <= 11)
    low = (ax >= 69) & (ax <= GOAL_X)
    deep = (ax >= 54) & (ax <= GOAL_X)
    ez[(ez == "") & low & slot] = "low_slot"
    ez[(ez == "") & deep & ~low & slot] = "high_slot"
    m = (ez == "") & deep & ~slot & (np.abs(y) <= 30)
    ez[m] = np.char.add("circle_", lr[m].astype(str))
    m = (ez == "") & deep & (np.abs(y) > 30)
    ez[m] = np.char.add("boards_", lr[m].astype(str))
    m = (ez == "") & (ax >= BLUE_X) & (ax < 54)
    ez[m] = np.char.add("point_", _side(y[m], 11).astype(str))
    end = ax >= BLUE_X
    out[ok & end & (x > 0)] = np.char.add("OZ_", ez[ok & end & (x > 0)].astype(str))
    out[ok & end & (x < 0)] = np.char.add("DZ_", ez[ok & end & (x < 0)].astype(str))
    nz = ok & ~end
    half = np.where(x >= 0, "NZ_att_", "NZ_own_")
    out[nz] = np.char.add(half[nz].astype(str), _side(y[nz], 14).astype(str))
    return out


def coarse_of(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.where(x >= BLUE_X, "OZ", np.where(x <= -BLUE_X, "DZ", "NZ")).astype(object)
    out[np.isnan(x)] = "NA"
    return out


def grid_of(x: np.ndarray, y: np.ndarray, dx: float, dy: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    nx, ny = int(round(200 / dx)), int(round(85 / dy))
    ix = np.clip(np.floor((np.nan_to_num(x) + 100) / dx), 0, nx - 1).astype(int)
    iy = np.clip(np.floor((np.nan_to_num(y) + 42.5) / dy), 0, ny - 1).astype(int)
    out = np.char.add(np.char.add(ix.astype(str), "_"), iy.astype(str)).astype(object)
    out[~ok] = "NA"
    return out


def grid_centers(dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = int(round(200 / dx)), int(round(85 / dy))
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx = -100 + (ix + 0.5) * dx
    cy = -42.5 + (iy + 0.5) * dy
    ids = np.char.add(np.char.add(ix.astype(str), "_"), iy.astype(str))
    return ids.ravel(), cx.ravel(), cy.ravel()


SCHEMES = {"zones": None, "grid10": (10.0, 8.5), "grid20": (20.0, 17.0)}


def spatial_of(scheme: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if scheme == "zones":
        return zone_of(x, y)
    dx, dy = SCHEMES[scheme]
    return grid_of(x, y, dx, dy)


def state_kind(obs_kind: np.ndarray, role: np.ndarray | None = None) -> np.ndarray:
    """Map observation kinds to state kinds (the *value lookup* class)."""
    k = np.asarray(obs_kind, dtype=object)
    out = k.copy()
    out[np.isin(k, list(CONTROL_OBS))] = "control"
    return out


def location_precision(kind: np.ndarray) -> np.ndarray:
    """'fine' | 'coarse' | 'none' spatial precision for each state kind."""
    k = np.asarray(kind, dtype=object)
    out = np.full(k.shape, "coarse", dtype=object)
    out[(k == "control") | (k == "faceoff") | np.isin(k, list(PRECISE_LOOSE))] = "fine"
    out[k == "loose_unobserved"] = "none"
    return out


def add_state_columns(df: pl.DataFrame, scheme: str, use_score: bool = False) -> pl.DataFrame:
    """Attach kind / spatial / parent / coarse / state_id columns to an observation frame.

    The frame needs ``obs_kind, x, y, strength, score_diff``.
    """
    x = df["x"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    y = df["y"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    kind = state_kind(df["obs_kind"].to_numpy())
    prec = location_precision(kind)
    fine = spatial_of(scheme, x, y)
    zone = zone_of(x, y)
    coarse = coarse_of(x)
    spatial = np.where(prec == "fine", fine, np.where(prec == "coarse", coarse, "NA"))
    parent = np.where(prec == "fine", zone if scheme != "zones" else coarse, np.where(prec == "coarse", "ALL", "NA"))
    sg = df["strength"].replace_strict(STRENGTH_GROUP, default="other").to_numpy()
    score = np.clip(df["score_diff"].fill_null(0).to_numpy(), -2, 2) if use_score else np.zeros(len(df), dtype=int)
    base = np.char.add(np.char.add(sg.astype(str), "|"), kind.astype(str))
    sid = np.char.add(np.char.add(base, "|"), spatial.astype(str))
    pid = np.char.add(np.char.add(base, "|"), parent.astype(str))
    if use_score:
        sc = np.char.add("|s", score.astype(str))
        sid, pid = np.char.add(sid, sc), np.char.add(pid, sc)
    return df.with_columns(
        pl.Series("kind", kind.astype(str)),
        pl.Series("spatial", spatial.astype(str)),
        pl.Series("zone", zone.astype(str)),
        pl.Series("coarse", coarse.astype(str)),
        pl.Series("sgroup", sg.astype(str)),
        pl.Series("state_id", sid.astype(str)),
        pl.Series("parent_id", pid.astype(str)),
        pl.Series("kind_id", base.astype(str)),
    )
