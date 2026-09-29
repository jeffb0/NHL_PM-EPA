"""Pluggable prior for the persistent shooter effect (methodology §5).

    u_p ~ N(Z_p gamma, sigma2_{u, group(p)})

``PriorSpec`` supplies (i) the player covariates Z (default none: zero prior mean) and (ii) the
variance group of each player (default: position group). gamma is estimated jointly and is
unpenalized, so an informative covariate (e.g. NHL EDGE shot speed) moves the prior mean of every
player who has it, while players without it fall back to the group mean (missingness indicator).

Plugging in NHL EDGE data later needs no change to the model:

    edge = pl.read_csv("edge_shooting.csv")      # player_id, shot_speed_max, skating_speed_max, ...
    fit(..., prior=CovariatePrior(edge, columns=["shot_speed_max", "skating_speed_max"]))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class PriorSpec:
    """Zero-mean prior with position-specific variances (the default)."""

    name: str = "position"

    def covariates(self, players: list[int]) -> tuple[np.ndarray, list[str]]:
        return np.zeros((len(players), 0)), []


@dataclass
class CovariatePrior(PriorSpec):
    """Prior mean linear in player covariates. Columns are z-scored on the players that have them;
    missing values are set to 0 with a per-column missingness indicator (also unpenalized)."""

    table: pl.DataFrame | None = None
    columns: list[str] | None = None
    name: str = "covariate"

    def covariates(self, players: list[int]) -> tuple[np.ndarray, list[str]]:
        cols = self.columns or [c for c in self.table.columns if c != "player_id"]
        tab = self.table.select(pl.col("player_id").cast(pl.Int64), *cols)
        t = pl.DataFrame({"player_id": pl.Series(players, dtype=pl.Int64)}).join(tab, on="player_id", how="left")
        out, names = [], []
        for c in cols:
            v = t[c].cast(pl.Float64).to_numpy()
            miss = np.isnan(v)
            mu, sd = np.nanmean(v), np.nanstd(v)
            z = np.where(miss, 0.0, (v - mu) / (sd if sd > 0 else 1.0))
            out.append(z)
            names.append(c)
            if miss.any():
                out.append(miss.astype(float))
                names.append(f"{c}__missing")
        return (np.column_stack(out) if out else np.zeros((len(players), 0))), names
