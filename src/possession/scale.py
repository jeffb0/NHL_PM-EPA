"""Rescale the possession component in the projection panel (docs/possession/methodology.md §3)."""

from __future__ import annotations

import polars as pl

from src.projection.data import COMPS


def rescale(p: pl.DataFrame, kappa: dict[str, float] | None) -> pl.DataFrame:
    """Multiply the possession rate by kappa["F"] or kappa["D"] and recompute the total rate. ``None`` or all ones returns the panel unchanged."""
    if not kappa or all(abs(v - 1.0) < 1e-12 for v in kappa.values()):
        return p
    k = pl.when(pl.col("pos_group") == "D").then(kappa["D"]).otherwise(kappa["F"])
    return p.with_columns((pl.col("r_pm") * k).alias("r_pm")).with_columns(pl.sum_horizontal([f"r_{c}" for c in COMPS]).alias("r_total"))
