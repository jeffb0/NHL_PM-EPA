"""Rescale components in the projection panel (docs/possession/methodology.md §3; docs/deshrink/methodology.md §2)."""

from __future__ import annotations

import polars as pl

from src.projection.data import COMPS

PLAY5 = ("pm", "def", "to", "fo")  # the 5v5 play block: possession, defense, turnovers, faceoffs


def rescale(p: pl.DataFrame, kappa: dict[str, float] | None) -> pl.DataFrame:
    """Multiply the possession rate by kappa["F"] or kappa["D"], and the whole 5v5 play block by kappa["play5"], then
    recompute the total rate. ``None`` or all ones returns the panel unchanged."""
    if not kappa or all(abs(v - 1.0) < 1e-12 for v in kappa.values()):
        return p
    if "F" in kappa or "D" in kappa:
        k = pl.when(pl.col("pos_group") == "D").then(kappa.get("D", 1.0)).otherwise(kappa.get("F", 1.0))
        p = p.with_columns((pl.col("r_pm") * k).alias("r_pm"))
    if "play5" in kappa:
        p = p.with_columns([(pl.col(f"r_{c}") * kappa["play5"]).alias(f"r_{c}") for c in PLAY5])
    return p.with_columns(pl.sum_horizontal([f"r_{c}" for c in COMPS]).alias("r_total"))
