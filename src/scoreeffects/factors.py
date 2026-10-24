"""Score-state weights for 5v5 expected goals (docs/scoreeffects/methodology.md §2)."""

from __future__ import annotations

import polars as pl

CAP = 2
PERIODS = (1, 2, 3)


def state_rates(proc) -> pl.DataFrame:
    """Per (season, period, d): 5v5 expected goals for by teams in score state d (own minus opponent goals, capped at
    +/-2) and the 5v5 minutes teams spent in that state."""
    sh = pl.read_parquet(
        proc / "fin_shots.parquet", columns=["season", "period", "score_diff", "strength", "xg_v1"]
    ).filter((pl.col("strength") == "5v5") & pl.col("period").is_in(PERIODS))
    x = (
        sh.with_columns(pl.col("score_diff").clip(-CAP, CAP).alias("d"))
        .group_by("season", "period", "d")
        .agg(pl.col("xg_v1").sum().alias("xg"))
    )
    st = pl.read_parquet(proc / "def_stints.parquet", columns=["season", "period", "score_home", "minutes"]).filter(
        pl.col("period").is_in(PERIODS)
    )
    m = (
        pl.concat(
            [
                st.select("season", "period", pl.col("score_home").alias("d"), "minutes"),
                st.select("season", "period", (-pl.col("score_home")).alias("d"), "minutes"),
            ]
        )
        .group_by("season", "period", "d")
        .agg(pl.col("minutes").sum())
    )
    return x.join(m.with_columns(pl.col("d").cast(x["d"].dtype)), on=["season", "period", "d"], how="inner")


def weights(rates: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """w(d, period) = rate when tied / rate in state d, in the same period, pooled over ``seasons``. Empty input gives
    no rows (no adjustment)."""
    r = (
        rates.filter(pl.col("season").is_in(seasons))
        .group_by("period", "d")
        .agg(pl.col("xg").sum(), pl.col("minutes").sum())
        .with_columns((pl.col("xg") / pl.col("minutes") * 60).alias("rate60"))
    )
    tied = r.filter(pl.col("d") == 0).select("period", pl.col("rate60").alias("tied60"))
    return (
        r.join(tied, on="period")
        .with_columns((pl.col("tied60") / pl.col("rate60")).alias("w"))
        .select("period", "d", "rate60", "w")
        .sort("period", "d")
    )


def apply(shots: pl.DataFrame, w: pl.DataFrame, xg: str = "xg") -> pl.DataFrame:
    """Add ``xg_adj``: 5v5 shots in periods 1-3 are multiplied by the weight of their score state; everything else is
    unchanged. Needs columns period, score_diff, strength and ``xg``."""
    if w.height == 0:
        return shots.with_columns(pl.col(xg).alias("xg_adj"))
    j = shots.with_columns(pl.col("score_diff").clip(-CAP, CAP).cast(w["d"].dtype).alias("_d")).join(
        w.select(pl.col("period").cast(shots["period"].dtype), pl.col("d").alias("_d"), "w"),
        on=["period", "_d"],
        how="left",
    )
    return j.with_columns(
        pl.when((pl.col("strength") == "5v5") & pl.col("w").is_not_null())
        .then(pl.col(xg) * pl.col("w"))
        .otherwise(pl.col(xg))
        .alias("xg_adj")
    ).drop("_d", "w")
