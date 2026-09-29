"""Projection data: player bio, the component-rate panel (rolling-refit finishing and faceoffs), history features (docs/projection/methodology.md §2)."""

from __future__ import annotations

import gzip
import json

import polars as pl

COMPS = ["pm", "def", "fin", "to", "fo", "pen", "pp", "pk", "4v4", "3v3"]
MARCEL_W = (5.0, 4.0, 3.0)
M0 = 1000.0
MIN_TARGET_MIN = 200.0


def player_bio(proc, raw) -> pl.DataFrame:
    """player_id, birth_date, height_in, weight_lb from the raw roster files (cached)."""
    path = proc / "player_bio.parquet"
    if path.exists():
        return pl.read_parquet(path)
    rows = {}
    for f in sorted((raw / "rosters").glob("*/*.json.gz")):
        d = json.load(gzip.open(f))
        for grp in ("forwards", "defensemen", "goalies"):
            for p in d.get(grp, []):
                rows[p["id"]] = (p["id"], p.get("birthDate"), p.get("heightInInches"), p.get("weightInPounds"))
    out = pl.DataFrame(list(rows.values()), schema={"player_id": pl.Int64, "birth_date": pl.Utf8, "height_in": pl.Float64, "weight_lb": pl.Float64}, orient="row")
    out = out.with_columns(pl.col("birth_date").str.to_date("%Y-%m-%d", strict=False))
    out.write_parquet(path)
    return out


def age_at(season: pl.Expr, birth: pl.Expr) -> pl.Expr:
    """Age in years on 1 October of the season's start year (seasons are labelled by start year)."""
    return ((pl.date(season, 10, 1) - birth).dt.total_days() / 365.25)


def panel(proc, roll: pl.DataFrame) -> pl.DataFrame:
    """One row per skater-season: minutes, the ten component rates per 60 (finishing and faceoffs rolling-refit), and their sum."""
    t = pl.read_parquet(proc / "player_value_calibrated_all.parquet")
    m = pl.col("toi_5v5") + sum((pl.col(f"st_{c}_min") for c in ("pp", "pk", "4v4", "3v3")), pl.lit(0.0))
    j = t.with_columns(m.alias("m")).join(roll.with_columns(pl.col("player_id").cast(t["player_id"].dtype), pl.col("season").cast(t["season"].dtype)), on=["player_id", "season"], how="left")
    j = j.with_columns([pl.col(c).fill_null(0.0) for c in j.columns if c.startswith("roll_")]).filter(pl.col("m") > 0)
    gar = {"pm": pl.col("pm_gar"), "def": pl.col("def_gar"), "fin": pl.col("roll_fin_gar"), "to": pl.col("to_gar"), "fo": pl.col("roll_fo_gar"), "pen": pl.col("pen_gar"),
           **{c: pl.col(f"st_{c}_gar") - pl.col(f"st_{c}_fin_gar") + pl.col(f"roll_st_{c}_fin") for c in ("pp", "pk", "4v4", "3v3")}}
    out = j.select(pl.col("player_id").cast(pl.Int64), pl.col("season").cast(pl.Int64), "pos_group", "team", "m", *[(g / pl.col("m") * 60).alias(f"r_{c}") for c, g in gar.items()])
    return out.with_columns(pl.sum_horizontal([f"r_{c}" for c in COMPS]).alias("r_total"))


def history(p: pl.DataFrame, m0: float = M0) -> pl.DataFrame:
    """Per (player_id, t): shrunk history features from seasons t, t-1, t-2 (weights 5/4/3 times minutes), M_eff and last-season minutes.

    A player is present at t if he has any of the three seasons; the features use only seasons <= t."""
    parts = []
    for s, w in enumerate(MARCEL_W):
        parts.append(p.select("player_id", (pl.col("season") + s).alias("t"), pl.lit(w).alias("wt"), "m", *[f"r_{c}" for c in COMPS], "r_total", pl.lit(s).alias("lag")))
    long = pl.concat(parts).filter(pl.col("t") <= p["season"].max())
    long = long.with_columns((pl.col("wt") * pl.col("m")).alias("w"))
    agg = long.group_by("player_id", "t").agg(*[((pl.col("w") * pl.col(f"r_{c}")).sum() / pl.col("w").sum()).alias(f"rbar_{c}") for c in [*COMPS, "total"]],
                                              (pl.col("w").sum() / 5.0).alias("m_eff"),
                                              pl.col("m").filter(pl.col("lag") == 0).sum().alias("m_last"),
                                              (pl.col("m").sum() / len(MARCEL_W)).alias("m_mean3"))
    shrink = pl.col("m_eff") / (pl.col("m_eff") + m0)
    return agg.with_columns(*[(pl.col(f"rbar_{c}") * shrink).alias(f"rt_{c}") for c in [*COMPS, "total"]])


def design_frame(p: pl.DataFrame, bio: pl.DataFrame, k: int, m0: float = M0, targets_only: bool = True, floor: float = MIN_TARGET_MIN) -> pl.DataFrame:
    """Rows (player, target season T) with history features from t = T - k, age at T, bio, and the target rate and minutes.

    With ``targets_only`` False every player-season is returned (for prediction, including rows below the minutes floor)."""
    h = history(p, m0).with_columns((pl.col("t") + k).alias("season"))
    d = p.join(h.drop("t"), on=["player_id", "season"], how="left").join(bio, on="player_id", how="left")
    d = d.with_columns(pl.col("m_eff").is_null().cast(pl.Float64).alias("no_history"), age_at(pl.col("season"), pl.col("birth_date")).alias("age"))
    fill = [pl.col(c).fill_null(0.0) for c in d.columns if c.startswith(("rt_", "rbar_")) or c in ("m_eff", "m_last", "m_mean3")]
    d = d.with_columns(*fill)
    for c in ("height_in", "weight_lb", "age"):
        med = d.group_by("pos_group").agg(pl.col(c).median().alias("_med"))
        d = d.join(med, on="pos_group", how="left").with_columns(pl.col(c).fill_null(pl.col("_med"))).drop("_med")
    return d.filter(pl.col("m") >= floor) if targets_only else d
