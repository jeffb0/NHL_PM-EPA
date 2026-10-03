"""Pre-NHL features: league translation factors, NHL-equivalent production, draft position (docs/prospects/methodology.md §3)."""

from __future__ import annotations

import gzip
import json

import numpy as np
import polars as pl

from src.projection import data as pd_
from src.projection.model import Spec

YP = ("yp_nhle", "yp_has_nhle", "yp_pick", "yp_undrafted")
MIN_GP_TRANS, MIN_GP_FEAT, PSEUDO, UNDRAFTED_PICK = 20, 10, 10.0, 250.0


def load_landing(raw) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(season rows: player_id, season [start year], league, gp, points; draft: player_id, overall pick or null)."""
    rows, draft = [], []
    for f in sorted((raw / "player_landing").glob("*.json.gz")):
        d = json.load(gzip.open(f))
        pid = int(d["playerId"])
        dd = d.get("draftDetails") or {}
        draft.append((pid, dd.get("overallPick")))
        for s in d.get("seasonTotals") or []:
            if s.get("gameTypeId") != 2 or not s.get("gamesPlayed"):
                continue
            rows.append((pid, int(str(s["season"])[:4]), str(s.get("leagueAbbrev")), float(s["gamesPlayed"]), float(s.get("points") or 0)))
    st = pl.DataFrame(rows, schema={"player_id": pl.Int64, "season": pl.Int64, "league": pl.Utf8, "gp": pl.Float64, "points": pl.Float64}, orient="row")
    st = st.group_by("player_id", "season", "league").agg(pl.col("gp").sum(), pl.col("points").sum())
    return st, pl.DataFrame(draft, schema={"player_id": pl.Int64, "overall": pl.Float64}, orient="row")


def transitions(st: pl.DataFrame) -> pl.DataFrame:
    """A season in a non-NHL league followed by an NHL season (both >= 20 games): league, NHL season, both points-per-game rates."""
    nhl = st.filter((pl.col("league") == "NHL") & (pl.col("gp") >= MIN_GP_TRANS)).select("player_id", "season", (pl.col("points") / pl.col("gp")).alias("nhl_ppg"))
    lg = st.filter((pl.col("league") != "NHL") & (pl.col("gp") >= MIN_GP_TRANS)).select("player_id", (pl.col("season") + 1).alias("season"), "league", (pl.col("points") / pl.col("gp")).alias("lg_ppg"))
    return lg.join(nhl, on=["player_id", "season"], how="inner")


def factors(tr: pl.DataFrame, T: int) -> tuple[dict, float]:
    """League -> shrunk translation factor from transitions whose NHL season is before T; and the pooled factor."""
    t = tr.filter(pl.col("season") < T)
    if t.height == 0:
        return {}, 0.3
    pooled = float(t["nhl_ppg"].mean() / t["lg_ppg"].mean())
    g = t.group_by("league").agg(pl.len().alias("n"), (pl.col("nhl_ppg").mean() / pl.col("lg_ppg").mean()).alias("r"))
    g = g.with_columns(((pl.col("n") * pl.col("r") + PSEUDO * pooled) / (pl.col("n") + PSEUDO)).alias("f"))
    return dict(g.select("league", "f").iter_rows()), pooled


def features(st: pl.DataFrame, draft: pl.DataFrame, tr: pl.DataFrame, T: int) -> pl.DataFrame:
    """Per player entering season T: NHLe from his most recent non-NHL season among T-1, T-2, and draft position. Uses rows from seasons < T only."""
    f, pooled = factors(tr, T)
    c = st.filter((pl.col("league") != "NHL") & (pl.col("gp") >= MIN_GP_FEAT) & pl.col("season").is_in([T - 1, T - 2]))
    c = c.sort(["player_id", "season", "gp"], descending=[False, True, True]).unique("player_id", keep="first", maintain_order=True)
    c = c.with_columns(pl.col("league").replace_strict(f, default=pooled, return_dtype=pl.Float64).alias("f")).select("player_id", (pl.col("points") / pl.col("gp") * pl.col("f")).alias("nhle"), pl.col("league").alias("yp_league"))
    out = draft.join(c, on="player_id", how="left")
    return out.select("player_id", pl.lit(T, pl.Int64).alias("season"), pl.col("nhle").fill_null(0.0), pl.col("nhle").is_not_null().cast(pl.Float64).alias("has_nhle"),
                      pl.col("overall").fill_null(UNDRAFTED_PICK).log().alias("pick"), pl.col("overall").is_null().cast(pl.Float64).alias("undrafted"), "yp_league")


def table(raw, seasons: list[int]) -> pl.DataFrame:
    """Feature rows for every fetched player and every target season in ``seasons``."""
    st, draft = load_landing(raw)
    tr = transitions(st)
    return pl.concat([features(st, draft, tr, T) for T in seasons])


def with_yp(frame: pl.DataFrame, yp: pl.DataFrame | None, m0: float = pd_.M0) -> pl.DataFrame:
    """Add the four faded columns to a design frame (player_id, season, m_eff). Players without a career page get the undrafted, no-NHLe values."""
    if yp is None:
        return frame
    j = frame.join(yp.select("player_id", "season", "nhle", "has_nhle", "pick", "undrafted"), on=["player_id", "season"], how="left")
    w = m0 / (pl.col("m_eff") + m0)
    return j.with_columns((w * pl.col("nhle").fill_null(0.0)).alias("yp_nhle"), (w * pl.col("has_nhle").fill_null(0.0)).alias("yp_has_nhle"),
                          (w * pl.col("pick").fill_null(float(np.log(UNDRAFTED_PICK)))).alias("yp_pick"), (w * pl.col("undrafted").fill_null(1.0)).alias("yp_undrafted")).drop("nhle", "has_nhle", "pick", "undrafted")


def spec(yp: pl.DataFrame | None) -> Spec:
    return Spec(extra=YP) if yp is not None else Spec()
