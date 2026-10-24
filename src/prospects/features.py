"""Pre-NHL features: league translation factors, NHL-equivalent production, draft position
(docs/prospects/methodology.md §3)."""

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
        draft.append((pid, dd.get("overallPick"), dd.get("year")))
        for s in d.get("seasonTotals") or []:
            if s.get("gameTypeId") != 2 or not s.get("gamesPlayed"):
                continue
            rows.append(
                (
                    pid,
                    int(str(s["season"])[:4]),
                    str(s.get("leagueAbbrev")),
                    float(s["gamesPlayed"]),
                    float(s.get("points") or 0),
                )
            )
    st = pl.DataFrame(
        rows,
        schema={"player_id": pl.Int64, "season": pl.Int64, "league": pl.Utf8, "gp": pl.Float64, "points": pl.Float64},
        orient="row",
    )
    st = st.group_by("player_id", "season", "league").agg(pl.col("gp").sum(), pl.col("points").sum())
    return st, pl.DataFrame(
        draft, schema={"player_id": pl.Int64, "overall": pl.Float64, "draft_year": pl.Int64}, orient="row"
    )


def transitions(st: pl.DataFrame) -> pl.DataFrame:
    """A season in a non-NHL league followed by an NHL season (both >= 20 games): league, NHL season, both
    points-per-game rates."""
    nhl = st.filter((pl.col("league") == "NHL") & (pl.col("gp") >= MIN_GP_TRANS)).select(
        "player_id", "season", (pl.col("points") / pl.col("gp")).alias("nhl_ppg")
    )
    lg = st.filter((pl.col("league") != "NHL") & (pl.col("gp") >= MIN_GP_TRANS)).select(
        "player_id", (pl.col("season") + 1).alias("season"), "league", (pl.col("points") / pl.col("gp")).alias("lg_ppg")
    )
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
    """Per player entering season T: NHLe from his most recent non-NHL season among T-1, T-2, and draft position. Uses
    rows from seasons < T only."""
    f, pooled = factors(tr, T)
    c = st.filter((pl.col("league") != "NHL") & (pl.col("gp") >= MIN_GP_FEAT) & pl.col("season").is_in([T - 1, T - 2]))
    c = c.sort(["player_id", "season", "gp"], descending=[False, True, True]).unique(
        "player_id", keep="first", maintain_order=True
    )
    c = c.with_columns(pl.col("league").replace_strict(f, default=pooled, return_dtype=pl.Float64).alias("f")).select(
        "player_id", (pl.col("points") / pl.col("gp") * pl.col("f")).alias("nhle"), pl.col("league").alias("yp_league")
    )
    out = draft.join(c, on="player_id", how="left")
    return out.select(
        "player_id",
        pl.lit(T, pl.Int64).alias("season"),
        pl.col("nhle").fill_null(0.0),
        pl.col("nhle").is_not_null().cast(pl.Float64).alias("has_nhle"),
        pl.col("overall").fill_null(UNDRAFTED_PICK).log().alias("pick"),
        pl.col("overall").is_null().cast(pl.Float64).alias("undrafted"),
        "yp_league",
    )


def table(raw, seasons: list[int]) -> pl.DataFrame:
    """Feature rows for every fetched player and every target season in ``seasons``."""
    st, draft = load_landing(raw)
    tr = transitions(st)
    return pl.concat([features(st, draft, tr, T) for T in seasons])


ELITE = ("yp_nhle", "yp_has_nhle", "yp_top3", "yp_p4_10", "yp_p11_32", "yp_d0", "yp_d1")
DEFAULTS = {
    "nhle": 0.0,
    "has_nhle": 0.0,
    "pick": float(np.log(UNDRAFTED_PICK)),
    "undrafted": 1.0,
    "top3": 0.0,
    "p4_10": 0.0,
    "p11_32": 0.0,
    "d0": 0.0,
    "d1": 0.0,
}


def elite_features(st: pl.DataFrame, draft: pl.DataFrame, tr: pl.DataFrame, T: int) -> pl.DataFrame:
    """docs/prospects/elite.md §1A: NHL-equivalent production, draft slot bins and debut timing for a player entering
    season T (rows from seasons < T only;
    a player with no NHL game before T debuts in T). ``nhl_gp_before`` is for test subsets, not a feature."""
    base = features(st, draft, tr, T).select("player_id", "season", "nhle", "has_nhle")
    nhl = st.filter((pl.col("league") == "NHL") & (pl.col("gp") > 0))
    before = (
        nhl.filter(pl.col("season") < T)
        .group_by("player_id")
        .agg(pl.col("season").min().alias("debut"), pl.col("gp").sum().alias("nhl_gp_before"))
    )
    d = draft.join(before, on="player_id", how="left").with_columns(
        pl.col("debut").fill_null(T), pl.col("nhl_gp_before").fill_null(0.0)
    )
    ov = pl.col("overall")
    d = d.select(
        "player_id",
        (ov <= 3).fill_null(False).cast(pl.Float64).alias("top3"),
        ((ov > 3) & (ov <= 10)).fill_null(False).cast(pl.Float64).alias("p4_10"),
        ((ov > 10) & (ov <= 32)).fill_null(False).cast(pl.Float64).alias("p11_32"),
        (pl.col("debut") == pl.col("draft_year")).fill_null(False).cast(pl.Float64).alias("d0"),
        (pl.col("debut") == pl.col("draft_year") + 1).fill_null(False).cast(pl.Float64).alias("d1"),
        "nhl_gp_before",
        "overall",
        "draft_year",
    )
    return base.join(d, on="player_id", how="left")


def elite_table(raw, seasons: list[int]) -> pl.DataFrame:
    st, draft = load_landing(raw)
    tr = transitions(st)
    return pl.concat([elite_features(st, draft, tr, T) for T in seasons])


def _names(yp: pl.DataFrame) -> tuple[str, ...]:
    return ELITE if "top3" in yp.columns else YP


def with_yp(frame: pl.DataFrame, yp: pl.DataFrame | None, m0: float = pd_.M0) -> pl.DataFrame:
    """Add the faded pre-NHL columns to a design frame (player_id, season, m_eff): the docs/prospects set, or the elite
    set when the table carries draft-slot bins.
    Players without a career page get the undrafted, no-production values."""
    if yp is None:
        return frame
    base = [n[3:] for n in _names(yp)]
    j = frame.join(yp.select("player_id", "season", *base), on=["player_id", "season"], how="left")
    w = m0 / (pl.col("m_eff") + m0)
    return j.with_columns([(w * pl.col(b).fill_null(DEFAULTS[b])).alias(f"yp_{b}") for b in base]).drop(base)


def spec(yp: pl.DataFrame | None) -> Spec:
    return Spec(extra=_names(yp)) if yp is not None else Spec()
