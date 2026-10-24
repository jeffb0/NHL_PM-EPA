"""Picks ledger: lock the day's picks, settle finished games, report profit:  python -m src.live.picks [YYYY-MM-DD]
(docs/market/picks.md)

Record-keeping for a personal betting ledger; nothing here places a bet."""

from __future__ import annotations

import csv
import logging
import sys
from datetime import UTC, date, datetime

import numpy as np
import polars as pl

from src.config import ROOT
from src.live import paths
from src.live import score as sc
from src.live.odds import implied
from src.viz import style
from src.viz.style import plt

log = logging.getLogger("live")
LOGS = paths.LOGS
PICKS = LOGS / "picks_log.csv"
COLS = [
    "locked_utc",
    "game_id",
    "game_date",
    "start_utc",
    "away",
    "home",
    "pick",
    "against",
    "side",
    "book",
    "american",
    "units",
    "p_model",
    "p_market",
    "p_sizing",
    "edge_vs_breakeven",
    "forecast_made_utc",
]
BANKROLL, KELLY, SHRINK, MIN_EDGE, MAX_UNITS = 100.0, 0.25, 0.5, 0.03, 3


def payout(american: int) -> float:
    """Profit per unit staked."""
    return american / 100.0 if american > 0 else 100.0 / -american


def units(p_model: float, p_market: float, american: int) -> tuple[int, float]:
    """Whole units (0 = no pick) and the sizing probability: quarter Kelly at the probability halfway between market and
    model, on a 100-unit reference bankroll, limited to 1-3."""
    if p_model - implied(american) < MIN_EDGE:
        return 0, p_market
    p = p_market + SHRINK * (p_model - p_market)
    b = payout(american)
    f = (p * b - (1 - p)) / b
    if f <= 0:
        return 0, p
    return int(min(max(round(KELLY * f * BANKROLL), 1), MAX_UNITS)), p


def best_books() -> pl.DataFrame:
    """Per game and capture: the best moneyline price for each side among the picks books (after the source-error guard)
    and which book has it."""
    o = pl.read_csv(LOGS / "odds_log.csv").filter(pl.col("market") == "moneyline")
    w = (
        o.with_columns(pl.col("american").map_elements(implied, return_dtype=pl.Float64).alias("q"))
        .pivot(on="side", index=["game_id", "captured_utc", "book"], values=["american", "q"])
        .drop_nulls()
    )
    w = w.with_columns((pl.col("q_home") / (pl.col("q_home") + pl.col("q_away"))).alias("m"))
    med = pl.col("m").filter(~pl.col("book").is_in(sc.SECOND_SOURCE)).median().over("game_id", "captured_utc")
    w = w.with_columns((pl.col("m") - med).abs().alias("_dev")).filter(
        (pl.col("_dev") <= sc.OUTLIER) & pl.col("book").is_in(sc.PICK_BOOKS)
    )
    h = (
        w.sort(["american_home", "book"], descending=[True, False])
        .unique(["game_id", "captured_utc"], keep="first", maintain_order=True)
        .select("game_id", "captured_utc", pl.col("book").alias("book_home"))
    )
    a = (
        w.sort(["american_away", "book"], descending=[True, False])
        .unique(["game_id", "captured_utc"], keep="first", maintain_order=True)
        .select("game_id", "captured_utc", pl.col("book").alias("book_away"))
    )
    return h.join(a, on=["game_id", "captured_utc"])


def _read() -> pl.DataFrame | None:
    return pl.read_csv(PICKS) if PICKS.exists() else None


def lock(day: str) -> list[dict]:
    """Lock picks for games on ``day`` that have not started and are not in the ledger, from the latest logged forecast
    and its prices."""
    d = sc.scored_frame().filter((pl.col("game_date") == day) & pl.col("mkt_home").is_not_null())
    have = set(_read()["game_id"].to_list()) if PICKS.exists() else set()
    bb = best_books()
    now = datetime.now(UTC)
    rows = []
    for r in d.join(bb, left_on=["game_id", "made_utc"], right_on=["game_id", "captured_utc"], how="left").iter_rows(
        named=True
    ):
        if r["game_id"] in have or datetime.fromisoformat(r["start_utc"].replace("Z", "+00:00")) <= now:
            continue
        cands = []
        for side, team, opp, p, mk, price, book in (
            ("home", r["home"], r["away"], r["p_home"], r["mkt_home"], r["pick_best_home"], r["book_home"]),
            ("away", r["away"], r["home"], r["p_away"], 1 - r["mkt_home"], r["pick_best_away"], r["book_away"]),
        ):
            if price is None:
                continue
            u, ps = units(p, mk, int(price))
            if u:
                cands.append(
                    {
                        "locked_utc": now.isoformat(timespec="seconds"),
                        "game_id": r["game_id"],
                        "game_date": r["game_date"],
                        "start_utc": r["start_utc"],
                        "away": r["away"],
                        "home": r["home"],
                        "pick": team,
                        "against": opp,
                        "side": side,
                        "book": book,
                        "american": int(price),
                        "units": u,
                        "p_model": round(p, 4),
                        "p_market": round(mk, 4),
                        "p_sizing": round(ps, 4),
                        "edge_vs_breakeven": round(p - implied(int(price)), 4),
                        "forecast_made_utc": r["made_utc"],
                    }
                )
        if cands:
            rows.append(max(cands, key=lambda c: c["edge_vs_breakeven"]))
    if rows:
        new = not PICKS.exists()
        with PICKS.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerows(rows)
    return rows


def settled() -> pl.DataFrame | None:
    """The ledger with results and profit (null while a game is unplayed)."""
    p = _read()
    if p is None:
        return None
    res = ROOT / "data/live/processed/games.parquet"
    r = (
        pl.read_parquet(res).select("game_id", "home_final", "away_final", "last_period_type")
        if res.exists()
        else pl.DataFrame(
            schema={"game_id": pl.Int64, "home_final": pl.Int64, "away_final": pl.Int64, "last_period_type": pl.Utf8}
        )
    )
    p = p.join(r, on="game_id", how="left")
    won = ((pl.col("side") == "home") & (pl.col("home_final") > pl.col("away_final"))) | (
        (pl.col("side") == "away") & (pl.col("away_final") > pl.col("home_final"))
    )
    pay = pl.when(pl.col("american") > 0).then(pl.col("american") / 100.0).otherwise(100.0 / -pl.col("american"))
    return (
        p.with_columns(
            pl.when(pl.col("home_final").is_null())
            .then(None)
            .when(won)
            .then(pl.lit("won"))
            .otherwise(pl.lit("lost"))
            .alias("result")
        )
        .with_columns(
            pl.when(pl.col("result") == "won")
            .then(pl.col("units") * pay)
            .when(pl.col("result") == "lost")
            .then(-pl.col("units").cast(pl.Float64))
            .otherwise(None)
            .alias("profit")
        )
        .sort("start_utc", "game_id")
    )


PUBLIC_RECORD = paths.PUBLIC / "picks_record.csv"
PUBLIC_COLS = [
    "locked_utc",
    "game_date",
    "start_utc",
    "away",
    "home",
    "pick",
    "against",
    "units",
    "decimal_odds",
    "book",
    "p_model",
    "away_final",
    "home_final",
    "last_period_type",
    "result",
    "profit",
]


def public_record(s: pl.DataFrame) -> pl.DataFrame:
    """The ledger as published: each pick in units at its decimal odds, the model's probability, and the result; the
    market probability and sizing inputs stay private."""
    paths.PUBLIC.mkdir(parents=True, exist_ok=True)
    return s.select(PUBLIC_COLS)


def summary(s: pl.DataFrame) -> dict:
    d = s.filter(pl.col("result").is_not_null())
    out = {
        "picks": s.height,
        "settled": d.height,
        "pending": s.height - d.height,
        "pending_units": int(s.filter(pl.col("result").is_null())["units"].sum()),
    }
    if d.height:
        pr, st = d["profit"].to_numpy(), d["units"].to_numpy().astype(float)
        rng = np.random.default_rng(0)
        idx = [rng.integers(0, len(pr), len(pr)) for _ in range(2000)] if len(pr) > 1 else []
        boots = [pr[i].sum() / st[i].sum() for i in idx]
        out.update(
            {
                "won": int((d["result"] == "won").sum()),
                "lost": int((d["result"] == "lost").sum()),
                "staked": float(st.sum()),
                "profit": float(pr.sum()),
                "roi": float(pr.sum() / st.sum()),
                "roi_lo95": float(np.quantile(boots, 0.025)) if boots else None,
                "roi_hi95": float(np.quantile(boots, 0.975)) if boots else None,
            }
        )
    return out


def decimal(american: int) -> float:
    """Decimal odds (total return per unit staked): -125 -> 1.80, +230 -> 3.30."""
    return 1.0 + payout(american)


def fmt(a: int) -> str:
    """Prices are shown as decimal odds; the logs keep the American price as captured."""
    return f"{decimal(a):.2f}"


def report(s: pl.DataFrame, m: dict, today: str) -> str:
    L = [
        "# Picks ledger",
        "",
        "Rules: [docs/market/picks.md](../../docs/market/picks.md). One unit = $1; 1–3 units per pick; decimal odds as locked at the daily run. **Bookkeeping for personal wagers, not advice; the model has no demonstrated edge over the market.**",
        "",
    ]
    if m.get("settled"):
        ci = f" (95% interval {m['roi_lo95']:+.0%} to {m['roi_hi95']:+.0%})" if m.get("roi_lo95") is not None else ""
        L += [
            f"**Season to date: {m['profit']:+.2f} units (${m['profit']:+.2f}) on {m['staked']:.0f} units staked; record {m['won']}–{m['lost']}; return {m['roi']:+.1%} per unit{ci}.** "
            f"{m['pending']} pick(s) pending ({m['pending_units']} units).",
            "",
            "![cumulative profit](picks_profit.png)",
            "",
        ]
    else:
        L += [f"No settled picks yet. {m['pending']} pick(s) pending ({m['pending_units']} units).", ""]
    pend = s.filter(pl.col("result").is_null())
    if pend.height:
        L += [
            "## Locked, not yet played",
            "",
            "| date | pick | units | odds (decimal) | book | model |",
            "|---|---|---|---|---|---|",
        ]
        L += [
            f"| {r['game_date']} | **{r['pick']}** over {r['against']} | {r['units']} (${r['units']}) | {fmt(r['american'])} | {r['book']} | {r['p_model']:.1%} |"
            for r in pend.iter_rows(named=True)
        ]
        L.append("")
    done = s.filter(pl.col("result").is_not_null()).sort("start_utc", descending=True)
    if done.height:
        by = (
            done.group_by("game_date")
            .agg(
                pl.len().alias("picks"),
                pl.col("units").sum(),
                pl.col("profit").sum(),
                (pl.col("result") == "won").sum().alias("won"),
            )
            .sort("game_date", descending=True)
        )
        L += ["## By day", "", "| date | picks | won | units staked | profit |", "|---|---|---|---|---|"]
        L += [
            f"| {r['game_date']} | {r['picks']} | {r['won']} | {r['units']} | {r['profit']:+.2f} |"
            for r in by.head(30).iter_rows(named=True)
        ]
        L += [
            "",
            "## Settled picks (latest first)",
            "",
            "| date | pick | units | odds (decimal) | book | score (away–home) | result | profit |",
            "|---|---|---|---|---|---|---|---|",
        ]
        L += [
            f"| {r['game_date']} | {r['pick']} over {r['against']} | {r['units']} | {fmt(r['american'])} | {r['book']} | {r['away']} {r['away_final']}–{r['home_final']} {r['home']} ({r['last_period_type']}) | {r['result']} | {r['profit']:+.2f} |"
            for r in done.head(60).iter_rows(named=True)
        ]
        L.append("")
    return "\n".join(L)


def chart(s: pl.DataFrame) -> None:
    """Cumulative net profit by date (x: game date; y: units)."""
    from datetime import date as _date

    d = (
        s.filter(pl.col("result").is_not_null())
        .group_by("game_date")
        .agg(pl.col("profit").sum(), pl.col("units").sum(), pl.len().alias("n"))
        .sort("game_date")
    )
    if d.height == 0:
        return
    days = [_date.fromisoformat(x) for x in d["game_date"].to_list()]
    cum = np.cumsum(d["profit"].to_numpy())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(0, color=style.TEXT2, lw=1)
    ax.plot(days, cum, color=style.SERIES[0], marker="o", markersize=5)
    ax.fill_between(days, 0, cum, where=cum >= 0, color=style.SERIES[2], alpha=0.15, interpolate=True)
    ax.fill_between(days, 0, cum, where=cum < 0, color=style.SERIES[7], alpha=0.15, interpolate=True)
    ax.annotate(
        f"{cum[-1]:+.2f} units",
        (days[-1], cum[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontweight="bold",
        color=style.SERIES[0],
    )
    ax.set_xlabel("date")
    ax.set_ylabel("net profit (units; 1 unit = $1)")
    ax.margins(x=0.12)
    fig.autofmt_xdate()
    ax.set_title(
        f"Picks ledger: cumulative net profit through {d['game_date'].max()} ({int(d['n'].sum())} settled picks, {int(d['units'].sum())} units staked)"
    )
    fig.tight_layout()
    style.save(fig, paths.PUBLIC_REPORTS, "picks_profit")


def run(day: str | None = None) -> dict:
    day = day or date.today().isoformat()
    new = lock(day)
    s = settled()
    if s is None:
        log.info("no picks locked yet")
        return {"locked_now": new}
    m = summary(s)
    chart(s)
    paths.PUBLIC_REPORTS.mkdir(parents=True, exist_ok=True)
    (paths.PUBLIC_REPORTS / "picks.md").write_text(report(s, m, day), encoding="utf-8")
    full = s.with_columns(
        (1.0 + pl.when(pl.col("american") > 0).then(pl.col("american") / 100.0).otherwise(100.0 / -pl.col("american")))
        .round(3)
        .alias("decimal_odds")
    )
    full.write_csv(LOGS / "picks_settled.csv")
    public_record(full).write_csv(PUBLIC_RECORD)
    log.info(
        "picks: %d locked now; %d settled, %d pending; profit %s units",
        len(new),
        m.get("settled", 0),
        m["pending"],
        f"{m['profit']:+.2f}" if m.get("settled") else "n/a",
    )
    return {"locked_now": new, **m}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(sys.argv[1] if len(sys.argv) > 1 else None)
