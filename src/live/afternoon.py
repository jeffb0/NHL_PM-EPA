"""Afternoon goalie update:  python -m src.live.afternoon [YYYY-MM-DD]   (docs/market/methodology.md §8 item 11)

For today's games whose starting goalie was not confirmed at the morning run and that have not started: fetches the lineup feed and prices again, re-forecasts, appends to the forecast and odds logs
(so the market test scores the later forecast), and writes reports/live/update_digest.md and .html for the afternoon email. It does not refresh game data, does not lock or change picks, and does not touch the morning digest.
When no game qualifies the digest's first line is "NO UPDATE" and no email is sent."""

from __future__ import annotations

import html
import logging
import sys
from datetime import UTC, date, datetime

import polars as pl

from src.config import ROOT
from src.live import daily as dl
from src.live import digest as dg
from src.live import odds as od
from src.live import picks as pk
from src.live import predict as lp
from src.live import score as sc
from src.live.odds import implied

log = logging.getLogger("live")
OUT, OUT_HTML = ROOT / "reports/live/update_digest.md", ROOT / "reports/live/update_digest.html"


def morning(day: str) -> pl.DataFrame:
    """The first forecast logged today for each of today's games."""
    f = pl.read_csv(dl.LOGS / "forecast_log.csv", infer_schema_length=None)
    return f.filter((pl.col("game_date") == day) & pl.col("made_utc").str.starts_with(day)).sort("made_utc").unique("game_id", keep="first", maintain_order=True)


def involved(m: pl.DataFrame, now: datetime) -> list[dict]:
    """Morning rows with an unconfirmed starter on either side, for games that have not started."""
    return [r for r in m.iter_rows(named=True) if dg.unconfirmed(r) and datetime.fromisoformat(r["start_utc"].replace("Z", "+00:00")) > now]


def pick_status(pick: dict | None, r: dict) -> str:
    """How a locked pick (or the lack of one) looks against the afternoon forecast ``r`` (a scored_frame row)."""
    if r["mkt_home"] is None:
        return "No prices captured this afternoon."
    if pick is None:
        q = [(team, p, int(a), p - implied(int(a))) for team, p, a in ((r["home"], r["p_home"], r["pick_best_home"]), (r["away"], r["p_away"], r["pick_best_away"])) if a is not None]
        q = [x for x in q if x[3] >= pk.MIN_EDGE]
        if not q:
            return "No pick this morning, and none qualifies now."
        team, _, price, edge = max(q, key=lambda x: x[3])
        return f"No pick this morning. {team} qualifies now at {pk.fmt(price)} (edge {edge:+.1%}); it is not in the ledger."
    home = pick["side"] == "home"
    p, price = (r["p_home"], r["pick_best_home"]) if home else (r["p_away"], r["pick_best_away"])
    e_locked, e_now = p - implied(int(pick["american"])), p - implied(int(price))
    verdict = "still qualifies" if e_locked >= pk.MIN_EDGE else "NO LONGER QUALIFIES"
    return (f"Morning pick {pick['pick']} over {pick['against']}, {pick['units']} unit{'s' if pick['units'] != 1 else ''} at {pk.fmt(int(pick['american']))}: {verdict}. "
            f"Model now {p:.1%} (was {pick['p_model']:.1%}); edge at the locked price {e_locked:+.1%}, at the best price now ({pk.fmt(int(price))}) {e_now:+.1%}.")


def build(day: str, rows: list[dict], now_rows: dict, picks: dict) -> tuple[str, str]:
    if not rows:
        return "NO UPDATE\n\nEvery starting goalie for today's remaining games was confirmed at the morning run, or no games remain.\n", ""
    conf = sum(1 for r in rows if not dg.unconfirmed(now_rows[r["game_id"]]))
    subject = f"NHL model {day} goalie update: {conf} of {len(rows)} games now confirmed"
    T = [f"Subject: {subject}", "", f"GAMES WITH AN UNCONFIRMED STARTER THIS MORNING ({day})", ""]
    H = [f'<div style="font-family:Arial,Helvetica,sans-serif;color:{dg.INK};max-width:680px">', f'<h2 style="font-size:18px;margin:0 0 10px 0">{html.escape(subject)}</h2>']
    tab = []
    for r in rows:
        n = now_rows[r["game_id"]]
        gk = lambda x, s: x[f"{s}_goalie"] + ("" if (x.get(f"{s}_goalie_source") or "") == dg.CONFIRMED else "*")        # noqa: E731
        changed = [f"{n[s]}: {r[f'{s}_goalie']} -> {n[f'{s}_goalie']}" for s in ("away", "home") if r[f"{s}_goalie"] != n[f"{s}_goalie"]]
        still = [t for t, _, _ in dg.unconfirmed(n)]
        mk = f"{1 - n['mkt_home']:.0%}/{n['mkt_home']:.0%}" if n["mkt_home"] is not None else "no prices"
        st = pick_status(picks.get(r["game_id"]), n)
        T += [f"- {r['away']} at {r['home']}", f"    Goalies now: {gk(n, 'away')} / {gk(n, 'home')}" + (f"  (changed: {'; '.join(changed)})" if changed else "  (same names as this morning)")
              + (f"; still not confirmed: {', '.join(still)}" if still else "; both confirmed"),
              f"    Model away/home: {r['p_away']:.0%}/{r['p_home']:.0%} this morning, {n['p_away']:.0%}/{n['p_home']:.0%} now; market now {mk}", f"    {st}"]
        col = dg.RED if "NO LONGER" in st else dg.INK
        tab.append([f"{r['away']} at {r['home']}", html.escape(f"{gk(n, 'away')} / {gk(n, 'home')}") + (f'<br><font color={dg.MUTED}>changed: {html.escape("; ".join(changed))}</font>' if changed else ""),
                    f"{r['p_away']:.0%} / {r['p_home']:.0%}", f"{n['p_away']:.0%} / {n['p_home']:.0%}", mk.replace("/", " / "), f'<span style="color:{col}">{html.escape(st)}</span>'])
    H.append(dg._table(["game", "goalies now, away / home", "model this morning", "model now", "market now", "pick"], tab))
    foot = ("* = still not confirmed. Picks stay as locked this morning; this update only says whether each still clears the three-point edge with the afternoon's information. "
            "The model has no demonstrated edge over the market yet. This is record-keeping of your own picks, not betting advice. Prices are decimal odds as captured at this run and may have moved.")
    T += ["", foot, ""]
    H.append(f'<p style="font-size:11px;color:{dg.MUTED};margin:18px 0 0 0;border-top:1px solid #d8d7d2;padding-top:8px">{html.escape(foot)}</p></div>')
    return "\n".join(T), "\n".join(H)


def run(day: str | None = None, season: int = 2026) -> str:
    day = day or date.today().isoformat()
    now = datetime.now(UTC)
    rows = involved(morning(day), now)
    now_rows: dict = {}
    if rows:
        ctx = lp.prepare(season)
        feed = dl.lineup_feed(ctx, (False,))
        stamp = now.isoformat(timespec="seconds")
        f_rows, o_rows = [], []
        extra = od.fetch_extra(day)
        for r in rows:
            g = {k: r[k] for k in ("game_id", "game_date", "start_utc", "away", "home")}
            f_rows.append(dl.forecast_row(ctx, g, feed.get((r["away"], r["home"])), None, stamp))
            prices, url = od.fetch(r["away"], r["home"])
            o_rows += [{"captured_utc": stamp, **{k: g[k] for k in ("game_id", "game_date", "away", "home")}, **p, "source": url} for p in prices]
            o_rows += dl.extra_rows(extra, g, stamp)
        dl._append(dl.LOGS / "forecast_log.csv", dl.F_COLS, f_rows)
        dl._append(dl.LOGS / "odds_log.csv", dl.O_COLS, o_rows)
        now_rows = {r["game_id"]: r for r in sc.scored_frame().filter(pl.col("made_utc") == stamp).iter_rows(named=True)}
        rows = [r for r in rows if r["game_id"] in now_rows]
    led = pk._read()
    picks = {r["game_id"]: r for r in led.filter(pl.col("game_date") == day).iter_rows(named=True)} if led is not None else {}
    text, page = build(day, rows, now_rows, picks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    OUT_HTML.write_text(page, encoding="utf-8")
    log.info("afternoon update: %s", text.splitlines()[0])
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))
