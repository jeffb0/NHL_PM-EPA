"""Daily summary for email:  python -m src.live.digest [YYYY-MM-DD]

Builds reports/live/daily_digest.md (first line is the subject) from the logs: yesterday's settled picks, the season-to-date ledger, today's locked picks and today's forecasts against the market.
Run after src.live.daily (which calls it). Prices are decimal odds."""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta

import polars as pl

from src.config import ROOT
from src.live import picks as pk
from src.live import score as sc

log = logging.getLogger("live")
OUT = ROOT / "reports/live/daily_digest.md"


def build(day: str | None = None) -> str:
    day = day or date.today().isoformat()
    yday = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    s = pk.settled()
    m = pk.summary(s) if s is not None else {"picks": 0, "settled": 0, "pending": 0}
    d = sc.scored_frame()
    ms = sc.summarize(d)
    today_picks = s.filter((pl.col("game_date") == day) & pl.col("result").is_null()) if s is not None else None
    n_today = today_picks.height if today_picks is not None else 0
    season = f"{m['profit']:+.2f} units" if m.get("settled") else "no settled picks yet"
    L = [f"Subject: NHL model {day}: {n_today} pick{'s' if n_today != 1 else ''} today; season {season}", ""]

    L += [f"YESTERDAY'S PICKS ({yday})", ""]
    y = s.filter((pl.col("game_date") == yday) & pl.col("result").is_not_null()) if s is not None else None
    if y is not None and y.height:
        for r in y.iter_rows(named=True):
            L.append(f"- {r['pick']} over {r['against']}: {r['units']} unit{'s' if r['units'] != 1 else ''} at {pk.fmt(r['american'])} ({r['book']}) - {r['result'].upper()} {r['profit']:+.2f}  [{r['away']} {r['away_final']}, {r['home']} {r['home_final']}, {r['last_period_type']}]")
        L.append(f"  Day: {int((y['result'] == 'won').sum())}-{int((y['result'] == 'lost').sum())}, {y['profit'].sum():+.2f} units on {int(y['units'].sum())} staked.")
    else:
        L.append("- No picks settled from yesterday.")
    L += ["", "SEASON TO DATE", ""]
    if m.get("settled"):
        ci = f" (95% interval {m['roi_lo95']:+.0%} to {m['roi_hi95']:+.0%})" if m.get("roi_lo95") is not None else ""
        L.append(f"- Profit {m['profit']:+.2f} units (${m['profit']:+.2f}) on {m['staked']:.0f} units staked; record {m['won']}-{m['lost']}; return {m['roi']:+.1%} per unit{ci}.")
    else:
        L.append("- No settled picks yet.")
    if ms.get("scored"):
        z = f", z = {ms['z']:.2f}" if ms.get("z") is not None else ""
        L.append(f"- Model against market on {ms['scored']} scored games: log loss {ms['logloss_model']:.4f} against {ms['logloss_market']:.4f}{z}. "
                 + ("Too few games for any conclusion (300 needed)." if ms["scored"] < sc.MIN_GAMES else ""))

    L += ["", f"TODAY'S PICKS ({day})", ""]
    if n_today:
        for r in today_picks.iter_rows(named=True):
            L.append(f"- {r['pick']} over {r['against']}: {r['units']} unit{'s' if r['units'] != 1 else ''} (${r['units']}) at {pk.fmt(r['american'])} ({r['book']}); model {r['p_model']:.0%}, market {r['p_market']:.0%}")
        L.append(f"  Total at stake today: {int(today_picks['units'].sum())} units.")
    else:
        L.append("- No pick qualified today.")

    L += ["", f"TODAY'S FORECASTS ({day})", "", "away at home | model away/home | market away/home | expected score | goalies (away / home)"]
    g = d.filter((pl.col("game_date") == day) & pl.col("home_win").is_null()).sort("start_utc")
    flags = []
    for r in g.iter_rows(named=True):
        mk = f"{1 - r['mkt_home']:.0%}/{r['mkt_home']:.0%}" if r["mkt_home"] is not None else "no prices"
        L.append(f"- {r['away']} at {r['home']} | {r['p_away']:.0%}/{r['p_home']:.0%} | {mk} | {r['exp_goals_away']:.1f}-{r['exp_goals_home']:.1f} | {r['away_goalie']} / {r['home_goalie']}")
        if r["mkt_home"] is not None and abs(r["p_home"] - r["mkt_home"]) > 0.08:
            flags.append(f"{r['away']} at {r['home']}: model and market differ by {abs(r['p_home'] - r['mkt_home']):.0%} - check for roster news the model may be missing.")
        for side in ("away", "home"):
            src = r.get(f"{side}_goalie_source") or ""
            if src.startswith(("back-to-back", "override")):
                flags.append(f"{r[side]}: {r[f'{side}_goalie']} in goal by {src} - correct it in roster_overrides.json if wrong.")
    if g.height == 0:
        L.append("- No games remaining today in the log.")
    if flags:
        L += ["", "CHECK", ""] + [f"- {x}" for x in flags]
    L += ["", "The model has no demonstrated edge over the market yet. This is record-keeping of your own picks, not betting advice. Prices are decimal odds as captured at the daily run and may have moved.", ""]
    return "\n".join(L)


def run(day: str | None = None) -> str:
    text = build(day)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    log.info("digest written: %s", text.splitlines()[0])
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))
