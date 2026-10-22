"""Daily summary for email:  python -m src.live.digest [YYYY-MM-DD]

Builds data/private/reports/daily_digest.md (plain text; first line is the subject) and daily_digest.html (the same with charts drawn as HTML tables, so no image attachment is needed):
yesterday's settled picks with how each game went on expected goals, the season ledger with a cumulative-profit chart by date, today's locked picks and forecasts, and for the owner's teams (Jets, Senators)
a stacked-bar chart of each skater's 5v5 impact in yesterday's game. Run after src.live.daily (which calls it). Prices are decimal odds."""

from __future__ import annotations

import html
import logging
import sys
from datetime import date, timedelta

import polars as pl

from src.config import ROOT
from src.live import paths
from src.live import picks as pk
from src.live import score as sc
from src.live.odds import implied
from src.scoreeffects import factors as sf

log = logging.getLogger("live")
OUT = paths.PRIVATE_REPORTS / "daily_digest.md"
OUT_HTML = paths.PRIVATE_REPORTS / "daily_digest.html"
LIVE = ROOT / "data/live/processed"
FOCUS_TEAMS = ("WPG", "OTT")            # teams whose games get the per-player chart
BLUE, ORANGE, GREY, RED, GREEN, INK, MUTED = "#2a78d6", "#eb6834", "#9a9994", "#c9403f", "#1a8f63", "#0b0b0b", "#52514e"
OWN, MATES, AGAINST = "#2a78d6", "#1baf7a", "#eb6834"   # stacked-bar components: blue, aqua, orange (the reference palette's first three slots, colour-blind-safe in every pairing)
BLOCK = "\u2588"            # full block: the unit of the text bars
VERDICT = {(True, True): ("deserved", "won and had the better of the chances"), (True, False): ("fortunate", "won despite being out-chanced"),
           (False, True): ("unlucky", "lost despite having the better of the chances"), (False, False): ("outplayed", "lost and was out-chanced")}


def game_xg() -> pl.DataFrame:
    """Per completed game and team: expected goals (all situations) raw and score-adjusted."""
    ev = pl.read_parquet(LIVE / "events.parquet", columns=["game_id", "event_idx", "period", "score_diff"])
    sh = pl.read_parquet(LIVE / "shots_xg.parquet").filter(pl.col("type").is_in(["shot-on-goal", "missed-shot", "goal"])).join(ev, on=["game_id", "event_idx"])
    wp = ROOT / "data/outputs/score_effect_weights.csv"
    sh = sf.apply(sh, pl.read_csv(wp)) if wp.exists() else sh.with_columns(pl.col("xg").alias("xg_adj"))
    x = sh.group_by("game_id", "team_id").agg(pl.col("xg").sum(), pl.col("xg_adj").sum())
    g = pl.read_parquet(LIVE / "games.parquet").select("game_id", "game_date", "home_id", "away_id", "home_abbrev", "away_abbrev")
    return (g.join(x.rename({"team_id": "home_id", "xg": "xg_home", "xg_adj": "xga_home"}), on=["game_id", "home_id"]).join(x.rename({"team_id": "away_id", "xg": "xg_away", "xg_adj": "xga_away"}), on=["game_id", "away_id"]))


def with_xg(s: pl.DataFrame) -> pl.DataFrame:
    """Settled picks with the pick's and the opponent's expected goals and the verdict (on score-adjusted expected goals)."""
    j = s.join(game_xg().select("game_id", "xg_home", "xg_away", "xga_home", "xga_away"), on="game_id", how="left")
    home = pl.col("side") == "home"
    j = j.with_columns(pl.when(home).then(pl.col("xg_home")).otherwise(pl.col("xg_away")).alias("xg_pick"), pl.when(home).then(pl.col("xg_away")).otherwise(pl.col("xg_home")).alias("xg_opp"),
                       pl.when(home).then(pl.col("xga_home")).otherwise(pl.col("xga_away")).alias("xga_pick"), pl.when(home).then(pl.col("xga_away")).otherwise(pl.col("xga_home")).alias("xga_opp"))
    tag = [None if r["result"] is None or r["xga_pick"] is None else VERDICT[(r["result"] == "won", r["xga_pick"] > r["xga_opp"])][0] for r in j.iter_rows(named=True)]
    return j.with_columns(pl.Series("verdict", tag, dtype=pl.Utf8))


CONFIRMED = "lineup feed (confirmed)"


def unconfirmed(r: dict) -> list[tuple[str, str, str | None]]:
    """(team, listed goalie, other goalie) for each side of a logged forecast whose starter was not confirmed at that run."""
    return [(r[s], r[f"{s}_goalie"], r.get(f"{s}_goalie_alt") or None) for s in ("away", "home") if (r.get(f"{s}_goalie_source") or "") != CONFIRMED]


def goalie_note(r: dict, side: str, american: int) -> tuple[str, bool]:
    """For a pick on ``side`` of logged forecast ``r`` at price ``american``: a sentence on how the edge depends on unconfirmed starters, and whether the pick is goalie-dependent
    (its edge over break-even would fall below the picks threshold if the other goalie started)."""
    un = unconfirmed(r)
    if not un:
        return "both starting goalies confirmed", False
    be = implied(american)
    p = r["p_home"] if side == "home" else r["p_away"]
    lo, hi = r.get("p_home_min"), r.get("p_home_max")
    who = "; ".join(f"{g} for {t}" + (f", or {a}" if a else "") for t, g, a in un)
    if lo is None or hi is None:
        return f"starter not confirmed ({who})", False
    worst = (lo if side == "home" else 1 - hi) - be
    if worst >= pk.MIN_EDGE:
        return f"holds whichever goalie starts (edge at least {worst:+.1%}); not confirmed: {who}", False
    return f"GOALIE-DEPENDENT: edge {p - be:+.1%} with the listed starters, {worst:+.1%} at worst if the other goalie starts; not confirmed: {who}", True


def rel(p) -> str:
    """A path relative to the repository (no user name or machine folder in the email)."""
    from pathlib import Path
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).name


def cumulative(s: pl.DataFrame) -> pl.DataFrame:
    """Net profit by game date and its running total."""
    d = s.filter(pl.col("result").is_not_null()).group_by("game_date").agg(pl.col("profit").sum(), pl.col("units").sum(), pl.len().alias("picks")).sort("game_date")
    return d.with_columns(pl.col("profit").cum_sum().alias("cum"))


# ---------------------------------------------------------------- HTML pieces (tables only: email clients drop images that are not attached, scripts and SVG)
def _bar(width: float, colour: str, height: int = 11) -> str:
    """One coloured cell; attributes without quotes keep the email small (the whole page is passed through the mail tool as text)."""
    w = max(int(round(width)), 0)
    return f"<td width={w} height={height} bgcolor={colour}></td>" if w > 0 else ""


def html_profit_chart(cum: pl.DataFrame, max_days: int = 45) -> str:
    """Column chart of cumulative net profit by date (green above zero, red below), drawn with table cells and kept small: the page is passed through the mail tool as text."""
    c = cum.tail(max_days)
    if c.height == 0:
        return "<p>No settled picks yet.</p>"
    top, bot = max(float(c["cum"].max()), 0.0), max(-float(c["cum"].min()), 0.0)
    scale = 100.0 / max(top + bot, 0.5)
    up_h, dn_h = max(int(top * scale), 1), max(int(bot * scale), 1)
    w = max(min(int(520 / c.height), 40), 11)
    every = max(1, -(-c.height // 12))                                # label about a dozen columns at most
    up, dn, lab, val = [], [], [], []
    for k, r in enumerate(c.iter_rows(named=True)):
        h = max(int(abs(r["cum"]) * scale), 2)
        blk = lambda col, h=h: f"<table cellpadding=0 cellspacing=0 align=center><tr><td width={w - 4} height={h} bgcolor={col}></td></tr></table>"  # noqa: E731
        show = k % every == 0 or k == c.height - 1
        up.append(f"<td width={w} height={up_h} valign=bottom>{blk(GREEN) if r['cum'] >= 0 else ''}</td>")
        dn.append(f"<td height={dn_h} valign=top>{blk(RED) if r['cum'] < 0 else ''}</td>")
        lab.append(f"<td align=center>{r['game_date'][5:] if show else ''}</td>")
        val.append(f"<td align=center>{format(r['cum'], '+.1f') if show else ''}</td>")
    nl = "\n"
    note = f" (last {max_days} days)" if cum.height > max_days else ""
    return (f'<p style="margin:4px 0;font-size:12px;color:{MUTED}">Cumulative net profit in units (1 unit = $1) by date{note}</p>\n'
            f'<table cellpadding=0 cellspacing=0 style="border-collapse:collapse;font-size:10px">\n<tr>{nl.join(val)}</tr>\n<tr>{nl.join(up)}</tr>\n'
            f"<tr><td colspan={c.height} height=1 bgcolor={MUTED}></td></tr>\n<tr>{nl.join(dn)}</tr>\n<tr>{nl.join(lab)}</tr>\n</table>")


def html_impact(players: pl.DataFrame, title: str, width: int = 16) -> str:
    """Stacked horizontal bars per skater, drawn with block characters in a monospace block (small enough to pass through the mail tool as text):
    expected goals against at 5v5 to the left of the centre line (orange), for to the right (his own shots blue, linemates' aqua), net at the end; sorted by net within team."""
    lim = max(float(players["on_xgf_5v5"].max()), float(players["on_xga_5v5"].max()), 0.1)
    per = width / lim
    teams = players["team"].unique(maintain_order=True).to_list()
    out = [f'<p style="margin:14px 0 2px 0"><b>{html.escape(title)}</b></p>',
           f'<p style="margin:0 0 4px 0;font-size:11px;color:{MUTED}">Expected goals at 5v5 while on the ice: <b><font color={AGAINST}>{BLOCK * 2} against</font></b> (left of the line), <b><font color={OWN}>{BLOCK * 2} his own shots</font></b>, <b><font color={MATES}>{BLOCK * 2} linemates\' shots</font></b>; number = net for minus against. Sorted best to worst.</p>',
           '<pre style="font-family:Consolas,Menlo,monospace;font-size:12px;line-height:14px;margin:0">']
    for team in teams:
        t = players.filter(pl.col("team") == team).sort("on_xgd_5v5", descending=True)
        out.append(f"<b>{html.escape(team)}</b>")
        for r in t.iter_rows(named=True):
            a = min(int(round(r["on_xga_5v5"] * per)), width)
            own = min(int(round(r["own_xg_5v5"] * per)), width)
            tot = min(max(int(round(r["on_xgf_5v5"] * per)), own), width)
            name = f"{r['name']} ({r['position']})"[:21].ljust(21)
            out.append(f"{html.escape(name)} {' ' * (width - a)}<font color={AGAINST}>{BLOCK * a}</font>|<font color={OWN}>{BLOCK * own}</font><font color={MATES}>{BLOCK * (tot - own)}</font>{' ' * (width - tot)} <b>{r['on_xgd_5v5']:+.2f}</b>")
        out.append("")
    out.append("</pre>")
    return "\n".join(out)


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th align=left><font color={MUTED}>{h}</font></th>" for h in head)
    tr = "\n".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table cellpadding=4 cellspacing=0 style="border-collapse:collapse;font-size:13px"><tr>{th}</tr>{tr}</table>'


def focus_games(yday: str) -> list[dict]:
    """Yesterday's completed games involving the owner's teams, with the skater table from the game report (which is generated here)."""
    from src.live import game_report as gr
    g = pl.read_parquet(LIVE / "games.parquet").filter((pl.col("game_date") == yday) & (pl.col("home_abbrev").is_in(FOCUS_TEAMS) | pl.col("away_abbrev").is_in(FOCUS_TEAMS)))
    out = []
    for r in g.iter_rows(named=True):
        try:
            rep = gr.run(int(r["game_id"]))
        except Exception as exc:        # a report failure must not stop the daily summary
            log.warning("game report failed for %s: %s", r["game_id"], exc)
            continue
        ts = {x["team"]: x for x in rep["teams"].to_dicts()}
        a, h = ts[r["away_abbrev"]], ts[r["home_abbrev"]]
        out.append({"title": f"{r['away_abbrev']} {r['away_final']} at {r['home_abbrev']} {r['home_final']} ({r['last_period_type']})", "players": rep["players"], "dir": rep["dir"],
                    "line": f"Expected goals {a['team']} {a['xg']:.2f}, {h['team']} {h['xg']:.2f} (score-adjusted {a['xg_score_adj']:.2f} to {h['xg_score_adj']:.2f}; 5v5 {a['xg_5v5']:.2f} to {h['xg_5v5']:.2f})."})
    return out


def build(day: str | None = None) -> tuple[str, str]:
    day = day or date.today().isoformat()
    yday = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    s = pk.settled()
    m = pk.summary(s) if s is not None else {"picks": 0, "settled": 0, "pending": 0}
    d = sc.scored_frame()
    ms = sc.summarize(d)
    sx = with_xg(s) if s is not None else None
    today_picks = s.filter((pl.col("game_date") == day) & pl.col("result").is_null()) if s is not None else None
    n_today = today_picks.height if today_picks is not None else 0
    season = f"{m['profit']:+.2f} units" if m.get("settled") else "no settled picks yet"
    subject = f"NHL model {day}: {n_today} pick{'s' if n_today != 1 else ''} today; season {season}"
    T = [f"Subject: {subject}", ""]
    H = [f'<div style="font-family:Arial,Helvetica,sans-serif;color:{INK};max-width:680px">', f'<h2 style="font-size:18px;margin:0 0 10px 0">{html.escape(subject)}</h2>']

    # ---- yesterday's picks, with how the game went on expected goals
    T += [f"YESTERDAY'S PICKS ({yday})", ""]
    H.append(f'<h3 style="font-size:15px;margin:16px 0 6px 0">Yesterday\'s picks ({yday})</h3>')
    y = sx.filter((pl.col("game_date") == yday) & pl.col("result").is_not_null()) if sx is not None else None
    if y is not None and y.height:
        rows = []
        for r in y.iter_rows(named=True):
            v = VERDICT[(r["result"] == "won", r["xga_pick"] > r["xga_opp"])] if r["xga_pick"] is not None else ("", "no expected-goal data")
            xg = f"xG {r['pick']} {r['xg_pick']:.2f}, {r['against']} {r['xg_opp']:.2f} (score-adjusted {r['xga_pick']:.2f} to {r['xga_opp']:.2f})" if r["xg_pick"] is not None else ""
            T.append(f"- {r['pick']} over {r['against']}: {r['units']} unit{'s' if r['units'] != 1 else ''} at {pk.fmt(r['american'])} ({r['book']}) - {r['result'].upper()} {r['profit']:+.2f}  [{r['away']} {r['away_final']}, {r['home']} {r['home_final']}, {r['last_period_type']}]")
            T.append(f"    {xg}: {v[1]}.")
            col = GREEN if r["result"] == "won" else RED
            rows.append([f"<b>{r['pick']}</b> over {r['against']}", f"{r['units']} at {pk.fmt(r['american'])}", f"{r['away']} {r['away_final']}, {r['home']} {r['home_final']} ({r['last_period_type']})",
                         f'<b style="color:{col}">{r["result"]} {r["profit"]:+.2f}</b>', f"{r['xga_pick']:.2f} to {r['xga_opp']:.2f}" if r["xga_pick"] is not None else "", v[1]])
        T.append(f"  Day: {int((y['result'] == 'won').sum())}-{int((y['result'] == 'lost').sum())}, {y['profit'].sum():+.2f} units on {int(y['units'].sum())} staked.")
        H.append(_table(["pick", "units at odds", "score", "result", "xG, pick to opponent (score-adjusted)", "on the chances"], rows))
        H.append(f'<p style="font-size:13px;margin:6px 0">Day: {int((y["result"] == "won").sum())}-{int((y["result"] == "lost").sum())}, <b>{y["profit"].sum():+.2f} units</b> on {int(y["units"].sum())} staked.</p>')
    else:
        T.append("- No picks settled from yesterday.")
        H.append('<p style="font-size:13px">No picks settled from yesterday.</p>')

    # ---- season to date
    T += ["", "SEASON TO DATE", ""]
    H.append('<h3 style="font-size:15px;margin:16px 0 6px 0">Season to date</h3>')
    if m.get("settled"):
        ci = f" (95% interval {m['roi_lo95']:+.0%} to {m['roi_hi95']:+.0%})" if m.get("roi_lo95") is not None else ""
        line = f"Profit {m['profit']:+.2f} units (${m['profit']:+.2f}) on {m['staked']:.0f} units staked; record {m['won']}-{m['lost']}; return {m['roi']:+.1%} per unit{ci}."
        vc = sx.filter(pl.col("verdict").is_not_null()).group_by("verdict").len()
        cnt = {r["verdict"]: r["len"] for r in vc.iter_rows(named=True)}
        luck = (f"On the chances (score-adjusted expected goals): {cnt.get('deserved', 0)} won with the better chances, {cnt.get('fortunate', 0)} won while out-chanced, "
                f"{cnt.get('unlucky', 0)} lost with the better chances, {cnt.get('outplayed', 0)} lost while out-chanced.")
        T += [f"- {line}", f"- {luck}"]
        H += [f'<p style="font-size:13px;margin:4px 0"><b>{html.escape(line)}</b></p>', f'<p style="font-size:13px;margin:4px 0">{html.escape(luck)}</p>', html_profit_chart(cumulative(s))]
        cum = cumulative(s)
        T.append("- Cumulative profit by date: " + ", ".join(f"{r['game_date'][5:]} {r['cum']:+.2f}" for r in cum.tail(14).iter_rows(named=True)))
    else:
        T.append("- No settled picks yet.")
        H.append('<p style="font-size:13px">No settled picks yet.</p>')
    if ms.get("scored"):
        z = f", z = {ms['z']:.2f}" if ms.get("z") is not None else ""
        mk = (f"Model against market on {ms['scored']} scored games: log loss {ms['logloss_model']:.4f} against {ms['logloss_market']:.4f}{z}. "
              + ("Too few games for any conclusion (300 needed)." if ms["scored"] < sc.MIN_GAMES else ""))
        T.append(f"- {mk}")
        H.append(f'<p style="font-size:12px;color:{MUTED};margin:6px 0">{html.escape(mk)}</p>')

    # ---- today's picks and forecasts
    T += ["", f"TODAY'S PICKS ({day})", ""]
    H.append(f'<h3 style="font-size:15px;margin:16px 0 6px 0">Today\'s picks ({day})</h3>')
    if n_today:
        rows, logged, dep = [], {r["game_id"]: r for r in d.filter(pl.col("game_date") == day).iter_rows(named=True)}, 0
        for r in today_picks.iter_rows(named=True):
            note, flagged = goalie_note(logged[r["game_id"]], r["side"], r["american"]) if r["game_id"] in logged else ("", False)
            dep += flagged
            T.append(f"- {r['pick']} over {r['against']}: {r['units']} unit{'s' if r['units'] != 1 else ''} (${r['units']}) at {pk.fmt(r['american'])} ({r['book']}); model {r['p_model']:.0%}, market {r['p_market']:.0%}")
            T.append(f"    Goalies: {note}.")
            cell = f'<b style="color:{RED}">{html.escape(note)}</b>' if flagged else html.escape(note)
            rows.append([f"<b>{r['pick']}</b> over {r['against']}", f"{r['units']} (${r['units']})", pk.fmt(r["american"]), r["book"], f"{r['p_model']:.0%}", f"{r['p_market']:.0%}", cell])
        tot = f"Total at stake today: {int(today_picks['units'].sum())} units." + (f" {dep} pick{'s' if dep != 1 else ''} depend{'' if dep != 1 else 's'} on an unconfirmed goalie: consider waiting for the 5 PM update." if dep else "")
        T.append(f"  {tot}")
        H += [_table(["pick", "units", "odds", "book", "model", "market", "goalies"], rows), f'<p style="font-size:13px;margin:6px 0">{html.escape(tot)}</p>']
    else:
        T.append("- No pick qualified today.")
        H.append('<p style="font-size:13px">No pick qualified today.</p>')
    T += ["", f"TODAY'S FORECASTS ({day})", "", "away at home | model away/home | market away/home | expected score | goalies (away / home)"]
    H.append(f'<h3 style="font-size:15px;margin:16px 0 6px 0">Today\'s forecasts ({day})</h3>')
    g = d.filter((pl.col("game_date") == day) & pl.col("home_win").is_null()).sort("start_utc")
    flags, rows, open_games = [], [], []
    for r in g.iter_rows(named=True):
        mk = f"{1 - r['mkt_home']:.0%}/{r['mkt_home']:.0%}" if r["mkt_home"] is not None else "no prices"
        gk = " / ".join(r[f"{s}_goalie"] + ("" if (r.get(f"{s}_goalie_source") or "") == CONFIRMED else "*") for s in ("away", "home"))       # * = not confirmed at this run
        if unconfirmed(r):
            open_games.append(f"{r['away']} at {r['home']}")
        T.append(f"- {r['away']} at {r['home']} | {r['p_away']:.0%}/{r['p_home']:.0%} | {mk} | {r['exp_goals_away']:.1f}-{r['exp_goals_home']:.1f} | {gk}")
        rows.append([f"{r['away']} at {r['home']}", f"{r['p_away']:.0%} / {r['p_home']:.0%}", mk.replace("/", " / "), f"{r['exp_goals_away']:.1f} to {r['exp_goals_home']:.1f}", html.escape(gk)])
        if r["mkt_home"] is not None and abs(r["p_home"] - r["mkt_home"]) > 0.08:
            flags.append(f"{r['away']} at {r['home']}: model and market differ by {abs(r['p_home'] - r['mkt_home']):.0%} - check for roster news the model may be missing.")
        for side in ("away", "home"):
            src = r.get(f"{side}_goalie_source") or ""
            if src.startswith(("back-to-back", "override")):
                flags.append(f"{r[side]}: {r[f'{side}_goalie']} in goal by {src} - no starter was confirmed in the morning lineups.")
    if g.height:
        H.append(_table(["game", "model, away / home", "market, away / home", "expected score", "goalies, away / home"], rows))
        note = (f"* Starting goalie not confirmed at this run, in {len(open_games)} of {g.height} games ({', '.join(open_games)}). An update for those games follows at about 5 PM Eastern."
                if open_games else "All of today's starting goalies were confirmed at this run; no afternoon update will be sent.")
        T += ["", note]
        H.append(f'<p style="font-size:12px;color:{MUTED};margin:6px 0">{html.escape(note)}</p>')
    else:
        T.append("- No games remaining today in the log.")
        H.append('<p style="font-size:13px">No games remaining today in the log.</p>')
    if flags:
        T += ["", "CHECK", ""] + [f"- {x}" for x in flags]
        H.append('<h3 style="font-size:15px;margin:16px 0 6px 0">Check</h3><ul style="font-size:13px;margin:0;padding-left:18px">' + "".join(f"<li>{html.escape(x)}</li>" for x in flags) + "</ul>")

    # ---- the owner's teams: yesterday's game by player
    fg = focus_games(yday)
    if fg:
        T += ["", f"YOUR TEAMS YESTERDAY ({yday})", ""]
        H.append(f'<h3 style="font-size:15px;margin:18px 0 2px 0">Your teams yesterday ({yday})</h3>')
        for x in fg:
            T.append(f"- {x['title']}. {x['line']}")
            for team in x["players"]["team"].unique(maintain_order=True).to_list():
                t = x["players"].filter(pl.col("team") == team).sort("on_xgd_5v5", descending=True)
                T.append(f"    {team} by 5v5 on-ice xG differential: best " + ", ".join(f"{r['name']} {r['on_xgd_5v5']:+.2f}" for r in t.head(3).iter_rows(named=True))
                         + "; lowest " + ", ".join(f"{r['name']} {r['on_xgd_5v5']:+.2f}" for r in t.tail(3).iter_rows(named=True)) + ".")
            T.append(f"    Full charts: {rel(x['dir'])}")
            H += [html_impact(x["players"], x["title"]), f'<p style="font-size:12px;color:{MUTED};margin:4px 0">{html.escape(x["line"])}</p>']
    foot = ("The model has no demonstrated edge over the market yet. This is record-keeping of your own picks, not betting advice. Prices are decimal odds as captured at the daily run and may have moved. "
            "One game is a small sample: the per-game numbers describe what happened and are not ratings.")
    T += ["", foot, ""]
    H.append(f'<p style="font-size:11px;color:{MUTED};margin:18px 0 0 0;border-top:1px solid #d8d7d2;padding-top:8px">{html.escape(foot)}</p></div>')
    return "\n".join(T), "\n".join(H)          # one element per line: tools that read the file line by line never meet an over-long line


def run(day: str | None = None) -> str:
    text, page = build(day)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    OUT_HTML.write_text(page, encoding="utf-8")
    log.info("digest written: %s (html %d characters)", text.splitlines()[0], len(page))
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))
