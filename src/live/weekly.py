"""Sunday weekly report:  python -m src.live.weekly [YYYY-MM-DD]

Season-to-date expected goals from the live tables (data/live/processed; NHL data only, so the report is public): teams ranked by expected-goal differential, the league's top skaters,
and every Jets and Senators skater as season percentiles. Writes reports/live/weekly_digest.md (first line is the subject) and reports/live/weekly_digest.html (tables only, for email).

Percentiles (JFresh-style) rank a skater among skaters of his position group (forwards or defensemen) with at least MIN_TOI minutes at 5v5, on per-60 rates:
offence (on-ice xG for), defence (on-ice xG against, lower is better), net (for minus against), relative (net on ice minus the team's net while he is off), and shooting (his own xG)."""

from __future__ import annotations

import html
import logging
import sys
from datetime import date

import polars as pl

from src.config import ROOT
from src.live import digest as dg
from src.scoreeffects import factors as sf

log = logging.getLogger("live")
LIVE = ROOT / "data/live/processed"
OUT, OUT_HTML = ROOT / "reports/live/weekly_digest.md", ROOT / "reports/live/weekly_digest.html"
FOCUS = {"WPG": "Winnipeg Jets", "OTT": "Ottawa Senators"}
MIN_TOI = 60.0                                   # 5v5 minutes to be ranked (raised automatically as the season grows: see min_toi)
# Net first: the overall on-ice measure (offence and defence together); the others break it down or put it in context
METRICS = [("net60", "Net", "on-ice xG differential per 60 (the overall measure: offence and defence together)", True), ("off60", "Off", "offence: on-ice xG for per 60", True),
           ("def60", "Def", "defence: on-ice xG against per 60 (fewer is better)", False), ("rel60", "Rel", "relative: net per 60 on ice minus the team's net per 60 with him off", True),
           ("ixg60", "Shot", "shooting: his own xG per 60", True)]
DIVISIONS = {"Atlantic": ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"], "Metropolitan": ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
             "Central": ["CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"], "Pacific": ["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"]}
NAME_W, COL_W = 120, 31                          # px: the name column fits "J. Siegenthaler NJD" on one line; every other column the same width, in every table
TEAM_W = 34
# diverging scale for percentiles: red (low) - grey (50th) - blue (high), the reference palette's diverging pair
LOW, MID, HIGH = (184, 48, 47), (240, 239, 236), (28, 92, 171)


def shots() -> pl.DataFrame:
    """Unblocked attempts with score-adjusted expected goals (docs/scoreeffects)."""
    ev = pl.read_parquet(LIVE / "events.parquet", columns=["game_id", "event_idx", "period", "score_diff"])
    sh = pl.read_parquet(LIVE / "shots_xg.parquet").filter(pl.col("type").is_in(["shot-on-goal", "missed-shot", "goal"])).join(ev, on=["game_id", "event_idx"])
    wp = ROOT / "data/outputs/score_effect_weights.csv"
    return sf.apply(sh, pl.read_csv(wp)) if wp.exists() else sh.with_columns(pl.col("xg").alias("xg_adj"))


def team_table(sh: pl.DataFrame, g: pl.DataFrame) -> pl.DataFrame:
    """Per team: record, goals, expected goals (all situations and 5v5, raw and score-adjusted), ranked by score-adjusted expected-goal differential per game."""
    teams = pl.concat([g.select("game_id", pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team"), pl.col("away_id").alias("opp_id"), pl.col("home_final").alias("gf"), pl.col("away_final").alias("ga"), "last_period_type"),
                       g.select("game_id", pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team"), pl.col("home_id").alias("opp_id"), pl.col("away_final").alias("gf"), pl.col("home_final").alias("ga"), "last_period_type")])
    x = sh.group_by("game_id", "team_id").agg(pl.col("xg").sum().alias("xf"), pl.col("xg_adj").sum().alias("xfa"), pl.col("xg").filter(pl.col("strength") == "5v5").sum().alias("xf5"),
                                              pl.col("xg_adj").filter(pl.col("strength") == "5v5").sum().alias("xf5a"))
    t = (teams.join(x, on=["game_id", "team_id"], how="left").join(x.rename({"team_id": "opp_id", "xf": "xa", "xfa": "xaa", "xf5": "xa5", "xf5a": "xa5a"}), on=["game_id", "opp_id"], how="left")
         .with_columns(pl.col("^x.*$").fill_null(0.0)))
    win, otl = pl.col("gf") > pl.col("ga"), (pl.col("gf") < pl.col("ga")) & (pl.col("last_period_type") != "REG")
    out = t.group_by("team").agg(pl.len().alias("gp"), win.sum().alias("w"), ((pl.col("gf") < pl.col("ga")) & (pl.col("last_period_type") == "REG")).sum().alias("l"), otl.sum().alias("otl"),
                                 pl.col("gf").sum(), pl.col("ga").sum(), pl.col("xf").sum(), pl.col("xa").sum(), pl.col("xfa").sum(), pl.col("xaa").sum(), pl.col("xf5a").sum(), pl.col("xa5a").sum())
    out = out.with_columns((2 * pl.col("w") + pl.col("otl")).alias("pts"), (pl.col("xf") - pl.col("xa")).alias("xgd"), ((pl.col("xfa") - pl.col("xaa")) / pl.col("gp")).alias("xgd_adj_pg"),
                           (pl.col("xf5a") / (pl.col("xf5a") + pl.col("xa5a"))).alias("xgf_pct_5v5"), ((pl.col("gf") - pl.col("ga")) - (pl.col("xf") - pl.col("xa"))).alias("gd_minus_xgd"))
    return out.sort("xgd_adj_pg", descending=True).with_row_index("rank", offset=1)


def min_toi(gp_max: int) -> float:
    """5v5 minutes to be ranked: 60, or 7 per team game once the season is longer (a regular forward plays about 12-14)."""
    return max(MIN_TOI, 7.0 * gp_max)


def player_table(sh: pl.DataFrame) -> pl.DataFrame:
    """Per skater, season to date at 5v5: minutes, on-ice and off-ice expected goals, his own expected goals, the per-60 rates and their percentiles within position group."""
    pg = pl.read_parquet(LIVE / "player_game.parquet").filter(pl.col("position") != "G")
    s5 = sh.filter(pl.col("strength") == "5v5").group_by("game_id", "team_id").agg(pl.col("xg").sum().alias("t_xf"))
    gm = pl.read_parquet(LIVE / "games.parquet").select("game_id", "home_id", "away_id")
    s5 = s5.join(gm, on="game_id").with_columns(pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.col("away_id")).otherwise(pl.col("home_id")).alias("opp"))
    s5 = s5.join(s5.select("game_id", pl.col("opp").alias("team_id"), pl.col("t_xf").alias("t_xa")), on=["game_id", "team_id"], how="left").select("game_id", "team_id", "t_xf", pl.col("t_xa").fill_null(0.0))
    toi = pg.group_by("game_id", "team_id").agg((pl.col("toi_5v5").sum() / 5.0).alias("t_toi"))      # the team's 5v5 seconds: five skaters at a time
    d = pg.join(s5, on=["game_id", "team_id"], how="left").join(toi, on=["game_id", "team_id"], how="left").with_columns(pl.col("t_xf", "t_xa").fill_null(0.0))
    last_team = pg.sort("game_id").group_by("player_id").agg(pl.col("team").last())
    p = d.group_by("player_id").agg(pl.col("first_name").first(), pl.col("last_name").first(), pl.col("position").first(), pl.len().alias("gp"), (pl.col("toi_5v5").sum() / 60).alias("toi5"),
                                    pl.col("oi_xgf").sum().alias("xgf"), pl.col("oi_xga").sum().alias("xga"), pl.col("ixg_ev").sum().alias("ixg"),
                                    ((pl.col("t_xf") - pl.col("oi_xgf")).sum()).alias("off_xgf"), ((pl.col("t_xa") - pl.col("oi_xga")).sum()).alias("off_xga"),
                                    ((pl.col("t_toi") - pl.col("toi_5v5")).sum() / 60).alias("off_toi"))
    p = p.join(last_team, on="player_id").with_columns((pl.col("first_name").str.slice(0, 1) + ". " + pl.col("last_name")).alias("name"),
                                                       pl.when(pl.col("position") == "D").then(pl.lit("D")).otherwise(pl.lit("F")).alias("pos"))
    r = 60.0 / pl.col("toi5")
    p = p.with_columns((pl.col("xgf") * r).alias("off60"), (pl.col("xga") * r).alias("def60"), ((pl.col("xgf") - pl.col("xga")) * r).alias("net60"), (pl.col("ixg") * r).alias("ixg60"),
                       (pl.col("xgf") - pl.col("xga")).alias("xgd"),
                       (((pl.col("xgf") - pl.col("xga")) * r) - pl.when(pl.col("off_toi") > 0).then((pl.col("off_xgf") - pl.col("off_xga")) * 60.0 / pl.col("off_toi")).otherwise(0.0)).alias("rel60"))
    return percentiles(p, min_toi(int(pl.read_parquet(LIVE / "player_game.parquet").group_by("team").agg(pl.col("game_id").n_unique())["game_id"].max())))


def percentiles(p: pl.DataFrame, floor: float) -> pl.DataFrame:
    """Percentile (0-100) of each rate among skaters of the same position group with at least ``floor`` 5v5 minutes; skaters below the floor are placed against that population too, and flagged."""
    out = []
    for grp in ("F", "D"):
        g = p.filter(pl.col("pos") == grp)
        ref = g.filter(pl.col("toi5") >= floor)
        cols = {}
        for m, _, _, up in METRICS:
            v = sorted(ref[m].to_list())
            n = len(v)
            def pct(x, v=v, n=n, up=up):                                     # share of the population below x, ties counted half
                if n == 0 or x is None:
                    return None
                below = sum(1 for y in v if y < x) + 0.5 * sum(1 for y in v if y == x)
                q = 100.0 * below / n
                return q if up else 100.0 - q
            cols[f"p_{m}"] = [pct(x) for x in g[m].to_list()]
        out.append(g.with_columns(*[pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()], (pl.col("toi5") >= floor).alias("ranked"), pl.lit(ref.height).alias("pop")))
    return pl.concat(out).with_columns(pl.lit(floor).alias("floor"))


# ---------------------------------------------------------------- HTML
def colour(q: float | None) -> tuple[str, str]:
    """Background and text colour for a percentile cell: red below 50, grey at 50, blue above."""
    if q is None:
        return "#ffffff", dg.MUTED
    t = min(abs(q - 50.0) / 50.0, 1.0)
    end = HIGH if q >= 50 else LOW
    rgb = tuple(round(m + (e - m) * t) for m, e in zip(MID, end))
    return "#" + "".join(f"{c:02x}" for c in rgb), ("#ffffff" if t > 0.55 else dg.INK)


def pct_cell(q: float | None, ranked: bool = True) -> str:
    bg, fg = colour(q)
    txt = "" if q is None else f"{q:.0f}" + ("" if ranked else "*")
    return f"<td bgcolor={bg}><font color={fg}><b>{txt}</b></font></td>"


def _t(head: list[str], rows: list[str], widths: list[int]) -> str:
    """A fixed-layout table: the given column widths in every row, so tables line up and nothing wraps on a phone."""
    th = "".join(f'<th width={w} align={"left" if i == 0 else "center"} style="white-space:nowrap"><font color={dg.MUTED}>{h}</font></th>' for i, (h, w) in enumerate(zip(head, widths)))
    return (f'<table cellpadding=3 cellspacing=0 width={sum(widths)} style="border-collapse:collapse;table-layout:fixed;font-size:11px;width:{sum(widths)}px;text-align:center"><tr>{th}</tr>\n'
            + "\n".join(rows) + "</table>")


PLAYER_HEAD = ["{who}", "GP", "Min", "xGD", *[m[1] for m in METRICS]]          # the per-60 rate is what the Net percentile ranks
PLAYER_W = [NAME_W, *[COL_W] * (len(PLAYER_HEAD) - 1)]
TEAM_HEAD = ["Team", "GP", "Rec", "PTS", "xGF", "xGA", "xGD", "/GP", "xG rk", "xGF%", "GD−xG"]
TEAM_WIDTHS = [TEAM_W, 22, 44, 26, *[TEAM_W] * (len(TEAM_HEAD) - 4)]      # games, record and points are narrow; the measures share one width


def standings(tt: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The team table split by division, each sorted by points (then fewer games played, then wins)."""
    return {d: tt.filter(pl.col("team").is_in(teams)).sort(["pts", "gp", "w"], descending=[True, False, True]) for d, teams in DIVISIONS.items()}


def build(day: str | None = None) -> tuple[str, str]:
    day = day or date.today().isoformat()
    g = pl.read_parquet(LIVE / "games.parquet").filter(pl.col("game_date") <= day)
    sh = shots().filter(pl.col("game_id").is_in(g["game_id"]))
    tt = team_table(sh, g)
    pt = player_table(sh)
    floor = float(pt["floor"][0])
    subject = f"NHL model weekly {day}: xG standings, top skaters, Jets and Senators percentiles"
    T, H = [f"Subject: {subject}", ""], [f'<div style="font-family:Arial,Helvetica,sans-serif;color:{dg.INK};max-width:720px">', f'<h2 style="font-size:18px;margin:0 0 6px 0">{html.escape(subject)}</h2>',
                                       f'<p style="font-size:12px;color:{dg.MUTED};margin:0 0 10px 0">Season to date through {day}: {g.height} games. Expected goals (xG) are the model\'s shot-quality estimate of goals; '
                                       f'"score-adjusted" weighs 5v5 chances by score state (a team protecting a lead concedes more).</p>']

    # ---- teams, by division in standings order
    T += ["STANDINGS BY DIVISION, WITH EXPECTED GOALS", "", "team GP W-L-OT PTS | xGF xGA xGD | xGD/GP (score-adjusted) | xG rank | 5v5 xGF% | GD - xGD"]
    H.append('<h3 style="font-size:15px;margin:14px 0 4px 0">Standings by division, with expected goals</h3>')
    for div, d in standings(tt).items():
        T.append(f"{div}:")
        rows = []
        for r in d.iter_rows(named=True):
            focus = r["team"] in FOCUS
            T.append(f"  {r['team']} {r['gp']} {r['w']}-{r['l']}-{r['otl']} {r['pts']} | {r['xf']:.1f} {r['xa']:.1f} {r['xgd']:+.1f} | {r['xgd_adj_pg']:+.2f} | {r['rank']} | {r['xgf_pct_5v5']:.1%} | {r['gd_minus_xgd']:+.1f}")
            b = (lambda s: f"<b>{s}</b>") if focus else (lambda s: s)
            vals = (r["team"], r["gp"], f"{r['w']}-{r['l']}-{r['otl']}", r["pts"], f"{r['xf']:.1f}", f"{r['xa']:.1f}", f"{r['xgd']:+.1f}", f"{r['xgd_adj_pg']:+.2f}", r["rank"], f"{r['xgf_pct_5v5']:.0%}", f"{r['gd_minus_xgd']:+.1f}")
            rows.append("<tr" + (" bgcolor=#f0efec" if focus else "") + ">" + "".join((f'<td align=left>{b(c)}</td>' if i == 0 else f"<td>{b(c)}</td>") for i, c in enumerate(vals)) + "</tr>")
        H += [f'<p style="font-size:13px;margin:10px 0 2px 0"><b>{div}</b></p>', _t(TEAM_HEAD, rows, TEAM_WIDTHS)]
    H.append(f'<p style="font-size:11px;color:{dg.MUTED};margin:4px 0">Sorted by points within each division. Rec = W-L-OT; /GP is the score-adjusted expected-goal differential per game, all situations; xG rk is that figure\'s rank in the league (1 = best); xGF% is the team\'s share of 5v5 expected goals, score-adjusted. '
             "GD−xG is goal differential minus xG differential: goals beyond expected (shooting and goaltending, much of it luck over a few weeks).</p>")

    # ---- league leaders
    ranked = pt.filter(pl.col("ranked"))
    T += ["", f"TOP SKATERS BY 5v5 ON-ICE xG DIFFERENTIAL (at least {floor:.0f} minutes at 5v5)", ""]
    H.append('<h3 style="font-size:15px;margin:16px 0 4px 0">Top skaters by 5v5 on-ice xG differential</h3>')
    for grp, label, n in (("F", "Forwards", 15), ("D", "Defensemen", 10)):
        top = ranked.filter(pl.col("pos") == grp).sort("xgd", descending=True).head(n)
        T.append(f"{label}: " + "; ".join(f"{r['name']} ({r['team']}) {r['xgd']:+.1f}, {r['net60']:+.2f}/60" for r in top.iter_rows(named=True)))
        H += [f'<p style="font-size:13px;margin:8px 0 2px 0"><b>{label}</b></p>', _t([h.format(who="Skater (team)") for h in PLAYER_HEAD], [player_row(r) for r in top.iter_rows(named=True)], PLAYER_W)]

    # ---- the owner's teams
    for ab, full in FOCUS.items():
        tr = tt.filter(pl.col("team") == ab).row(0, named=True)
        line = (f"{full}: {tr['w']}-{tr['l']}-{tr['otl']}, {tr['pts']} points. Expected goals {tr['xf']:.1f} for, {tr['xa']:.1f} against ({tr['xgd']:+.1f}); score-adjusted {tr['xgd_adj_pg']:+.2f} per game, {ordinal(tr['rank'])} of {tt.height}; "
                f"5v5 xGF% {tr['xgf_pct_5v5']:.1%}. Actual goals {tr['gf']}-{tr['ga']} ({tr['gd_minus_xgd']:+.1f} against expected).")
        team = pt.filter(pl.col("team") == ab).sort(["pos", "net60"], descending=[True, True])
        T += ["", full.upper(), "", line, "skater (pos) GP 5v5min xGD | percentiles: " + " ".join(m[1] for m in METRICS)]
        for r in team.iter_rows(named=True):
            T.append(f"- {r['name']} ({r['position']}) {r['gp']} {r['toi5']:.0f} {r['xgd']:+.1f} | " + " ".join("-" if r[f"p_{m[0]}"] is None else f"{r[f'p_{m[0]}']:.0f}" for m in METRICS) + ("" if r["ranked"] else " (under the minutes floor)"))
        H += [f'<h3 style="font-size:15px;margin:18px 0 4px 0">{html.escape(full)}</h3>', f'<p style="font-size:13px;margin:0 0 6px 0">{html.escape(line)}</p>',
              _t([h.format(who="Skater (pos)") for h in PLAYER_HEAD], [player_row(r, team_col=False) for r in team.iter_rows(named=True)], PLAYER_W)]

    key = "; ".join(f"{m[1]} = {m[2]}" for m in METRICS)
    foot = (f"Columns: GP = games; Min = 5v5 minutes; xGD = 5v5 on-ice xG differential (Net ranks it per 60 minutes). "
            f"Percentiles rank each skater among forwards or defensemen with at least {floor:.0f} minutes at 5v5 this season (100 = best in the league; blue above the 50th, red below). "
            f"* = under that floor: placed against the same group, but on very few minutes. {key}. On-ice figures are raw 5v5 expected goals. A few weeks of games is a small sample: these describe the season so far and are not ratings.")
    T += ["", foot, ""]
    H.append(f'<p style="font-size:11px;color:{dg.MUTED};margin:16px 0 0 0;border-top:1px solid #d8d7d2;padding-top:8px">{html.escape(foot)}</p></div>')
    return "\n".join(T), "\n".join(H)


def player_row(r: dict, team_col: bool = True) -> str:
    who = f"{html.escape(r['name'])} <font color={dg.MUTED}>{r['team'] if team_col else r['position']}</font>"
    cells = [f'<td align=left style="white-space:nowrap;overflow:hidden">{who}</td>', *(f"<td>{v}</td>" for v in (r["gp"], f"{r['toi5']:.0f}", f"{r['xgd']:+.1f}"))]
    return "<tr>" + "".join(cells) + "".join(pct_cell(r[f"p_{m[0]}"], r["ranked"]) for m in METRICS) + "</tr>"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def run(day: str | None = None) -> str:
    text, page = build(day)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    OUT_HTML.write_text(page, encoding="utf-8")
    log.info("weekly digest written: %s (html %d characters)", text.splitlines()[0], len(page))
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))
