"""One-game summary (graphics, player table, takeaways):  python -m src.live.game_report <game_id> [season]

Reads the live tables written by src.live.update. Everything here is descriptive for one game: expected goals from the shot model,
possession value from the PM-EPA layer, and on-ice expected goals in every situation (even strength, power play, shorthanded). Season-level components (defense, finishing talent, special teams ratings) need many games and are not shown."""

from __future__ import annotations

import functools
import logging
import sys

import numpy as np
import polars as pl

from src.config import ROOT, load_config
from src.live.update import live_config
from src.scoreeffects import factors as sf
from src.viz import style
from src.viz.style import plt

log = logging.getLogger("live")
HOME_C, AWAY_C = style.SERIES[0], style.SERIES[1]
UNBLOCKED = ["shot-on-goal", "missed-shot", "goal"]
# impact-chart colours by situation (the reference palette's first three slots: the only three that stay distinct in every pairing for colour-blind readers);
# the side of the centre line says for or against, and texture separates a player's own even-strength shots (solid) from his linemates' (hatched / shaded)
ES_C, PP_C, SH_C = "#2a78d6", "#eb6834", "#1baf7a"           # even strength blue, power play orange, shorthanded aqua
ES_LIGHT = "#86b6ef"                                          # linemates' even-strength shots: a lighter step of the same blue (passes the colour-blind and normal-vision checks against the other three)
EVEN = ("5v5", "4v4", "3v3")
SIT = ["es_own", "es_mates", "pp_for", "sh_for", "es_against", "pk_against", "ppa_against", "es_danger_for", "es_danger_against"]


def load(game_id: int) -> dict:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    g = pl.read_parquet(proc / "games.parquet").filter(pl.col("game_id") == game_id).row(0, named=True)
    names = pl.read_parquet(proc / "players.parquet").select("player_id", (pl.col("first_name").str.slice(0, 1) + ". " + pl.col("last_name")).alias("name"), "last_name", "position")
    ev = pl.read_parquet(proc / "events.parquet").filter(pl.col("game_id") == game_id)
    sh = (pl.read_parquet(proc / "shots_xg.parquet").filter((pl.col("game_id") == game_id) & pl.col("type").is_in(UNBLOCKED))
          .join(ev.select("event_idx", "t_game", "period", "player2", "score_diff"), on="event_idx").join(names.select(pl.col("player_id").alias("player1"), "last_name"), on="player1", how="left").sort("t_game"))
    wp = ROOT / "data/outputs/score_effect_weights.csv"            # league score-state weights for 5v5 shots (docs/scoreeffects); without the file the adjusted figure equals the raw one
    sh = sf.apply(sh, pl.read_csv(wp) if wp.exists() else pl.DataFrame(schema={"period": pl.Int64, "d": pl.Int64, "rate60": pl.Float64, "w": pl.Float64}))
    pg = pl.read_parquet(proc / "player_game.parquet").filter(pl.col("game_id") == game_id)
    oi = pl.read_parquet(proc / "onice.parquet").filter(pl.col("game_id") == game_id).select("event_idx", "home_on", "away_on")
    obs = pl.read_parquet(proc / "obs_valued.parquet", columns=["poss_id", "game_id", "seq", "team_id", "strength", "v", "onice", "opp_onice"]).filter(pl.col("game_id") == game_id)
    poss = pl.read_parquet(proc / "poss_valued_epa.parquet", columns=["poss_id", "game_id", "end_class"]).filter(pl.col("game_id") == game_id)
    return {"g": g, "ev": ev, "sh": sh, "pg": pg, "names": names, "oi": oi, "obs": obs, "poss": poss}


def unconverted_danger(obs: pl.DataFrame, poss: pl.DataFrame) -> pl.DataFrame:
    """Per even-strength possession that ended without an unblocked shot (blocked shots included): the danger it built, its highest expected-goal value
    (PM-EPA possession value) minus its value at the start, floored at zero, with the skaters on ice for each side at that peak."""
    first = obs.sort("poss_id", "seq").group_by("poss_id").agg(pl.col("v").first().alias("v_start"))
    peak = obs.sort(["poss_id", "v", "seq"], descending=[False, True, False]).unique("poss_id", keep="first").select("poss_id", "team_id", "strength", pl.col("v").alias("v_peak"), "onice", "opp_onice")
    p = peak.join(first, on="poss_id").join(poss.select("poss_id", "end_class"), on="poss_id")
    return p.filter((pl.col("end_class") == 2) & pl.col("strength").is_in(EVEN)).with_columns((pl.col("v_peak") - pl.col("v_start")).clip(lower_bound=0.0).alias("built"))


def situations(d: dict) -> pl.DataFrame:
    """Per skater: on-ice expected goals by situation, from his team's point of view. For: his own even-strength shots, his linemates' even-strength shots,
    power play, shorthanded. Against: even strength, on the penalty kill, on the power play (shorthanded chances conceded). Empty-net shots are left out.
    Also the even-strength danger built without an unblocked shot (unconverted_danger): by his team (for) and by the opponent (against) while he was on the ice."""
    g = d["g"]
    s = d["sh"].filter(~pl.col("empty_net") & (pl.col("strength") != "EN")).join(d["oi"], on="event_idx", how="inner")
    home = pl.col("team_id") == g["home_id"]
    s = s.with_columns(pl.when(home).then(pl.col("home_on")).otherwise(pl.col("away_on")).alias("for_on"), pl.when(home).then(pl.col("away_on")).otherwise(pl.col("home_on")).alias("against_on"))
    es, pp, pk = pl.col("strength").is_in(EVEN), pl.col("strength") == "PP", pl.col("strength") == "PK"
    f = s.explode("for_on").group_by(pl.col("for_on").alias("player_id")).agg(
        pl.col("xg").filter(es & (pl.col("player1") == pl.col("for_on"))).sum().alias("es_own"), pl.col("xg").filter(es & (pl.col("player1") != pl.col("for_on"))).sum().alias("es_mates"),
        pl.col("xg").filter(pp).sum().alias("pp_for"), pl.col("xg").filter(pk).sum().alias("sh_for"))
    a = s.explode("against_on").group_by(pl.col("against_on").alias("player_id")).agg(
        pl.col("xg").filter(es).sum().alias("es_against"), pl.col("xg").filter(pp).sum().alias("pk_against"), pl.col("xg").filter(pk).sum().alias("ppa_against"))
    u = unconverted_danger(d["obs"], d["poss"])
    df = u.explode("onice").group_by(pl.col("onice").alias("player_id")).agg(pl.col("built").sum().alias("es_danger_for"))
    da = u.explode("opp_onice").group_by(pl.col("opp_onice").alias("player_id")).agg(pl.col("built").sum().alias("es_danger_against"))
    out = f.join(a, on="player_id", how="full", coalesce=True).join(df, on="player_id", how="full", coalesce=True).join(da, on="player_id", how="full", coalesce=True)
    return out.with_columns(pl.col(SIT).fill_null(0.0))


def team_summary(d: dict) -> pl.DataFrame:
    g, sh = d["g"], d["sh"]
    rows = []
    for tid, ab in ((g["away_id"], g["away_abbrev"]), (g["home_id"], g["home_abbrev"])):
        f, a = sh.filter(pl.col("team_id") == tid), sh.filter(pl.col("team_id") != tid)
        s5 = f.filter(pl.col("strength") == "5v5")
        # goalie line: every unblocked attempt against with a goalie in net (the xG model is defined on unblocked attempts, so misses count); the name is the goalie on the shots on goal
        gk = a.filter(~pl.col("empty_net"))
        who = (a.filter(pl.col("type").is_in(["shot-on-goal", "goal"]) & pl.col("player2").is_not_null())
               .join(d["names"].filter(pl.col("position") == "G").select(pl.col("player_id").alias("player2"), pl.col("name").alias("goalie")), on="player2", how="inner"))
        rows.append({"team": ab, "goals": g["away_final"] if tid == g["away_id"] else g["home_final"], "xg": float(f["xg"].sum()), "xg_score_adj": float(f["xg_adj"].sum()), "xg_5v5": float(s5["xg"].sum()), "xg_5v5_score_adj": float(s5["xg_adj"].sum()), "xg_pp": float(f.filter(pl.col("strength") == "PP")["xg"].sum()),
                     "xg_other": float(f.filter(~pl.col("strength").is_in(["5v5", "PP"]))["xg"].sum()), "shots_on_goal": int(f["type"].is_in(["shot-on-goal", "goal"]).sum()), "unblocked_attempts": f.height,
                     "goalie": " / ".join(who["goalie"].unique(maintain_order=True).to_list()) if who.height else None, "goalie_shots": who.height, "goalie_xg_faced": float(gk["xg"].sum()), "goalie_goals_against": int(gk["goal"].sum())})
    return pl.DataFrame(rows).with_columns((pl.col("goalie_xg_faced") - pl.col("goalie_goals_against")).alias("goalie_saved_above_expected"))


@functools.lru_cache(maxsize=4)
def league_rates(proc_dir: str) -> dict[str, float]:
    """League on-ice expected goals per skater-minute by situation, this season to date (empty-net shots left out): es_for = es_against at even strength;
    on the power play, for and against (shorthanded chances conceded); on the penalty kill, for (shorthanded chances) and against."""
    from pathlib import Path
    proc = Path(proc_dir)
    pg = pl.read_parquet(proc / "player_game.parquet", columns=["player_id", "position", "toi_5v5", "toi_4v4", "toi_3v3", "toi_PP", "toi_PK"]).filter(pl.col("position") != "G")
    skaters = pg["player_id"].unique().to_list()
    sh = (pl.read_parquet(proc / "shots_xg.parquet", columns=["game_id", "event_idx", "type", "team_id", "strength", "empty_net", "xg"])
          .filter(pl.col("type").is_in(UNBLOCKED) & ~pl.col("empty_net") & (pl.col("strength") != "EN"))
          .join(pl.read_parquet(proc / "onice.parquet"), on=["game_id", "event_idx"]).join(pl.read_parquet(proc / "games.parquet").select("game_id", "home_id"), on="game_id"))
    home = pl.col("team_id") == pl.col("home_id")
    sh = sh.with_columns(pl.when(home).then(pl.col("home_on")).otherwise(pl.col("away_on")).list.eval(pl.element().is_in(skaters)).list.sum().alias("nf"),
                         pl.when(home).then(pl.col("away_on")).otherwise(pl.col("home_on")).list.eval(pl.element().is_in(skaters)).list.sum().alias("na"))
    tot = lambda cond, n: float(sh.filter(cond).select((pl.col("xg") * pl.col(n)).sum()).item())      # noqa: E731  skater-weighted expected goals
    es, pp, pk = pl.col("strength").is_in(EVEN), pl.col("strength") == "PP", pl.col("strength") == "PK"
    m_es, m_pp, m_pk = float((pg["toi_5v5"].sum() + pg["toi_4v4"].sum() + pg["toi_3v3"].sum()) / 60), float(pg["toi_PP"].sum() / 60), float(pg["toi_PK"].sum() / 60)
    return {"es_for": tot(es, "nf") / m_es, "es_against": tot(es, "na") / m_es, "pp_for": tot(pp, "nf") / m_pp, "pp_against": tot(pk, "na") / m_pp,
            "pk_for": tot(pk, "nf") / m_pk, "pk_against": tot(pp, "na") / m_pk}


def adjusted(t: pl.DataFrame, r: dict[str, float]) -> pl.DataFrame:
    """Situation-adjusted net: in each situation, (on-ice for minus the league rate times his minutes) minus (on-ice against minus the league rate times his minutes), summed.
    A penalty killer is credited for allowing fewer chances than an average penalty kill would in his minutes, not charged for every chance; the same for the power play and even strength."""
    t_es, t_pp, t_pk = (pl.col("toi_5v5_s") + pl.col("toi_4v4") + pl.col("toi_3v3")) / 60, pl.col("toi_PP") / 60, pl.col("toi_PK") / 60
    return t.with_columns(((pl.col("es_own") + pl.col("es_mates") + pl.col("es_danger_for") - r["es_for"] * t_es) - (pl.col("es_against") + pl.col("es_danger_against") - r["es_against"] * t_es)).alias("adj_es"),
                          ((pl.col("pp_for") - r["pp_for"] * t_pp) - (pl.col("ppa_against") - r["pp_against"] * t_pp)).alias("adj_pp"),
                          ((pl.col("sh_for") - r["pk_for"] * t_pk) - (pl.col("pk_against") - r["pk_against"] * t_pk)).alias("adj_pk")).with_columns(
        (pl.col("adj_es") + pl.col("adj_pp") + pl.col("adj_pk")).alias("net_adj"))


def player_table(d: dict) -> pl.DataFrame:
    pg = d["pg"].join(d["names"].select("player_id", "name"), on="player_id", how="left")
    # impact order: situation-adjusted on-ice expected-goal differential, best first (the model owner's preference; ice time stays as a column)
    t = pg.select("team", "name", "position", (pl.col("toi_all") / 60).alias("toi"), "goals", (pl.col("a1") + pl.col("a2")).alias("assists"), "sog", "ixg", pl.col("pm_epa_total").alias("pm_epa"),
                     pl.col("oi_xgf").alias("on_xgf_5v5"), pl.col("oi_xga").alias("on_xga_5v5"), (pl.col("oi_xgf") - pl.col("oi_xga")).alias("on_xgd_5v5"), (pl.col("toi_5v5") / 60).alias("toi_5v5"),
                     pl.col("ixg_ev").alias("own_xg_5v5"), (pl.col("oi_xgf") - pl.col("ixg_ev")).clip(lower_bound=0.0).alias("mates_xg_5v5"), "player_id", pl.col("toi_5v5").alias("toi_5v5_s"), "toi_4v4", "toi_3v3", "toi_PP", "toi_PK")
    t = t.join(situations(d), on="player_id", how="left").with_columns(pl.col(SIT).fill_null(0.0))
    t = t.with_columns((pl.col("es_own") + pl.col("es_mates") + pl.col("es_danger_for") + pl.col("pp_for") + pl.col("sh_for")).alias("xgf_all"),
                       (pl.col("es_against") + pl.col("es_danger_against") + pl.col("pk_against") + pl.col("ppa_against")).alias("xga_all"))
    t = adjusted(t.with_columns((pl.col("xgf_all") - pl.col("xga_all")).alias("net_all")), league_rates(str(load_config().path(load_config().data.processed_dir))))
    return t.drop("player_id", "toi_5v5_s", "toi_4v4", "toi_3v3").with_columns((pl.col("toi_PP") / 60).alias("toi_pp"), (pl.col("toi_PK") / 60).alias("toi_pk")).drop("toi_PP", "toi_PK").sort(
        "team", "net_adj", descending=[False, True])


def fig_impact(d: dict, pt: pl.DataFrame):
    """Stacked bars of each skater's on-ice expected goals in every situation: for (right of the line) his own and his linemates' even-strength shots, power play and shorthanded;
    against (left) at even strength, on the penalty kill and on the power play. The number is the situation-adjusted net (each situation against the league rate for his minutes); sorted by it."""
    g = d["g"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.8), sharex=True)
    lim = 1.3 * max(float(pt["xgf_all"].max()), float(pt["xga_all"].max()), 0.1)            # room on the right for the net figure
    bar = {"height": 0.72, "edgecolor": style.SURFACE, "linewidth": 2}
    for ax, ab in ((axes[0], g["away_abbrev"]), (axes[1], g["home_abbrev"])):
        t = pt.filter(pl.col("team") == ab).sort("net_adj")
        y = np.arange(t.height)
        v = {c: t[c].to_numpy() for c in SIT}
        left = np.zeros(t.height)
        play = v["es_mates"] + v["es_danger_for"]
        for val, col, hatch in ((v["es_own"], ES_C, None), (play, ES_LIGHT, "////"), (v["pp_for"], PP_C, None), (v["sh_for"], SH_C, None)):
            ax.barh(y, val, left=left, color=col, hatch=hatch, **bar)
            left += val
        left = np.zeros(t.height)
        for val, col, hatch in ((v["es_against"], ES_C, None), (v["es_danger_against"], ES_LIGHT, "////"), (v["pk_against"], SH_C, None), (v["ppa_against"], PP_C, None)):
            ax.barh(y, -val, left=left, color=col, hatch=hatch, **bar)
            left -= val
        net = t["net_adj"].to_numpy()
        ax.text(lim, t.height - 0.3, "net vs\naverage", va="bottom", ha="right", fontsize=8, color=style.TEXT2, linespacing=1.0)
        for yi, n in zip(y, net):
            ax.text(lim, yi, f"{n:+.2f}", va="center", ha="right", fontsize=8.5, color=style.TEXT, fontweight="bold")
        ax.set_yticks(y, [f"{n}  ({p})" for n, p in zip(t["name"].to_list(), t["position"].to_list())], fontsize=9)
        ax.axvline(0, color=style.TEXT2, lw=1)
        ax.set_xlim(-lim / 1.3 * 1.05, lim)
        ax.set_ylim(-0.6, t.height - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(ab, color=style.TEXT, fontsize=13)
        ax.set_xlabel("expected goals while on the ice, all situations   (against  ←  →  for)")
    h = axes[0]
    h.barh([-9], [0], color=ES_C, label="EV-Shots: even-strength shots (his own for; all against)")
    h.barh([-9], [0], color=ES_LIGHT, hatch="////", label="EV-Play: linemates' shots and danger built without a shot (for), danger allowed without a shot (against)")
    h.barh([-9], [0], color=PP_C, label="power play")
    h.barh([-9], [0], color=SH_C, label="shorthanded / penalty kill")
    fig.legend(*h.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.025), fontsize=9)
    fig.text(0.5, 0.004, "Number at right: net versus a league-average skater in the same minutes of each situation (even strength, power play, penalty kill), so special-teams minutes do not tilt it.",
             ha="center", fontsize=8.5, color=style.TEXT2)
    fig.suptitle(f"{g['away_abbrev']} at {g['home_abbrev']}, {g['game_date']}: impact by skater, all situations (sorted by situation-adjusted net)", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    return fig


def fig_flow(d: dict, ts: pl.DataFrame):
    g, sh = d["g"], d["sh"]
    end = max(60.0, float(d["ev"]["t_game"].max()) / 60)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 9.2), gridspec_kw={"height_ratios": [1, 1.05]})
    for tid, ab, col in ((g["home_id"], g["home_abbrev"], HOME_C), (g["away_id"], g["away_abbrev"], AWAY_C)):
        s = sh.filter(pl.col("team_id") == tid)
        t, c = np.concatenate([[0], s["t_game"].to_numpy() / 60, [end]]), np.concatenate([[0], np.cumsum(s["xg"].to_numpy())])
        a1.step(t, np.concatenate([c, [c[-1]]]), where="post", color=col)
        a1.text(end + 1.8, c[-1], f"{ab} {c[-1]:.2f}", color=col, va="center", fontweight="bold")
        gl = s.with_columns(pl.col("xg").cum_sum().alias("cum")).filter(pl.col("goal") == 1)
        a1.scatter(gl["t_game"].to_numpy() / 60, gl["cum"].to_numpy(), s=70, color=col, edgecolor=style.SURFACE, linewidth=1.5, zorder=5)
        for r in gl.iter_rows(named=True):
            a1.annotate(r["last_name"], (r["t_game"] / 60, r["cum"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=style.TEXT2)
    for x in (20, 40, 60):
        if x < end:
            a1.axvline(x, color=style.GRID, lw=1)
    a1.set_xlim(0, end + 1.0)
    a1.set_ylim(0, None)
    a1.set_xlabel("game minutes")
    a1.set_ylabel("cumulative expected goals")
    a1.set_title(f"{g['away_abbrev']} {g['away_final']} at {g['home_abbrev']} {g['home_final']} ({g['last_period_type']}), {g['game_date']}: expected goals through the game (dots are goals)")
    a1.spines["right"].set_visible(False)
    for tid, col, sign in ((g["home_id"], HOME_C, 1.0), (g["away_id"], AWAY_C, -1.0)):
        s = sh.filter(pl.col("team_id") == tid)
        miss, goal = s.filter(pl.col("goal") == 0), s.filter(pl.col("goal") == 1)
        a2.scatter(sign * miss["x"].to_numpy(), sign * miss["y"].to_numpy(), s=20 + 700 * miss["xg"].to_numpy(), facecolor="none", edgecolor=col, linewidth=1.2, alpha=0.9, zorder=6)
        a2.scatter(sign * goal["x"].to_numpy(), sign * goal["y"].to_numpy(), s=20 + 700 * goal["xg"].to_numpy(), color=col, edgecolor=style.TEXT, linewidth=1.0, zorder=7)
    style.draw_rink(a2)
    hx, ax_ = ts.filter(pl.col("team") == g["home_abbrev"]).row(0, named=True), ts.filter(pl.col("team") == g["away_abbrev"]).row(0, named=True)
    a2.text(62, 47, f"{g['home_abbrev']} shoots this way: {hx['unblocked_attempts']} attempts, {hx['xg']:.2f} xG", color=HOME_C, ha="center", fontweight="bold")
    a2.text(-62, 47, f"{g['away_abbrev']} shoots this way: {ax_['unblocked_attempts']} attempts, {ax_['xg']:.2f} xG", color=AWAY_C, ha="center", fontweight="bold")
    a2.text(0, -48.5, "Unblocked shot attempts; circle area grows with expected goals; filled circles are goals.", ha="center", color=style.TEXT2, fontsize=9)
    a2.set_ylim(-51, 50)
    fig.tight_layout()
    return fig


COLS = [("toi", "TOI", "{:.1f}", None), ("goals", "G", "{:.0f}", None), ("assists", "A", "{:.0f}", None), ("ixg", "ixG", "{:.2f}", "seq"), ("pm_epa", "puck\nvalue", "{:+.2f}", "div"),
        ("on_xgd_5v5", "5v5 on-ice\nxG diff", "{:+.2f}", "div"), ("pp_for", "PP\nxGF", "{:.2f}", None), ("pk_against", "PK\nxGA", "{:.2f}", None), ("net_all", "net xG\nall sit.", "{:+.2f}", None), ("net_adj", "net vs\naverage", "{:+.2f}", "div")]


def fig_players(d: dict, pt: pl.DataFrame):
    g = d["g"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.2))
    lim = {c: float(pt[c].abs().max()) or 1.0 for c, _, _, kind in COLS if kind}
    for ax, ab, col in ((axes[0], g["away_abbrev"], AWAY_C), (axes[1], g["home_abbrev"], HOME_C)):
        t = pt.filter(pl.col("team") == ab)
        t = pl.concat([t.filter(pl.col("position") != "D"), t.filter(pl.col("position") == "D")])          # each group already in impact order
        n = t.height
        ax.set_xlim(0, 3.2 + len(COLS))
        ax.set_ylim(n + 0.2, -1.6)
        ax.axis("off")
        ax.text(0, -0.5, ab, fontsize=13, fontweight="bold", color=col)
        for j, (_, label, _, _) in enumerate(COLS):
            ax.text(3.7 + j, -0.12, label, ha="center", va="bottom", fontsize=8.5, color=style.TEXT2, fontweight="bold", linespacing=1.1)
        nf = t.filter(pl.col("position") != "D").height
        for i, r in enumerate(t.iter_rows(named=True)):
            y = i + (0.25 if i >= nf else 0.0)
            ax.text(0, y + 0.5, r["name"], va="center", fontsize=9)
            ax.text(2.9, y + 0.5, r["position"], va="center", ha="center", fontsize=8, color=style.TEXT2)
            for j, (c, _, fmt, kind) in enumerate(COLS):
                v = r[c]
                if kind:
                    z = v / lim[c]
                    face = style.SEQ(0.05 + 0.6 * max(z, 0)) if kind == "seq" else style.DIV(0.5 - 0.42 * z)      # blue = good for the player's team
                    ax.add_patch(plt.Rectangle((3.2 + j + 0.04, y + 0.06), 0.92, 0.88, color=face, lw=0))
                dark = kind == "div" and abs(v / lim[c]) > 0.75 or kind == "seq" and v / lim[c] > 0.8
                ax.text(3.7 + j, y + 0.5, fmt.format(v), ha="center", va="center", fontsize=8.5, color="white" if dark else style.TEXT)
    fig.suptitle(f"{g['away_abbrev']} at {g['home_abbrev']}, {g['game_date']}: skaters (forwards, then defensemen, by situation-adjusted net)", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.02, 0.015, "ixG: expected goals from the player's own shots.  Puck value: goals of expected value added by his credited puck actions (PM-EPA, all situations).  "
             "5v5 on-ice xG diff: for minus against at 5v5.  PP xGF / PK xGA: on-ice for on the power play / against on the penalty kill.  Net all sit.: on-ice for minus against in every situation (empty net left out).  Net vs average: the same, each situation measured against a league-average skater in his minutes there.\nOne game is a small sample: these describe what happened, not how good a player is.", fontsize=8.5, color=style.TEXT2, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


def takeaways(d: dict, ts: pl.DataFrame, pt: pl.DataFrame) -> str:
    g = d["g"]
    a, h = ts.row(0, named=True), ts.row(1, named=True)
    win = a if a["goals"] > h["goals"] else h
    xw = a if a["xg"] > h["xg"] else h
    L = [f"# {a['team']} {a['goals']} at {h['team']} {h['goals']} ({g['last_period_type']}), {g['game_date']}", "",
         "| team | goals | expected goals | score-adjusted | 5v5 | 5v5 score-adjusted | power play | other | shots on goal | unblocked attempts |", "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['team']} | {r['goals']} | {r['xg']:.2f} | {r['xg_score_adj']:.2f} | {r['xg_5v5']:.2f} | {r['xg_5v5_score_adj']:.2f} | {r['xg_pp']:.2f} | {r['xg_other']:.2f} | {r['shots_on_goal']} | {r['unblocked_attempts']} |" for r in (a, h)]
    L += ["", "## Takeaways", ""]
    L.append(f"* **{win['team']} won {max(a['goals'], h['goals'])}–{min(a['goals'], h['goals'])}; {xw['team']} had more expected goals ({xw['xg']:.2f} to {min(a['xg'], h['xg']):.2f}).** "
             f"At 5v5 it was {a['team']} {a['xg_5v5']:.2f}, {h['team']} {h['xg_5v5']:.2f}; on the power play {a['team']} {a['xg_pp']:.2f}, {h['team']} {h['xg_pp']:.2f}.")
    L.append(f"* **Adjusted for the score:** {a['team']} {a['xg_score_adj']:.2f}, {h['team']} {h['xg_score_adj']:.2f} (5v5: {a['xg_5v5_score_adj']:.2f} to {h['xg_5v5_score_adj']:.2f}). "
             "5v5 shots are re-weighted to what they would be worth at even score in the same period (league averages); the correction is small because at 5v5 the main effect is a trailing team's extra chances in the third period.")
    for r in (a, h):
        if r["goalie"]:
            L.append(f"* **{r['team']} goaltending:** {r['goalie']} faced {r['goalie_shots']} shots on goal; the unblocked attempts against were worth {r['goalie_xg_faced']:.2f} expected goals and {r['goalie_goals_against']} went in "
                     f"({r['goalie_saved_above_expected']:+.2f} goals saved above expected).")
    for ab in (a["team"], h["team"]):
        t = pt.filter(pl.col("team") == ab)
        best, worst = t.sort("net_adj", descending=True).row(0, named=True), t.sort("net_adj").row(0, named=True)
        sh, pv = t.sort("ixg", descending=True).row(0, named=True), t.sort("pm_epa", descending=True).row(0, named=True)
        L.append(f"* **{ab}:** best situation-adjusted net {best['name']} ({best['net_adj']:+.2f}; raw all situations {best['net_all']:+.2f}, 5v5 {best['on_xgd_5v5']:+.2f} in {best['toi_5v5']:.1f} min), lowest {worst['name']} ({worst['net_adj']:+.2f}); "
                 f"most dangerous shooter {sh['name']} ({sh['ixg']:.2f} ixG, {sh['goals']} {'goal' if sh['goals'] == 1 else 'goals'} on {sh['sog']} shots on goal); most puck value {pv['name']} ({pv['pm_epa']:+.2f}).")
    L += ["", "*One game is a small sample. These numbers describe what happened in this game; they are not ratings of the players. Expected goals use the shot model fitted on 2017–2025; puck value is the possession-value (PM-EPA) credit for the player's own puck actions.*", ""]
    return "\n".join(L)


def run(game_id: int, season: int = 2026) -> dict:
    live_config(season)
    d = load(game_id)
    g = d["g"]
    out = ROOT / "reports" / "games" / f"{g['game_date']}_{g['away_abbrev']}_at_{g['home_abbrev']}"
    ts, pt = team_summary(d), player_table(d)
    style.save(fig_flow(d, ts), out, "game_flow")
    style.save(fig_players(d, pt), out, "players")
    style.save(fig_impact(d, pt), out, "impact")
    pt.write_csv(out / "players.csv", float_precision=3)
    ts.write_csv(out / "teams.csv", float_precision=3)
    (out / "summary.md").write_text(takeaways(d, ts, pt), encoding="utf-8")
    log.info("report written to %s", out)
    return {"dir": out, "teams": ts, "players": pt}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2026)
