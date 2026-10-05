"""One-game summary (graphics, player table, takeaways):  python -m src.live.game_report <game_id> [season]

Reads the live tables written by src.live.update. Everything here is descriptive for one game: expected goals from the shot model,
possession value from the PM-EPA layer, and 5v5 on-ice expected goals. Season-level components (defense, finishing talent, special teams ratings) need many games and are not shown."""

from __future__ import annotations

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
    return {"g": g, "ev": ev, "sh": sh, "pg": pg, "names": names}


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


def player_table(d: dict) -> pl.DataFrame:
    pg = d["pg"].join(d["names"].select("player_id", "name"), on="player_id", how="left")
    # impact order: 5v5 on-ice expected-goal differential, best first (the model owner's preference; ice time stays as a column)
    return pg.select("team", "name", "position", (pl.col("toi_all") / 60).alias("toi"), "goals", (pl.col("a1") + pl.col("a2")).alias("assists"), "sog", "ixg", pl.col("pm_epa_total").alias("pm_epa"),
                     pl.col("oi_xgf").alias("on_xgf_5v5"), pl.col("oi_xga").alias("on_xga_5v5"), (pl.col("oi_xgf") - pl.col("oi_xga")).alias("on_xgd_5v5"), (pl.col("toi_5v5") / 60).alias("toi_5v5"),
                     pl.col("ixg_ev").alias("own_xg_5v5"), (pl.col("oi_xgf") - pl.col("ixg_ev")).clip(lower_bound=0.0).alias("mates_xg_5v5")).sort("team", "on_xgd_5v5", descending=[False, True])


def fig_impact(d: dict, pt: pl.DataFrame):
    """Stacked bars of each skater's 5v5 on-ice expected goals: his own shots and his linemates' shots (for), against (negative), with the net marked; sorted by net."""
    g = d["g"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6), sharex=True)
    lim = 1.3 * max(float(pt["on_xgf_5v5"].max()), float(pt["on_xga_5v5"].max()))             # room on the right for the net figure
    for ax, ab, col in ((axes[0], g["away_abbrev"], AWAY_C), (axes[1], g["home_abbrev"], HOME_C)):
        t = pt.filter(pl.col("team") == ab).sort("on_xgd_5v5")
        y = np.arange(t.height)
        own, mates, ag, net = (t[c].to_numpy() for c in ("own_xg_5v5", "mates_xg_5v5", "on_xga_5v5", "on_xgd_5v5"))
        ax.barh(y, own, color=col, height=0.72, edgecolor=style.SURFACE, linewidth=1.5)
        ax.barh(y, mates, left=own, color=col, alpha=0.45, height=0.72, edgecolor=style.SURFACE, linewidth=1.5)
        ax.barh(y, -ag, color="#9a9994", height=0.72, edgecolor=style.SURFACE, linewidth=1.5)
        ax.scatter(net, y, s=46, color=style.TEXT, zorder=6, edgecolor=style.SURFACE, linewidth=1.2)
        for yi, n in zip(y, net):
            ax.text(lim, yi, f"{n:+.2f}", va="center", ha="right", fontsize=8.5, color=style.TEXT, fontweight="bold")
        ax.set_yticks(y, [f"{n}  ({p})" for n, p in zip(t["name"].to_list(), t["position"].to_list())], fontsize=9)
        ax.axvline(0, color=style.TEXT2, lw=1)
        ax.set_xlim(-lim / 1.3 * 1.05, lim)
        ax.set_ylim(-0.6, t.height - 0.4)
        ax.grid(axis="y", visible=False)
        ax.set_title(ab, color=col, fontsize=13)
        ax.set_xlabel("expected goals at 5v5 while on the ice   (against  ←  →  for)")
    h = axes[0]
    h.barh([-9], [0], color=AWAY_C, label="his own shots")
    h.barh([-9], [0], color=AWAY_C, alpha=0.45, label="linemates' shots")
    h.barh([-9], [0], color="#9a9994", label="against")
    h.scatter([], [], s=46, color=style.TEXT, label="net (number at right)")
    fig.legend(*h.get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{g['away_abbrev']} at {g['home_abbrev']}, {g['game_date']}: net 5v5 impact by skater (sorted best to worst)", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
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
        ("on_xgf_5v5", "on-ice\nxGF", "{:.2f}", None), ("on_xga_5v5", "on-ice\nxGA", "{:.2f}", None), ("on_xgd_5v5", "on-ice\nxG diff", "{:+.2f}", "div")]


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
    fig.suptitle(f"{g['away_abbrev']} at {g['home_abbrev']}, {g['game_date']}: skaters (forwards, then defensemen, by 5v5 on-ice xG differential)", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.02, 0.015, "ixG: expected goals from the player's own shots.  Puck value: goals of expected value added by his credited puck actions (PM-EPA, all situations).  "
             "xGF/xGA on: expected goals for and against while he was on the ice at 5v5.\nOne game is a small sample: these describe what happened, not how good a player is.", fontsize=8.5, color=style.TEXT2, va="bottom")
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
        best, worst = t.sort("on_xgd_5v5", descending=True).row(0, named=True), t.sort("on_xgd_5v5").row(0, named=True)
        sh, pv = t.sort("ixg", descending=True).row(0, named=True), t.sort("pm_epa", descending=True).row(0, named=True)
        L.append(f"* **{ab}:** best 5v5 on-ice expected-goal differential {best['name']} ({best['on_xgd_5v5']:+.2f} in {best['toi_5v5']:.1f} min), lowest {worst['name']} ({worst['on_xgd_5v5']:+.2f}); "
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
