"""Extract real, representative possessions for docs + figures (deterministic selection).

Writes reports/metrics/example_plays.json (for the rink diagrams) and regenerates the section
between the GENERATED markers in docs/pm_epa/attribution_examples.md.
"""

from __future__ import annotations

import json
import logging
import re

import polars as pl

from src.config import ROOT, load_config

log = logging.getLogger(__name__)
BEGIN, END = "<!-- BEGIN GENERATED EXAMPLES -->", "<!-- END GENERATED EXAMPLES -->"


def _pick(cfg) -> list[tuple[str, int]]:
    proc = cfg.path(cfg.data.processed_dir)
    tr = pl.read_parquet(proc / "transitions.parquet").filter(
        (pl.col("season") == cfg.data.seasons[-1]) & (pl.col("strength") == "5v5")
    )
    poss = pl.read_parquet(proc / "poss_valued_epa.parquet").filter(~pl.col("excluded"))
    tr = tr.join(poss.select("poss_id", "end_type", "start_type", "G"), on="poss_id")
    goalies = pl.read_parquet(proc / "players.parquet").filter(pl.col("position") == "G")["player_id"].to_list()
    tr = tr.filter(
        ~pl.col("start_player").is_in(goalies).fill_null(False) & ~pl.col("end_player").is_in(goalies).fill_null(False)
    )
    picks = []

    def first(df: pl.DataFrame, title: str, sort: str, desc: bool = True):
        df = df.sort([sort, "poss_id"], descending=[desc, False])
        if df.height:
            picks.append((title, int(df["poss_id"][0])))

    def median_row(df: pl.DataFrame, title: str, col: str):
        df = df.sort([col, "poss_id"])
        if df.height:
            picks.append((title, int(df["poss_id"][df.height // 2])))

    first(
        tr.filter(
            (pl.col("seg_type") == "carry")
            & (pl.col("from_kind") == "takeaway")
            & (pl.col("x") < 0)
            & (pl.col("to_x") > 54)
        ),
        "Observed carry: defensive-half takeaway carried to a scoring-area shot by the same player",
        "delta",
    )
    median_row(
        tr.filter(
            (pl.col("seg_type") == "split")
            & (pl.col("from_kind") == "loose_opp_miss")
            & (pl.col("end_type") == "shot_sog")
            & (pl.col("to_x") > 69)
        ),
        "Unobserved transition: recovery after an opponent miss, shot from the low slot (median Δ)",
        "delta",
    )
    median_row(
        tr.filter(
            (pl.col("from_kind") == "faceoff")
            & (pl.col("x") > 25)
            & (pl.col("end_type") == "shot_block")
            & (pl.col("to_x") < 54)
        ),
        "Offensive-zone faceoff win → point shot blocked",
        "delta",
    )
    median_row(
        tr.filter(
            (pl.col("from_kind") == "takeaway")
            & (pl.col("to_kind") == "giveaway")
            & (pl.col("to_x") < -25)
            & (pl.col("x") > -89)
        ),
        "Takeaway followed by a defensive-zone giveaway (median Δ)",
        "delta",
    )
    median_row(
        tr.filter(
            (pl.col("seg_type") == "split")
            & (pl.col("from_kind") == "loose_opp_giveaway")
            & (pl.col("to_kind") == "shot")
        ),
        "Opponent giveaway converted into a shot",
        "delta",
    )
    return picks


def run() -> None:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    obs = pl.read_parquet(proc / "obs_valued.parquet")
    credits = pl.read_parquet(proc / "credits.parquet")
    poss = pl.read_parquet(proc / "poss_valued_epa.parquet")
    players = pl.read_parquet(proc / "players.parquet").select(
        "player_id", (pl.col("first_name") + " " + pl.col("last_name")).alias("name")
    )
    games = pl.read_parquet(proc / "games.parquet")
    names = dict(zip(players["player_id"], players["name"]))
    out, md = [], []
    for i, (title, pid) in enumerate(_pick(cfg), 1):
        o = obs.filter(pl.col("poss_id") == pid).sort("seq")
        p = poss.filter(pl.col("poss_id") == pid).row(0, named=True)
        g = games.filter(pl.col("game_id") == p["game_id"]).row(0, named=True)
        team = g["home_abbrev"] if p["team_id"] == g["home_id"] else g["away_abbrev"]
        c = credits.filter(pl.col("poss_id") == pid).with_columns(
            pl.col("player_id").replace_strict(names, default=None, return_dtype=pl.Utf8).alias("player")
        )
        pts = [
            {
                "x": r["x"],
                "y": r["y"],
                "v": r["v"],
                "kind": r["obs_kind"],
                "label": f"{r['obs_kind']}"
                + (f" ({names.get(r['player_id'], '?').split(' ')[-1]})" if r["player_id"] else ""),
            }
            for r in o.iter_rows(named=True)
        ]
        out.append({"title": f"Ex. R{i}: {title}", "points": pts})
        md.append(f"### R{i}. {title}\n")
        md.append(
            f"*{g['game_date']} {g['away_abbrev']} @ {g['home_abbrev']}, possession by {team}, period {p['period']}; "
            f"start `{p['start_type']}` → end `{p['end_type']}`; terminal xG G = {p['G']:.3f}.*\n"
        )
        md.append("| seq | observation | player | t (s) | x, y | V(s) |\n|---|---|---|---|---|---|")
        for r in o.iter_rows(named=True):
            xy = "—" if r["x"] is None else f"{r['x']:.0f}, {r['y']:.0f}"
            md.append(
                f"| {r['seq']} | `{r['obs_kind']}` | {names.get(r['player_id'], '—')} | {r['t']:.0f} | {xy} | {r['v']:.4f} |"
            )
        cc = c.group_by("player", "component").agg(pl.col("value").sum()).sort("component", "player")
        md.append("\n| credited to | component | value (xG) |\n|---|---|---|")
        for r in cc.iter_rows(named=True):
            md.append(f"| {r['player'] or '*team / unattributed*'} | `{r['component']}` | {r['value']:+.4f} |")
        tot = c["value"].sum()
        md.append(
            f"\nSum of all components = {tot:+.4f} (G = {p['G']:.4f}; they are equal when λ = 0, and differ by the λ-weighted opponent terms otherwise).\n"
        )
    (cfg.path(cfg.data.metrics_dir) / "example_plays.json").write_text(json.dumps(out, indent=2))
    doc = ROOT / "docs/pm_epa/attribution_examples.md"
    text = doc.read_text(encoding="utf-8")
    gen = f"{BEGIN}\n\n" + "\n".join(md) + f"\n{END}"
    doc.write_text(re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: gen, text, flags=re.S), encoding="utf-8")
    log.info("wrote %d real examples", len(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
