"""Projected rate for a player under a chosen destination team (context-aware projection).

    python -m src.teamcontext.move --player "Jason Robertson" --team TOR [--team NYR ...]

Fits the projection ridge with the context features (docs/teamcontext/methodology.md §3C) on every available transition and predicts the next
season for a player who played the latest season, with his destination team's latest-season context (leave-one-out) as ``ctx_dest`` and his
current team's as ``ctx_origin``. ``--team`` may be repeated. The effect of context is modest (about a fifth of the difference between two
teams' prior-season mean rates passes through), it bundles coaching, system, linemates and arena, and it is not player-specific. The projected change splits into a
context effect (destination context minus current context) and an average move effect (the coefficient on the mover flag, identical for every
destination: movers are on average projected differently from stayers, through selection and adjustment).
"""

from __future__ import annotations

import argparse

import polars as pl

from src.config import load_config
from src.projection import data as pd_
from src.projection import model as pm
from src.projection import run as pr
from src.projection.model import Spec
from src.teamcontext.data import add_context

SPEC = Spec(extra=("ctx_dest", "ctx_origin", "mover"))


def team_contexts(p: pl.DataFrame, season: int) -> pl.DataFrame:
    """Minute-weighted mean total value rate of each team's skaters in ``season`` (with the sums, for leave-one-out)."""
    return p.filter(pl.col("season") == season).group_by("team").agg((pl.col("m") * pl.col("r_total")).sum().alias("S"), pl.col("m").sum().alias("M")).with_columns((pl.col("S") / pl.col("M")).alias("ctx"))


def project(dest_teams: list[str], player: str) -> pl.DataFrame:
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    p = pd_.panel(proc, pl.read_parquet(proc / "wm_rolling_values.parquet"))
    bio = pd_.player_bio(proc, cfg.path(cfg.data.raw_dir))
    p2 = add_context(p)
    ft = pd_.design_frame(p2, bio, 1, floor=pr.REVISED_FLOOR)
    models = pm.fit_models(ft, SPEC)
    last = int(p["season"].max())
    names = pl.read_parquet(proc / "players.parquet").select(pl.col("player_id").cast(pl.Int64), (pl.col("first_name") + " " + pl.col("last_name")).alias("name"))
    who = names.filter(pl.col("name").str.to_lowercase() == player.lower())
    if who.height != 1:
        raise SystemExit(f"player {player!r} not found uniquely")
    pid = who["player_id"][0]
    fut = pr.future_frame(p, bio, last + 1, 1).filter(pl.col("player_id") == pid)
    if fut.height == 0:
        raise SystemExit(f"{player} did not play in {last}")
    tc = team_contexts(p, last)
    own = p.filter((pl.col("season") == last) & (pl.col("player_id") == pid)).select("team", "m", "r_total").row(0, named=True)
    own_s, own_m = own["m"] * own["r_total"], own["m"]

    def ctx_for(team: str) -> float:
        r = tc.filter(pl.col("team") == team).row(0, named=True)
        s, m = r["S"] - (own_s if team == own["team"] else 0.0), r["M"] - (own_m if team == own["team"] else 0.0)
        return s / m

    coef = dict(zip(pm.feature_names(SPEC), models[fut["pos_group"][0]].raw_coef().tolist()))
    rows = []
    for team in [own["team"], *dest_teams]:
        moving = team != own["team"]
        d = fut.with_columns(pl.lit(ctx_for(team)).alias("ctx_dest"), pl.lit(ctx_for(own["team"])).alias("ctx_origin"), pl.lit(float(team != own["team"])).alias("mover"))
        rate = float(pm.predict_models(models, d, SPEC)[0])
        rows.append({"player": player, "team": team, "is_current_team": team == own["team"], "team_context_per60": ctx_for(team), "projected_rate_per60": rate,
                     "context_effect_per60": coef["ctx_dest"] * (ctx_for(team) - ctx_for(own["team"])) if moving else 0.0, "average_move_effect_per60": coef["mover"] if moving else 0.0})
    out = pl.DataFrame(rows)
    return out.with_columns((pl.col("projected_rate_per60") - out["projected_rate_per60"][0]).alias("change_vs_current_team_per60"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--team", action="append", required=True, help="destination team abbreviation (repeatable)")
    a = ap.parse_args()
    with pl.Config(tbl_rows=20, tbl_cols=10, fmt_str_lengths=40):
        print(project([t.upper() for t in a.team], a.player))


if __name__ == "__main__":
    main()
