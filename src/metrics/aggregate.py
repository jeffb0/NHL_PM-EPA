"""Player / team / game aggregation of PM-EPA, rate stats, bootstrap CIs and leaderboards.

Rates are per 60 minutes of the matching strength's TOI from official shift charts.
Games without usable shift data (missing chart or <90% of 5v5 events with 5+5 skaters
matched) are dropped from *both* numerator and denominator.
"EV" in this project means 5v5 with both goalies in net.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.config import load_config
from src.epa.engine import DIRECT, PM_COMPONENTS, POOLED

log = logging.getLogger(__name__)

SG_OF = {"5v5": "ev", "PP": "pp", "PK": "pk", "4v4": "4v4_3v3", "3v3": "4v4_3v3"}
POS_GROUP = {"C": "F", "L": "F", "R": "F", "D": "D", "G": "G"}


def good_games(proc) -> pl.DataFrame:
    q = pl.read_parquet(proc / "shift_quality.parquet")
    return q.filter(pl.col("onice_match_rate") >= 0.9).select("game_id")


MOVEMENT = ["carry", "move_start", "move_end", "move_pool"]


def pm_components(include_turnovers: bool = True) -> list[str]:
    """Components summed into PM-EPA (config ``epa.include_turnovers``)."""
    return PM_COMPONENTS if include_turnovers else MOVEMENT


def player_game(credits: pl.DataFrame, proc, include_turnovers: bool = True) -> pl.DataFrame:
    """PM-EPA by player-game, split by strength group and direct/pooled; plus play counts."""
    comps = pm_components(include_turnovers)
    ok = good_games(proc)
    c = credits.filter(pl.col("player_id").is_not_null()).join(ok, on="game_id")
    c = c.with_columns(pl.col("strength").replace_strict(SG_OF, default="other").alias("sg"))
    pm = c.filter(pl.col("component").is_in(comps))
    aggs = [
        pl.col("value").sum().alias("pm_epa_total"),
        pl.col("value").filter(pl.col("component").is_in(DIRECT)).sum().alias("pm_direct_total"),
        pl.col("value").filter(pl.col("component").is_in(POOLED)).sum().alias("pm_pooled_total"),
    ]
    for sg in ["ev", "pp", "pk", "4v4_3v3"]:
        aggs.append(pl.col("value").filter(pl.col("sg") == sg).sum().alias(f"pm_epa_{sg}"))
    for sg in ["ev"]:
        aggs.append(pl.col("value").filter((pl.col("sg") == sg) & pl.col("component").is_in(DIRECT)).sum().alias(f"pm_direct_{sg}"))
        aggs.append(pl.col("value").filter((pl.col("sg") == sg) & pl.col("component").is_in(POOLED)).sum().alias(f"pm_pooled_{sg}"))
    for comp in PM_COMPONENTS:
        aggs.append(pl.col("value").filter((pl.col("component") == comp) & (pl.col("sg") == "ev")).sum().alias(f"ev_{comp}"))
    direct = pm.filter(pl.col("component").is_in(DIRECT))
    plays = direct.group_by("player_id", "game_id").agg(
        pl.len().alias("n_plays"), (pl.col("value") > 0).sum().alias("plays_pos"), (pl.col("value") < 0).sum().alias("plays_neg"),
        pl.col("value").sum().alias("_sum_plays"), (pl.col("value") ** 2).sum().alias("_sumsq_plays"),
    )
    poss = pm.group_by("player_id", "game_id").agg(pl.col("poss_id").n_unique().alias("n_possessions_involved"))
    sd = c.filter(pl.col("component") == "shot_quality").group_by("player_id", "game_id").agg(
        pl.col("value").sum().alias("shot_quality_total"),
        pl.col("value").filter(pl.col("sg") == "ev").sum().alias("shot_quality_ev"))
    out = pm.group_by("player_id", "game_id").agg(aggs)
    return out.join(plays, on=["player_id", "game_id"], how="left").join(poss, on=["player_id", "game_id"], how="left").join(
        sd, on=["player_id", "game_id"], how="full", coalesce=True)


def boxscore_game(proc) -> pl.DataFrame:
    """Individual + on-ice (5v5) counting stats per player-game for validation baselines."""
    ev = pl.read_parquet(proc / "events.parquet", columns=["game_id", "event_idx", "type", "team_id", "player1", "assist1", "assist2", "strength"])
    shots = pl.read_parquet(proc / "shots_xg.parquet").select("game_id", "event_idx", "xg")
    ev = ev.join(shots, on=["game_id", "event_idx"], how="left")
    sh = ev.filter(pl.col("type").is_in(["shot-on-goal", "missed-shot", "goal"]))
    ind = sh.group_by(pl.col("player1").alias("player_id"), "game_id").agg(
        (pl.col("type") == "goal").sum().alias("goals"),
        ((pl.col("type") == "goal") & (pl.col("strength") == "5v5")).sum().alias("goals_ev"),
        pl.col("type").is_in(["shot-on-goal", "goal"]).sum().alias("sog"),
        pl.len().alias("iff"),
        (pl.col("strength") == "5v5").sum().alias("iff_ev"),
        pl.col("xg").sum().alias("ixg"),
        pl.col("xg").filter(pl.col("strength") == "5v5").sum().alias("ixg_ev"),
    )
    g = ev.filter(pl.col("type") == "goal")
    a1 = g.group_by(pl.col("assist1").alias("player_id"), "game_id").agg(pl.len().alias("a1"), (pl.col("strength") == "5v5").sum().alias("a1_ev"))
    a2 = g.group_by(pl.col("assist2").alias("player_id"), "game_id").agg(pl.len().alias("a2"), (pl.col("strength") == "5v5").sum().alias("a2_ev"))
    tv = ev.filter(pl.col("type").is_in(["giveaway", "takeaway"])).group_by(pl.col("player1").alias("player_id"), "game_id").agg(
        ((pl.col("type") == "giveaway") & (pl.col("strength") == "5v5")).sum().alias("gv_ev"),
        ((pl.col("type") == "takeaway") & (pl.col("strength") == "5v5")).sum().alias("tk_ev"))
    # on-ice 5v5 for / against
    oi = pl.read_parquet(proc / "onice.parquet")
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "home_id")
    s5 = sh.filter(pl.col("strength") == "5v5").join(oi, on=["game_id", "event_idx"]).join(games, on="game_id")
    s5 = s5.with_columns(
        pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.col("home_on")).otherwise(pl.col("away_on")).alias("for_on"),
        pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.col("away_on")).otherwise(pl.col("home_on")).alias("against_on"),
        (pl.col("type") == "goal").cast(pl.Int32).alias("g"), pl.col("type").is_in(["shot-on-goal", "goal"]).cast(pl.Int32).alias("sog1"),
    )
    f = s5.explode("for_on").group_by(pl.col("for_on").alias("player_id"), "game_id").agg(
        pl.col("xg").sum().alias("oi_xgf"), pl.col("g").sum().alias("oi_gf"), pl.col("sog1").sum().alias("oi_sogf"), pl.len().alias("oi_ff"))
    a = s5.explode("against_on").group_by(pl.col("against_on").alias("player_id"), "game_id").agg(
        pl.col("xg").sum().alias("oi_xga"), pl.col("g").sum().alias("oi_ga"), pl.col("sog1").sum().alias("oi_soga"))
    out = f.join(a, on=["player_id", "game_id"], how="full", coalesce=True)
    for d in (ind, a1, a2, tv):
        out = out.join(d.filter(pl.col("player_id").is_not_null()), on=["player_id", "game_id"], how="full", coalesce=True)
    return out.fill_null(0)


def player_game_table(credits: pl.DataFrame, proc, include_turnovers: bool = True) -> pl.DataFrame:
    toi = pl.read_parquet(proc / "toi_game.parquet").join(good_games(proc), on="game_id")
    games = pl.read_parquet(proc / "games.parquet").select("game_id", "season", "home_id", "home_abbrev", "away_abbrev")
    pg = toi.join(player_game(credits, proc, include_turnovers), on=["player_id", "game_id"], how="left")
    pg = pg.join(boxscore_game(proc), on=["player_id", "game_id"], how="left").join(games, on="game_id")
    num = [c for c, t in pg.schema.items() if t in (pl.Float64, pl.Int64, pl.UInt32, pl.Int32) and c not in ("player_id", "game_id", "team_id", "season", "home_id")]
    return pg.with_columns([pl.col(c).fill_null(0) for c in num]).with_columns(
        pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.col("home_abbrev")).otherwise(pl.col("away_abbrev")).alias("team"))


def player_season(pg: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    skip = {"player_id", "game_id", "team_id", "season", "home_id", "home_abbrev", "away_abbrev", "team"}
    sums = [pl.col(c).sum() for c, t in pg.schema.items() if c not in skip and t.is_numeric()]
    team = pg.group_by("player_id", "season", "team").agg(pl.col("toi_all").sum().alias("_t")).sort("_t", descending=True).group_by(
        "player_id", "season").agg(pl.col("team").str.join("/").alias("team"), pl.col("team").first().alias("primary_team"))
    ps = pg.group_by("player_id", "season").agg([pl.col("game_id").n_unique().alias("gp"), *sums]).join(team, on=["player_id", "season"])
    ps = ps.join(players.select("player_id", (pl.col("first_name") + " " + pl.col("last_name")).alias("player_name"), "position"), on="player_id", how="left")
    ps = ps.with_columns(pl.col("position").replace_strict(POS_GROUP, default="F").alias("pos_group"))
    m = lambda c: pl.col(c) / 60.0  # noqa: E731
    ps = ps.with_columns(
        m("toi_all").alias("toi_total"), m("toi_5v5").alias("toi_ev"), m("toi_PP").alias("toi_pp"), m("toi_PK").alias("toi_pk"),
        (m("toi_4v4") + m("toi_3v3")).alias("toi_4v4_3v3"),
    )
    rate = lambda num, den: pl.when(pl.col(den) > 0).then(pl.col(num) * 60 / pl.col(den)).otherwise(None)  # noqa: E731
    ps = ps.with_columns(
        rate("pm_epa_total", "toi_total").alias("pm_epa_per60_total"),
        rate("pm_epa_ev", "toi_ev").alias("pm_epa_per60_ev"),
        rate("pm_epa_pp", "toi_pp").alias("pm_epa_per60_pp"),
        rate("pm_epa_pk", "toi_pk").alias("pm_epa_per60_pk"),
        rate("pm_direct_ev", "toi_ev").alias("pm_direct_per60_ev"),
        rate("pm_pooled_ev", "toi_ev").alias("pm_pooled_per60_ev"),
        pl.when(pl.col("n_possessions_involved") > 0).then(100 * pl.col("pm_epa_total") / pl.col("n_possessions_involved")).alias("pm_epa_per100_poss"),
        pl.when(pl.col("n_plays") > 0).then(pl.col("_sum_plays") / pl.col("n_plays")).alias("mean_epa_per_play"),
        pl.when(pl.col("n_plays") > 1).then(((pl.col("_sumsq_plays") - pl.col("_sum_plays") ** 2 / pl.col("n_plays")) / (pl.col("n_plays") - 1)).clip(0, None).sqrt()).alias("std_epa_per_play"),
        rate("ixg_ev", "toi_ev").alias("ixg_per60_ev"),
        rate("goals_ev", "toi_ev").alias("g_per60_ev"),
        ((pl.col("goals_ev") + pl.col("a1_ev")) * 60 / pl.col("toi_ev")).alias("p1_per60_ev"),
        ((pl.col("goals_ev") + pl.col("a1_ev") + pl.col("a2_ev")) * 60 / pl.col("toi_ev")).alias("pts_per60_ev"),
        (pl.col("a1_ev") * 60 / pl.col("toi_ev")).alias("a1_per60_ev"),
        rate("oi_xgf", "toi_ev").alias("oi_xgf_per60_ev"),
        rate("oi_xga", "toi_ev").alias("oi_xga_per60_ev"),
        (pl.col("oi_xgf") / (pl.col("oi_xgf") + pl.col("oi_xga"))).alias("oi_xgf_pct_ev"),
        (pl.col("oi_gf") / pl.col("oi_sogf")).alias("oi_sh_pct_ev"),
        (pl.col("goals") / pl.col("sog")).alias("ind_sh_pct"),
        (pl.col("goals") - pl.col("ixg")).alias("g_minus_xg"),
        (1 - pl.col("oi_ga") / pl.col("oi_soga")).alias("oi_sv_pct_ev"),
        rate("gv_ev", "toi_ev").alias("gv_per60_ev"),
        rate("tk_ev", "toi_ev").alias("tk_per60_ev"),
        rate("shot_quality_ev", "toi_ev").alias("shot_quality_per60_ev"),
    )
    return ps.with_columns((pl.col("oi_sh_pct_ev") + pl.col("oi_sv_pct_ev")).alias("oi_pdo_ev"))


def bootstrap_ci(pg: pl.DataFrame, num: str, den: str, reps: int, level: float, seed: int) -> pl.DataFrame:
    """Game-level (block) bootstrap of a per-60 rate for each player-season."""
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    d = pg.filter(pl.col(den) > 0).select("player_id", "season", num, den).sort("player_id", "season")
    for (pid, season), g in d.group_by(["player_id", "season"], maintain_order=True):
        x, t = g[num].to_numpy(), g[den].to_numpy()
        k = len(x)
        idx = rng.integers(0, k, size=(reps, k))
        r = x[idx].sum(1) * 3600 / np.maximum(t[idx].sum(1), 1e-9)
        rows.append((pid, season, float(np.quantile(r, lo_q)), float(np.quantile(r, hi_q)), float(r.std())))
    return pl.DataFrame(rows, schema=["player_id", "season", "uncertainty_ci_low_per60", "uncertainty_ci_high_per60", "boot_se_per60"], orient="row")


LEADERBOARD_COLS = [
    "player_id", "player_name", "team", "position", "season", "gp", "toi_total", "toi_ev", "toi_pp", "toi_pk",
    "pm_epa_total", "pm_epa_ev", "pm_epa_pp", "pm_epa_pk",
    "pm_epa_per60_total", "pm_epa_per60_ev", "pm_epa_per60_pp", "pm_epa_per60_pk",
    "pm_epa_per100_poss", "n_plays", "n_possessions_involved", "plays_pos", "plays_neg", "mean_epa_per_play", "std_epa_per_play",
    "percentile_ev", "uncertainty_ci_low_per60", "uncertainty_ci_high_per60",
]
EXTRA_COLS = ["pos_group", "pm_direct_per60_ev", "pm_pooled_per60_ev", "pm_epa_per60_ev_lam1", "shot_quality_per60_ev", "qualified_ev", "boot_se_per60"]


def run(credits_path: str = "credits.parquet") -> None:
    cfg = load_config()
    proc, out = cfg.path(cfg.data.processed_dir), cfg.path(cfg.data.output_dir)
    credits = pl.read_parquet(proc / credits_path)
    players = pl.read_parquet(proc / "players.parquet")
    pg = player_game_table(credits, proc, cfg.epa.include_turnovers)
    pg = pg.join(players.select("player_id", "position"), on="player_id", how="left").filter(pl.col("position") != "G").drop("position")
    pg.write_parquet(proc / "player_game.parquet")
    ps = player_season(pg, players)
    ci = bootstrap_ci(pg, "pm_epa_ev", "toi_5v5", cfg.metrics.bootstrap_reps, cfg.metrics.ci_level, cfg.seed)
    ps = ps.join(ci, on=["player_id", "season"], how="left")
    ps = ps.with_columns((pl.col("toi_ev") >= cfg.metrics.min_toi_ev).alias("qualified_ev"))
    pct = ps.filter(pl.col("qualified_ev")).with_columns(
        (pl.col("pm_epa_per60_ev").rank("average").over("season", "pos_group") / pl.len().over("season", "pos_group") * 100).round(1).alias("percentile_ev")
    ).select("player_id", "season", "percentile_ev")
    ps = ps.join(pct, on=["player_id", "season"], how="left")
    lam1 = proc / "credits_lam1.parquet"
    if lam1.exists():  # risk-adjusted companion (lambda = 1): own-zone giveaways also cost the opponent's gain
        c1 = pl.read_parquet(lam1).filter(pl.col("player_id").is_not_null() & (pl.col("strength") == "5v5")
                                          & pl.col("component").is_in(pm_components(cfg.epa.include_turnovers)))
        c1 = c1.join(good_games(proc), on="game_id").group_by("player_id", "season").agg(pl.col("value").sum().alias("_pm1"))
        ps = ps.join(c1, on=["player_id", "season"], how="left").with_columns(
            (pl.col("_pm1").fill_null(0) * 60 / pl.col("toi_ev")).alias("pm_epa_per60_ev_lam1")).drop("_pm1")
    else:
        ps = ps.with_columns(pl.lit(None, pl.Float64).alias("pm_epa_per60_ev_lam1"))
    ps.write_parquet(proc / "player_season.parquet")
    ps.select(LEADERBOARD_COLS + EXTRA_COLS).sort("season", "pm_epa_per60_ev", descending=[False, True]).write_csv(out / "pm_epa_player_leaderboards.csv", float_precision=5)

    # team-season and team-game
    c = credits.join(pl.read_parquet(proc / "games.parquet").select("game_id", "home_id", "home_abbrev", "away_abbrev"), on="game_id")
    c = c.with_columns(pl.when(pl.col("team_id") == pl.col("home_id")).then(pl.col("home_abbrev")).otherwise(pl.col("away_abbrev")).alias("team"))
    tg = c.group_by("season", "game_id", "team").agg(
        pl.col("value").filter(pl.col("component").is_in(PM_COMPONENTS)).sum().alias("pm_epa_attributed"),
        pl.col("value").filter(pl.col("component").is_in(DIRECT)).sum().alias("pm_epa_direct"),
        pl.col("value").filter(pl.col("component").is_in(POOLED)).sum().alias("pm_epa_pooled"),
        pl.col("value").filter(pl.col("component") == "unattributed").sum().alias("unattributed"),
        pl.col("value").filter(pl.col("component") == "shot_quality").sum().alias("shot_quality"),
        pl.col("value").filter((pl.col("component").is_in(PM_COMPONENTS)) & (pl.col("strength") == "5v5")).sum().alias("pm_epa_ev"),
    )
    tg.sort("season", "game_id", "team").write_csv(out / "pm_epa_team_game.csv", float_precision=5)
    tg.group_by("season", "team").agg(pl.len().alias("gp"), pl.exclude("game_id", "season", "team").sum()).sort("season", "pm_epa_attributed", descending=[False, True]).write_csv(
        out / "pm_epa_team_season.csv", float_precision=5)
    pg.select("player_id", "game_id", "season", "team", "toi_5v5", "toi_all", "pm_epa_total", "pm_epa_ev", "pm_direct_ev", "pm_pooled_ev", "n_plays").write_csv(
        out / "pm_epa_player_game.csv", float_precision=5)
    log.info("player-seasons=%d qualified(ev>=%d)=%d", ps.height, cfg.metrics.min_toi_ev, ps.filter(pl.col("qualified_ev")).height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
