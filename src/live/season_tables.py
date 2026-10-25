"""Season-to-date tables for the dashboard:  python -m src.live.season_tables [season]

Reads the live tables written by src.live.update and writes five CSVs to data/live/outputs/dashboard/:
skaters.csv (one row per skater), skater_games.csv (one row per skater and game), teams.csv, team_games.csv and
goalies.csv, plus season.csv (one row: games played, date of the last game, league rates).

The skater card components are the game report's (src.live.game_report.adjusted), summed over games, with the league
rates for the season to date."""

from __future__ import annotations

import logging
import sys

import polars as pl

from src.config import load_config
from src.live.game_report import CARD, UNBLOCKED, adjusted, league_rates, on_ice, penalties, penalty_costs
from src.live.update import live_config

log = logging.getLogger("live")
COMPONENTS = [c for c, _, _ in CARD]
# league average skaters on the ice: a team's power-play minutes are its skaters' power-play minutes over five, its
# penalty-kill minutes its skaters' over four
PP_SKATERS, PK_SKATERS = 5, 4


def load(proc) -> dict:
    games = pl.read_parquet(proc / "games.parquet")
    sh = pl.read_parquet(proc / "shots_xg.parquet").filter(pl.col("type").is_in(UNBLOCKED))
    oi = pl.read_parquet(proc / "onice.parquet")
    obs = pl.read_parquet(
        proc / "obs_valued.parquet",
        columns=["poss_id", "game_id", "seq", "team_id", "strength", "v", "onice", "opp_onice"],
    )
    poss = pl.read_parquet(proc / "poss_valued_epa.parquet", columns=["poss_id", "game_id", "end_class"])
    pg = pl.read_parquet(proc / "player_game.parquet")
    ev = pl.read_parquet(
        proc / "events.parquet",
        columns=[
            "game_id",
            "period",
            "t_period",
            "type",
            "team_id",
            "player1",
            "player2",
            "penalty_type",
            "penalty_min",
        ],
    )
    return {"games": games, "sh": sh, "oi": oi, "obs": obs, "poss": poss, "pg": pg, "ev": ev}


def team_names(games: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        [
            games.select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")),
            games.select(pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team")),
        ]
    ).unique("team_id")


def skater_games(d: dict, rates: dict[str, float]) -> pl.DataFrame:
    """One row per skater and game: box score, puck value (PM-EPA) and the seven card components."""
    g = d["games"]
    sit = on_ice(d["sh"], d["oi"], d["obs"], d["poss"], g.select("game_id", "home_id"))
    pen = penalties(d["ev"], penalty_costs())
    t = (
        d["pg"]
        .filter(pl.col("position") != "G")
        .join(sit, on=["game_id", "player_id"], how="left")
        .with_columns(pl.col(sit.columns[2:]).fill_null(0.0))
        .join(pen, on=["game_id", "player_id"], how="left")
        .with_columns(pl.col("pen_taken", "pen_drawn").fill_null(0), pl.col("pen").fill_null(0.0))
        .join(g.select("game_id", "game_date"), on="game_id")
        .with_columns(
            pl.when(pl.col("team") == pl.col("home_abbrev"))
            .then(pl.col("away_abbrev"))
            .otherwise("@" + pl.col("home_abbrev"))
            .alias("opponent"),
            pl.col("toi_5v5").alias("toi_5v5_s"),
        )
    )
    t = adjusted(t, rates)
    return t.select(
        "game_id",
        "game_date",
        "player_id",
        (pl.col("first_name") + " " + pl.col("last_name")).alias("name"),
        "team",
        pl.when(pl.col("position") == "D").then(pl.lit("D")).otherwise(pl.lit("F")).alias("pos"),
        "opponent",
        (pl.col("toi_all") / 60).alias("toi"),
        ((pl.col("toi_5v5") + pl.col("toi_4v4") + pl.col("toi_3v3")) / 60).alias("toi_es"),
        (pl.col("toi_PP") / 60).alias("toi_pp"),
        (pl.col("toi_PK") / 60).alias("toi_pk"),
        pl.col("goals").cast(pl.Int64),
        (pl.col("a1") + pl.col("a2")).cast(pl.Int64).alias("assists"),
        pl.col("sog").cast(pl.Int64),
        "ixg",
        pl.col("pm_epa_total").alias("pm_epa"),
        "pm_epa_ev",
        "pm_epa_pp",
        "pm_epa_pk",
        pl.col("oi_xgf").alias("xgf_5v5"),
        pl.col("oi_xga").alias("xga_5v5"),
        "pen_drawn",
        "pen_taken",
        *COMPONENTS,
        "net_adj",
    )


def per60(df: pl.DataFrame, cols: list[str], minutes: str) -> list[pl.Expr]:
    return [
        pl.when(pl.col(minutes) > 0).then(pl.col(c) * 60 / pl.col(minutes)).otherwise(None).alias(f"{c}_60")
        for c in cols
    ]


def skaters(sg: pl.DataFrame) -> pl.DataFrame:
    """Season totals per skater, with rates per 60 minutes (each component per 60 of its own situation's minutes)."""
    s = (
        sg.sort("game_date")
        .group_by("player_id")
        .agg(
            pl.col("name").last(),
            pl.col("team").last(),
            pl.col("pos").last(),
            pl.len().alias("gp"),
            *[
                pl.col(c).sum()
                for c in [
                    "toi",
                    "toi_es",
                    "toi_pp",
                    "toi_pk",
                    "goals",
                    "assists",
                    "sog",
                    "ixg",
                    "pm_epa",
                    "pm_epa_ev",
                    "pm_epa_pp",
                    "pm_epa_pk",
                    "xgf_5v5",
                    "xga_5v5",
                    "pen_drawn",
                    "pen_taken",
                    *COMPONENTS,
                    "net_adj",
                ]
            ],
        )
        .with_columns(
            (pl.col("goals") + pl.col("assists")).alias("points"),
            (pl.col("goals") - pl.col("ixg")).alias("gax"),
            pl.when(pl.col("xgf_5v5") + pl.col("xga_5v5") > 0)
            .then(pl.col("xgf_5v5") / (pl.col("xgf_5v5") + pl.col("xga_5v5")))
            .alias("xg_share_5v5"),
        )
    )
    s = s.with_columns(
        *per60(s, ["net_adj", "pm_epa", "ixg", "gax", "goals", "points", "pen"], "toi"),
        # each component per 60 of all his minutes: these add up to net_adj_60 (for stacked bars)
        *[pl.when(pl.col("toi") > 0).then(pl.col(c) * 60 / pl.col("toi")).alias(f"{c}_60toi") for c in COMPONENTS],
        *per60(s, ["ev_off", "ev_def", "pm_epa_ev"], "toi_es"),
        *per60(s, ["pp_off", "pp_def", "pm_epa_pp"], "toi_pp"),
        *per60(s, ["pk_off", "pk_def", "pm_epa_pk"], "toi_pk"),
        (pl.col("toi") / pl.col("gp")).alias("toi_gp"),
    )
    return s.sort("net_adj", descending=True)


def team_games(d: dict, sg: pl.DataFrame) -> pl.DataFrame:
    """One row per team and game: result, expected goals by situation, power-play and penalty-kill minutes."""
    g = d["games"]
    sides = pl.concat(
        [
            g.select(
                "game_id",
                "game_date",
                pl.col("home_id").alias("team_id"),
                pl.col("away_id").alias("opp_id"),
                pl.col("home_abbrev").alias("team"),
                pl.col("away_abbrev").alias("opponent"),
                pl.col("home_final").alias("gf"),
                pl.col("away_final").alias("ga"),
                "last_period_type",
            ),
            g.select(
                "game_id",
                "game_date",
                pl.col("away_id").alias("team_id"),
                pl.col("home_id").alias("opp_id"),
                pl.col("away_abbrev").alias("team"),
                ("@" + pl.col("home_abbrev")).alias("opponent"),
                pl.col("away_final").alias("gf"),
                pl.col("home_final").alias("ga"),
                "last_period_type",
            ),
        ]
    )
    sh = d["sh"].filter(~pl.col("empty_net") & (pl.col("strength") != "EN"))
    by = sh.group_by("game_id", "team_id").agg(
        pl.col("xg").sum().alias("xg"),
        pl.col("xg").filter(pl.col("strength") == "5v5").sum().alias("xg_5v5"),
        pl.col("xg").filter(pl.col("strength") == "PP").sum().alias("xg_pp"),
        pl.col("goal").sum().cast(pl.Int64).alias("goals_unblocked"),
    )
    mins = sg.group_by("game_id", "team").agg(
        (pl.col("toi_pp").sum() / PP_SKATERS).alias("pp_min"),
        (pl.col("toi_pk").sum() / PK_SKATERS).alias("pk_min"),
        pl.col("pm_epa").sum().alias("pm_epa"),
        pl.col("net_adj").sum().alias("net_adj"),
    )
    f = by.rename({"xg": "xgf", "xg_5v5": "xgf_5v5", "xg_pp": "xgf_pp"}).drop("goals_unblocked")
    a = by.select(
        "game_id",
        pl.col("team_id").alias("opp_id"),
        pl.col("xg").alias("xga"),
        pl.col("xg_5v5").alias("xga_5v5"),
        # chances the opponent had on its power play are this team's penalty kill
        pl.col("xg_pp").alias("xga_pk"),
    )
    t = (
        sides.join(f, on=["game_id", "team_id"], how="left")
        .join(a, on=["game_id", "opp_id"], how="left")
        .join(mins, on=["game_id", "team"], how="left")
        .with_columns(pl.col("xgf", "xgf_5v5", "xgf_pp", "xga", "xga_5v5", "xga_pk").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("gf") > pl.col("ga"))
            .then(pl.lit("W"))
            .when(pl.col("last_period_type") == "REG")
            .then(pl.lit("L"))
            .otherwise(pl.lit("OTL"))
            .alias("result"),
        )
    )
    return t.drop("team_id", "opp_id").sort("game_date", "team")


def teams(tg: pl.DataFrame, gl: pl.DataFrame) -> pl.DataFrame:
    t = tg.group_by("team").agg(
        pl.len().alias("gp"),
        (pl.col("result") == "W").sum().alias("w"),
        (pl.col("result") == "L").sum().alias("l"),
        (pl.col("result") == "OTL").sum().alias("otl"),
        *[
            pl.col(c).sum()
            for c in ["gf", "ga", "xgf", "xga", "xgf_5v5", "xga_5v5", "xgf_pp", "xga_pk", "pp_min", "pk_min", "pm_epa"]
        ],
        pl.col("net_adj").sum().alias("skater_net_adj"),
    )
    t = t.with_columns(
        (2 * pl.col("w") + pl.col("otl")).alias("pts"),
        (pl.col("xgf") / (pl.col("xgf") + pl.col("xga"))).alias("xg_share"),
        (pl.col("xgf_5v5") / (pl.col("xgf_5v5") + pl.col("xga_5v5"))).alias("xg_share_5v5"),
        (pl.col("xgf") / pl.col("gp")).alias("xgf_gp"),
        (pl.col("xga") / pl.col("gp")).alias("xga_gp"),
        (pl.col("gf") - pl.col("ga")).alias("gd"),
        (pl.col("xgf") - pl.col("xga")).alias("xgd"),
        pl.when(pl.col("pp_min") > 0).then(pl.col("xgf_pp") * 60 / pl.col("pp_min")).alias("xgf_pp_60"),
        pl.when(pl.col("pk_min") > 0).then(pl.col("xga_pk") * 60 / pl.col("pk_min")).alias("xga_pk_60"),
    )
    gsax = gl.group_by("team").agg(pl.col("gsax").sum())
    t = t.join(gsax, on="team", how="left").with_columns(pl.col("gsax").fill_null(0.0))
    return t.with_columns(
        (pl.col("pts") / (2 * pl.col("gp"))).alias("pts_pct"),
        (pl.col("w").cast(pl.String) + "-" + pl.col("l").cast(pl.String) + "-" + pl.col("otl").cast(pl.String)).alias(
            "record"
        ),
    ).sort("pts", "xg_share", descending=True)


def goalies(d: dict, proc) -> pl.DataFrame:
    """Per goalie: every unblocked attempt against while he was in net (empty net excluded), its expected goals and the
    goals that went in. The goalie in net is the one on a shift for the defending team at the shot's time."""
    g = d["games"]
    names = pl.read_parquet(proc / "players.parquet").filter(pl.col("position") == "G")
    shifts = (
        pl.read_parquet(proc / "shifts.parquet")
        .join(names.select("player_id"), on="player_id")
        .rename({"team_id": "def_id"})
    )
    ev = pl.read_parquet(proc / "events.parquet", columns=["game_id", "event_idx", "period", "t_period"])
    sh = (
        d["sh"]
        .filter(~pl.col("empty_net") & (pl.col("strength") != "EN"))
        .join(ev, on=["game_id", "event_idx"])
        .join(g.select("game_id", "home_id", "away_id"), on="game_id")
        .with_columns(
            pl.when(pl.col("team_id") == pl.col("home_id"))
            .then(pl.col("away_id"))
            .otherwise(pl.col("home_id"))
            .alias("def_id")
        )
    )
    inn = (
        sh.join(shifts, on=["game_id", "def_id", "period"])
        .filter((pl.col("start") <= pl.col("t_period")) & (pl.col("t_period") <= pl.col("end")))
        .unique(["game_id", "event_idx"], keep="first")
    )
    log.info("goalies: %d of %d attempts matched to a goalie in net", inn.height, sh.height)
    out = (
        inn.join(team_names(g).rename({"team_id": "def_id"}), on="def_id")
        .sort("game_id")
        .group_by("player_id")
        .agg(
            pl.col("team").last(),
            pl.col("game_id").n_unique().alias("gp"),
            pl.len().alias("attempts"),
            pl.col("type").is_in(["shot-on-goal", "goal"]).sum().alias("shots"),
            pl.col("goal").sum().cast(pl.Int64).alias("ga"),
            pl.col("xg").sum().alias("xga"),
        )
        .join(
            names.select("player_id", (pl.col("first_name") + " " + pl.col("last_name")).alias("name")), on="player_id"
        )
        .with_columns(
            (pl.col("xga") - pl.col("ga")).alias("gsax"),
            (1 - pl.col("ga") / pl.col("shots")).alias("sv_pct"),
            (1 - pl.col("xga") / pl.col("shots")).alias("xsv_pct"),
        )
        .with_columns((pl.col("gsax") / pl.col("shots") * 100).alias("gsax_100"))
    )
    cols = [
        "player_id",
        "name",
        "team",
        "gp",
        "shots",
        "attempts",
        "ga",
        "xga",
        "gsax",
        "gsax_100",
        "sv_pct",
        "xsv_pct",
    ]
    return out.select(cols).sort("gsax", descending=True)


def run(season: int = 2026) -> dict:
    live_config(season)
    cfg = load_config()
    proc = cfg.path(cfg.data.processed_dir)
    out = cfg.path(cfg.data.output_dir) / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    d = load(proc)
    rates = league_rates(str(proc))
    sg = skater_games(d, rates)
    tg = team_games(d, sg)
    gl = goalies(d, proc)
    tables = {
        "skater_games": sg.sort("game_date", "team", "name"),
        "skaters": skaters(sg),
        "team_games": tg,
        "teams": teams(tg, gl),
        "goalies": gl,
        "season": pl.DataFrame(
            {
                "season": [f"{season}-{(season + 1) % 100:02d}"],
                "games": [d["games"].height],
                "through": [d["games"]["game_date"].max()],
                # league on-ice expected goals per 60 skater-minutes, by situation
                **{f"rate_{k}_60": [60 * v] for k, v in rates.items()},
            }
        ),
    }
    for name, df in tables.items():
        df.write_csv(out / f"{name}.csv", float_precision=4)
        log.info("%s: %d rows", name, df.height)
    return tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
