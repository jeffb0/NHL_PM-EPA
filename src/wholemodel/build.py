"""Team-season validation table (docs/wholemodel/methodology.md §2-§3)."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.valuation import recon

COMPONENTS = {"pm": "pm_gar", "def": "def_gar", "fin": "fin_gar", "to": "to_gar", "fo": "fo_gar", "pen": "pen_gar",
              "pp": "st_pp_gar", "pk": "st_pk_gar", "4v4": "st_4v4_gar", "3v3": "st_3v3_gar"}
GOALIE_K = 1000.0


def _abbr(proc) -> pl.DataFrame:
    g = pl.read_parquet(proc / "games.parquet")
    return pl.concat([g.select(pl.col("home_id").alias("team_id"), pl.col("home_abbrev").alias("team")), g.select(pl.col("away_id").alias("team_id"), pl.col("away_abbrev").alias("team"))]).unique("team_id")


def games_played(proc) -> pl.DataFrame:
    g = pl.read_parquet(proc / "games.parquet")
    both = pl.concat([g.select("season", pl.col("home_abbrev").alias("team")), g.select("season", pl.col("away_abbrev").alias("team"))])
    return both.group_by("season", "team").agg(pl.len().alias("gp"))


def team_xgd(proc) -> pl.DataFrame:
    sh = pl.read_parquet(proc / "fin_shots.parquet", columns=["season", "team_id", "opp_team_id", "xg_v1"])
    f = sh.group_by("season", "team_id").agg(pl.col("xg_v1").sum().alias("xgf"))
    a = sh.group_by("season", pl.col("opp_team_id").alias("team_id")).agg(pl.col("xg_v1").sum().alias("xga"))
    return f.join(a, on=["season", "team_id"]).join(_abbr(proc), on="team_id").select("season", "team", (pl.col("xgf") - pl.col("xga")).alias("xgd_all"))


def goalie_term(proc, k: float = GOALIE_K) -> pl.DataFrame:
    """season T, team: sum over goalies of shots faced in T x the goalie's prior-season shrunk (xG - goals) per shot."""
    sh = pl.read_parquet(proc / "fin_shots.parquet", columns=["season", "goalie", "opp_team_id", "goal", "xg_v1"]).drop_nulls("goalie")
    gs = sh.group_by("season", "goalie").agg((pl.col("xg_v1").sum() - pl.col("goal").sum()).alias("gsax"), pl.len().alias("n")).with_columns((pl.col("gsax") / (pl.col("n") + k)).alias("rate"))
    prev = gs.select("goalie", (pl.col("season") + 1).alias("season"), pl.col("rate").alias("rate_prev"))
    faced = sh.group_by("season", "goalie", "opp_team_id").agg(pl.len().alias("n_team"))
    j = faced.join(prev, on=["season", "goalie"], how="left").with_columns(pl.col("rate_prev").fill_null(0.0)).join(_abbr(proc), left_on="opp_team_id", right_on="team_id")
    return j.group_by("season", "team").agg((pl.col("n_team") * pl.col("rate_prev")).sum().alias("G"))


def goalie_roll(proc) -> pl.DataFrame | None:
    """season T, team: sum over goalies of shots faced in T x the goalie's rolling season-(T-1) persistent goals saved per shot
    (goalies with no prior estimate get the fold's replacement rate). Rates come from folds refit on data through T-1 only."""
    path = proc / "gk_rolling_rates.parquet"
    if not path.exists():
        return None
    rates = pl.read_parquet(path)
    rep = rates.group_by("season").agg(pl.col("rep_v").first())
    faced = pl.read_parquet(proc / "gk_shots.parquet", columns=["season", "goalie", "team_id"]).group_by("season", "goalie", "team_id").agg(pl.len().alias("n"))
    j = (faced.join(rates.select("goalie", "season", "rate_v"), on=["goalie", "season"], how="left").join(rep, on="season", how="left")
         .with_columns(pl.coalesce("rate_v", "rep_v").alias("r")).join(_abbr(proc), on="team_id"))
    return j.group_by("season", "team").agg((pl.col("n") * pl.col("r")).sum().alias("G_roll"))


def onice_xgd_rate(proc) -> pl.DataFrame:
    """Raw on-ice all-strength xG differential per minute by player-season (5v5 and special-teams stints), for the naive baseline B3."""
    from src.defense.stints import COMPONENTS as LEDGER
    hx, ax = [f"h_{c}" for c in LEDGER], [f"a_{c}" for c in LEDGER]
    s5 = (pl.read_parquet(proc / "def_stints.parquet", columns=["season", "minutes", "home_on", "away_on", *hx, *ax])
          .select("season", "minutes", "home_on", "away_on", pl.sum_horizontal(hx).alias("xh"), pl.sum_horizontal(ax).alias("xa")))
    ss = pl.read_parquet(proc / "st_stints.parquet", columns=["season", "minutes", "home_on", "away_on", "h_xg", "a_xg"]).rename({"h_xg": "xh", "a_xg": "xa"})
    st = pl.concat([s5, ss.select(s5.columns)])
    home = st.select("season", "minutes", (pl.col("xh") - pl.col("xa")).alias("x"), pl.col("home_on").alias("p"))
    away = st.select("season", "minutes", (pl.col("xa") - pl.col("xh")).alias("x"), pl.col("away_on").alias("p"))
    long = pl.concat([home, away]).explode("p").rename({"p": "player_id"})
    return long.group_by("player_id", "season").agg((pl.col("x").sum() / pl.col("minutes").sum()).alias("x_rate"))


def variants(t: pl.DataFrame) -> dict[str, pl.Expr]:
    """Alternative totals (methodology §3), as expressions over the calibrated all-strength table."""
    tot = pl.col("total_gar_all")
    out = {"headline": tot, "calibrated": pl.col("total_gar_all_cal"), "five_v_five_total": pl.col("total_gar"), "original_turnover": tot - pl.col("to_gar") + pl.col("to_gar_orig"),
           "leakage_clean": (pl.col("pm_gar") + pl.col("def_gar") + pl.col("fin_raw_gar") + pl.col("to_gar") + pl.col("fo_raw_gar") + pl.col("pen_gar")
                             + pl.col("st_pp_raw_gar") + pl.col("st_pk_raw_gar") + pl.col("st_4v4_raw_gar") + pl.col("st_3v3_raw_gar"))}
    # exploratory (added after the WM-5 result; methodology §8): no component that uses a persistent player effect fitted on every season
    out["no_persistent_effects"] = (pl.col("pm_gar") + pl.col("def_gar") + pl.col("to_gar") + pl.col("pen_gar")
                                    + pl.col("st_pp_rapm_gar") + pl.col("st_pk_rapm_gar") + pl.col("st_4v4_rapm_gar") + pl.col("st_3v3_rapm_gar"))
    for c, col in COMPONENTS.items():
        out[f"drop_{c}"] = tot - pl.col(col)
    return out


def build(proc) -> pl.DataFrame:
    """One row per team-season T with the target, the goalie proxy, baselines, and the roster-weighted sums for every variant."""
    t = pl.read_parquet(proc / "player_value_calibrated_all.parquet")
    m_all = pl.col("toi_5v5") + sum((pl.col(f"st_{c}_min") for c in ("pp", "pk", "4v4", "3v3")), pl.lit(0.0))
    var = variants(t)
    base = t.with_columns(m_all.alias("m")).filter(pl.col("m") > 0)
    rates = base.select("player_id", "season", "m", "team", *[(e / pl.col("m")).alias(f"r_{k}") for k, e in var.items()])
    x = onice_xgd_rate(proc).with_columns(pl.col("player_id").cast(rates["player_id"].dtype), pl.col("season").cast(rates["season"].dtype))
    rates = rates.join(x, on=["player_id", "season"], how="left").with_columns(pl.col("x_rate").fill_null(0.0))
    cur = rates.select("player_id", "season", "team", pl.col("m").alias("m_T"))
    prev = rates.select("player_id", (pl.col("season") + 1).alias("season"), *[c for c in rates.columns if c.startswith("r_") or c == "x_rate"])
    j = cur.join(prev, on=["player_id", "season"], how="left").with_columns([pl.col(c).fill_null(0.0) for c in prev.columns if c.startswith("r_") or c == "x_rate"])
    sums = j.group_by("season", "team").agg(*[(pl.col(f"r_{k}") * pl.col("m_T")).sum().alias(f"S_{k}") for k in var], (pl.col("x_rate") * pl.col("m_T")).sum().alias("S_naive"))
    # no-oracle roster: previous team, previous minutes
    prv = rates.select("player_id", (pl.col("season") + 1).alias("season"), pl.col("team"), pl.col("m").alias("m_prev"), pl.col("r_headline").alias("r_h_prev"))
    no = prv.group_by("season", "team").agg((pl.col("r_h_prev") * pl.col("m_prev")).sum().alias("S_no_oracle"))
    gd = recon.team_gd_all(proc).join(games_played(proc), on=["season", "team"]).join(team_xgd(proc), on=["season", "team"])
    pg = gd.select((pl.col("season") + 1).alias("season"), "team", (pl.col("gd_all") / pl.col("gp")).alias("gd_pg_prev"), (pl.col("xgd_all") / pl.col("gp")).alias("xgd_pg_prev"))
    df = (gd.join(sums, on=["season", "team"]).join(no, on=["season", "team"], how="left").join(goalie_term(proc), on=["season", "team"], how="left")
          .join(pg, on=["season", "team"]).with_columns(pl.col("S_no_oracle").fill_null(0.0), pl.col("G").fill_null(0.0))
          .with_columns((pl.col("gd_pg_prev") * pl.col("gp")).alias("B_gd"), (pl.col("xgd_pg_prev") * pl.col("gp")).alias("B_xgd")))
    gr = goalie_roll(proc)
    if gr is not None:              # rolling goalie component (src/goaltending); replaces the crude proxy G in the goaltending-aware models
        df = df.join(gr, on=["season", "team"], how="left").with_columns(pl.col("G_roll").fill_null(0.0))
    return df.sort("season", "team")


def expanding_fit(df: pl.DataFrame, cols: list[str], test_seasons: list[int], fixed_slope: str | None = None) -> pl.DataFrame:
    """Expanding-window OLS of ``gd_all`` on ``cols`` (with intercept); with ``fixed_slope`` that column enters with coefficient 1."""
    rows = []
    for T in test_seasons:
        tr, te = df.filter(pl.col("season") < T), df.filter(pl.col("season") == T)
        y = tr["gd_all"].to_numpy().astype(float)
        off_tr = tr[fixed_slope].to_numpy() if fixed_slope else 0.0
        off_te = te[fixed_slope].to_numpy() if fixed_slope else 0.0
        X = np.column_stack([np.ones(tr.height)] + [tr[c].to_numpy() for c in cols]) if cols else np.ones((tr.height, 1))
        Xt = np.column_stack([np.ones(te.height)] + [te[c].to_numpy() for c in cols]) if cols else np.ones((te.height, 1))
        beta = np.linalg.lstsq(X, y - off_tr, rcond=None)[0]
        rows.append(pl.DataFrame({"season": T, "team": te["team"], "y": te["gd_all"].to_numpy().astype(float), "pred": Xt @ beta + off_te}))
    return pl.concat(rows)
