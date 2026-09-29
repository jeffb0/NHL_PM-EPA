"""PM-EPA computation and attribution (brief §7; spec: docs/pm_epa/methodology.md §5).

Inputs per observation: V at the observation (``v``), possessing team, player on puck,
on-ice skaters. Every unit of value change is booked to exactly one component:

  component              who                     in PM-EPA   what
  ---------------------  ----------------------  ----------  ---------------------------------------------
  carry                  the carrier             direct      same player at both ends of a segment (<= carry_max_gap_s)
  move_start             segment start player    direct      start_share x delta of an unobserved-pass segment
  move_end               segment end player      direct      end_share x delta
  move_pool              on-ice skaters (equal)  pooled      residual share (+ shares of unknown endpoints)
  takeaway               taker                   direct      V_T(s) + lam * V_opp(mirror s)
  takeaway_against       opp. on-ice skaters     pooled      -(V_opp(mirror s) + lam * V_T(s))
  giveaway               giver                   direct      -(V_T(s) + lam * V_opp(loose_opp_giveaway at mirror s))
  shot_attempt           shooter                 direct      V_att(s) - V_dec(s): getting an attempt off at s instead of
                                                            the average decision outcome there (mean-zero with giveaways)
  unobserved_loss        on-ice skaters          pooled      -(V_T(s_last) + lam * V_opp(loose_unobserved))
  stoppage_loss          on-ice skaters          pooled      -V_T(s_last) for icing / offside
  shot_quality           shooter                 excluded    xG_shot - V_att(s): the release itself (type, block, rebound
                                                            luck) — the outcome the brief excludes (§4.2)
  unattributed           team only               excluded    start values of faceoffs / loose recoveries, rebounds,
                                                            neutral stoppages, pool rows with no shift data

Identity (tested): with lam = 0, the components of a possession sum to its terminal xG G_p.
"""

from __future__ import annotations

import polars as pl

PM_COMPONENTS = ["carry", "move_start", "move_end", "move_pool", "takeaway", "takeaway_against", "giveaway",
                 "shot_attempt", "unobserved_loss", "stoppage_loss"]
DIRECT = ["carry", "move_start", "move_end", "takeaway", "giveaway", "shot_attempt"]
POOLED = ["move_pool", "takeaway_against", "unobserved_loss", "stoppage_loss"]
CHARGED_STOPPAGES = {"stoppage:icing", "stoppage:offside"}
SHOT_ENDS = {"shot_sog", "shot_miss", "shot_block", "goal"}
FLIP = {"PP": "PK", "PK": "PP"}


def flip_strength(s: pl.Expr) -> pl.Expr:
    return s.replace(FLIP)


def counterfactual_frames(obs: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """States whose value is needed but that are not themselves observations (opponent views)."""
    mirror = [(-pl.col("x")).alias("x"), (-pl.col("y")).alias("y"), flip_strength(pl.col("strength")).alias("strength"),
              (-pl.col("score_diff")).alias("score_diff")]
    base = ["game_id", "poss_id", "seq", "season"]
    gv = obs.filter(pl.col("obs_kind") == "giveaway").select(base + mirror + [pl.lit("loose_opp_giveaway").alias("obs_kind")])
    tk = obs.filter(pl.col("obs_kind").is_in(["takeaway"])).select(base + mirror + [pl.lit("takeaway").alias("obs_kind")])
    return {"gv_opp": gv, "tk_opp": tk}


def attribute(obs: pl.DataFrame, poss: pl.DataFrame, cfg_epa, cfg_poss, lam: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (transitions, credits).

    ``obs`` must carry: poss_id, game_id, season, team_id, seq, role, obs_kind, player_id, t, x, y,
    strength, event_idx, v, onice (list of the possessing team's skaters at the event), and for
    giveaway / takeaway rows ``v_cf`` (the opponent counterfactual value).
    ``poss`` must carry: poss_id, end_type, start_type, G, excluded (bool).
    """
    ss, es = cfg_epa.start_share, cfg_epa.end_share
    fo_share = getattr(cfg_epa, "faceoff_start_share", 0.0)
    obs = obs.join(poss.select("poss_id", "end_type", "G", "excluded"), on="poss_id").filter(~pl.col("excluded"))
    obs = obs.sort("poss_id", "seq")

    # ------------------------------------------------------------- movement segments
    nxt = {c: pl.col(c).shift(-1).over("poss_id") for c in ["seq", "player_id", "t", "v", "onice", "event_idx", "obs_kind", "x", "y"]}
    seg = obs.with_columns(**{f"n_{c}": e for c, e in nxt.items()}).filter(pl.col("n_seq").is_not_null())
    gap = pl.col("n_t") - pl.col("t")
    start_known = pl.col("player_id").is_not_null()
    end_known = pl.col("n_player_id").is_not_null()
    is_fo = pl.col("obs_kind") == "faceoff"
    seg = seg.with_columns(
        (pl.col("n_v") - pl.col("v")).alias("delta"),
        gap.alias("gap"),
        pl.when(pl.col("obs_kind") == "rebound").then(pl.lit("rebound"))
        .when(gap > cfg_poss.max_segment_gap_s).then(pl.lit("long_gap"))
        .when(start_known & end_known & (pl.col("player_id") == pl.col("n_player_id")) & (gap <= cfg_poss.carry_max_gap_s) & ~is_fo)
        .then(pl.lit("carry"))
        .otherwise(pl.lit("split")).alias("seg_type"),
    ).with_columns(
        pl.when(pl.col("seg_type") == "carry").then(1.0)
        .when(pl.col("seg_type") == "split").then(
            pl.when(~start_known).then(0.0).when(is_fo).then(fo_share).otherwise(ss)
        ).otherwise(0.0).alias("w_start"),
        pl.when(pl.col("seg_type") == "split").then(pl.when(end_known).then(es).otherwise(0.0)).otherwise(0.0).alias("w_end"),
    ).with_columns(
        pl.when(pl.col("seg_type").is_in(["split", "long_gap"])).then(1.0 - pl.col("w_start") - pl.col("w_end"))
        .otherwise(0.0).alias("w_pool"),
        pl.when(pl.col("seg_type") == "rebound").then(1.0).otherwise(0.0).alias("w_unatt"),
    )
    transitions = seg.select(
        "game_id", "season", "poss_id", "team_id", "seq", "strength", pl.col("obs_kind").alias("from_kind"),
        pl.col("n_obs_kind").alias("to_kind"), "x", "y", pl.col("n_x").alias("to_x"), pl.col("n_y").alias("to_y"),
        pl.col("player_id").alias("start_player"), pl.col("n_player_id").alias("end_player"), "gap",
        pl.col("v").alias("v_before"), pl.col("n_v").alias("v_after"), "delta", "seg_type", "w_start", "w_end", "w_pool",
        pl.col("n_event_idx").alias("event_idx"), pl.col("n_onice").alias("onice"),
    )
    base = ["game_id", "season", "poss_id", "team_id", "strength", "event_idx"]
    moves = [
        transitions.filter(pl.col("seg_type") == "carry").select(base + [pl.col("start_player").alias("player_id"), pl.lit("carry").alias("component"), pl.col("delta").alias("value")]),
        transitions.filter((pl.col("seg_type") == "split") & (pl.col("w_start") > 0)).select(base + [pl.col("start_player").alias("player_id"), pl.lit("move_start").alias("component"), (pl.col("delta") * pl.col("w_start")).alias("value")]),
        transitions.filter((pl.col("seg_type") == "split") & (pl.col("w_end") > 0)).select(base + [pl.col("end_player").alias("player_id"), pl.lit("move_end").alias("component"), (pl.col("delta") * pl.col("w_end")).alias("value")]),
        _pool(transitions.filter(pl.col("w_pool") > 0).with_columns((pl.col("delta") * pl.col("w_pool")).alias("value")), base, "move_pool", cfg_epa.onice_pool),
        transitions.filter(pl.col("seg_type") == "rebound").select(base + [pl.lit(None, pl.Int64).alias("player_id"), pl.lit("unattributed").alias("component"), pl.col("delta").alias("value")]),
    ]

    # ------------------------------------------------------------- possession starts
    starts = []
    first = obs.filter(pl.col("role") == "start")
    # possessions whose start was not observed: the value at the first observation appears "from nowhere"
    no_start = obs.filter((pl.col("seq") == 0) & (pl.col("role") != "start"))
    starts.append(no_start.select(base + [pl.lit(None, pl.Int64).alias("player_id"), pl.lit("unattributed").alias("component"), pl.col("v").alias("value")]))
    starts.append(first.filter(pl.col("obs_kind") != "takeaway").select(base + [pl.lit(None, pl.Int64).alias("player_id"), pl.lit("unattributed").alias("component"), pl.col("v").alias("value")]))
    tk = first.filter(pl.col("obs_kind") == "takeaway")
    starts.append(tk.select(base + [pl.col("player_id"), pl.lit("takeaway").alias("component"), (pl.col("v") + lam * pl.col("v_cf")).alias("value")]))
    # the loser's debit: booked to the losing possession if it was open, else to an orphan row; on-ice of the loser
    tk_loss = tk.select(
        "game_id", "season", "event_idx", pl.col("opp_team_id").alias("team_id"), flip_strength(pl.col("strength")).alias("strength"),
        pl.col("start_link").alias("poss_id"), (-(pl.col("v_cf") + lam * pl.col("v"))).alias("value"), pl.col("opp_onice").alias("onice"),
    )
    # ------------------------------------------------------------- possession ends
    # (the takeaway swing's debit is terminal for the losing possession)
    credits = [_pool(tk_loss, base, "takeaway_against", cfg_epa.onice_pool)]
    last = obs.filter(pl.col("seq") == pl.col("seq").max().over("poss_id"))
    shot = last.filter(pl.col("end_type").is_in(list(SHOT_ENDS)) & (pl.col("obs_kind") == "shot"))
    v_att = pl.col("v_att").fill_null(pl.col("v")) if "v_att" in shot.columns else pl.col("v")
    credits.append(shot.select(base + [pl.col("player_id"), pl.lit("shot_attempt").alias("component"), (v_att - pl.col("v")).alias("value")]))
    credits.append(shot.select(base + [pl.col("player_id"), pl.lit("shot_quality").alias("component"), (pl.col("G") - v_att).alias("value")]))
    gv = last.filter(pl.col("obs_kind") == "giveaway")
    credits.append(gv.select(base + [pl.col("player_id"), pl.lit("giveaway").alias("component"), (-(pl.col("v") + lam * pl.col("v_cf"))).alias("value")]))
    # lost_takeaway: value already booked by the takeaway swing -> nothing here
    # losses revealed later than the last observation are charged to skaters on ice when the possession ended
    end_oi = [pl.col("end_onice").alias("onice")] if "end_onice" in last.columns else []
    unobs = last.filter(pl.col("end_type") == "unobserved_loss").with_columns(*end_oi, (-(pl.col("v") + lam * pl.col("v_cf_unobs"))).alias("value"))
    credits.append(_pool(unobs, base, "unobserved_loss", cfg_epa.onice_pool))
    stop = last.filter(pl.col("end_type").is_in(list(CHARGED_STOPPAGES))).with_columns(*end_oi, (-pl.col("v")).alias("value"))
    credits.append(_pool(stop, base, "stoppage_loss", cfg_epa.onice_pool))
    other = last.filter(
        ~pl.col("end_type").is_in(list(SHOT_ENDS | CHARGED_STOPPAGES | {"giveaway", "unobserved_loss", "takeaway_against"}))
    )
    credits.append(other.select(base + [pl.lit(None, pl.Int64).alias("player_id"), pl.lit("unattributed").alias("component"), (-pl.col("v")).alias("value")]))
    # a shot end whose shot observation is missing (no coordinates) -> unattributed remainder
    # phase (start / move / terminal) lets consumers such as the defensive model split a possession's ledger
    credits = pl.concat([
        c.select(base + ["player_id", "component", "value"]).with_columns(
            pl.col("player_id").cast(pl.Int64), pl.col("poss_id").cast(pl.Int64), pl.lit(phase).alias("phase"))
        for phase, frames in (("start", starts), ("move", moves), ("terminal", credits)) for c in frames
    ])
    return transitions.drop("onice"), credits


def _pool(df: pl.DataFrame, base: list[str], component: str, use_pool: bool) -> pl.DataFrame:
    """Split ``value`` equally across the on-ice list; rows without on-ice data -> unattributed."""
    if df.height == 0:
        return pl.DataFrame(schema={**{c: pl.Int64 for c in base}, "strength": pl.Utf8, "player_id": pl.Int64, "component": pl.Utf8, "value": pl.Float64}).select(base + ["player_id", "component", "value"])
    n = pl.col("onice").list.len()
    has = n > 0 if use_pool else pl.lit(False)
    with_pool = df.filter(has).with_columns((pl.col("value") / n).alias("value")).explode("onice").select(
        base + [pl.col("onice").alias("player_id"), pl.lit(component).alias("component"), "value"])
    none = df.filter(~has).select(base + [pl.lit(None, pl.Int64).alias("player_id"), pl.lit("unattributed").alias("component"), "value"])
    return pl.concat([with_pool.with_columns(pl.col("player_id").cast(pl.Int64)), none])


def possession_identity_residual(credits: pl.DataFrame, poss: pl.DataFrame) -> pl.DataFrame:
    """Per-possession sum of components vs G (valid as an identity when lam = 0)."""
    s = credits.filter(pl.col("poss_id").is_not_null()).group_by("poss_id").agg(pl.col("value").sum().alias("sum_components"))
    return poss.filter(~pl.col("excluded")).select("poss_id", "G").join(s, on="poss_id", how="left").with_columns(
        (pl.col("sum_components").fill_null(0) - pl.col("G")).alias("residual"))
