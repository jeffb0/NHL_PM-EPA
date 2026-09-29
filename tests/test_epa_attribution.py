"""PM-EPA attribution on hand-built possessions.

The numbers here are the worked examples in docs/pm_epa/attribution_examples.md; if the
engine changes, both must change together. Toy config: start/end shares 0.35, pool 0.30,
faceoff start share 0, lambda = 1, 5 skaters on ice (players 1-5 home, 21-25 away).
"""

from types import SimpleNamespace

import polars as pl
import pytest

from src.epa.engine import attribute, possession_identity_residual

EPA = SimpleNamespace(start_share=0.35, end_share=0.35, onice_pool=True, faceoff_start_share=0.0)
POSS = SimpleNamespace(max_segment_gap_s=45, carry_max_gap_s=12)
HOME_ON, AWAY_ON = [1, 2, 3, 4, 5], [21, 22, 23, 24, 25]


def o(poss, seq, role, kind, player, t, v, x=0.0, y=0.0, v_cf=None, v_cf_unobs=0.0, team=10, start_link=None, ev=None, v_att=None):
    return {
        "poss_id": poss, "game_id": 1, "season": 2024, "team_id": team, "opp_team_id": 20 if team == 10 else 10,
        "seq": seq, "role": role, "obs_kind": kind, "player_id": player, "t": float(t), "x": x, "y": y,
        "strength": "5v5", "score_diff": 0, "event_idx": ev if ev is not None else poss * 10 + seq, "v": v,
        "v_cf": v_cf, "v_cf_unobs": v_cf_unobs, "start_link": start_link, "v_att": v_att,
        "onice": HOME_ON if team == 10 else AWAY_ON, "opp_onice": AWAY_ON if team == 10 else HOME_ON,
    }


def run(obs_rows, poss_rows, lam=1.0):
    obs = pl.DataFrame(obs_rows, schema_overrides={"v_cf": pl.Float64, "start_link": pl.Int64, "player_id": pl.Int64, "v_att": pl.Float64})
    poss = pl.DataFrame(poss_rows).with_columns(pl.lit(False).alias("excluded"))
    tr, cr = attribute(obs, poss, EPA, POSS, lam)
    return tr, cr, poss


def total(cr, player=None, component=None):
    d = cr
    if player is not None:
        d = d.filter(pl.col("player_id") == player)
    if component is not None:
        d = d.filter(pl.col("component") == component)
    return d["value"].sum()


def test_example1_steal_and_solo_carry_to_shot():
    rows = [o(1, 0, "start", "takeaway", 1, 100, 0.020, v_cf=0.012), o(1, 1, "end", "shot", 1, 106, 0.060, v_att=0.075)]
    _, cr, _ = run(rows, [{"poss_id": 1, "end_type": "shot_sog", "G": 0.08}])
    assert total(cr, 1, "takeaway") == pytest.approx(0.032)
    assert total(cr, 1, "carry") == pytest.approx(0.040)
    assert total(cr, 1, "shot_attempt") == pytest.approx(0.015)
    assert total(cr, 1, "shot_quality") == pytest.approx(0.005)
    pm = cr.filter((pl.col("player_id") == 1) & pl.col("component").is_in(["takeaway", "carry", "shot_attempt"]))["value"].sum()
    assert pm == pytest.approx(0.087)
    assert total(cr, 21, "takeaway_against") == pytest.approx(-0.032 / 5)
    assert total(cr, component="takeaway_against") == pytest.approx(-0.032)


def test_example2_loose_recovery_unobserved_transition_split():
    rows = [o(2, 0, "start", "loose_opp_miss", None, 200, 0.015), o(2, 1, "end", "shot", 2, 214, 0.090, v_att=0.110)]
    tr, cr, _ = run(rows, [{"poss_id": 2, "end_type": "shot_sog", "G": 0.12}])
    assert tr["seg_type"][0] == "split" and tr["w_start"][0] == 0 and tr["w_pool"][0] == pytest.approx(0.65)
    assert total(cr, 2, "move_end") == pytest.approx(0.35 * 0.075)
    assert total(cr, 2, "move_pool") == pytest.approx(0.65 * 0.075 / 5)
    assert total(cr, 1) == pytest.approx(0.65 * 0.075 / 5)
    assert total(cr, component="unattributed") == pytest.approx(0.015)
    assert total(cr, 2, "shot_attempt") == pytest.approx(0.02)
    assert total(cr, 2, "shot_quality") == pytest.approx(0.01)


def test_example3_faceoff_winner_gets_no_start_share():
    rows = [o(3, 0, "start", "faceoff", 3, 300, 0.045), o(3, 1, "end", "shot", 4, 304, 0.030, v_att=0.035)]
    _, cr, _ = run(rows, [{"poss_id": 3, "end_type": "shot_block", "G": 0.0}])
    assert total(cr, 3, "move_start") == 0
    assert total(cr, 4, "move_end") == pytest.approx(0.35 * -0.015)
    assert total(cr, component="move_pool") == pytest.approx(0.65 * -0.015)
    assert total(cr, 4, "shot_attempt") == pytest.approx(0.005)
    assert total(cr, 4, "shot_quality") == pytest.approx(-0.035)


def test_example4_takeaway_then_own_zone_giveaway():
    rows = [o(4, 0, "start", "takeaway", 1, 400, 0.010, v_cf=0.035), o(4, 1, "end", "giveaway", 1, 403, 0.008, v_cf=0.110)]
    _, cr, _ = run(rows, [{"poss_id": 4, "end_type": "giveaway", "G": 0.0}])
    assert total(cr, 1, "takeaway") == pytest.approx(0.045)
    assert total(cr, 1, "carry") == pytest.approx(-0.002)
    assert total(cr, 1, "giveaway") == pytest.approx(-0.118)
    pm = cr.filter(pl.col("component").is_in(["takeaway", "carry", "giveaway"]) & (pl.col("player_id") == 1))
    assert pm["value"].sum() == pytest.approx(-0.075)


def test_example5_rebound_is_not_movement_credit():
    rows = [o(5, 0, "start", "rebound", None, 500, 0.20), o(5, 1, "end", "shot", 3, 502, 0.09, v_att=0.10)]
    _, cr, _ = run(rows, [{"poss_id": 5, "end_type": "goal", "G": 0.25}])
    assert total(cr, component="move_pool") == 0 and total(cr, component="move_end") == 0
    assert total(cr, component="unattributed") == pytest.approx(0.20 - 0.11)
    assert total(cr, 3, "shot_attempt") == pytest.approx(0.01)
    assert total(cr, 3, "shot_quality") == pytest.approx(0.15)


def test_example6_icing_charged_to_on_ice_pool():
    rows = [o(6, 0, "start", "faceoff", 3, 600, 0.010)]
    _, cr, _ = run(rows, [{"poss_id": 6, "end_type": "stoppage:icing", "G": 0.0}])
    assert total(cr, component="stoppage_loss") == pytest.approx(-0.010)
    assert cr.filter(pl.col("component") == "stoppage_loss").height == 5


def test_example7_unobserved_loss_uses_opponent_loose_value():
    rows = [o(7, 0, "start", "takeaway", 1, 700, 0.020, v_cf=0.012, v_cf_unobs=0.018)]
    _, cr, _ = run(rows, [{"poss_id": 7, "end_type": "unobserved_loss", "G": 0.0}])
    assert total(cr, component="unobserved_loss") == pytest.approx(-(0.020 + 0.018))


def test_carry_requires_same_player_and_short_gap():
    rows = [o(8, 0, "start", "takeaway", 1, 800, 0.02, v_cf=0.01), o(8, 1, "end", "shot", 1, 830, 0.06)]
    tr, _, _ = run(rows, [{"poss_id": 8, "end_type": "shot_miss", "G": 0.05}])
    assert tr["seg_type"][0] == "split"  # 30 s > carry_max_gap_s


def test_long_gap_goes_entirely_to_pool():
    rows = [o(9, 0, "start", "takeaway", 1, 900, 0.02, v_cf=0.01), o(9, 1, "end", "shot", 2, 960, 0.06)]
    tr, cr, _ = run(rows, [{"poss_id": 9, "end_type": "shot_miss", "G": 0.05}])
    assert tr["seg_type"][0] == "long_gap"
    assert total(cr, component="move_pool") == pytest.approx(0.04)
    assert total(cr, component="move_end") == 0


def test_missing_shift_data_sends_pool_share_to_unattributed():
    r = o(10, 1, "end", "shot", 2, 1010, 0.09)
    r["onice"] = []
    rows = [o(10, 0, "start", "loose_opp_miss", None, 1000, 0.015), r]
    _, cr, _ = run(rows, [{"poss_id": 10, "end_type": "shot_sog", "G": 0.1}])
    assert total(cr, component="move_pool") == 0
    assert total(cr, component="unattributed") == pytest.approx(0.015 + 0.65 * 0.075)


def test_identity_components_sum_to_terminal_xg_when_lambda_zero():
    rows = [
        o(1, 0, "start", "takeaway", 1, 100, 0.020, v_cf=0.012), o(1, 1, "end", "shot", 1, 106, 0.060, v_att=0.075),
        o(2, 0, "start", "loose_opp_miss", None, 200, 0.015), o(2, 1, "end", "shot", 2, 214, 0.090, v_att=0.110),
        o(4, 0, "start", "takeaway", 1, 400, 0.010, v_cf=0.035), o(4, 1, "end", "giveaway", 1, 403, 0.008, v_cf=0.110),
        o(6, 0, "start", "faceoff", 3, 600, 0.010),
        o(11, 0, "end", "shot", 4, 1100, 0.05),  # unobserved start
    ]
    poss = [{"poss_id": 1, "end_type": "shot_sog", "G": 0.08}, {"poss_id": 2, "end_type": "shot_sog", "G": 0.12},
            {"poss_id": 4, "end_type": "giveaway", "G": 0.0}, {"poss_id": 6, "end_type": "stoppage:icing", "G": 0.0},
            {"poss_id": 11, "end_type": "shot_miss", "G": 0.07}]
    _, cr, p = run(rows, poss, lam=0.0)
    res = possession_identity_residual(cr, p)
    assert res["residual"].abs().max() < 1e-12


def test_phase_tags_split_each_possession_into_start_move_terminal():
    rows = [
        o(1, 0, "start", "takeaway", 1, 100, 0.020, v_cf=0.012), o(1, 1, "end", "shot", 1, 106, 0.060, v_att=0.075),
        o(5, 0, "start", "rebound", None, 500, 0.20), o(5, 1, "end", "shot", 3, 502, 0.09, v_att=0.10),
        o(11, 0, "end", "shot", 4, 1100, 0.05),  # unobserved start
        o(20, 0, "start", "faceoff", 21, 0, 0.03, team=20), o(20, 1, "end", "lost_takeaway", None, 5, 0.012, team=20),
        o(21, 0, "start", "takeaway", 1, 5, 0.020, v_cf=0.012, start_link=20),
    ]
    poss = [{"poss_id": 1, "end_type": "shot_sog", "G": 0.08}, {"poss_id": 5, "end_type": "goal", "G": 0.25},
            {"poss_id": 11, "end_type": "shot_miss", "G": 0.07}, {"poss_id": 20, "end_type": "takeaway_against", "G": 0.0},
            {"poss_id": 21, "end_type": "period_end", "G": 0.0}]
    _, cr, _ = run(rows, poss, lam=0.0)
    ph = cr.group_by("poss_id", "phase").agg(pl.col("value").sum())
    get = lambda p, f: ph.filter((pl.col("poss_id") == p) & (pl.col("phase") == f))["value"].sum()  # noqa: E731
    assert set(cr["phase"].unique()) == {"start", "move", "terminal"}
    assert get(1, "start") == pytest.approx(0.020) and get(1, "move") == pytest.approx(0.040) and get(1, "terminal") == pytest.approx(0.020)
    assert get(5, "start") == pytest.approx(0.20) and get(5, "move") == pytest.approx(-0.11)  # rebound segment is movement
    assert get(11, "start") == pytest.approx(0.05)  # unobserved start: value at first observation
    assert get(20, "terminal") == pytest.approx(-0.012)  # takeaway debit ends the loser's possession


def test_takeaway_swing_is_zero_sum_when_lambda_one():
    rows = [
        o(20, 0, "start", "faceoff", 21, 0, 0.03, team=20),
        o(20, 1, "end", "lost_takeaway", None, 5, 0.012, team=20),
        o(21, 0, "start", "takeaway", 1, 5, 0.020, v_cf=0.012, start_link=20),
    ]
    poss = [{"poss_id": 20, "end_type": "takeaway_against", "G": 0.0}, {"poss_id": 21, "end_type": "period_end", "G": 0.0}]
    _, cr, _ = run(rows, poss)
    assert total(cr, component="takeaway") + total(cr, component="takeaway_against") == pytest.approx(0.0)
    # loser's debit is booked to the losing possession
    assert set(cr.filter(pl.col("component") == "takeaway_against")["poss_id"].to_list()) == {20}


def test_exclusion_propagates_across_takeaway_links():
    from src.epa.run import propagate_exclusion

    poss = pl.DataFrame({
        "poss_id": [1, 2, 3, 4], "start_type": ["faceoff", "takeaway", "takeaway", "faceoff"],
        "start_link": [None, 1, 2, None], "excluded": [False, False, True, False],
    }, schema_overrides={"start_link": pl.Int64})
    out = propagate_exclusion(poss)
    assert out["excluded"].to_list() == [True, True, True, False]  # 3 excluded -> 2 -> 1 (chain)


def test_decision_terms_are_mean_zero_at_a_location():
    """At one spot with V_dec = P(attempt) * V_att, attempt credits and loss debits cancel (lambda = 0)."""
    # 4 decisions at the same spot: 2 attempts (V_att 0.10), 2 giveaways -> V_dec = 0.05
    rows, poss = [], []
    for i, kind in enumerate(["shot", "shot", "giveaway", "giveaway"]):
        rows.append(o(30 + i, 0, "end", kind, 7, 10 * i, 0.05, v_att=0.10 if kind == "shot" else None, v_cf=0.0))
        poss.append({"poss_id": 30 + i, "end_type": "shot_sog" if kind == "shot" else "giveaway", "G": 0.10 if kind == "shot" else 0.0})
    _, cr, _ = run(rows, poss, lam=0.0)
    assert total(cr, 7, "shot_attempt") + total(cr, 7, "giveaway") == pytest.approx(0.0)
