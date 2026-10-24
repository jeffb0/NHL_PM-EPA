"""Possession-chain inference: core rules and edge cases (definitions.md §3)."""

from src.possession.chains import ChainParams, build_game
from tests.helpers import AWAY, HOME, ev

P = ChainParams(rebound_window_s=3, max_inferred_start_gap_s=25)


def run(events, params=P):
    return build_game(events, params)


def test_faceoff_then_shot_is_one_possession_with_two_observations():
    poss, obs = run([ev(0, 0, "faceoff", HOME, 10, 0, 0), ev(1, 8, "shot-on-goal", HOME, 11, 60, 5)])
    assert len(poss) == 1
    assert poss[0]["start_type"] == "faceoff" and poss[0]["end_type"] == "shot_sog"
    assert [o["obs_kind"] for o in obs] == ["faceoff", "shot"]
    assert [o["player_id"] for o in obs] == [10, 11]


def test_takeaway_closes_opponent_possession_with_mirrored_loss_point():
    poss, obs = run(
        [
            ev(0, 0, "faceoff", AWAY, 20, 0, 0),
            ev(1, 6, "takeaway", HOME, 11, 30, -10),  # home frame
        ]
    )
    away = [p for p in poss if p["team_id"] == AWAY][0]
    assert away["end_type"] == "takeaway_against"
    lost = [o for o in obs if o["obs_kind"] == "lost_takeaway"][0]
    assert (lost["x"], lost["y"]) == (-30, 10)  # away's frame
    home = [p for p in poss if p["team_id"] == HOME][0]
    assert home["start_type"] == "takeaway" and home["start_link"] == away["poss_id"]


def test_stoppage_closes_possession():
    poss, _ = run([ev(0, 0, "faceoff", HOME, 10, 69, 22), ev(1, 5, "stoppage", reason="icing")])
    assert poss[0]["end_type"] == "stoppage:icing"


def test_recovery_after_opponent_miss_is_inferred_loose_start_in_own_frame():
    poss, obs = run(
        [
            ev(0, 0, "faceoff", AWAY, 20, 69, 22),
            ev(1, 4, "missed-shot", AWAY, 21, 70, -15),
            ev(2, 14, "shot-on-goal", HOME, 11, 75, 3),
        ]
    )
    home = [p for p in poss if p["team_id"] == HOME][0]
    assert home["start_type"] == "loose_opp_miss"
    start = [o for o in obs if o["poss_id"] == home["poss_id"]][0]
    assert (start["x"], start["y"]) == (-70, 15)  # opponent's shot spot mirrored into home frame
    assert start["player_id"] is None and start["t"] == 4


def test_rebound_same_team_within_window():
    poss, _ = run(
        [
            ev(0, 0, "faceoff", HOME, 10, 69, 22),
            ev(1, 5, "shot-on-goal", HOME, 11, 80, 3),
            ev(2, 7, "shot-on-goal", HOME, 12, 85, -2),
        ]
    )
    assert poss[1]["start_type"] == "rebound"


def test_same_team_shot_outside_rebound_window_is_offensive_recovery():
    poss, _ = run(
        [
            ev(0, 0, "faceoff", HOME, 10, 69, 22),
            ev(1, 5, "missed-shot", HOME, 11, 60, 3),
            ev(2, 15, "shot-on-goal", HOME, 12, 55, -2),
        ]
    )
    assert poss[1]["start_type"] == "loose_own_miss"


def test_long_gap_gives_unobserved_start_without_start_observation():
    poss, obs = run(
        [
            ev(0, 0, "faceoff", AWAY, 20, 69, 22),
            ev(1, 4, "missed-shot", AWAY, 21, 70, -15),
            ev(2, 60, "shot-on-goal", HOME, 11, 75, 3),
        ]
    )
    home = [p for p in poss if p["team_id"] == HOME][0]
    assert home["start_type"] == "unobserved" and home["n_obs"] == 1


def test_opponent_event_without_takeaway_closes_as_unobserved_loss():
    poss, obs = run(
        [
            ev(0, 0, "faceoff", HOME, 10, 0, 0),
            ev(1, 9, "shot-on-goal", AWAY, 21, 60, 0),  # home lost it unobserved (e.g. dump-in)
        ]
    )
    assert poss[0]["end_type"] == "unobserved_loss"
    assert poss[1]["start_type"] == "loose_unobserved"
    start = [o for o in obs if o["poss_id"] == poss[1]["poss_id"]][0]
    assert start["x"] is None  # no location for an unobserved loss


def test_giveaway_hands_opponent_a_loose_start_at_mirrored_location():
    poss, obs = run(
        [
            ev(0, 0, "faceoff", HOME, 10, -69, 22),
            ev(1, 3, "giveaway", HOME, 12, -80, 30),
            ev(2, 5, "shot-on-goal", AWAY, 21, 82, -10),
        ]
    )
    assert poss[0]["end_type"] == "giveaway"
    assert poss[1]["start_type"] == "loose_opp_giveaway"
    start = [o for o in obs if o["poss_id"] == poss[1]["poss_id"]][0]
    assert (start["x"], start["y"]) == (80, -30)


def test_takeaway_by_team_already_in_possession_implies_unobserved_loss():
    poss, _ = run([ev(0, 0, "faceoff", HOME, 10, 0, 0), ev(1, 10, "takeaway", HOME, 11, 40, 0)])
    assert poss[0]["end_type"] == "unobserved_loss"
    assert poss[1]["start_type"] == "takeaway" and poss[1]["start_link"] is None


def test_hits_ignored_by_default_and_mid_observation_when_enabled():
    events = [
        ev(0, 0, "faceoff", HOME, 10, 69, 22),
        ev(1, 3, "hit", AWAY, 20, -85, 38, p2=12),  # away hits home player 12 in home's OZ corner
        ev(2, 6, "shot-on-goal", HOME, 13, 70, 5),
    ]
    _, obs = run(events)
    assert [o["obs_kind"] for o in obs] == ["faceoff", "shot"]
    _, obs = run(events, ChainParams(hits_as_observations=True))
    mid = [o for o in obs if o["obs_kind"] == "hit_mid"][0]
    assert mid["player_id"] == 12 and (mid["x"], mid["y"]) == (85, -38)


def test_hit_on_non_possessing_team_is_never_an_observation():
    events = [
        ev(0, 0, "faceoff", HOME, 10, 69, 22),
        ev(1, 3, "hit", HOME, 11, 85, 38, p2=21),  # home hits away player: away does not possess
        ev(2, 6, "shot-on-goal", HOME, 13, 70, 5),
    ]
    _, obs = run(events, ChainParams(hits_as_observations=True))
    assert "hit_mid" not in [o["obs_kind"] for o in obs]


def test_period_change_closes_open_possession():
    poss, _ = run([ev(0, 1190, "faceoff", HOME, 10, 0, 0), ev(1, 0, "faceoff", AWAY, 20, 0, 0, period=2)])
    assert poss[0]["end_type"] == "period_end"


def test_strength_is_from_possessing_team_perspective():
    _, obs = run([ev(0, 0, "faceoff", AWAY, 20, 69, 22, home_sk=4, away_sk=5)])
    assert obs[0]["strength"] == "PP"
    _, obs = run([ev(0, 0, "faceoff", HOME, 10, 69, 22, home_sk=4, away_sk=5)])
    assert obs[0]["strength"] == "PK"


def test_empty_net_strength():
    _, obs = run([ev(0, 0, "faceoff", HOME, 10, 0, 0, home_sk=6, home_g=0)])
    assert obs[0]["strength"] == "EN"


def test_score_diff_from_possessing_team_perspective():
    _, obs = run([ev(0, 0, "faceoff", AWAY, 20, 0, 0, home_score=3, away_score=1)])
    assert obs[0]["score_diff"] == -2


def test_blocked_shot_then_blocker_team_recovers():
    poss, _ = run(
        [
            ev(0, 0, "faceoff", HOME, 10, 69, 22),
            ev(1, 4, "blocked-shot", HOME, 11, 70, 0, p2=25),
            ev(2, 9, "giveaway", AWAY, 25, -60, 0),
        ]
    )
    assert poss[1]["team_id"] == AWAY and poss[1]["start_type"] == "loose_opp_block"


def test_possession_ids_unique_and_every_observation_belongs_to_a_possession():
    events = [
        ev(0, 0, "faceoff", HOME, 10, 0, 0),
        ev(1, 5, "takeaway", AWAY, 20, 10, 0),
        ev(2, 9, "shot-on-goal", AWAY, 20, 70, 0),
        ev(3, 11, "shot-on-goal", AWAY, 21, 80, 0),
        ev(4, 20, "giveaway", HOME, 12, -60, 0),
        ev(5, 25, "stoppage", reason="offside"),
    ]
    poss, obs = run(events)
    ids = [p["poss_id"] for p in poss]
    assert len(ids) == len(set(ids))
    assert {o["poss_id"] for o in obs} <= set(ids)
    assert sum(p["n_obs"] for p in poss) == len(obs)


def test_goal_never_seeds_a_loose_start():
    poss, _ = run(
        [
            ev(0, 0, "faceoff", HOME, 10, 69, 22),
            ev(1, 5, "goal", HOME, 11, 80, 3),
            ev(2, 5, "giveaway", AWAY, 21, -60, 0),  # out-of-order PBP artefact
        ]
    )
    assert poss[1]["start_type"] == "unobserved"


def test_out_of_order_time_never_creates_inferred_start():
    poss, _ = run(
        [
            ev(0, 0, "faceoff", AWAY, 20, 69, 22),
            ev(1, 10, "missed-shot", AWAY, 21, 70, -15),
            ev(2, 8, "shot-on-goal", HOME, 11, 75, 3),
        ]
    )
    assert poss[1]["start_type"] == "unobserved"


def test_end_event_idx_is_the_event_that_revealed_the_end():
    poss, _ = run([ev(0, 0, "faceoff", HOME, 10, 0, 0), ev(1, 30, "shot-on-goal", AWAY, 21, 60, 0)])
    assert poss[0]["end_type"] == "unobserved_loss" and poss[0]["end_event_idx"] == 1
    assert poss[0]["terminal_event_idx"] is None
