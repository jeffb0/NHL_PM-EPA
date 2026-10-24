"""Data-level edge cases: parsing, strength, direction inference, xG context flags, RAPM design."""

import numpy as np
import polars as pl
import pytest

from src.ingestion.parse import _mmss, _parse_situation, infer_direction, strength_label
from src.validation.rapm import design
from src.xg.features import build_shot_features


def test_clock_parsing_and_missing():
    assert _mmss("12:34") == 754
    assert np.isnan(_mmss(None))


@pytest.mark.parametrize(
    "code,expected",
    [
        ("1551", (1, 5, 5, 1)),
        ("0651", (0, 6, 5, 1)),
        ("1331", (1, 3, 3, 1)),
        (None, (1, 5, 5, 1)),
        ("bad", (1, 5, 5, 1)),
    ],
)
def test_situation_code_parsing(code, expected):
    assert _parse_situation(code) == expected


@pytest.mark.parametrize(
    "own,opp,og,pg,label",
    [
        (5, 5, 1, 1, "5v5"),
        (5, 4, 1, 1, "PP"),
        (4, 5, 1, 1, "PK"),
        (4, 4, 1, 1, "4v4"),
        (3, 3, 1, 1, "3v3"),
        (6, 5, 0, 1, "EN"),
        (5, 6, 1, 0, "EN"),
        (5, 3, 1, 1, "PP"),
    ],
)
def test_strength_labels(own, opp, og, pg, label):
    assert strength_label(own, opp, og, pg) == label


def test_direction_inference_votes_and_falls_back_to_defending_side():
    games = pl.DataFrame({"game_id": [1, 2], "home_id": [10, 10]})
    ev = pl.DataFrame(
        {
            "game_id": [1, 1, 1, 2],
            "period": [1, 1, 1, 1],
            "type": ["shot-on-goal", "hit", "blocked-shot", "faceoff"],
            # game 1: home offensive-zone events at +x -> home attacks +x; the blocked shot must be ignored
            "zone_code_raw": ["O", "O", "D", "N"],
            "x_raw": [70, 60, -80, 0],
            "owner_team_raw": [10, 10, 10, 10],
            "home_defending_side": [None, None, None, "right"],
        }
    )
    d = infer_direction(ev, games).sort("game_id")
    assert d["home_attack_sign"].to_list() == [1, -1]  # game 2: no votes -> defending right => attacks -x


def _events(rows):
    base = {
        "game_id": 1,
        "period": 1,
        "team_id": 10,
        "x": 0.0,
        "y": 0.0,
        "opp_goalie": 1,
        "own_goalie": 1,
        "score_diff": 0,
        "shot_type": "wrist",
        "player1": 99,
        "season": 2024,
        "strength": "5v5",
        "is_home": True,
    }
    return pl.DataFrame([{**base, **r} for r in rows], schema_overrides={"x": pl.Float64, "y": pl.Float64})


def test_rebound_and_rush_flags():
    ev = _events(
        [
            {"event_idx": 0, "type": "takeaway", "t_period": 10, "x": -10.0},
            {"event_idx": 1, "type": "shot-on-goal", "t_period": 13, "x": 70.0},  # rush: prev in NZ 3 s ago
            {"event_idx": 2, "type": "missed-shot", "t_period": 15, "x": 80.0},  # rebound: same-team shot 2 s ago
            {"event_idx": 3, "type": "shot-on-goal", "t_period": 30, "x": 60.0},  # neither
        ]
    )
    players = pl.DataFrame({"player_id": [99], "shoots": ["L"]})
    s = build_shot_features(ev, players).sort("event_idx")
    assert s["rush"].to_list() == [True, False, False]
    assert s["rebound"].to_list() == [False, True, False]


def test_shots_without_y_are_dropped_from_xg_features():
    ev = _events([{"event_idx": 0, "type": "shot-on-goal", "t_period": 5, "x": 70.0, "y": None}])
    assert build_shot_features(ev, pl.DataFrame({"player_id": [99], "shoots": ["L"]})).height == 0


def test_off_wing_uses_handedness_in_attacking_frame():
    ev = _events(
        [
            {"event_idx": 0, "type": "shot-on-goal", "t_period": 5, "x": 70.0, "y": -10.0},
            {"event_idx": 1, "type": "shot-on-goal", "t_period": 40, "x": 70.0, "y": 10.0},
        ]
    )
    s = build_shot_features(ev, pl.DataFrame({"player_id": [99], "shoots": ["L"]})).sort("event_idx")
    assert s["off_wing"].to_list() == [True, False]  # left shot on the right side (y < 0) is off-wing


def test_rapm_design_offense_defense_blocks_and_rates():
    st = pl.DataFrame(
        {
            "game_id": [1],
            "minutes": [2.0],
            "home_on": [[1, 2, 3, 4, 5]],
            "away_on": [[6, 7, 8, 9, 10]],
            "home_xg": [0.2],
            "away_xg": [0.1],
        }
    )
    X, y, w, groups, players = design(st)
    A = X.toarray()
    assert A.shape == (2, 21)
    assert A[0, :5].sum() == 5 and A[0, 15:20].sum() == 5  # home offense, away defense
    assert A[1, 5:10].sum() == 5 and A[1, 10:15].sum() == 5  # away offense, home defense
    assert A[:, 20].tolist() == [1.0, 0.0]  # home-offense indicator
    assert y.tolist() == pytest.approx([6.0, 3.0]) and w.tolist() == [2.0, 2.0]
