from datetime import date

import numpy as np
import polars as pl

from src.angles.features import ANGLES, FOLK, game_angles, team_states
from src.angles.run import bh, record


def _games():
    # (id, date, home, away, home goals, away goals, period type)
    rows = [
        (1, date(2025, 10, 1), "AAA", "BBB", 5, 0, "REG"),
        (2, date(2025, 10, 2), "CCC", "BBB", 2, 3, "OT"),
        (3, date(2025, 10, 3), "AAA", "BBB", 1, 2, "REG"),
        (4, date(2025, 10, 5), "CCC", "BBB", 0, 1, "REG"),
        (5, date(2025, 10, 8), "BBB", "AAA", 2, 4, "REG"),
    ]
    return pl.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": [2025] * 5,
            "date": [r[1] for r in rows],
            "home_abbrev": [r[2] for r in rows],
            "away_abbrev": [r[3] for r in rows],
            "home_final": [r[4] for r in rows],
            "away_final": [r[5] for r in rows],
            "last_period_type": [r[6] for r in rows],
            "xgd": [1.0, 0.0, 0.5, None, 0.0],
        }
    )


def test_team_states():
    s = {(r["game_id"], r["team"]): r for r in team_states(_games()).iter_rows(named=True)}
    assert set(FOLK) == set(ANGLES)
    # the motivating example: played yesterday and lost by three or more (5-0, so also shut out); plays again tomorrow
    b2 = s[(2, "BBB")]
    assert (
        b2["b2b_after_blowout"] == 1
        and b2["blowout_loss"] == 1
        and b2["shutout_loss"] == 1
        and b2["front_b2b"] == 1
        and b2["revenge"] == 0
    )
    b3 = s[(3, "BBB")]  # back-to-back after overtime, facing the team that beat it 5-0
    assert (
        b3["b2b_after_ot"] == 1
        and b3["prev_ot"] == 1
        and b3["revenge"] == 1
        and b3["revenge_blowout"] == 1
        and b3["rematch_lost"] == 0
        and b3["trip_finale"] == 0
    )
    assert s[(3, "AAA")]["blowout_win"] == 1 and s[(3, "AAA")]["scored5"] == 1 and s[(3, "AAA")]["revenge"] == 0
    b4 = s[(4, "BBB")]  # fourth straight road game and the next one is at home
    assert (
        b4["trip_finale"] == 1
        and b4["streak_w3"] == 0
        and s[(4, "CCC")]["rematch_lost"] == 1
        and s[(4, "CCC")]["revenge"] == 1
    )
    b5 = s[(5, "BBB")]  # home after four on the road, three straight wins
    assert (
        b5["home_after_trip"] == 1
        and b5["streak_w3"] == 1
        and b5["streak_l3"] == 0
        and abs(b5["h2h"] - (1 - 1) / 2) < 1e-12
        and b5["last10"] == 0.0
    )
    assert s[(5, "AAA")]["home_after_trip"] == 0 and s[(5, "AAA")]["revenge"] == 1


def test_game_angles_are_home_minus_away():
    g = {r["game_id"]: r for r in game_angles(_games()).iter_rows(named=True)}
    assert (
        g[3]["revenge"] == -1
        and g[3]["blowout_win"] == 1
        and g[5]["home_after_trip"] == 1
        and g[4]["trip_finale"] == -1
        and g[5]["streak_w3"] == 1
    )


def test_bh_and_record():
    r = bh({"a": 0.001, "b": 0.04, "c": 0.5, "d": 0.9}, q=0.10)
    assert r == {"a": True, "b": True, "c": False, "d": False}
    assert not any(bh({"a": 0.2, "b": 0.6}).values())
    rec = record(np.array([True, True, False, True]), np.array([1.0, 0.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.9, 0.5]))
    assert (
        (rec["wins"], rec["losses"]) == (2, 1)
        and abs(rec["expected_wins"] - 1.5) < 1e-12
        and abs(rec["z"] - 0.5 / np.sqrt(0.75)) < 1e-12
    )
