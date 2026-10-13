from datetime import date

import numpy as np
import polars as pl

from src.gamemodel.data import Games
from src.infoquality.run import goalie_guesses, lineup_overlap


def _games():
    t = pl.DataFrame({"game_id": [1, 2, 3, 4], "season": [2025] * 4, "date": [date(2025, 10, 1), date(2025, 10, 3), date(2025, 10, 4), date(2025, 10, 6)], "home_id": [10, 10, 10, 10], "away_id": [20, 20, 20, 20]})
    gk = {(1, 10): [(1, 30.0, 0.0)], (2, 10): [(1, 30.0, 0.0)], (3, 10): [(2, 30.0, 0.0)], (4, 10): [(2, 30.0, 0.0)], **{(g, 20): [(9, 30.0, 0.0)] for g in (1, 2, 3, 4)}}
    a, b = np.array([1, 2, 3]), np.array([1, 2, 4])
    lu = {(1, 10): (a, np.array([20.0, 20.0, 20.0])), (2, 10): (a, np.array([20.0, 20.0, 20.0])), (3, 10): (b, np.array([30.0, 20.0, 10.0])), (4, 10): (b, np.array([20.0, 20.0, 20.0])),
          **{(g, 20): (a, np.array([20.0, 20.0, 20.0])) for g in (1, 2, 3, 4)}}
    return Games(t, lu, gk)


def test_goalie_guesses():
    rule, prev, q = goalie_guesses(_games())
    assert rule[(1, 10)] == 1 and prev[(1, 10)] == 1                    # first game: the actual starter
    assert rule[(2, 10)] == 1                                           # most-used starter
    assert rule[(3, 10)] == 1 and prev[(3, 10)] == 1                    # back-to-back, but no other goalie has started yet: both guesses miss
    assert rule[(4, 10)] == 1 and prev[(4, 10)] == 2                    # two starts against one: the rule still names goalie 1
    r = {(x["game_id"], x["team_id"]): x for x in q.iter_rows(named=True)}
    assert r[(3, 10)]["b2b"] and not r[(3, 10)]["rule_right"] and r[(4, 10)]["prev_right"] and r[(1, 10)]["first_game"] and r[(4, 20)]["rule_right"]


def test_lineup_overlap():
    o = {r["game_id"]: r for r in lineup_overlap(_games()).with_row_index().filter(pl.col("index") % 2 == 0).iter_rows(named=True)}       # the home team's rows
    assert 1 not in o and o[2]["share_players"] == 1.0 and o[2]["new_players"] == 0
    assert abs(o[3]["share_players"] - 2 / 3) < 1e-12 and abs(o[3]["share_minutes"] - 50 / 60) < 1e-12 and o[3]["new_players"] == 1
