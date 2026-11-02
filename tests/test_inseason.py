import numpy as np
import polars as pl
import pytest

from src.inseason.finishing import FinUpdate
from src.inseason.individual import CHECK, Individual, signal_row


def test_finishing_update_is_one_laplace_step_scaled_to_goals_per_60():
    # two games; shooter 1 scores once in each (2 of 3 attempts at p0 = 0.1), shooter 2 has no attempts
    cur = pl.DataFrame(
        {
            "game_id": [10, 10, 11],
            "shooter": [1, 1, 1],
            "goal": [1, 0, 1],
            "p0": [0.1, 0.1, 0.1],
        }
    )
    up = FinUpdate(cur, last={1: 9.0}, prec={1: 50.0, 2: 50.0}, minutes_last={1: 1000.0})
    assert up.adj(1) == 0.0  # nothing observed yet
    up.observe(10, 7, np.array([1, 2]), np.array([20.0, 18.0]))
    sg, sw = 1 - 0.2, 2 * 0.09
    assert up.adj(1) == pytest.approx(sg / (50 + sw) * (9 + sw) * 60 / 1020)
    assert up.adj(2) == 0.0
    up.observe(11, 7, np.array([1, 2]), np.array([20.0, 18.0]))
    sg, sw = 2 - 0.3, 3 * 0.09
    assert up.adj(1) == pytest.approx(sg / (50 + sw) * (9 + sw) * 60 / 1040)


def test_individual_signals_relative_to_team_off_ice_and_last_seasons_minutes():
    # 4 games: 80 min, 60 at 5v5, puck value 0.4, ixG 1.2, 2 points, on-ice xGF 3 / xGA 2, off-ice xGD -1 over 120 min
    c = np.array([4.0, 80.0, 60.0, 0.4, 1.2, 2.0, 3.0, 2.0, -1.0, 120.0])
    s = signal_row(c, toi_last=18.0)
    assert s == pytest.approx([20.0, 0.3, 0.9, 1.5, 1.0, 1.5, 2.0])
    assert signal_row(c, toi_last=None)[-1] == 0.0


def test_individual_adjustment_waits_for_the_first_checkpoint():
    class Flat:
        def predict(self, X):
            return np.array([0.05])

    line = np.array([20.0, 15.0, 0.1, 0.3, 0.5, 0.8, 0.6, -0.2, 30.0])
    ind = Individual({("F", "old", CHECK[0]): Flat()}, {1: ("F", "old")}, {(g, 1): line for g in range(10)}, {1: 18.0})
    for g in range(CHECK[0] - 1):
        ind.observe(g, 7, np.array([1, 2]), np.array([20.0, 18.0]))
        assert ind.adj(1) == 0.0
    ind.observe(CHECK[0] - 1, 7, np.array([1, 2]), np.array([20.0, 18.0]))
    assert ind.adj(1) == pytest.approx(0.05)
    assert ind.adj(2) == 0.0  # not an established skater in this season's table
