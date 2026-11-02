import numpy as np
import polars as pl
import pytest

from src.inseason.finishing import FinUpdate


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
    up.observe(10, np.array([1, 2]), np.array([20.0, 18.0]))
    sg, sw = 1 - 0.2, 2 * 0.09
    assert up.adj(1) == pytest.approx(sg / (50 + sw) * (9 + sw) * 60 / 1020)
    assert up.adj(2) == 0.0
    up.observe(11, np.array([1, 2]), np.array([20.0, 18.0]))
    sg, sw = 2 - 0.3, 3 * 0.09
    assert up.adj(1) == pytest.approx(sg / (50 + sw) * (9 + sw) * 60 / 1040)
