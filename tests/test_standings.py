import numpy as np
import polars as pl

from src.standings.forecast import allocate_minutes
from src.standings.run import _z, ols


def test_allocate_minutes_keeps_top_and_hits_team_totals():
    rows = [{"team_id": 1, "player_id": i, "pos_group": "F"} for i in range(16)] + [
        {"team_id": 1, "player_id": 100 + i, "pos_group": "D"} for i in range(9)
    ]
    d = pl.DataFrame(rows)
    m = np.concatenate([np.linspace(1500, 300, 16), np.linspace(1500, 400, 9)])
    a = allocate_minutes(d, m, {"F": 9000.0, "D": 5000.0})
    assert a.filter(pl.col("pos_group") == "F").height == 13 and a.filter(pl.col("pos_group") == "D").height == 7
    assert abs(a.filter(pl.col("pos_group") == "F")["mins"].sum() - 9000.0) < 1e-6
    assert abs(a.filter(pl.col("pos_group") == "D")["mins"].sum() - 5000.0) < 1e-6


def test_paired_z_sign_and_ols():
    rng = np.random.default_rng(0)
    good, bad = rng.normal(0, 1, 200), rng.normal(0, 3, 200)
    assert _z(good, bad) > 2 and _z(bad, good) < -2
    x = rng.normal(size=(300, 1))
    beta = ols(x, 2.0 + 3.0 * x[:, 0])
    assert abs(beta[0] - 2.0) < 1e-8 and abs(beta[1] - 3.0) < 1e-8
