import numpy as np
import polars as pl

from src.chainfree.run import _yoy


def test_yoy_recovers_persistence_and_centres_by_season():
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(size=n)
    rows = []
    for s, noise in ((2018, 0.5), (2019, 0.5)):
        for i in range(n):
            rows.append(
                {
                    "player_id": i,
                    "season": s,
                    "pos_group": "F",
                    "m": 900.0,
                    "x": base[i] + rng.normal(0, noise) + (3.0 if s == 2019 else 0.0),
                }
            )
    r = _yoy(pl.DataFrame(rows), "x", "m", 500)
    # true correlation is 1 / (1 + 0.25) = 0.8; the season mean shift is removed
    assert 0.6 < r["F"]["r"] < 0.9 and r["F"]["n"] == n
