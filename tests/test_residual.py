import numpy as np
import polars as pl

from src.residual.decomp import carried, ols0


def test_carried_follows_players_by_minutes_and_position():
    p = pl.DataFrame(
        {
            "player_id": [1, 2, 3, 1, 2, 3],
            "season": [2020, 2020, 2020, 2021, 2021, 2021],
            "team": ["A", "A", "B", "A", "B", "B"],
            "pos_group": ["F", "D", "F", "F", "D", "F"],
            "m": [1000.0, 1000.0, 1000.0, 800.0, 600.0, 500.0],
        }
    )
    t = pl.DataFrame({"season": [2020, 2020], "team": ["A", "B"], "rho": [0.01, -0.02], "rho_play5": [0.004, -0.01]})
    c = {r["team"]: r for r in carried(p, t, ["rho_play5", "rho"]).filter(pl.col("season") == 2021).to_dicts()}
    assert abs(c["A"]["c_rho"] - 0.01 * 800) < 1e-12  # player 1 stayed
    # player 2 arrived from A, player 3 stayed
    assert abs(c["B"]["c_rho"] - (0.01 * 600 - 0.02 * 500)) < 1e-12
    assert abs(c["B"]["c_D"] - 0.01 * 600) < 1e-12 and abs(c["B"]["c_F"] + 0.02 * 500) < 1e-12
    assert abs(c["B"]["c_rho_play5"] - (0.004 * 600 - 0.01 * 500)) < 1e-12


def test_ols_through_origin():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    b, se = ols0(X, X @ np.array([0.9, 0.3]) + rng.normal(0, 0.1, 500))
    assert abs(b[0] - 0.9) < 0.03 and abs(b[1] - 0.3) < 0.03 and (se < 0.02).all()
