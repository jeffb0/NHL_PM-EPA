"""Role-adjusted turnover: involvement counting and the expectation fit."""

import numpy as np
import polars as pl
import pytest

from src.components import turnover_ra


def test_involvement_counts_only_successful_observed_events_at_5v5_in_usable_games(tmp_path):
    rows = []
    for comp, strength, gid, pid in [
        ("carry", "5v5", 1, 7),
        ("move_start", "5v5", 1, 7),
        ("shot_attempt", "5v5", 1, 7),
        ("giveaway", "5v5", 1, 7),
        ("takeaway", "5v5", 1, 7),
        ("carry", "PP", 1, 7),
        ("move_end", "5v5", 2, 7),
        ("carry", "5v5", 1, 8),
    ]:
        rows.append(
            {
                "game_id": gid,
                "season": 2024,
                "poss_id": 1,
                "team_id": 10,
                "strength": strength,
                "event_idx": 1,
                "player_id": pid,
                "component": comp,
                "value": 0.01,
                "phase": "move",
            }
        )
    pl.DataFrame(rows).write_parquet(tmp_path / "credits.parquet")
    # game 2 has no usable shift data
    pl.DataFrame({"game_id": [1], "onice_match_rate": [0.99]}).write_parquet(tmp_path / "shift_quality.parquet")
    inv = turnover_ra.involvement(tmp_path).sort("player_id")
    # carry, move_start, shot_attempt; not giveaway/takeaway/PP/unusable game
    assert dict(zip(inv["player_id"], inv["inv_n"])) == {7: 3, 8: 1}


def test_expected_rate_removes_the_involvement_gradient_and_keeps_the_mean():
    rng = np.random.default_rng(0)
    n = 500
    toi = rng.uniform(250, 1500, n)
    inv = rng.uniform(5, 40, n)
    rate = -0.05 - 0.006 * inv + rng.normal(0, 0.03, n)  # more involvement -> larger charge
    d = pl.DataFrame(
        {"player_id": np.arange(n), "season": 2024, "pos_group": "F", "toi_ev": toi, "inv60": inv, "rate": rate}
    )
    out, r2 = turnover_ra.expected_rate(d)
    resid = out["rate"] - out["expected"]
    assert abs(np.corrcoef(resid, out["inv60"])[0, 1]) < 0.05  # the gradient is gone
    # and the weighted mean is preserved (~0 by least squares)
    assert abs(np.average(resid, weights=out["toi_ev"])) < 5e-3
    # the involvement gradient explained most of the variance here
    assert r2["2024_F"] > 0.5
    assert np.corrcoef(rate, inv)[0, 1] < -0.6 and pytest.approx(np.corrcoef(resid, inv)[0, 1], abs=0.05) == 0.0
