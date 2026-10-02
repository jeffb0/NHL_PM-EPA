import polars as pl

from src.standings.candidates import carried_residual, decide


def test_carried_residual_follows_the_player():
    # last season: team A outscored its player values by +10 over 1,000 skater-minutes, team B by -5 over 1,000
    p = pl.DataFrame({"player_id": [1, 2, 3], "season": [2024, 2024, 2024], "team": ["A", "A", "B"]})
    resid = pl.DataFrame({"season": [2024, 2024], "team": ["A", "B"], "rho": [10 / 1000, -5 / 1000]})
    # this season player 1 stays on team 10, player 2 moves to team 20, player 3 joins team 10, player 4 has no history
    alloc = pl.DataFrame({"team_id": [10, 20, 10, 20], "player_id": [1, 2, 3, 4], "mins": [500.0, 500.0, 200.0, 800.0]})
    r = {x["team_id"]: x["Rc"] for x in carried_residual(alloc, p, resid, 2025).to_dicts()}
    assert abs(r[10] - (0.01 * 500 - 0.005 * 200)) < 1e-12
    assert abs(r[20] - 0.01 * 500) < 1e-12


def test_decision_rule():
    assert decide(2.5, False) and decide(2.5, True)
    assert decide(0.3, True) and not decide(0.3, False)
    assert not decide(-1.5, True)
