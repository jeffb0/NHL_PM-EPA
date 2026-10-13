from datetime import date

import polars as pl

from src.structure.features import environment, goalie_flags, home_ice_team, volume_target


def test_goalie_flags():
    g = pl.DataFrame({"game_id": [1, 2, 3, 4], "season": [2025] * 4, "date": [date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3), date(2025, 10, 20)],
                      "home_id": [10, 30, 10, 10], "away_id": [20, 20, 20, 30]})
    st = {(1, 10): 1, (1, 20): 2, (2, 30): 3, (2, 20): 2, (3, 10): 1, (3, 20): 9, (4, 10): 1, (4, 30): 3}
    f = {r["game_id"]: r for r in goalie_flags(g, st).iter_rows(named=True)}
    assert f[2]["tired_a"] == 1 and f[2]["gk_tired"] == -1              # the away starter also started last night
    assert f[3]["tired_a"] == 0 and f[3]["gk_tired"] == 0               # third night running, but the other goalie starts
    assert f[4]["rusty_h"] == 1 and f[4]["rusty_a"] == 1 and f[4]["gk_rusty"] == 0 and f[1]["rusty_h"] == 0


def test_home_ice_team_uses_earlier_seasons_only():
    g = pl.DataFrame({"game_id": [1, 2, 3, 4, 5], "season": [2024, 2024, 2024, 2024, 2025], "home_id": [10, 20, 10, 20, 10], "away_id": [20, 10, 20, 10, 20], "gd": [3.0, 1.0, 3.0, 1.0, -9.0]})
    f = dict(home_ice_team(g).iter_rows())
    assert f[1] == 0.0                                                   # no earlier season
    # team 10: +3 at home, -1 away -> (3 - (-1)) / 2 = 2; the league's home edge is 2; so no extra, and the 2025 result is not used
    assert abs(f[5]) < 1e-12


def test_environment_and_volume():
    g = pl.DataFrame({"game_id": [1, 2], "season": [2025, 2025], "date": [date(2025, 10, 1), date(2025, 10, 3)], "home_id": [10, 10], "away_id": [20, 30], "xg_total": [8.0, 6.0]})
    e = {r["game_id"]: r for r in environment(g).iter_rows(named=True)}
    assert e[1]["min_played"] == 0 and e[2]["min_played"] == 0 and abs(e[1]["exp_total"] - 7.0) < 1e-12        # prior: the league average
    assert abs(e[2]["exp_total"] - ((20 * 7 + 8) / 21 + 7) / 2) < 1e-12
    shots = pl.DataFrame({"game_id": [1, 1, 1], "season": [2025] * 3, "team_id": [10, 10, 20], "xg_v1": [0.1, 0.1, 0.1]})
    v = volume_target(shots, pl.DataFrame({"game_id": [1], "home_id": [10], "away_id": [20]}))
    assert abs(v["vol_gd"][0] - 0.1) < 1e-12
