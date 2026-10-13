from datetime import date

import polars as pl

from src.schedule.features import ARENAS, BLOCKS, game_features, km


def _games():
    rows = [(1, date(2025, 10, 1), "AAA", "BBB"), (2, date(2025, 10, 2), "CCC", "BBB"), (3, date(2025, 10, 4), "AAA", "BBB"), (4, date(2025, 10, 10), "BBB", "AAA")]
    return pl.DataFrame({"game_id": [r[0] for r in rows], "season": [2025] * 4, "date": [r[1] for r in rows], "home_abbrev": [r[2] for r in rows], "away_abbrev": [r[3] for r in rows]})


def test_distance_and_arenas():
    assert abs(km("NYR", "LAK") - 3940) < 60 and km("BOS", "BOS") == 0.0
    assert len(ARENAS) == 33 and all(-8 <= v[2] <= -5 for v in ARENAS.values())


def test_schedule_features(monkeypatch):
    monkeypatch.setitem(ARENAS, "AAA", ARENAS["NYR"])
    monkeypatch.setitem(ARENAS, "BBB", ARENAS["LAK"])
    monkeypatch.setitem(ARENAS, "CCC", ARENAS["CHI"])
    f = {r["game_id"]: r for r in game_features(_games()).iter_rows(named=True)}
    assert set(f[1]) == {"game_id", *sum(BLOCKS.values(), [])}
    # opener: both rested, the visitor has crossed the continent and is three zones east of home
    assert f[1]["rest_adv"] == 0 and f[1]["km_h"] == 0 and abs(f[1]["km_a"] - 3.94) < 0.06 and f[1]["east_a"] == 3 and f[1]["west_a"] == 0 and f[1]["trip_a"] == 1
    # the visitor's second road game, on a back-to-back, against a rested home team
    assert f[2]["rest_adv"] == 2 and f[2]["trip_a"] == 2 and f[2]["east_a"] == 2 and f[2]["dense_a"] == 0
    # third game in four nights for the visitor; the home team is in its second straight home game
    assert f[3]["dense_a"] == 1 and f[3]["dense_h"] == 0 and f[3]["trip_a"] == 3 and f[3]["stand_h"] == 2 and f[3]["rest_adv"] == 1
    # six days later the roles reverse: both off a layoff, the former home team now travels west
    assert f[4]["layoff_h"] == 1 and f[4]["layoff_a"] == 1 and f[4]["west_a"] == 3 and f[4]["trip_a"] == 1 and f[4]["stand_h"] == 1 and f[4]["km_h"] > 3.8
