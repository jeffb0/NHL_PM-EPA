import math

import polars as pl

from src.prospects.features import factors, features, transitions, with_yp


def _st():
    rows = []
    # 30 players: 0.8 points per game in league "AHL" in 2020, then 0.4 in the NHL in 2021 -> factor 0.5
    for i in range(30):
        rows += [(i, 2020, "AHL", 60.0, 48.0), (i, 2021, "NHL", 70.0, 28.0)]
    # player 100: a 2022 "AHL" season and nothing in the NHL yet; player 101: only a later season
    rows += [(100, 2022, "AHL", 50.0, 50.0), (101, 2024, "AHL", 50.0, 50.0)]
    return pl.DataFrame(rows, schema={"player_id": pl.Int64, "season": pl.Int64, "league": pl.Utf8, "gp": pl.Float64, "points": pl.Float64}, orient="row")


def test_factor_uses_only_earlier_transitions():
    tr = transitions(_st())
    assert tr.height == 30
    f, pooled = factors(tr, 2022)
    assert abs(f["AHL"] - 0.5) < 1e-9 and abs(pooled - 0.5) < 1e-9
    assert factors(tr, 2021)[0] == {}                       # the NHL seasons are 2021: not known before 2021


def test_features_respect_the_season_boundary_and_fade():
    st = _st()
    draft = pl.DataFrame({"player_id": [100, 101], "overall": [10.0, None]})
    f = features(st, draft, transitions(st), 2023)
    a = f.filter(pl.col("player_id") == 100).row(0, named=True)
    b = f.filter(pl.col("player_id") == 101).row(0, named=True)
    assert abs(a["nhle"] - 0.5) < 1e-9 and a["has_nhle"] == 1.0 and abs(a["pick"] - math.log(10)) < 1e-9 and a["undrafted"] == 0.0
    assert b["nhle"] == 0.0 and b["has_nhle"] == 0.0 and b["undrafted"] == 1.0      # his 2024 season is in the future of 2023
    frame = pl.DataFrame({"player_id": [100, 100], "season": [2023, 2023], "m_eff": [0.0, 3000.0]})
    w = with_yp(frame, f)
    assert abs(w["yp_nhle"][0] - 0.5) < 1e-9 and abs(w["yp_nhle"][1] - 0.125) < 1e-9
