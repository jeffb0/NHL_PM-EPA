import math

import polars as pl

from src.prospects.features import ELITE, elite_features, factors, features, spec, transitions, with_yp


def _st():
    rows = []
    # 30 players: 0.8 points per game in league "AHL" in 2020, then 0.4 in the NHL in 2021 -> factor 0.5
    for i in range(30):
        rows += [(i, 2020, "AHL", 60.0, 48.0), (i, 2021, "NHL", 70.0, 28.0)]
    # player 100: a 2022 "AHL" season and nothing in the NHL yet; player 101: only a later season
    rows += [(100, 2022, "AHL", 50.0, 50.0), (101, 2024, "AHL", 50.0, 50.0)]
    return pl.DataFrame(
        rows,
        schema={"player_id": pl.Int64, "season": pl.Int64, "league": pl.Utf8, "gp": pl.Float64, "points": pl.Float64},
        orient="row",
    )


def test_factor_uses_only_earlier_transitions():
    tr = transitions(_st())
    assert tr.height == 30
    f, pooled = factors(tr, 2022)
    assert abs(f["AHL"] - 0.5) < 1e-9 and abs(pooled - 0.5) < 1e-9
    assert factors(tr, 2021)[0] == {}  # the NHL seasons are 2021: not known before 2021


def test_features_respect_the_season_boundary_and_fade():
    st = _st()
    draft = pl.DataFrame(
        {"player_id": [100, 101], "overall": [10.0, None], "draft_year": [2021, None]},
        schema={"player_id": pl.Int64, "overall": pl.Float64, "draft_year": pl.Int64},
    )
    f = features(st, draft, transitions(st), 2023)
    a = f.filter(pl.col("player_id") == 100).row(0, named=True)
    b = f.filter(pl.col("player_id") == 101).row(0, named=True)
    assert (
        abs(a["nhle"] - 0.5) < 1e-9
        and a["has_nhle"] == 1.0
        and abs(a["pick"] - math.log(10)) < 1e-9
        and a["undrafted"] == 0.0
    )
    # his 2024 season is in the future of 2023
    assert b["nhle"] == 0.0 and b["has_nhle"] == 0.0 and b["undrafted"] == 1.0
    frame = pl.DataFrame({"player_id": [100, 100], "season": [2023, 2023], "m_eff": [0.0, 3000.0]})
    w = with_yp(frame, f)
    assert abs(w["yp_nhle"][0] - 0.5) < 1e-9 and abs(w["yp_nhle"][1] - 0.125) < 1e-9


def test_elite_features_slot_bins_and_debut_timing():
    st = _st()
    # player 200: first overall in 2023, no NHL game before 2023 -> debuts in his draft year (D0); player 201: 7th in
    # 2021, first NHL season 2022 (D1); player 202: undrafted veteran
    st = pl.concat(
        [
            st,
            pl.DataFrame(
                [(201, 2022, "NHL", 60.0, 20.0), (202, 2019, "NHL", 70.0, 30.0)], schema=st.schema, orient="row"
            ),
        ]
    )
    draft = pl.DataFrame(
        {"player_id": [200, 201, 202], "overall": [1.0, 7.0, None], "draft_year": [2023, 2021, None]},
        schema={"player_id": pl.Int64, "overall": pl.Float64, "draft_year": pl.Int64},
    )
    f = {r["player_id"]: r for r in elite_features(st, draft, transitions(st), 2023).to_dicts()}
    assert (f[200]["top3"], f[200]["d0"], f[200]["d1"], f[200]["nhl_gp_before"]) == (1.0, 1.0, 0.0, 0.0)
    assert (f[201]["p4_10"], f[201]["d0"], f[201]["d1"], f[201]["nhl_gp_before"]) == (1.0, 0.0, 1.0, 60.0)
    assert (f[202]["top3"], f[202]["p4_10"], f[202]["p11_32"], f[202]["d0"], f[202]["d1"]) == (0.0, 0.0, 0.0, 0.0, 0.0)
    tab = pl.DataFrame(list(f.values()))
    assert spec(tab).extra == ELITE
    w = with_yp(pl.DataFrame({"player_id": [200, 200], "season": [2023, 2023], "m_eff": [0.0, 1000.0]}), tab)
    # the prior fades as NHL minutes accumulate
    assert w["yp_top3"].to_list() == [1.0, 0.5] and w["yp_d0"].to_list() == [1.0, 0.5]


def test_early_adjustment_applies_only_to_thin_players_with_enough_games():
    import numpy as np

    from src.projection.model import wridge
    from src.prospects.hybrid import CHECK, Early, signal_row

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    # only the first signal (ice time per game) matters
    m = wridge(X, 0.05 * X[:, 0] + rng.normal(0, 0.01, 200), np.ones(200), 0.01)
    e = Early({1}, {"F": {2: m, 5: m}}, {1: "F", 2: "F"})
    c = np.array([3.0, 60.0, 45.0, 0.1, 0.5, 2.0, 1.0, 0.5])  # 3 games, 60 minutes
    assert abs(signal_row(c)[0] - 20.0) < 1e-12 and abs(signal_row(c)[4] - (0.5 / 45 * 60)) < 1e-12
    assert e.adj(2, c) == 0.0  # not a thin-history player
    # one game: below the first checkpoint
    assert e.adj(1, np.array([1.0, 20.0, 15.0, 0, 0, 0, 0, 0])) == 0.0
    assert abs(e.adj(1, c) - float(m.predict(signal_row(c)[None, :])[0])) < 1e-12 and CHECK[0] == 2
