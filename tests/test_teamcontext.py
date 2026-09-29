import polars as pl

from src.teamcontext.data import add_context


def test_context_is_leave_one_out_uses_only_the_previous_season_and_flags_movers():
    rows = [
        # season 2018: team A has players 1 (rate 1.0, 100 min) and 2 (rate 3.0, 300 min); team B has player 3 (rate -1.0, 200 min)
        (1, 2018, "A", 100.0, 1.0), (2, 2018, "A", 300.0, 3.0), (3, 2018, "B", 200.0, -1.0),
        # season 2019: player 1 stays with A, player 2 moves to B, player 3 stays with B, player 4 is new on A
        (1, 2019, "A", 100.0, 9.0), (2, 2019, "B", 300.0, 9.0), (3, 2019, "B", 200.0, 9.0), (4, 2019, "A", 100.0, 9.0),
    ]
    p = pl.DataFrame(rows, schema=["player_id", "season", "team", "m", "r_total"], orient="row").with_columns(pl.lit("F").alias("pos_group"))
    x = add_context(p).filter(pl.col("season") == 2019).sort("player_id")
    r = {row["player_id"]: row for row in x.iter_rows(named=True)}
    # player 1 stayed on A: context = other A players in 2018 = player 2 (3.0); the 2019 rate 9.0 must not enter
    assert abs(r[1]["ctx_dest"] - 3.0) < 1e-9 and abs(r[1]["ctx_origin"] - 3.0) < 1e-9 and r[1]["mover"] == 0.0
    # player 2 moved from A to B: destination B in 2018 = player 3 (-1.0); origin A without himself = player 1 (1.0)
    assert abs(r[2]["ctx_dest"] + 1.0) < 1e-9 and abs(r[2]["ctx_origin"] - 1.0) < 1e-9 and r[2]["mover"] == 1.0
    # player 4 has no 2018 season: destination A in 2018 = (100 * 1 + 300 * 3) / 400 = 2.5; not a mover
    assert abs(r[4]["ctx_dest"] - 2.5) < 1e-9 and r[4]["mover"] == 0.0
