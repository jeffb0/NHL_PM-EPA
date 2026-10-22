import polars as pl

from src.live.weekly import METRICS, colour, min_toi, ordinal, percentiles


def test_percentiles_by_position_with_floor_and_direction():
    n = 5
    p = pl.DataFrame({"pos": ["F"] * n + ["D"], "toi5": [100.0, 100.0, 100.0, 100.0, 10.0, 100.0],
                      **{m: [1.0, 2.0, 3.0, 4.0, 9.0, 0.0] for m, *_ in METRICS}})
    q = percentiles(p, floor=50.0)
    f = q.filter(pl.col("pos") == "F")
    assert f["p_off60"].to_list() == [12.5, 37.5, 62.5, 87.5, 100.0]            # ranked against the four with enough minutes
    assert f["p_def60"].to_list() == [87.5, 62.5, 37.5, 12.5, 0.0]              # xG against: fewer is better
    assert f["ranked"].to_list() == [True, True, True, True, False] and q.filter(pl.col("pos") == "D")["p_net60"][0] == 50.0


def test_colour_scale_and_helpers():
    assert colour(50.0)[0] == "#f0efec" and colour(100.0)[0] == "#1c5cab" and colour(0.0)[0] == "#b8302f" and colour(None)[0] == "#ffffff"
    assert colour(95.0)[1] == "#ffffff" and colour(60.0)[1] != "#ffffff"
    assert min_toi(3) == 60.0 and min_toi(20) == 140.0
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 32)] == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "32nd"]
