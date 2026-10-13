import polars as pl

from src.deshrink.run import kappa_before
from src.possession.scale import rescale
from src.projection.data import COMPS


def _panel():
    base = {f"r_{c}": [0.1, -0.2] for c in COMPS}
    return pl.DataFrame({"player_id": [1, 2], "pos_group": ["F", "D"], **base}).with_columns(pl.sum_horizontal([f"r_{c}" for c in COMPS]).alias("r_total"))


def test_rescale_play_block_only():
    p = _panel()
    q = rescale(p, {"play5": 1.5})
    assert q["r_pm"].to_list() == [0.15000000000000002, -0.30000000000000004] and q["r_def"][0] == q["r_to"][0] == q["r_fo"][0] and q["r_fin"].to_list() == [0.1, -0.2]
    assert abs(q["r_total"][0] - (4 * 0.15 + 6 * 0.1)) < 1e-12
    assert rescale(p, None) is p and rescale(p, {"play5": 1.0}) is p
    assert abs(rescale(p, {"F": 2.0, "D": 1.0})["r_pm"][0] - 0.2) < 1e-12 and rescale(p, {"F": 2.0, "D": 1.0})["r_pm"][1] == -0.2      # the possession-only form still works


def test_kappa_uses_earlier_seasons_only():
    t = pl.DataFrame({"season": [2020, 2020, 2021, 2021], "o_play5": [2.0, -2.0, 9.0, -9.0], "e_play5": [1.0, -1.0, 1.0, -1.0]})
    assert abs(kappa_before(t, 2021) - 2.0) < 1e-12 and abs(kappa_before(t) - 5.5) < 1e-12
