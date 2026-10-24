import numpy as np
import polars as pl

from src.transition.opm import add_transition


def test_tr_is_orthogonal_to_pm_within_position():
    rng = np.random.default_rng(0)
    n = 300
    rows = []
    for pos, beta in (("D", 3.0), ("F", 4.0)):
        pm = rng.normal(0, 0.2, n)
        opm = beta * pm + rng.normal(0, 0.5, n)
        for i in range(n):
            rows.append(
                {
                    "player_id": len(rows),
                    "season": 2020,
                    "pos_group": pos,
                    "fringe": False,
                    "toi_5v5": 800.0,
                    "off_val": 0.0,
                    "opm_rate60": 0.0,
                    "opm_gar": float(opm[i]),
                    "pm_gar": float(pm[i]),
                }
            )
    df = pl.DataFrame(rows)
    opm = df.drop("pm_gar")
    t = df.select("player_id", "season", "pm_gar")
    tr, beta = add_transition(opm, t)
    assert abs(beta["D"] - 3.0) < 0.3 and abs(beta["F"] - 4.0) < 0.3
    for pos in ("D", "F"):
        s = tr.filter(pl.col("pos_group") == pos)
        assert abs(float(np.corrcoef(s["tr_gar"], s["pm_gar"])[0, 1])) < 0.05  # no PM-EPA left in TR
