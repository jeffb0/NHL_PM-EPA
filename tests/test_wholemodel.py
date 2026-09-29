import numpy as np
import polars as pl

from src.wholemodel import build as wb
from src.wholemodel.run import paired


def _df():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(2018, 2024):
        for t in range(12):
            x = rng.normal(0, 10)
            rows.append({"season": s, "team": f"T{t}", "gd_all": 2.0 * x + rng.normal(0, 3), "G": rng.normal(0, 1), "S": x})
    return pl.DataFrame(rows)


def test_expanding_fit_uses_only_past_seasons_and_recovers_slope():
    df = _df()
    p = wb.expanding_fit(df, ["G", "S"], [2020, 2021, 2022])
    assert p["season"].unique().sort().to_list() == [2020, 2021, 2022] and p.height == 36
    assert float(((p["y"] - p["pred"]) ** 2).mean()) < 30                      # the true slope is 2, the noise sd 3
    fixed = wb.expanding_fit(df, ["G"], [2020, 2021, 2022], fixed_slope="S")
    assert float(((fixed["y"] - fixed["pred"]) ** 2).mean()) > float(((p["y"] - p["pred"]) ** 2).mean())   # a slope fixed at 1 is worse when the truth is 2


def test_paired_sign():
    a = pl.DataFrame({"season": [1, 1, 1, 1], "team": list("abcd"), "y": [1.0, 2.0, 3.0, 4.0], "pred": [1.1, 2.1, 3.1, 4.1]})
    b = a.with_columns((pl.col("pred") + 1.0).alias("pred"))
    r = paired(a, b)
    assert r["mean_diff"] > 0                                                  # b - a squared error: positive favors a
