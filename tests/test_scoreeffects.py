import polars as pl

from src.scoreeffects.factors import apply, weights


def _rates():
    rows = []
    for p, base in ((1, 2.5), (3, 2.4)):
        for d, mult in ((-2, 1.1), (-1, 1.1), (0, 1.0), (1, 1.0), (2, 0.9)):
            m = 1000.0
            rows.append(
                {"season": 2020, "period": p, "d": d, "xg": base * (mult if p == 3 else 1.0) * m / 60, "minutes": m}
            )
    return pl.DataFrame(rows)


def test_weights_are_relative_to_the_tied_rate_in_the_same_period():
    w = {(r["period"], r["d"]): r["w"] for r in weights(_rates(), [2020]).iter_rows(named=True)}
    assert abs(w[(1, -1)] - 1.0) < 1e-12 and abs(w[(3, 0)] - 1.0) < 1e-12
    # trailing shots discounted, leading shots marked up
    assert abs(w[(3, -1)] - 1 / 1.1) < 1e-12 and abs(w[(3, 2)] - 1 / 0.9) < 1e-12
    # no earlier season: no adjustment
    assert weights(_rates(), [2019]).height == 0


def test_apply_touches_only_5v5_shots_in_regulation_and_caps_the_differential():
    w = weights(_rates(), [2020])
    shots = pl.DataFrame(
        {
            "period": [3, 3, 3, 4, 1],
            "score_diff": [-1, -5, -1, -1, 0],
            "strength": ["5v5", "5v5", "PP", "5v5", "5v5"],
            "xg": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    a = apply(shots, w)["xg_adj"].to_list()
    # down five is treated as down two
    assert abs(a[0] - 1 / 1.1) < 1e-12 and abs(a[1] - 1 / 1.1) < 1e-12
    # power play, overtime and tied shots are unchanged
    assert a[2] == 1.0 and a[3] == 1.0 and a[4] == 1.0
    assert apply(shots, weights(_rates(), [2019]))["xg_adj"].to_list() == [1.0] * 5
