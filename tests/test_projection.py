import numpy as np
import polars as pl

from src.projection import data as pd_
from src.projection import model as pm


def _panel():
    rows = []
    for pid in (1, 2):
        for s in (2017, 2018, 2019, 2020):
            rows.append({"player_id": pid, "season": s, "pos_group": "F", "team": "AAA", "m": 1000.0, **{f"r_{c}": (0.1 * pid + 0.01 * s % 7) for c in pd_.COMPS}})
    p = pl.DataFrame(rows)
    return p.with_columns(pl.sum_horizontal([f"r_{c}" for c in pd_.COMPS]).alias("r_total"))


def test_history_uses_only_seasons_up_to_t_and_weights_recent_more():
    p = _panel()
    p_future = p.with_columns(pl.when(pl.col("season") == 2020).then(pl.col("r_total") + 5.0).otherwise(pl.col("r_total")).alias("r_total"))
    a = pd_.history(p).filter((pl.col("player_id") == 1) & (pl.col("t") == 2019))
    b = pd_.history(p_future).filter((pl.col("player_id") == 1) & (pl.col("t") == 2019))
    assert a["rbar_total"][0] == b["rbar_total"][0]                      # a later season cannot change features at t = 2019
    r = p.filter(pl.col("player_id") == 1).sort("season")["r_total"].to_list()
    expect = (5 * r[2] + 4 * r[1] + 3 * r[0]) / 12                       # weights 5/4/3 with equal minutes: the last season counts most
    assert abs(a["rbar_total"][0] - expect) < 1e-9


def test_ridge_recovers_a_linear_signal_and_age_basis_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(0, 0.1, 500)
    m = pm.wridge(X, y, np.ones(500), 1e-4)
    assert np.allclose(m.raw_coef(), [2.0, -1.0, 0.0], atol=0.05)
    b = pm.age_basis(np.array([20.0, 27.0, 34.0]))
    assert b.shape == (3, 4) and b[1, 0] == 0.0 and b[2, 3] == 2.0


def test_variance_model_and_optional_prior_covariates():
    m = np.array([300.0, 600.0, 1200.0, 2400.0] * 50)
    res = np.random.default_rng(1).normal(0, np.sqrt(0.01 + 20.0 / m))
    a, b = pm.variance_model(m, res, m)
    assert a >= 0 and b > 0
    spec = pm.Spec(extra=("edge_speed",))
    assert "edge_speed" in pm.feature_names(spec)
    d = pl.DataFrame({**{f"rt_{c}": [0.0, 0.1] for c in pd_.COMPS}, "no_history": [0.0, 1.0], "age": [25.0, 30.0], "height_in": [72.0, 74.0], "weight_lb": [190.0, 200.0], "edge_speed": [22.0, 23.5]})
    assert pm.xmat(d, spec).shape == (2, len(pm.feature_names(spec)))
