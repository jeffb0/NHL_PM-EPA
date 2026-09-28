"""Value-function estimators: shrinkage, Markov solve, fallbacks, scoring."""

import numpy as np
import polars as pl
import pytest

from src.value.estimators import GBMValue, TabularValue, _eb_shrink, method_of_moments_k
from src.value.fit import score


def frame(rows):
    return pl.DataFrame(rows, schema_overrides={"x": pl.Float64, "y": pl.Float64, "G": pl.Float64}, infer_schema_length=None)


def obs(kind, x, y, g, strength="5v5", **kw):
    return {"obs_kind": kind, "x": x, "y": y, "strength": strength, "score_diff": 0, "G": g, **kw}


def test_eb_shrink_limits():
    n = np.array([0, 10, 1e9])
    out = _eb_shrink(n, np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.1, 0.1]), k=10)
    assert out[0] == pytest.approx(0.1)          # no data -> prior
    assert out[1] == pytest.approx(0.3)          # n == k -> midpoint
    assert out[2] == pytest.approx(0.5, abs=1e-6)  # lots of data -> raw mean


def test_method_of_moments_k_large_when_no_between_variance():
    n = np.full(20, 100.0)
    k = method_of_moments_k(n, np.full(20, 0.05), var_within=0.01)
    assert k >= 1e4


def test_tabular_mc_recovers_cell_means_with_k0():
    rows = [obs("takeaway", 80, 0, 0.2)] * 50 + [obs("takeaway", 80, 0, 0.0)] * 50 + [obs("takeaway", -80, 0, 0.01)] * 100
    m = TabularValue("zones", k=0).fit(frame(rows))
    p = m.predict(frame([obs("takeaway", 80, 0, 0), obs("shot", 82, 1, 0), obs("takeaway", -80, 0, 0)]))
    assert p[0] == pytest.approx(0.1) and p[1] == pytest.approx(0.1)  # 'shot' location valued as control
    assert p[2] == pytest.approx(0.01)


def test_unseen_cell_falls_back_to_parent_then_kind():
    rows = [obs("takeaway", 80, 0, 0.1)] * 10 + [obs("takeaway", -80, 0, 0.0)] * 10
    m = TabularValue("zones", k=0).fit(frame(rows))
    # OZ_circle_L unseen -> parent OZ (0.1)
    assert m.predict(frame([obs("takeaway", 75, 20, 0)]))[0] == pytest.approx(0.1)
    # PK unseen entirely -> strength-group fallback chain ends at 0 without crashing
    assert np.isfinite(m.predict(frame([obs("takeaway", 75, 20, 0, strength="PK")]))[0])


def test_shrinkage_pulls_thin_cells_toward_parent():
    rows = [obs("takeaway", 80, 0, 1.0)] * 2 + [obs("takeaway", 60, 0, 0.0)] * 200
    raw = TabularValue("zones", k=0).fit(frame(rows)).predict(frame([obs("takeaway", 80, 0, 0)]))[0]
    shr = TabularValue("zones", k=50).fit(frame(rows)).predict(frame([obs("takeaway", 80, 0, 0)]))[0]
    assert raw == pytest.approx(1.0) and shr < 0.1


def test_markov_solve_matches_analytic_two_step_chain():
    # state A (NZ takeaway) always moves to mid state B (OZ hit_mid); B ends with xG 0.2 half the time
    rows = []
    for i in range(100):
        g = 0.2 if i % 2 == 0 else 0.0
        rows.append(obs("takeaway", 10, 0, g, next_idx=2 * i + 1, R=g))
        rows.append(obs("hit_mid", 80, 0, g, next_idx=None, R=g))
    f = frame(rows).with_columns(pl.col("next_idx").cast(pl.Float64))
    m = TabularValue("zones", k=0, method="markov").fit(f)
    va, vb = m.predict(frame([obs("takeaway", 10, 0, 0), obs("hit_mid", 80, 0, 0)]))
    assert vb == pytest.approx(0.1)
    assert va == pytest.approx(0.1)  # V(A) = V(B) under deterministic A->B


def test_markov_bootstraps_where_mc_cannot():
    # A's own returns are noisy (few rows) but B is well-estimated: Markov uses V(B), MC uses A's returns
    rows = [obs("hit_mid", 80, 0, 0.1 if i % 2 else 0.0, next_idx=None, R=0.1 if i % 2 else 0.0) for i in range(1000)]
    rows += [obs("takeaway", 10, 0, 0.0, next_idx=0, R=0.0) for _ in range(3)]  # A -> B, all three got 0
    f = frame(rows).with_columns(pl.col("next_idx").cast(pl.Float64))
    mk = TabularValue("zones", k=0, method="markov").fit(f).predict(frame([obs("takeaway", 10, 0, 0)]))[0]
    mc = TabularValue("zones", k=0, method="mc").fit(f).predict(frame([obs("takeaway", 10, 0, 0)]))[0]
    assert mc == pytest.approx(0.0) and mk == pytest.approx(0.05, abs=1e-3)


def test_gbm_learns_location_signal():
    rng = np.random.default_rng(0)
    x = rng.uniform(-99, 99, 6000)
    g = np.where(x > 60, 0.15, 0.01) * rng.uniform(0.5, 1.5, 6000)
    f = frame([obs("takeaway", xi, 0.0, gi) for xi, gi in zip(x, g)])
    m = GBMValue(seed=0, max_iter=100).fit(f)
    p = m.predict(frame([obs("takeaway", 80, 0, 0), obs("takeaway", -80, 0, 0)]))
    assert p[0] > 5 * p[1]


def test_score_perfect_and_constant_predictions():
    g = np.array([0.0, 0.1, 0.2, 0.3] * 25)
    s = score(g.copy(), g, (g > 0.25).astype(float))
    assert s["mse"] < 1e-10 and s["calib_slope"] == pytest.approx(1.0, rel=1e-4)  # 0s clipped to 1e-6
    s0 = score(np.full_like(g, g.mean()), g, g * 0)
    assert s0["r2_vs_const"] == pytest.approx(0.0, abs=1e-9)


def test_auto_shrinkage_is_per_level_and_never_worse_than_parent_level():
    # many fine cells with identical means inside two very different zones: the fine level has ~no
    # between-cell variance (huge k), but zone means must NOT be shrunk away.
    rng = np.random.default_rng(1)
    rows = []
    for x in np.linspace(70, 88, 30):
        rows += [obs("takeaway", x, 0.0, g) for g in rng.binomial(1, 0.2, 60) * 0.5]
    for x in np.linspace(-88, -70, 30):
        rows += [obs("takeaway", x, 0.0, g) for g in rng.binomial(1, 0.02, 60) * 0.5]
    m = TabularValue("grid10", k="auto").fit(frame(rows))
    hi, lo = m.predict(frame([obs("takeaway", 80, 0, 0), obs("takeaway", -80, 0, 0)]))
    assert hi > 0.08 and lo < 0.03
    assert set(m.k_used) == {"kind", "parent", "state"}


def test_vectorized_predict_matches_reference_loop():
    rng = np.random.default_rng(3)
    rows = [obs(k, float(x), float(y), float(g), strength=st) for k, x, y, g, st in zip(
        rng.choice(["takeaway", "faceoff", "loose_opp_miss", "loose_unobserved"], 3000), rng.uniform(-99, 99, 3000),
        rng.uniform(-42, 42, 3000), rng.uniform(0, 0.2, 3000), rng.choice(["5v5", "PP", "PK", "3v3"], 3000))]
    m = TabularValue("grid10", k="auto").fit(frame(rows))
    test = frame(rows[:500] + [obs("takeaway", 1.0, 1.0, 0, strength="4v4"), obs("shot", None, None, 0)])
    assert np.allclose(m.predict(test), m._predict_loop(test))
