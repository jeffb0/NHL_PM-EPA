"""Defensive PM-EPA: ledger classification, stint assignment and the ridge solver."""

import numpy as np
import polars as pl
import pytest

from src.defense import ridge
from src.defense.stints import COMPONENTS, _game, classify

C = len(COMPONENTS)


def test_classify_maps_every_ledger_row():
    df = pl.DataFrame({
        "phase": ["move", "terminal", "terminal", "start", "start", "start", "start", "start", "start"],
        "component": ["carry", "shot_attempt", "shot_quality", "takeaway", "unattributed", "unattributed", "unattributed", "unattributed", "unattributed"],
        "start_type": ["faceoff", "faceoff", "faceoff", "takeaway", "loose_opp_giveaway", "faceoff", "loose_own_miss", "rebound", "unobserved"],
    })
    got = df.select(classify(pl.col("phase"), pl.col("component"), pl.col("start_type")))[:, 0].to_list()
    assert got == ["move", "term", "quality", "start_turnover", "start_turnover", "start_faceoff", "start_recovery", "start_recovery", "start_unobserved"]


def _toy_game():
    home, away1, away2 = [1, 2, 3, 4, 5], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]
    sh = {
        "period": np.ones(15, dtype=int),
        "start": np.array([0.0] * 5 + [0.0] * 5 + [30.0] * 5),
        "end": np.array([60.0] * 5 + [30.0] * 5 + [60.0] * 5),
        "player_id": np.array(home + away1 + away2),
        "home": np.array([True] * 5 + [False] * 10),
    }
    # t=0 faceoff won by home at centre; t=10 shot; t=30 stoppage then faceoff won by home at x=-69 (home DZ);
    # t=45 away goal; home leads 1-0 throughout until the goal
    ev = {
        "period": np.ones(6, dtype=int), "t": np.array([0.0, 10.0, 30.0, 30.0, 45.0, 60.0]), "code": np.full(6, 1551),
        "is_fo": np.array([True, False, False, True, False, False]), "is_goal": np.array([False, False, False, False, True, False]),
        "x": np.array([0.0, 50.0, np.nan, -69.0, 80.0, np.nan]), "home": np.array([True, True, False, True, False, False]),
        "home_sd": np.array([1, 1, 1, 1, 1, 0]),
    }
    led = {
        "period": np.ones(3, dtype=int), "t": np.array([10.0, 30.0, 30.0]), "is_fo": np.array([False, False, True]),
        "att_home": np.array([True, False, False]),
        "comp": np.array([COMPONENTS.index("move"), COMPONENTS.index("term"), COMPONENTS.index("start_faceoff")]),
        "value": np.array([0.05, 0.02, 0.01]), "is_5v5": np.ones(3, dtype=bool),
    }
    return _game((99, sh, ev, led))


def test_stints_split_on_line_changes_and_faceoffs_with_zone_and_score():
    rows, q = _toy_game()
    assert len(rows) == 2
    s1, s2 = rows
    assert (s1[2], s1[3], s2[2], s2[3]) == (0.0, 30.0, 30.0, 60.0)
    assert s1[6] == [21, 22, 23, 24, 25] and s2[6] == [26, 27, 28, 29, 30]
    assert s1[7] == "NZ" and s2[7] == "DZ"  # home view: second faceoff is in home's defensive zone
    assert s1[8] == 1 and s2[8] == 1


def test_ledger_rows_follow_the_onice_convention():
    rows, q = _toy_game()
    s1, s2 = rows
    vals1, vals2 = np.array(s1[9:]), np.array(s2[9:])
    assert vals1[COMPONENTS.index("move")] == pytest.approx(0.05)  # home attacking
    assert vals1[C + COMPONENTS.index("term")] == pytest.approx(0.02)  # t=30 non-faceoff closes (0, 30]
    assert vals2[C + COMPONENTS.index("start_faceoff")] == pytest.approx(0.01)  # t=30 faceoff opens [30, 60)
    assert vals2[2 * C + 1] == 1 and vals1[2 * C + 1] == 0  # away goal at t=45
    assert q["ledger_abs_assigned"] == pytest.approx(q["ledger_abs_5v5"])


def test_first_event_after_a_penalty_expires_is_in_a_valid_stint():
    # away player 25 serves a penalty until t=20 (4 away skaters); the next event, a home shot at t=25, is coded 5v5
    home, away = [1, 2, 3, 4, 5], [21, 22, 23, 24]
    sh = {"period": np.ones(10, dtype=int), "start": np.array([0.0] * 9 + [20.0]), "end": np.full(10, 60.0),
          "player_id": np.array(home + away + [25]), "home": np.array([True] * 5 + [False] * 5)}
    ev = {"period": np.ones(3, dtype=int), "t": np.array([0.0, 25.0, 60.0]), "code": np.array([1451, 1551, 1551]),
          "is_fo": np.array([True, False, False]), "is_goal": np.zeros(3, dtype=bool), "x": np.array([0.0, 50.0, np.nan]),
          "home": np.array([True, True, False]), "home_sd": np.zeros(3, dtype=int)}
    led = {"period": np.ones(1, dtype=int), "t": np.array([25.0]), "is_fo": np.array([False]), "att_home": np.array([True]),
           "comp": np.array([COMPONENTS.index("term")]), "value": np.array([0.03]), "is_5v5": np.array([True])}
    rows, q = _game((1, sh, ev, led))
    assert [(r[2], r[3]) for r in rows] == [(20.0, 25.0), (25.0, 60.0)]  # the code change at t=25 is a boundary
    assert rows[0][9 + COMPONENTS.index("term")] == pytest.approx(0.03)
    assert q["ledger_abs_assigned"] == pytest.approx(0.03)


def _random_stints(n=400, seed=0):
    rng = np.random.default_rng(seed)
    players = np.arange(1, 41)
    data = {"game_id": rng.integers(0, 40, n), "minutes": rng.uniform(0.2, 1.5, n),
            "zone_home": rng.choice(["fly", "OZ", "NZ", "DZ"], n), "score_home": rng.integers(-2, 3, n)}
    data["home_on"] = [sorted(rng.choice(players[:20], 5, replace=False).tolist()) for _ in range(n)]
    data["away_on"] = [sorted(rng.choice(players[20:], 5, replace=False).tolist()) for _ in range(n)]
    for side in "ha":
        for c in COMPONENTS:
            data[f"{side}_{c}"] = rng.normal(0.01, 0.02, n) * data["minutes"]
        data[f"{side}_goals"] = rng.poisson(0.05, n).astype(float)
    return pl.DataFrame(data)


def test_ridge_matches_dense_penalized_least_squares():
    d = ridge.build_design(_random_stints(), ["xg"])
    beta = ridge.fit(d, {"xg": 50.0})[:, 0]
    X = d.X.toarray()
    y, w = d.y()[:, 0], d.w
    ref = np.linalg.solve(X.T @ (w[:, None] * X) + 50.0 * np.diag(d.pen), X.T @ (w * y))
    assert np.allclose(beta, ref, atol=1e-10)


def test_component_coefficients_sum_to_total_at_common_alpha():
    d = ridge.build_design(_random_stints(), COMPONENTS + ["xg"])
    beta = ridge.fit(d, {t: 300.0 for t in d.targets})
    assert np.abs(beta[:, :C].sum(1) - beta[:, C]).max() < 1e-8


def test_prior_pulls_coefficients_toward_prior_mean():
    d = ridge.build_design(_random_stints(), ["xg"])
    prev = ridge.coef_frame(d, np.zeros((d.X.shape[1], 1))).with_columns(pl.lit(1.0).alias("def_xg"))
    b0 = ridge.fit(d, {"xg": 1e6})[1 + d.P:1 + 2 * d.P, 0]
    b1 = ridge.fit(d, {"xg": 1e6}, ridge.prior_vector(d, prev, 0.5))[1 + d.P:1 + 2 * d.P, 0]
    assert np.allclose(b0, 0, atol=1e-3) and np.allclose(b1, 0.5, atol=1e-3)


def test_applying_a_fit_to_new_stints_drops_unknown_players():
    st = _random_stints()
    d = ridge.build_design(st, ["xg"], players=[1, 2, 21])
    assert d.X.shape[1] == 1 + 2 * 3 + len(ridge.COVARIATES)
