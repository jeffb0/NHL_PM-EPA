"""Joint calibration: ridge-to-1, cross-validation, fixed-effect regression, unshrunk builders."""

import numpy as np
import polars as pl
import pytest

from src.calibration import player, team


def test_ridge_to_one_recovers_weights_with_strong_data_and_returns_one_at_infinite_penalty():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 3, (400, 3))
    w_true = np.array([1.6, 0.7, 1.0])
    y = 2.0 + X @ w_true + rng.normal(0, 1.0, 400)
    a, w = team.ridge_to_one(X, y, lam=1e-6)
    assert np.allclose(w, w_true, atol=0.05) and a == pytest.approx(2.0, abs=0.2)
    a1, w1 = team.ridge_to_one(X, y, lam=np.inf)
    assert np.all(w1 == 1.0) and a1 == pytest.approx(np.mean(y - X.sum(1)))
    _, wm = team.ridge_to_one(X, y, lam=1e5)
    assert np.all(np.abs(wm - 1.0) < np.abs(w - 1.0) + 1e-9)         # a heavy penalty pulls every weight toward 1


def test_cross_validation_prefers_calibration_when_the_scale_is_wrong_and_ones_when_it_is_right():
    rng = np.random.default_rng(1)
    n, seasons = 270, np.repeat(np.arange(9), 30)
    X = rng.normal(0, 4, (n, 2))
    for w_true, expect_calibrated in ((np.array([1.7, 0.6]), True), (np.array([1.0, 1.0]), False)):
        y = X @ w_true + rng.normal(0, 3, n)
        cv = team.loso_cv(X, y, seasons, team.LAMBDAS)
        best = int(np.argmin(cv["total"][:-1]))
        diff = cv["sq_err"][-1] - cv["sq_err"][best]
        z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
        assert (z > 2) == expect_calibrated


def test_within_group_regression_removes_team_effects_and_matches_the_true_slope():
    rng = np.random.default_rng(2)
    n_groups, per = 60, 25
    g = np.repeat([f"g{i}" for i in range(n_groups)], per)
    team_fx = np.repeat(rng.normal(0, 5, n_groups), per)               # strong team effects unrelated to x
    x = rng.normal(0, 1, n_groups * per) + np.repeat(rng.normal(0, 2, n_groups), per) * 0.0
    w = rng.uniform(200, 1500, n_groups * per)
    y = team_fx + 0.8 * x + rng.normal(0, 0.5, n_groups * per)
    fit = player.wls_fe(y, x[:, None], w, g)
    assert fit["beta"][0] == pytest.approx(0.8, abs=0.05)
    assert 0.5 * fit["se"][0] < 0.05 and fit["r2_within"] > 0.5
    assert set(fit["sse_by_group"]) == set(np.unique(g))


def test_apply_weights_scales_shrunk_components_and_leaves_penalties_and_uncertainty_consistent():
    t = pl.DataFrame({"pm_gar": [1.0, 2.0], "def_gar": [0.5, -1.0], "fin_gar": [1.0, 0.0], "to_gar": [-0.3, 0.1], "fo_gar": [0.2, 0.0], "pen_gar": [0.4, -0.1],
                      "pm_gar_sd": [0.2, 0.2], "def_gar_sd": [0.3, 0.3], "fin_gar_sd": [0.4, 0.4], "to_gar_sd": [0.1, 0.1], "fo_gar_sd": [0.1, 0.1], "pen_gar_sd": [0.05, 0.05],
                      "total_gar": [2.8, 1.0]})   # equals the sum of the six components
    w = {"pm": 1.5, "def": 1.0, "fin": 2.0, "to": 1.0, "fo": 1.0}
    c = team.apply_weights(t, w, gpw=6.0)
    assert c["total_gar_cal"].to_list() == pytest.approx([1.5 + 0.5 + 2.0 - 0.3 + 0.2 + 0.4, 3.0 - 1.0 + 0.0 + 0.1 + 0.0 - 0.1])
    assert c["total_war_cal"][0] == pytest.approx(c["total_gar_cal"][0] / 6.0)
    ones = team.apply_weights(t, {k: 1.0 for k in w}, gpw=6.0)
    assert ones["cal_change_gar"].abs().max() == pytest.approx(0.0)
    assert ones["total_gar_cal_sd"][0] == pytest.approx(np.sqrt(0.2**2 + 0.3**2 + 0.4**2 + 0.1**2 + 0.1**2 + 0.05**2))


def test_team_table_sums_unshrunk_and_shrunk_columns_by_primary_team(tmp_path):
    games = pl.DataFrame({"game_id": [1], "season": [2024], "home_id": [10], "away_id": [20], "home_abbrev": ["AAA"], "away_abbrev": ["BBB"], "home_final": [3], "away_final": [1]})
    ev = pl.DataFrame({"game_id": [1] * 4, "season": [2024] * 4, "type": ["goal"] * 4, "team_id": [10, 10, 20, 10], "opp_team_id": [20, 20, 10, 20], "strength": ["5v5"] * 4})
    games.write_parquet(tmp_path / "games.parquet")
    ev.write_parquet(tmp_path / "events.parquet")
    cols = {f"{c}": [1.0, 2.0, 5.0] for c in ("pm_raw_gar", "pm_gar", "def_gar", "fin_raw_gar", "fin_gar", "to_raw_gar", "to_gar", "fo_raw_gar", "fo_gar")}
    t = pl.DataFrame({"season": [2024] * 3, "team": ["AAA", "AAA", "BBB"], **cols})
    j = team.team_table(t, tmp_path)
    a = j.filter(pl.col("team") == "AAA").row(0, named=True)
    assert a["pm_raw_gar"] == pytest.approx(3.0) and a["gd5"] == 2 and j.filter(pl.col("team") == "BBB")["gd5"][0] == -2   # 3 goals for, 1 against


def test_apply_weights_all():
    import polars as pl

    from src.calibration.team import SHR, apply_weights_all
    comps = ["pm", "pen", "pp"]
    t = pl.DataFrame({SHR["pm"]: [1.0, 2.0], SHR["pen"]: [0.5, -0.5], SHR["pp"]: [2.0, 1.0], "pm_gar_sd": [0.1, 0.1], "pen_gar_sd": [0.1, 0.1], "st_pp_gar_sd": [0.1, 0.1]})
    t = t.with_columns((pl.col("pm_gar") + pl.col("pen_gar") + pl.col("st_pp_gar")).alias("total_gar_all"))
    same = apply_weights_all(t, {c: 1.0 for c in comps}, 5.9, "cal")
    assert same["cal_change_gar"].abs().max() < 1e-12
    w = apply_weights_all(t, {"pm": 2.0, "pen": 1.0, "pp": 0.5}, 5.9, "ridge")
    assert abs(w["total_gar_all_ridge"][0] - (2.0 * 1.0 + 0.5 + 0.5 * 2.0)) < 1e-12 and abs(w["ridge_change_gar"][0] - (3.5 - 3.5)) < 1e-12
