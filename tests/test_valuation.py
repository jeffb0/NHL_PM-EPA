"""Value integration: joins, scale conversions, replacement definition, and shrinkage."""

import numpy as np
import polars as pl
import pytest

from src.valuation import combine as cb
from src.valuation import gpw
from src.valuation.replacement import status


def test_goals_per_win_recovers_the_generating_slope_and_does_not_wrap_negative_margins():
    rng = np.random.default_rng(0)
    n = 300
    gd = rng.normal(0, 40, n)                       # includes many negative goal differentials
    gp = np.full(n, 82)
    wins = gp / 2 + gd / 6.0 + rng.normal(0, 2, n)  # true: 6 goals per win
    ts = pl.DataFrame({"season": rng.integers(2017, 2026, n), "gd": gd, "excess_wins": wins - gp / 2})
    res = gpw.goals_per_win(ts, reps=200, seed=1)
    assert res["goals_per_win"] == pytest.approx(6.0, rel=0.05)
    assert res["ci95"][0] < 6.0 < res["ci95"][1]


def test_team_seasons_goal_differential_is_signed(tmp_path):
    # team 10 loses 1-3 and 1-2 (goal differential -3); team 20 wins both (+3). Unsigned counts would wrap here.
    games = pl.DataFrame({"game_id": [1, 2], "season": [2024, 2024], "home_id": [10, 10], "away_id": [20, 20], "home_final": [1, 1], "away_final": [3, 2]})
    ev = pl.DataFrame({"game_id": [1, 1, 1, 1, 2, 2, 2], "type": ["goal"] * 7, "team_id": [10, 20, 20, 20, 10, 20, 20], "strength": ["5v5"] * 7})
    games.write_parquet(tmp_path / "games.parquet")
    ev.write_parquet(tmp_path / "events.parquet")
    ts = gpw.team_seasons(tmp_path).sort("team_id")
    assert ts["gd"].to_list() == [-3, 3]
    assert ts["wins"].to_list() == [0, 2]
    assert ts["excess_wins"].to_list() == [-1.0, 1.0]


def ps_frame():
    rows = []
    for team in ("AAA", "BBB"):
        for i in range(16):        # 16 forwards per team
            rows.append((f"{team}F{i}", team, "F", 1000.0 - 20 * i))
        for i in range(8):         # 8 defensemen per team
            rows.append((f"{team}D{i}", team, "D", 900.0 - 30 * i))
    return pl.DataFrame({"player_id": list(range(len(rows))), "season": 2024, "pos_group": [r[2] for r in rows], "primary_team": [r[1] for r in rows], "toi_total": [r[3] for r in rows]})


def test_regular_fringe_split_is_per_team_and_position():
    st = status(ps_frame(), n_f=12, n_d=6)
    assert st.filter(pl.col("pos_group") == "F")["fringe"].sum() == 2 * 4      # 16 - 12 per team
    assert st.filter(pl.col("pos_group") == "D")["fringe"].sum() == 2 * 2      # 8 - 6 per team
    top = st.filter(pl.col("team_rank") == 1)
    assert not top["fringe"].any()
    assert st.filter(pl.col("pos_group") == "D")["team_rank"].max() == 8


def test_pm_component_shrinks_noisy_rates_and_anchors_replacement_at_the_pooled_fringe_rate():
    ps = ps_frame().with_columns(pl.lit(2024).alias("season"))
    n = ps.height
    rng = np.random.default_rng(0)
    toi = ps["toi_total"].to_numpy()
    se = 0.9 / np.sqrt(toi)                           # more minutes -> smaller SE (0.04 at 500 min, comparable to the true spread)
    true = rng.normal(0.05, 0.04, n)
    rate = true + rng.normal(0, se)
    ps = ps.with_columns(pl.Series("toi_ev", toi), pl.Series("pm_epa_per60_ev", rate), pl.Series("boot_se_per60", se), pl.Series("pm_epa_ev", rate * toi / 60))
    st = status(ps, 12, 6)
    out, rep = cb.pm_component(ps, st, min_toi=500)
    j = ps.join(out, on=["player_id", "season"])
    assert (j["pm_post_per60"] - j["pm_rep_per60"]).abs().min() >= 0
    fringe_ids = st.filter(pl.col("fringe") & (pl.col("pos_group") == "F"))["player_id"].to_list()
    pool = ps.filter(pl.col("player_id").is_in(fringe_ids))
    pooled = float((pool["pm_epa_ev"].sum() / pool["toi_ev"].sum()) * 60)
    assert rep.filter(pl.col("pos_group") == "F")["pm_rep_per60"][0] == pytest.approx(pooled)
    # shrinkage: the posterior is pulled toward the group mean more for the noisier (low-TOI) player
    B = j.filter(pl.col("pos_group") == "F").sort("toi_ev")["pm_shrink_B"].to_numpy()   # same position -> same tau^2
    assert B[0] < B[-1] and np.all(np.diff(B) >= 0)
    # value in goals = (rate - replacement) * TOI / 60
    r = j.row(0, named=True)
    assert r["pm_gar"] == pytest.approx((r["pm_post_per60"] - r["pm_rep_per60"]) * r["toi_ev"] / 60)


def test_combined_table_adds_components_converts_to_wins_and_propagates_uncertainty():
    ps = pl.DataFrame({"player_id": [1, 2], "season": [2024, 2024], "player_name": ["A", "B"], "position": ["C", "D"], "pos_group": ["F", "D"], "primary_team": ["X", "X"],
                       "toi_ev": [1000.0, 900.0], "gp": [80, 70], "pm_epa_per60_ev": [0.1, 0.0], "qualified_ev": [True, True]})
    pm = pl.DataFrame({"player_id": [1, 2], "season": [2024, 2024], "pm_gar": [2.0, 0.5], "pm_gar_sd": [0.3, 0.4], "pm_post_per60": [0.1, 0.0], "pm_rep_per60": [0.0, 0.0], "pm_shrink_B": [0.5, 0.5]})
    dp = pl.DataFrame({"player_id": [1, 2], "season": [2024, 2024], "def_gar": [-0.5, 1.5], "def_gar_sd": [0.4, 0.3], "dpm_var_per60": [0.0, 0.0], "dpm_rep_per60": [0.0, 0.0], "dpm_val_per60": [0.0, 0.0]})
    fin = pl.DataFrame({"player_id": [1], "season": [2024], "fin_gar": [1.2]})
    sd = pl.DataFrame({"player_id": [1], "season": [2024], "fin_gar_sd": [0.5]})
    t = cb.combine(ps, pm, dp, fin, sd, None, gpw=6.0).sort("player_id")
    assert t["total_gar"].to_list() == pytest.approx([2.7, 2.0])                         # missing finishing -> 0 for player 2
    assert t["total_war"].to_list() == pytest.approx([2.7 / 6, 2.0 / 6])
    assert t["total_gar_sd"][0] == pytest.approx(np.sqrt(0.3**2 + 0.4**2 + 0.5**2))
    assert t["pm_wins"][0] == pytest.approx(2.0 / 6)


def test_finishing_from_shifts_uses_the_fringe_pool_as_reference_and_only_5v5_shots():
    rows = []
    rng = np.random.default_rng(0)
    for p, T in ((1, 0.2), (2, 0.0), (3, -0.1)):          # player 3 is the fringe pool
        for _ in range(200):
            rows.append((p, 2024, "F", "5v5" if rng.random() < 0.8 else "PP", float(rng.normal(-2.5, 0.5)), T))
    sh = pl.DataFrame(rows, schema=["shooter", "season", "pos_group", "strength_g", "l", "T"], orient="row")
    st = pl.DataFrame({"player_id": [1, 2, 3], "season": [2024] * 3, "fringe": [False, False, True]})
    out = cb.fin_from_shifts(sh, st).sort("player_id")
    g = dict(zip(out["player_id"], out["fin_gar"]))
    assert g[1] > g[2] > 0 and g[3] == pytest.approx(0.0, abs=1e-9)                        # the pool sits at its own reference
    n5 = sh.filter((pl.col("shooter") == 1) & (pl.col("strength_g") == "5v5")).height
    assert n5 < 200                                                                         # some shots were non-5v5 and excluded
