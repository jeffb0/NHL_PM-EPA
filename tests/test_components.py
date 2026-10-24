"""Turnover, faceoff and penalty components: joins, signs, accounting, shrinkage and scale conversions."""

import numpy as np
import polars as pl
import pytest

from src.components import common, faceoff, penalty, turnover
from src.valuation import combine as cb


# ------------------------------------------------------------------ faceoffs
def test_snap_maps_to_the_nine_dots_and_rejects_far_points():
    x = np.array([0.0, 69.0, -20.0, 20.0, 50.0, -68.0])
    y = np.array([0.0, 22.0, -22.0, 22.0, 0.0, -21.0])
    k = faceoff.snap(x, y)
    assert k[0] == 0 and faceoff.DOTS[k[1]] == (69.0, 22.0) and faceoff.DOTS[k[2]] == (-20.0, -22.0)
    assert k[4] == -1  # (50, 0) is not a dot
    assert faceoff.DOTS[k[5]] == (-69.0, -22.0)  # within tolerance


def _draw_frame(n=6):
    return pl.DataFrame(
        {
            "home_taker": [1, 1, 2, 2, 3, 3][:n],
            "away_taker": [4, 5, 4, 5, 4, 5][:n],
            "season": [2024] * n,
            "dot": [0, 5, 0, 5, 1, 2][:n],
            "side": [0.0, 1.0, -1.0, 0.0, 2.0, 0.0][:n],
            "frd": [0.0] * n,
            "y": [1, 0, 1, 1, 0, 0][:n],
        }
    )


def test_faceoff_design_has_signed_taker_columns_and_side_feature():
    d = _draw_frame()
    lay = faceoff.Layout(d, {1: "C", 2: "C", 3: "O", 4: "C", 5: "C"}, fringe=False)
    X = faceoff.design(d, lay).toarray()
    assert X[0, lay.u[1]] == 1.0 and X[0, lay.u[4]] == -1.0  # home taker +1, away taker -1
    assert X[0, lay.dots[0]] == 1.0 and X[1, lay.dots[5]] == 1.0
    assert X[4, lay.side_col] == 2.0
    assert X[0, lay.d[1 * 10000 + 2024]] == 1.0 and X[0, lay.d[4 * 10000 + 2024]] == -1.0


def test_faceoff_model_recovers_taker_skill_and_values_the_better_taker_positively():
    rng = np.random.default_rng(0)
    n_t = 40
    skill = rng.normal(0, 0.3, n_t)
    rows = []
    for s in (2021, 2022, 2023):
        for _ in range(2500):
            h, a = rng.choice(n_t, 2, replace=False)
            eta = skill[h] - skill[a]
            rows.append(
                (h + 1, a + 1, s, int(rng.integers(0, 9)), 0.0, 0.0, int(rng.random() < 1 / (1 + np.exp(-eta))), 0.02)
            )
    d = pl.DataFrame(rows, schema=["home_taker", "away_taker", "season", "dot", "side", "frd", "y", "S"], orient="row")
    grp = {i + 1: "C" for i in range(n_t)}
    lay, mode, s2, trace = faceoff.fit(d, grp, False, em_iters=30, em_tol=1e-4)
    est = np.array([mode.beta[lay.u[i + 1]] for i in range(n_t)])
    assert np.corrcoef(est, skill)[0, 1] > 0.9
    d = d.with_columns(pl.lit(1.0).alias("fr_home"), pl.lit(1.0).alias("fr_away"))
    lay2, mode2, _, _ = faceoff.fit(d, grp, True, s2=s2)
    meta = pl.DataFrame(
        {"player_id": list(range(1, n_t + 1)) * 3, "season": sorted([2021, 2022, 2023] * n_t), "grp": "C"}
    )
    eff = faceoff.effects(lay2, mode2, s2, d, meta)
    top = eff.group_by("player_id").agg(pl.col("fo_gaa").sum())
    best = np.argsort(skill)[-5:] + 1
    worst = np.argsort(skill)[:5] + 1
    assert (
        top.filter(pl.col("player_id").is_in(best.tolist()))["fo_gaa"].mean()
        > 0
        > top.filter(pl.col("player_id").is_in(worst.tolist()))["fo_gaa"].mean()
    )


# ------------------------------------------------------------------ penalties
def _pen_events(tmp_path, with_coincident=False):
    rows = []

    def ev(i, t, typ, team, hs=5, as_=5, strength="5v5", p1=None, p2=None, pmin=None, ptype=None):
        rows.append(
            {
                "game_id": 1,
                "event_idx": i,
                "season": 2024,
                "period": 1,
                "t_period": t,
                "type": typ,
                "team_id": team,
                "home_skaters": hs,
                "away_skaters": as_,
                "home_goalie": 1,
                "away_goalie": 1,
                "strength": strength,
                "player1": p1,
                "player2": p2,
                "penalty_min": pmin,
                "penalty_type": ptype,
            }
        )

    ev(1, 0, "faceoff", 10)
    ev(2, 100, "penalty", 10, p1=11, p2=21, pmin=2, ptype="MIN")  # team 10 (home) is penalized
    ev(3, 100, "faceoff", 20, hs=4, as_=5, strength="PP")
    ev(4, 110, "shot-on-goal", 20, hs=4, as_=5, strength="PP")
    ev(5, 120, "shot-on-goal", 10, hs=4, as_=5, strength="PK")
    ev(6, 150, "missed-shot", 20, hs=4, as_=5, strength="PP")
    # even strength again, but beyond the 120 s cap
    ev(7, 400, "faceoff", 10)
    if with_coincident:
        ev(8, 100, "penalty", 20, p1=22, p2=12, pmin=2, ptype="MIN")
    pl.DataFrame(rows).write_parquet(tmp_path / "events.parquet")
    pl.DataFrame(
        {"game_id": [1, 1, 1], "event_idx": [4, 5, 6], "team_id": [20, 10, 20], "xg": [0.10, 0.05, 0.20]}
    ).write_parquet(tmp_path / "shots_xg.parquet")


def test_penalty_cost_is_net_xg_for_the_advantaged_team_over_the_capped_power_play(tmp_path):
    _pen_events(tmp_path)
    pen = penalty.qualifying(tmp_path, {2: 120, 4: 240, 5: 300})
    assert pen.height == 1
    r = pen.row(0, named=True)
    assert r["net_xg"] == pytest.approx(0.10 + 0.20 - 0.05)  # (20's xG) - (10's xG) after the whistle
    # capped at the minor's length (even strength only at t=400)
    assert r["adv_s"] == 120.0
    assert (r["committer"], r["drawer"]) == (11, 21)


def test_coincident_penalties_are_excluded(tmp_path):
    _pen_events(tmp_path, with_coincident=True)
    assert penalty.qualifying(tmp_path, {2: 120, 4: 240, 5: 300}).height == 0


def test_gamma_poisson_shrinks_low_exposure_players_more_and_recovers_the_group_rate():
    rng = np.random.default_rng(1)
    n = 400
    e = rng.uniform(100, 1500, n)
    lam = rng.gamma(4.0, 0.01 / 4.0, n)  # mean 0.01 per minute, shape 4
    c = rng.poisson(lam * e).astype(float)
    d = pl.DataFrame(
        {"player_id": np.arange(n), "season": 2024, "pos_group": "F", "fringe": False, "toi_ev": e, "w": c}
    )
    out = penalty.gamma_poisson(d, "w", min_toi=500)
    post, raw = out["w_post"].to_numpy(), out["w_raw"].to_numpy()
    m = c.sum() / e.sum()
    lo, hi = e < 300, e > 1200
    assert np.mean(np.abs(post[lo] - m)) < np.mean(np.abs(raw[lo] - m))  # shrunk toward the group mean
    # less shrinkage with more exposure
    assert np.mean(np.abs(post[hi] - raw[hi])) < np.mean(np.abs(post[lo] - raw[lo]))
    assert 1.0 < float(out["alpha"][0]) < 1e4


def test_poisson_deviance_is_zero_at_the_fit_and_positive_elsewhere():
    n = np.array([0.0, 3.0, 5.0])
    assert penalty.deviance(n, np.array([1e-6, 3.0, 5.0]))[1:].sum() == pytest.approx(0.0, abs=1e-9)
    d = penalty.deviance(n, np.array([1.0, 1.0, 1.0]))
    assert np.all(d > 0) and d[0] == pytest.approx(2.0)


# ------------------------------------------------------------------ turnover
def _credits(tmp_path):
    def rows(scale):
        base = [
            ("giveaway", 1, -0.03),
            ("unobserved_loss", 1, -0.02),
            ("takeaway_against", 1, -0.01),
            ("takeaway", 2, 0.05),
            ("giveaway", 2, -0.04),
        ]
        out = []
        for comp, pid, v in base:
            out.append(
                {
                    "game_id": 1,
                    "season": 2024,
                    "poss_id": 1,
                    "team_id": 10,
                    "strength": "5v5",
                    "event_idx": 1,
                    "player_id": pid,
                    "component": comp,
                    "value": v * scale,
                    "phase": "terminal",
                }
            )
        out.append(
            {
                "game_id": 1,
                "season": 2024,
                "poss_id": 2,
                "team_id": 10,
                "strength": "PP",
                "event_idx": 2,
                "player_id": 1,
                "component": "giveaway",
                "value": -9.0,
                "phase": "terminal",
            }
        )
        return pl.DataFrame(out)

    a = rows(1.0)
    b = rows(1.0).with_columns(
        pl.when(pl.col("component") == "giveaway")
        .then(pl.col("value") - 0.02)
        .when(pl.col("component") == "takeaway_against")
        .then(pl.col("value") - 0.01)
        .when(pl.col("component") == "unobserved_loss")
        .then(pl.col("value") - 0.03)
        .when(pl.col("component") == "takeaway")
        .then(pl.col("value") + 0.04)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    a.write_parquet(tmp_path / "credits.parquet")
    b.write_parquet(tmp_path / "credits_lam1.parquet")
    pl.DataFrame({"game_id": [1], "onice_match_rate": [0.99]}).write_parquet(tmp_path / "shift_quality.parquet")


def test_turnover_charge_is_the_lambda_difference_of_the_three_charge_components_at_5v5(tmp_path):
    _credits(tmp_path)
    g = turnover.game_terms(tmp_path).sort("player_id")
    p1, p2 = (
        g.filter(pl.col("player_id") == 1).row(0, named=True),
        g.filter(pl.col("player_id") == 2).row(0, named=True),
    )
    # giveaway + unobserved loss + takeaway_against (PP row ignored)
    assert p1["to_total"] == pytest.approx(-0.02 - 0.03 - 0.01)
    assert p1["to_direct"] == pytest.approx(-0.02) and p1["to_pooled"] == pytest.approx(-0.04)
    # player 2: only a giveaway charge; the taker credit (+0.04) is excluded
    assert p2["to_total"] == pytest.approx(-0.02)


def test_arena_effects_recover_a_planted_effect_from_visitors_and_vanish_without_one(tmp_path):
    rng = np.random.default_rng(3)
    teams = list(range(1, 9))
    games, toi, terms = [], [], []
    gid = 0
    for _ in range(14):
        for h in teams:
            a = int(rng.choice([t for t in teams if t != h]))
            gid += 1
            games.append((gid, 2024, h))
            for team in (h, a):
                for k in range(12):
                    pid = team * 100 + k
                    toi.append((gid, pid, team, 900.0))
                    effect = 0.06 if h == 3 else 0.0  # arena 3 records extra charges
                    terms.append((gid, 2024, pid, (-0.2 - effect + rng.normal(0, 0.05)) * 900.0 / 3600.0))
    pl.DataFrame(games, schema=["game_id", "season", "home_id"], orient="row").write_parquet(tmp_path / "games.parquet")
    pl.DataFrame(toi, schema=["game_id", "player_id", "team_id", "toi_5v5"], orient="row").write_parquet(
        tmp_path / "toi_game.parquet"
    )
    pl.DataFrame({"game_id": list(range(1, gid + 1)), "onice_match_rate": 0.99}).write_parquet(
        tmp_path / "shift_quality.parquet"
    )
    gt = pl.DataFrame(terms, schema=["game_id", "season", "player_id", "to_total"], orient="row")
    arena, summary = turnover.arena_effects(tmp_path, gt)
    e = dict(zip(arena["home_id"], arena["arena_effect"]))
    # planted arena is extreme; effects are centred before shrinkage
    assert e[3] < -0.02 and abs(np.mean([e[t] for t in teams])) < 5e-4
    assert all(abs(e[t]) < abs(e[3]) / 2 for t in teams if t != 3)
    assert summary[2024]["arena_sd_excess"] > 0.01


def test_eb_rate_component_replacement_is_the_pooled_fringe_rate_and_gar_scales_with_toi():
    rng = np.random.default_rng(2)
    n = 60
    toi = rng.uniform(300, 1500, n)
    rate = rng.normal(-0.2, 0.04, n)
    d = pl.DataFrame(
        {
            "player_id": np.arange(n),
            "season": 2024,
            "pos_group": "F",
            "fringe": [i >= 45 for i in range(n)],
            "toi_ev": toi,
            "total": rate * toi / 60,
            "se60": 0.9 / np.sqrt(toi),
        }
    )
    out, rep = common.eb_rate_component(d, min_toi=500)
    fr = d.filter(pl.col("fringe"))
    assert rep["rep_per60"][0] == pytest.approx(fr["total"].sum() / fr["toi_ev"].sum() * 60)
    r = out.join(d.select("player_id", "toi_ev"), on="player_id").row(0, named=True)
    assert r["gar"] == pytest.approx((r["post_per60"] - r["rep_per60"]) * r["toi_ev"] / 60)


# ------------------------------------------------------------------ integration
def test_combine_adds_the_extra_components_to_the_total_and_converts_them_to_wins():
    ps = pl.DataFrame(
        {
            "player_id": [1, 2],
            "season": [2024, 2024],
            "player_name": ["A", "B"],
            "position": ["C", "D"],
            "pos_group": ["F", "D"],
            "primary_team": ["X", "X"],
            "toi_ev": [1000.0, 900.0],
            "gp": [80, 70],
            "pm_epa_per60_ev": [0.1, 0.0],
            "qualified_ev": [True, True],
        }
    )
    pm = pl.DataFrame({"player_id": [1, 2], "season": [2024, 2024], "pm_gar": [2.0, 0.5], "pm_gar_sd": [0.3, 0.4]})
    dp = pl.DataFrame({"player_id": [1, 2], "season": [2024, 2024], "def_gar": [-0.5, 1.5], "def_gar_sd": [0.4, 0.3]})
    fin = pl.DataFrame({"player_id": [1], "season": [2024], "fin_gar": [1.2]})
    sd = pl.DataFrame({"player_id": [1], "season": [2024], "fin_gar_sd": [0.5]})
    extras = {
        "to": pl.DataFrame(
            {"player_id": [1, 2], "season": [2024, 2024], "to_gar": [-0.4, 0.6], "to_gar_sd": [0.2, 0.2]}
        ),
        "fo": pl.DataFrame({"player_id": [1], "season": [2024], "fo_gar": [0.9], "fo_gar_sd": [0.3]}),
        "pen": pl.DataFrame({"player_id": [2], "season": [2024], "pen_gar": [0.3], "pen_gar_sd": [0.1]}),
    }
    t = cb.combine(ps, pm, dp, fin, sd, None, gpw=6.0, extras=extras).sort("player_id")
    assert t["total_gar"].to_list() == pytest.approx(
        [2.0 - 0.5 + 1.2 - 0.4 + 0.9 + 0.0, 0.5 + 1.5 + 0.0 + 0.6 + 0.0 + 0.3]
    )
    assert t["total_gar_sd"][0] == pytest.approx(np.sqrt(0.3**2 + 0.4**2 + 0.5**2 + 0.2**2 + 0.3**2))
    assert t["fo_wins"][0] == pytest.approx(0.9 / 6.0) and t["pen_wins"][1] == pytest.approx(0.3 / 6.0)
