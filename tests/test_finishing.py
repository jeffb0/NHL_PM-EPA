"""Finishing model: solver correctness, variance recovery, prior plumbing, joins, and the four-term identity."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import optimize, sparse
from scipy.special import expit

from src.finishing import glmm, model
from src.finishing.prior import CovariatePrior, PriorSpec
from src.finishing.shots import _context, add_v2_features
from tests.test_epa_attribution import o, run

ROOT = Path(__file__).resolve().parents[1]


def simulate(
    n_players=250,
    n_goalies=15,
    seasons=(2021, 2022, 2023),
    shots_per=90,
    sd_u=0.25,
    sd_d=0.10,
    sd_v=0.12,
    seed=0,
    z_effect=0.0,
):
    rng = np.random.default_rng(seed)
    pid = np.arange(1000, 1000 + n_players)
    gid = np.arange(5000, 5000 + n_goalies)
    pos = rng.choice(["F", "D"], n_players, p=[0.65, 0.35])
    z = rng.standard_normal(n_players)
    u = z_effect * z + sd_u * rng.standard_normal(n_players)
    v = sd_v * rng.standard_normal(n_goalies)
    rows = []
    for s in seasons:
        d = sd_d * rng.standard_normal(n_players)
        for i, p in enumerate(pid):
            n = rng.poisson(shots_per)
            g = rng.integers(0, n_goalies, n)
            xg = np.clip(rng.lognormal(-2.6, 0.7, n), 0.005, 0.6)
            eta = np.log(xg / (1 - xg)) + u[i] + d[i] + v[g]
            y = rng.random(n) < expit(eta)
            rows.append(
                pl.DataFrame(
                    {
                        "shooter": p,
                        "goalie": gid[g],
                        "season": s,
                        "pos_group": pos[i],
                        "goal": y.astype(np.int8),
                        "xg_f": xg,
                        "strength_g": rng.choice(["5v5", "PP"], n, p=[0.85, 0.15]),
                        "tg": rng.choice(["base", "slap", "back", "tip"], n),
                        "fringe": False,
                    }
                )
            )
    df = pl.concat(rows)
    return model.with_offset(df), dict(zip(pid, u)), dict(zip(gid, v)), pid, z


def test_newton_matches_generic_optimizer():
    rng = np.random.default_rng(1)
    n, m = 400, 12
    X = sparse.csr_matrix((rng.random((n, m)) < 0.3).astype(float))
    off = rng.normal(-2, 0.5, n)
    y = (rng.random(n) < expit(off + X @ rng.normal(0, 0.4, m))).astype(float)
    prec = np.full(m, 3.0)
    mode = glmm.newton(X, y, off, prec, tol=1e-10)

    def nll(b):
        eta = X @ b + off
        return -(y * eta - np.logaddexp(0, eta)).sum() + 0.5 * (prec * b**2).sum()

    ref = optimize.minimize(nll, np.zeros(m), method="BFGS", options={"gtol": 1e-9}).x
    assert mode.converged and np.abs(mode.beta - ref).max() < 1e-5


def test_em_recovers_variance_components_and_effects():
    df, u_true, v_true, pid, _ = simulate()
    fit = model.fit_finishing(df, model.Spec(fringe=False), em_iters=40, em_tol_sd=1e-4)
    assert np.sqrt(fit.s2["u_F"]) == pytest.approx(0.25, abs=0.07) and np.sqrt(fit.s2["u_D"]) == pytest.approx(
        0.25, abs=0.09
    )
    assert np.sqrt(fit.s2["d_F"]) == pytest.approx(0.10, abs=0.05)
    assert np.sqrt(fit.s2["v"]) == pytest.approx(0.12, abs=0.07)
    lay = fit.layout
    est = np.array([fit.beta[lay.shooter[p]] for p in pid])
    # ceiling: ~270 shots/player at p~0.08 -> reliability ~0.54 -> corr <= sqrt(0.54) ~ 0.73 (observed 0.73)
    assert np.corrcoef(est, [u_true[p] for p in pid])[0, 1] > 0.65


def test_squarem_em_reaches_a_fixed_point_of_the_plain_em_map():
    # weakly separated persistent / season variances make plain EM crawl (rate ~0.98); a small per-step change is
    # not convergence. After SQUAREM, one more plain EM step must move every standard deviation by < 5e-4 (logit).
    # (The simulation has no goalie-season effect, so that component sits at the boundary and approaches 0 slowly:
    # convergence must be judged on the absolute sd scale, not relatively.)
    df, *_ = simulate(sd_u=0.15, sd_d=0.12, seed=7)
    fit = model.fit_finishing(df, model.Spec(fringe=False), em_iters=40, em_tol_sd=1e-4)
    assert fit.trace[-1]["converged"]
    lay = fit.layout
    X = model.design(df, lay)
    mode = glmm.newton(
        X,
        df["goal"].to_numpy().astype(float),
        df["offset"].to_numpy(),
        glmm.precision_vector(lay.n_cols, lay.groups, fit.s2),
    )
    diag = np.diag(mode.posterior_cov())
    for g, idx in lay.groups.items():
        step = float(np.mean(mode.beta[idx] ** 2 + diag[idx]))
        assert abs(np.sqrt(step) - np.sqrt(fit.s2[g])) < 5e-4, g


def test_covariate_prior_learns_gamma_and_tightens_variance():
    df, _, _, pid, z = simulate(z_effect=0.3, sd_u=0.12, seed=3)
    plain = model.fit_finishing(df, model.Spec(fringe=False), em_iters=12, em_tol_sd=0.03)
    tab = pl.DataFrame({"player_id": pid, "zfeat": z})
    cov = model.fit_finishing(
        df, model.Spec(fringe=False), prior=CovariatePrior(table=tab, columns=["zfeat"]), em_iters=12, em_tol_sd=0.03
    )
    lo, hi = cov.layout.blocks["gamma"]
    assert cov.beta[lo] == pytest.approx(0.3, abs=0.08)
    assert cov.s2["u_F"] < 0.6 * plain.s2["u_F"]


def test_prior_default_has_no_covariates():
    X, names = PriorSpec().covariates([1, 2, 3])
    assert X.shape == (3, 0) and names == []


def test_unseen_players_and_seasons_get_zero_effects():
    df, *_ = simulate(n_players=40, n_goalies=5, seasons=(2021, 2022), shots_per=60)
    fit = model.fit_finishing(
        df, model.Spec(fringe=False), s2={"u_F": 0.05, "u_D": 0.05, "d_F": 0.01, "d_D": 0.01, "v": 0.01, "e": 0.01}
    )
    new = df.head(10).with_columns(
        pl.lit(999999).alias("shooter"), pl.lit(2030).alias("season"), pl.lit(888888).alias("goalie")
    )
    parts = fit.parts(new)
    for k in ("u", "d", "v", "e", "cell"):
        assert np.all(parts[k] == 0)


def test_cell_shift_equalizes_predicted_and_observed_goals():
    rng = np.random.default_rng(0)
    eta = rng.normal(-2.5, 1, 5000)
    y = (rng.random(5000) < expit(eta + 0.2)).astype(float)
    cell = rng.integers(0, 3, 5000)
    sh = model.calibrate_cell_shift(eta, y, cell)
    for c in range(3):
        m = cell == c
        assert expit(eta[m] + sh[m]).sum() == pytest.approx(y[m].sum(), abs=1e-3)


def test_effects_sum_to_zero_goals_at_league_average_and_fringe_reference():
    df, *_ = simulate(n_players=60, n_goalies=6, seasons=(2022,), shots_per=80, seed=5)
    df = df.with_columns((pl.col("shooter") % 5 == 0).alias("fringe"))
    fit = model.fit_finishing(
        df, model.Spec(fringe=True, season_dev=False, goalie_season=False), s2={"u_F": 0.05, "u_D": 0.05, "v": 0.01}
    )
    eff = model.player_effects(fit, df, draws=0, seed=0)
    assert set(["fin_gaa_5v5", "fin_gar_5v5", "attempts_5v5", "goals_5v5", "xg_f_5v5"]) <= set(eff.columns)
    assert eff["attempts_5v5"].sum() == df.filter(pl.col("strength_g") == "5v5").height
    fr = set(df.filter(pl.col("fringe"))["shooter"].unique().to_list())
    # by construction the fringe pool sits at the replacement reference (its own weighted mean shift) -> ~0 goals above
    # replacement
    pool = eff.filter(pl.col("player_id").is_in(list(fr)))
    assert abs(pool["fin_gar_5v5"].sum()) < 0.05 * max(pool["goals_5v5"].sum(), 1)


def test_context_features_on_a_toy_event_stream():
    ev = pl.DataFrame(
        {
            "game_id": [1] * 6,
            "event_idx": [1, 2, 3, 4, 5, 6],
            "period": [1] * 6,
            "type": ["faceoff", "shot-on-goal", "missed-shot", "faceoff", "shot-on-goal", "hit"],
            "t_period": [0, 5, 8, 100, 130, 131],
            "team_id": [10, 10, 10, 20, 10, 20],
        }
    )
    c = _context(ev).sort("event_idx")
    assert c["secs_since_fo"].to_list() == [0, 5, 8, 0, 30, 31]
    assert c.filter(pl.col("event_idx") == 3)["team_att_10s"][0] == 1  # the 5 s shot is within 10 s
    assert c.filter(pl.col("event_idx") == 5)["team_att_10s"][0] == 0  # earlier shots are > 10 s old


def test_v2_features_use_owner_of_previous_event():
    s = pl.DataFrame(
        {
            "team_id": [10, 10, 10],
            "prev_team_id": [10, 20, None],
            "x": [70.0, 70.0, 70.0],
            "y": [-5.0, -5.0, -5.0],
            "prev_x_s": [50.0, 50.0, None],
            "prev_y_s": [20.0, 20.0, None],
            "dt_prev": [2.0, 2.0, 2.0],
        }
    )
    r = add_v2_features(s)
    assert r["prev_same"].to_list() == [True, False, False]
    assert r["lat_same"].to_list() == [25.0, 0.0, 0.0]
    assert r["cross_slot"].to_list() == [True, False, False]


@pytest.mark.skipif(not (ROOT / "reports/metrics/fin_shots_checks.json").exists(), reason="finishing shots not built")
def test_every_goal_is_in_the_finishing_table_or_a_documented_exclusion():
    c = json.loads((ROOT / "reports/metrics/fin_shots_checks.json").read_text())
    assert c["goals_unaccounted"] == 0 and c["goals_on_blocked_rows"] == 0
    assert c["rows_duplicate_keys"] == 0 and c["rows_missing_xg_v1"] == 0 and c["goalie_ids_not_goalies"] == 0
    assert (
        c["goals_in_events"]
        == c["goals_without_coordinates"]
        + c["goals_empty_net"]
        + c["goals_by_goalie_shooters"]
        + c["goals_in_finishing_table"]
    )


def test_four_term_identity_goal_minus_vdec_splits_into_pm_epa_quality_context_and_finishing():
    # methodology gate F6: goal - V_dec = (V_att - V_dec) + (xG - V_att) + (xG_f - xG) + (goal - xG_f)
    rows = [o(1, 0, "start", "takeaway", 1, 100, 0.020, v_cf=0.012), o(1, 1, "end", "shot", 1, 106, 0.060, v_att=0.075)]
    _, cr, _ = run(rows, [{"poss_id": 1, "end_type": "shot_sog", "G": 0.08}], lam=0.0)  # G = xG of the shot
    attempt = cr.filter(pl.col("component") == "shot_attempt")["value"].sum()
    quality = cr.filter(pl.col("component") == "shot_quality")["value"].sum()
    xg, xg_f, goal, v_dec = 0.08, 0.095, 1.0, 0.060
    assert attempt + quality + (xg_f - xg) + (goal - xg_f) == pytest.approx(goal - v_dec, abs=1e-12)
    goal = 0.0
    assert attempt + quality + (xg_f - xg) + (goal - xg_f) == pytest.approx(goal - v_dec, abs=1e-12)
