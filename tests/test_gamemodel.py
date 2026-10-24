import numpy as np

from src.gamemodel.model import logit_fit, logit_pred, logloss, skater_update
from src.gamemodel.run import paired


def test_skater_update_moves_team_strength_by_the_step():
    theta = {1: 0.2, 2: -0.1, 3: 0.0}
    ids, mins = np.array([1, 2, 3]), np.array([20.0, 15.0, 10.0])
    before = sum(theta[i] * m / 60 for i, m in zip(ids, mins))
    skater_update(theta, ids, mins, 0.3)
    after = sum(theta[i] * m / 60 for i, m in zip(ids, mins))
    assert abs((after - before) - 0.3) < 1e-12
    assert theta[1] - 0.2 > theta[3] - 0.0 > 0  # more ice time, larger move


def test_logit_recovers_coefficients_and_paired_sign():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 1))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.3 + 0.8 * x[:, 0])))).astype(float)
    b = logit_fit(x, y)
    assert abs(b[0] - 0.3) < 0.06 and abs(b[1] - 0.8) < 0.06
    good, flat = logloss(logit_pred(b, x), y), logloss(np.full(len(y), y.mean()), y)
    dates = np.repeat(np.arange(200), 100)
    r = paired(good, flat, dates)
    assert r["z"] > 2 and r["z_cluster_date"] > 2 and r["mean_diff"] > 0


def test_experience_weighted_update_keeps_the_team_step_and_moves_rookies_more():
    theta = {1: 0.0, 2: 0.0}
    ids, mins = np.array([1, 2]), np.array([15.0, 15.0])
    skater_update(theta, ids, mins, 0.2, gain=np.array([3.0, 1.0]))
    assert abs(sum(theta[i] * m / 60 for i, m in zip(ids, mins)) - 0.2) < 1e-12
    assert abs(theta[1] / theta[2] - 3.0) < 1e-12
