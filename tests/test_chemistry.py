import numpy as np
import polars as pl

from src.chemistry.model import design, fit_additive, fit_fm, objective, predict, roster_synergy


def _toy(n=4000, P=30, k=2, seed=0, synergy=True):
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + P)
    beta, V = rng.normal(0, 0.3, P), rng.normal(0, 0.5, (P, k))
    rows = []
    for g in range(n):
        pick = rng.choice(P, 10, replace=False)
        h, a = pick[:5], pick[5:]
        syn = lambda L: 0.5 * ((V[L].sum(0) ** 2).sum() - (V[L] ** 2).sum())  # noqa: E731
        y = beta[h].sum() - beta[a].sum() + (syn(h) - syn(a) if synergy else 0.0) + rng.normal(0, 0.3)
        rows.append({"game_id": g // 40, "season": 2020, "minutes": 1.0, "home_on": ids[h].tolist(), "away_on": ids[a].tolist(), "zone_OZ": 0.0, "zone_DZ": 0.0, "zone_NZ": 0.0, "score": 0.0, "y": float(y)})
    return pl.DataFrame(rows), ids, beta, V


def test_gradient_matches_finite_differences():
    d, _, _, _ = _toy(n=300, P=14)
    D, players, centre = design(d)
    rng = np.random.default_rng(1)
    theta = rng.normal(0, 0.1, 5 + len(players) * 3)
    val, grad = objective(theta, D, 2, 5.0, 7.0)
    for j in rng.choice(len(theta), 12, replace=False):
        t = theta.copy()
        t[j] += 1e-6
        assert abs((objective(t, D, 2, 5.0, 7.0)[0] - val) / 1e-6 - grad[j]) < 1e-4


def test_fm_recovers_planted_synergy_and_scores_unseen_data():
    d, _, _, V = _toy()
    tr, te = d.head(3000), d.tail(1000)
    D, players, centre = design(tr)
    add = fit_additive(D, players, centre, 1.0)
    fm = fit_fm(D, add, 2, 1.0, 1.0)
    Dt, _, _ = design(te, players, centre)
    err = lambda p: float(((Dt.y - predict(Dt, p)) ** 2).mean())  # noqa: E731
    assert err(fm) < 0.5 * err(add)                                   # interactions carry to unseen stints
    G_true, G_fit = V @ V.T, fm.V @ fm.V.T
    iu = np.triu_indices(len(players), 1)
    assert np.corrcoef(G_true[iu], G_fit[iu])[0, 1] > 0.9             # pair synergies are recovered (traits only up to rotation)


def test_roster_synergy_matches_pair_sum():
    rng = np.random.default_rng(0)
    V, pi = rng.normal(size=(6, 3)), rng.random(6)
    direct = sum(pi[i] * pi[j] * V[i] @ V[j] for i in range(6) for j in range(i + 1, 6))
    assert abs(roster_synergy(V, pi) - direct) < 1e-10
