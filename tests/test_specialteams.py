import numpy as np
import polars as pl

from src.defense import ridge
from src.specialteams import model, stints


def test_group_of():
    assert stints.group_of(5, 4) == "adv" and stints.group_of(6, 5) == "adv" and stints.group_of(3, 6) == "adv"
    assert stints.group_of(4, 4) == "4v4" and stints.group_of(3, 3) == "3v3"
    assert stints.group_of(5, 5) is None and stints.group_of(6, 6) is None and stints.group_of(2, 5) is None


def _toy(n=60):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        adv_home = i % 2 == 0
        h = [int(x) for x in rng.choice(np.arange(1, 9), 5, replace=False)] if adv_home else [int(x) for x in rng.choice(np.arange(1, 9), 4, replace=False)]
        a = [int(x) for x in rng.choice(np.arange(11, 19), 4, replace=False)] if adv_home else [int(x) for x in rng.choice(np.arange(11, 19), 5, replace=False)]
        rows.append({"game_id": i // 3, "period": 1, "a": 0.0, "b": 1.0, "minutes": 1.0, "home_on": h, "away_on": a, "nh": len(h), "na": len(a), "zone_home": "fly",
                     "score_home": 0, "h_xg": 0.1, "a_xg": 0.05, "h_goals": 0.0, "a_goals": 0.0, "season": 2020, "grp": "adv"})
    return pl.DataFrame(rows)


def test_roles_and_design_shapes():
    st = _toy()
    f = model.as_stints_frame(st, "adv")
    # the advantaged side's players carry role 1, the disadvantaged side's role 2
    r0 = f.row(0, named=True)
    assert all(p % 10 == 1 for p in r0["home_on"]) and all(p % 10 == 2 for p in r0["away_on"])
    r1 = f.row(1, named=True)
    assert all(p % 10 == 2 for p in r1["home_on"]) and all(p % 10 == 1 for p in r1["away_on"])
    lookup = {(2020, p): (p % 3 == 0, p > 15) for p in list(range(1, 9)) + list(range(11, 19))}
    d = model.make_design(st, "adv", lookup)
    k = model.n_fringe_cols("adv")
    assert k == 8
    assert d.X.shape[0] == 2 * st.height
    # fringe counts are the last k columns and count at most the skaters on the ice
    fr = d.X[:, -k:].toarray()
    assert fr.min() >= 0 and fr.sum(1).max() <= 9


def test_ridge_recovers_role_specific_effect(monkeypatch):
    monkeypatch.setattr(ridge, "JITTER", 1e-6)   # the toy has empty covariate columns, as thin real states can
    st = _toy(2000)
    rng = np.random.default_rng(1)
    # player 1 (as advantaged-side skater) raises the advantaged side's xG rate by 3 per 60
    boost = st["home_on"].list.contains(1) & (st["nh"] > st["na"])
    y_home = np.where(boost.to_numpy(), 0.10 + 3 / 60, 0.10) + rng.normal(0, 0.02, st.height)
    st = st.with_columns(pl.Series("h_xg", y_home))
    lookup = {(2020, p): (False, False) for p in list(range(1, 9)) + list(range(11, 19))}
    d = model.make_design(st, "adv", lookup)
    beta = ridge.fit(d, {"xg": 1.0})
    coefs = ridge.coef_frame(d, beta).filter(pl.col("player_id") == 11)     # pseudo id 1 * 10 + role 1
    assert coefs["off_xg"][0] > 1.0
