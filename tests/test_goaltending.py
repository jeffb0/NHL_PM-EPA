import numpy as np
import polars as pl

from src.goaltending import model as gm


def _shots(n_per=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    good, bad = 0, 1
    for season in (2018, 2019):
        for g, eff in ((good, -0.3), (bad, 0.3), (2, 0.0)):
            xg = rng.uniform(0.03, 0.2, n_per)
            p = 1 / (1 + np.exp(-(np.log(xg / (1 - xg)) + eff)))
            for i in range(n_per):
                rows.append(
                    {
                        "game_id": int(rng.integers(0, 200)),
                        "season": season,
                        "goalie": g,
                        "team_id": g,
                        "goal": int(rng.random() < p[i]),
                        "strength_g": "5v5",
                        "offset": float(np.log(xg[i] / (1 - xg[i]))),
                        "gfringe": False,
                    }
                )
    return pl.DataFrame(rows)


def test_model_recovers_goalie_effects_and_probabilities_are_ordered():
    df = _shots()
    lay = gm.build_layout(df, team_effect=False, fringe=True)
    mode, _, _ = gm.fit(df, lay, s2={"u": 0.05, "d": 0.01})
    p = gm.shot_probs(df, lay, mode.beta)
    saved = p["base"] - p["full"]
    g = df.with_columns(pl.Series("saved", saved)).group_by("goalie").agg(pl.col("saved").sum())
    s = dict(g.iter_rows())
    # the goalie with the negative effect saves goals, the other allows them
    assert s[0] > 0 > s[1]
    assert np.all(p["with_status"] > 0) and np.all(p["rep"] >= 0)


def test_returning_goalie_keeps_his_career_effect():
    # the good goalie (0) sits out the last season: he is absent from its shots but keeps a rate from his effect
    df = _shots(2000).filter(~((pl.col("season") == 2019) & (pl.col("goalie") == 0)))
    lay = gm.build_layout(df, team_effect=False, fringe=True)
    mode, _, _ = gm.fit(df, lay, s2={"u": 0.05, "d": 0.01})
    back = gm.returning_rates(df.filter(pl.col("season") == 2019), lay, mode.beta)
    assert back["goalie"].to_list() == [0]
    assert back["rate_v"][0] > 0.005  # saves goals relative to an average goalie


def test_unseen_ids_get_zero_effect():
    df = _shots(500)
    lay = gm.build_layout(df, team_effect=True)
    new = df.head(50).with_columns(pl.lit(99).alias("goalie"), pl.lit(2030).alias("season"))
    X = gm.design(new, lay, cell_season=pl.lit(2019))
    assert (
        X.shape == (50, lay.n_cols)
        and X[:, lay.blocks["u"][0] : lay.blocks["u"][1]].sum() == 0
        and X[:, lay.blocks["d"][0] : lay.blocks["d"][1]].sum() == 0
    )
