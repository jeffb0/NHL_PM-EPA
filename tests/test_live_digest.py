import polars as pl

from src.live.digest import VERDICT, _segments, cumulative, html_impact, html_profit_chart


def test_verdicts_cover_the_four_outcomes():
    assert VERDICT[(True, True)][0] == "deserved" and VERDICT[(True, False)][0] == "fortunate"
    assert VERDICT[(False, True)][0] == "unlucky" and VERDICT[(False, False)][0] == "outplayed"


def test_cumulative_profit_is_by_date_and_ignores_unsettled_picks():
    s = pl.DataFrame(
        {
            "game_date": ["2026-10-04", "2026-10-04", "2026-10-05", "2026-10-06"],
            "result": ["won", "lost", "lost", None],
            "profit": [0.8, -1.0, -2.0, None],
            "units": [1, 1, 2, 1],
        }
    )
    c = cumulative(s)
    assert c["game_date"].to_list() == ["2026-10-04", "2026-10-05"]
    assert [round(x, 2) for x in c["cum"].to_list()] == [-0.2, -2.2]
    page = html_profit_chart(c)
    # drawn with table cells, no image
    assert "10-04" in page and "-2.2" in page and "<img" not in page


def test_impact_chart_lists_each_team_sorted_by_net():
    p = pl.DataFrame(
        {
            "team": ["AAA", "AAA", "BBB"],
            "name": ["X. Low", "Y. High", "Z. Solo"],
            "position": ["C", "D", "L"],
            "ev_off": [0.3, 1.1, 0.5],
            "ev_def": [-0.9, -0.3, -0.5],
            "pp_off": [0.0, 0.2, 0.0],
            "pp_def": [0.0, -0.05, 0.0],
            "pk_off": [0.2, 0.0, 0.0],
            "pk_def": [-0.15, 0.0, 0.0],
            "pen": [0.0, 0.0, -0.175],
        }
    )
    p = p.with_columns(
        pl.sum_horizontal("ev_off", "ev_def", "pp_off", "pp_def", "pk_off", "pk_def", "pen").alias("net_adj")
    )
    page = html_impact(p, "AAA 1 at BBB 2 (REG)")
    assert page.index("Y. High") < page.index("X. Low") < page.index("Z. Solo")
    # solid offense, striped defense
    assert "+0.95" in page and "-0.55" in page and "<script" not in page and "\u2588" in page and "\u258c" in page
    assert all(
        lab in page
        for lab in ("EV-Offense", "EV-Defense", "PP-Offense", "PP-Defense", "PK-Offense", "PK-Defense", "Penalties")
    )
    # small: it is passed through the mail tool as text
    assert len(page) < 3500


def test_penalties_credit_the_drawer_charge_the_taker_and_skip_offsetting_calls():
    from src.live.game_report import penalties

    ev = pl.DataFrame(
        {
            "game_id": [9] * 6,
            "period": [1, 1, 2, 2, 3, 3],
            "t_period": [100, 300, 50, 50, 400, 600],
            "type": ["penalty"] * 5 + ["faceoff"],
            "team_id": [10, 10, 10, 20, 20, 20],
            "player1": [1, 1, 2, 7, None, 7],
            "player2": [7, 8, 8, 2, None, 1],
            "penalty_type": ["MIN", "MIN", "MIN", "MIN", "BEN", None],
            "penalty_min": [2, 4, 2, 2, 2, None],
        }
    )
    p = {r["player_id"]: r for r in penalties(ev, {2: 0.2, 4: 0.35}).iter_rows(named=True)}
    # player 1 took a minor and a double minor; the period-2 pair offsets; the bench minor names nobody
    assert p[1]["pen_taken"] == 2 and abs(p[1]["pen"] + 0.55) < 1e-12
    assert abs(p[7]["pen"] - 0.2) < 1e-12 and abs(p[8]["pen"] - 0.35) < 1e-12
    assert 2 not in p


def test_segments_keep_small_pieces_and_total_length():
    # rounded on the running total: 1.2 blocks in all
    assert _segments([0.04, 0.04, 0.04], per=10.0, width=18) == [0, 1, 0]
    assert sum(_segments([1.0, 0.5, 0.26], per=10.0, width=18)) == 18  # capped at the bar width


def test_unconverted_danger_counts_only_even_strength_possessions_without_an_unblocked_shot():
    from src.live.game_report import unconverted_danger

    obs = pl.DataFrame(
        {
            "poss_id": [1, 1, 1, 2, 2, 3, 3],
            "game_id": [9] * 7,
            "seq": [0, 1, 2, 0, 1, 0, 1],
            "team_id": [10, 10, 10, 20, 20, 10, 10],
            "strength": ["5v5", "5v5", "5v5", "5v5", "5v5", "PP", "PP"],
            "v": [0.02, 0.15, 0.05, 0.03, 0.20, 0.02, 0.30],
            "onice": [[1, 2]] * 7,
            "opp_onice": [[7, 8]] * 7,
        }
    )
    # 2 = no unblocked shot; 1 = ended in one
    poss = pl.DataFrame({"poss_id": [1, 2, 3], "game_id": [9, 9, 9], "end_class": [2, 1, 2]})
    u = unconverted_danger(obs, poss)
    # the shot possession and the power play are excluded
    assert u["poss_id"].to_list() == [1] and abs(u["built"][0] - 0.13) < 1e-12 and u["opp_onice"][0].to_list() == [7, 8]
