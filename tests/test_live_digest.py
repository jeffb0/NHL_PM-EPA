import polars as pl

from src.live.digest import VERDICT, _segments, cumulative, html_impact, html_profit_chart


def test_verdicts_cover_the_four_outcomes():
    assert VERDICT[(True, True)][0] == "deserved" and VERDICT[(True, False)][0] == "fortunate"
    assert VERDICT[(False, True)][0] == "unlucky" and VERDICT[(False, False)][0] == "outplayed"


def test_cumulative_profit_is_by_date_and_ignores_unsettled_picks():
    s = pl.DataFrame({"game_date": ["2026-10-04", "2026-10-04", "2026-10-05", "2026-10-06"], "result": ["won", "lost", "lost", None], "profit": [0.8, -1.0, -2.0, None], "units": [1, 1, 2, 1]})
    c = cumulative(s)
    assert c["game_date"].to_list() == ["2026-10-04", "2026-10-05"]
    assert [round(x, 2) for x in c["cum"].to_list()] == [-0.2, -2.2]
    page = html_profit_chart(c)
    assert "10-04" in page and "-2.2" in page and "<img" not in page                      # drawn with table cells, no image


def test_impact_chart_lists_each_team_sorted_by_net():
    p = pl.DataFrame({"team": ["AAA", "AAA", "BBB"], "name": ["X. Low", "Y. High", "Z. Solo"], "position": ["C", "D", "L"],
                      "es_own": [0.1, 0.4, 0.2], "es_mates": [0.1, 0.6, 0.3], "pp_for": [0.0, 0.3, 0.0], "sh_for": [0.25, 0.0, 0.0],
                      "es_against": [0.5, 0.3, 0.5], "pk_against": [0.4, 0.0, 0.0], "ppa_against": [0.0, 0.1, 0.0],
                      "es_danger_for": [0.0, 0.1, 0.0], "es_danger_against": [0.2, 0.0, 0.0]})
    p = p.with_columns((pl.col("es_own") + pl.col("es_mates") + pl.col("es_danger_for") + pl.col("pp_for") + pl.col("sh_for")).alias("xgf_all"),
                       (pl.col("es_against") + pl.col("es_danger_against") + pl.col("pk_against") + pl.col("ppa_against")).alias("xga_all"))
    p = p.with_columns((pl.col("xgf_all") - pl.col("xga_all")).alias("net_all"), pl.Series("adj_pp", [0.0, 0.2, 0.0]), pl.Series("adj_pk", [-0.1, 0.0, 0.0]))
    p = p.with_columns((pl.col("net_all") * 0.5).alias("net_adj"))            # the email shows and sorts by the situation-adjusted net
    page = html_impact(p, "AAA 1 at BBB 2 (REG)")
    assert page.index("Y. High") < page.index("X. Low") < page.index("Z. Solo")
    assert "+0.50" in page and "-0.33" in page and "<script" not in page and "\u2588" in page and "\u2592" not in page and "#86b6ef" in page      # solid blocks; linemates in light blue
    assert all(lab in page for lab in ("PK", "PP", "EV-Shots", "EV-Play", "danger allowed without a shot", "net vs league average"))                 # the legend table
    assert len(page) < 3000                                                                # small: it is passed through the mail tool as text


def test_segments_keep_small_pieces_and_total_length():
    assert _segments([0.04, 0.04, 0.04], per=10.0, width=18) == [0, 1, 0]                  # rounded on the running total: 1.2 blocks in all
    assert sum(_segments([1.0, 0.5, 0.26], per=10.0, width=18)) == 18                       # capped at the bar width


def test_unconverted_danger_counts_only_even_strength_possessions_without_an_unblocked_shot():
    from src.live.game_report import unconverted_danger
    obs = pl.DataFrame({"poss_id": [1, 1, 1, 2, 2, 3, 3], "game_id": [9] * 7, "seq": [0, 1, 2, 0, 1, 0, 1], "team_id": [10, 10, 10, 20, 20, 10, 10],
                        "strength": ["5v5", "5v5", "5v5", "5v5", "5v5", "PP", "PP"], "v": [0.02, 0.15, 0.05, 0.03, 0.20, 0.02, 0.30],
                        "onice": [[1, 2]] * 7, "opp_onice": [[7, 8]] * 7})
    poss = pl.DataFrame({"poss_id": [1, 2, 3], "game_id": [9, 9, 9], "end_class": [2, 1, 2]})        # 2 = no unblocked shot; 1 = ended in one
    u = unconverted_danger(obs, poss)
    assert u["poss_id"].to_list() == [1] and abs(u["built"][0] - 0.13) < 1e-12 and u["opp_onice"][0].to_list() == [7, 8]   # the shot possession and the power play are excluded
