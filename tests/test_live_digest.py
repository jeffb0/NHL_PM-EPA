import polars as pl

from src.live.digest import VERDICT, cumulative, html_impact, html_profit_chart


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
    p = pl.DataFrame({"team": ["AAA", "AAA", "BBB"], "name": ["X. Low", "Y. High", "Z. Solo"], "position": ["C", "D", "L"], "on_xgf_5v5": [0.2, 1.0, 0.5], "on_xga_5v5": [0.9, 0.3, 0.5],
                      "on_xgd_5v5": [-0.7, 0.7, 0.0], "own_xg_5v5": [0.1, 0.4, 0.2], "mates_xg_5v5": [0.1, 0.6, 0.3]})
    page = html_impact(p, "AAA 1 at BBB 2 (REG)")
    assert page.index("Y. High") < page.index("X. Low") < page.index("Z. Solo")
    assert "+0.70" in page and "-0.70" in page and "<script" not in page and "\u2588" in page
    assert len(page) < 1500                                                                # small: it is passed through the mail tool as text
