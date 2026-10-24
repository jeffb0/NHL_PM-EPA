from datetime import UTC, datetime

from src.live import afternoon as af
from src.live.digest import CONFIRMED, goalie_note, unconfirmed


def _row(**kw):
    r = {
        "game_id": 1,
        "away": "AAA",
        "home": "BBB",
        "p_home": 0.60,
        "p_away": 0.40,
        "home_goalie": "H. One",
        "home_goalie_source": CONFIRMED,
        "away_goalie": "A. One",
        "away_goalie_source": "lineup feed (expected)",
        "home_goalie_alt": None,
        "away_goalie_alt": "A. Two",
        "p_home_min": 0.55,
        "p_home_max": 0.60,
        "start_utc": "2026-10-13T23:00:00Z",
        "mkt_home": 0.52,
        "pick_best_home": -110,
        "pick_best_away": 100,
    }
    return {**r, **kw}


def test_unconfirmed_lists_only_open_sides():
    assert unconfirmed(_row()) == [("AAA", "A. One", "A. Two")]
    assert unconfirmed(_row(away_goalie_source=CONFIRMED)) == []


def test_goalie_note():
    # break-even at -110 is 52.4%: edge +7.6 as listed, +2.6 at worst -> goalie-dependent
    note, flagged = goalie_note(_row(), "home", -110)
    assert (
        flagged
        and note.startswith("GOALIE-DEPENDENT")
        and "A. One for AAA, or A. Two" in note
        and "+7.6%" in note
        and "+2.6%" in note
    )
    note, flagged = goalie_note(_row(p_home_min=0.58), "home", -110)
    assert not flagged and note.startswith("holds whichever goalie starts")
    assert goalie_note(_row(away_goalie_source=CONFIRMED), "home", -110) == ("both starting goalies confirmed", False)
    # rows logged before the range existed
    note, flagged = goalie_note(_row(p_home_min=None, p_home_max=None), "home", -110)
    assert not flagged and note.startswith("starter not confirmed")
    # a pick on the away side is judged at the top of the home range
    assert goalie_note(_row(p_home=0.40, p_away=0.60, p_home_min=0.40, p_home_max=0.47), "away", -110)[1]


def test_afternoon_selection_and_status():
    now = datetime(2026, 10, 13, 21, 0, tzinfo=UTC)
    rows = [_row(), _row(game_id=2, away_goalie_source=CONFIRMED), _row(game_id=3, start_utc="2026-10-13T20:00:00Z")]
    # game 2 was confirmed, game 3 has started
    assert [r["game_id"] for r in af.involved(__import__("polars").DataFrame(rows), now)] == [1]
    pick = {"side": "home", "pick": "BBB", "against": "AAA", "units": 2, "american": -110, "p_model": 0.60}
    assert "still qualifies" in af.pick_status(pick, _row())
    assert "NO LONGER QUALIFIES" in af.pick_status(pick, _row(p_home=0.54, p_away=0.46))
    assert "BBB qualifies now at 1.91" in af.pick_status(None, _row()) and "none qualifies now" in af.pick_status(
        None, _row(p_home=0.53, p_away=0.47)
    )


def test_afternoon_digest():
    text, page = af.build("2026-10-13", [], {}, {})
    assert text.startswith("NO UPDATE") and page == ""
    m, n = _row(), _row(away_goalie="A. Two", away_goalie_source=CONFIRMED, p_home=0.55, p_away=0.45)
    text, page = af.build("2026-10-13", [m], {1: n}, {})
    assert (
        text.startswith("Subject: NHL model 2026-10-13 goalie update: 1 of 1 games now confirmed")
        and "AAA: A. One -> A. Two" in text
        and "both confirmed" in text
        and "<table" in page
    )
